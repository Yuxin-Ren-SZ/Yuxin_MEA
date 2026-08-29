#!/usr/bin/env python
"""QC the stability of Network-scan electrode selection across a culture's life.

Every MaxTwo Network run re-picks the ~1,000 electrodes it routes out of 26,400,
so "the same well" on two dates need not mean the same recording sites. This
script measures that for each (sample, chip, well) and emits three independently
selectable deliverables:

* ``map``    — cumulative selection map: how many scans picked each electrode,
               on the true chip footprint, white where never selected;
* ``anim``   — the same map animated, current scan emphasised and past scans
               greyed by how many *days* separate them from it;
* ``report`` — ``report.md`` with the scan list, the stability metrics and a
               plain-language verdict.

Like ``run_activity_scan.py`` this stays *outside* the pipeline: the unit is one
well across many recordings, which the per-(rec, well) task queue cannot express,
and only h5 metadata is read, so it never needs preprocessing or sorting to have
run. Discovery is from ``experiment_cache.json``; nothing here writes to
``pipeline_cache.json``.

    # everything, default D0-D21 window, no animations
    python scripts/run_electrode_selection_qc.py --config pipeline_config_local.json

    # one well end to end, over every scan it has
    python scripts/run_electrode_selection_qc.py --config pipeline_config_local.json \
        --sample CX138 --well well000 --all-scans --mode map,anim,report

Exit: 0 ok (or dry-run), 1 some wells failed, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger("run_electrode_selection_qc")

INDEX_FILE = "selection_index.parquet"
SCANS_FILE = "scans.parquet"
SUMMARY_FILE = "stability_summary.csv"
INDEX_MD = "index.md"
MAP_STEM = "selection_count_map"
ANIM_STEM = "selection_history"
REPORT_FILE = "report.md"


# --------------------------------------------------------------------------- #
# Extraction (stage 1)
# --------------------------------------------------------------------------- #
def load_experiment_cache(analysis_root: Path) -> dict:
    path = Path(analysis_root) / "experiment_cache.json"
    try:
        with path.open() as fh:
            cache = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}")
    return cache.get("entries", cache)


def build_index(entries: dict, data_root: Path, out_root: Path, *,
                samples=None, plates=None, wells=None, refresh: bool = False):
    """Read (or reuse) the selection index. Returns ``(index_df, scans_df)``.

    The index is the expensive-ish part (~0.4 s per Network file) and the only
    part that touches the NAS, so it is cached: re-rendering figures with
    different windows or colour maps costs nothing after the first run. A cache
    built under a narrower ``--sample``/``--plate`` filter is reused only when
    it already covers what is being asked for now.
    """
    import pandas as pd

    from yuxin_mea.analysis import electrode_selection_qc as Q

    index_path = out_root / INDEX_FILE
    scans_path = out_root / SCANS_FILE
    if not refresh and index_path.exists() and scans_path.exists():
        try:
            index_df = pd.read_parquet(index_path)
            scans_df = pd.read_parquet(scans_path)
        except Exception as exc:  # noqa: BLE001 — a bad cache must not be fatal
            logger.warning("could not reuse %s (%s); re-extracting", index_path, exc)
        else:
            have = set(scans_df["sample_id"].unique())
            if not samples or samples <= have:
                logger.info("reusing cached index: %d scan-well rows", len(scans_df))
                return index_df, scans_df
            logger.info("cached index lacks %s; re-extracting",
                        ", ".join(sorted(samples - have)))

    t0 = time.time()
    index_df, scans_df = Q.extract_selection_index(
        entries, data_root, samples=samples, plates=plates, wells=wells)
    logger.info("extracted %d scan-well rows / %d electrode rows in %.0fs",
                len(scans_df), len(index_df), time.time() - t0)
    out_root.mkdir(parents=True, exist_ok=True)
    if len(index_df):
        index_df.to_parquet(index_path, index=False)
        scans_df.to_parquet(scans_path, index=False)
    return index_df, scans_df


# --------------------------------------------------------------------------- #
# Worker (stages 2-3 + rendering, one well)
# --------------------------------------------------------------------------- #
def run_one(job: dict) -> dict:
    """Compute + render one well. Pickle-safe (module level, plain dict arg)."""
    import numpy as np
    import pandas as pd

    from yuxin_mea.analysis import electrode_selection_inspector as VI
    from yuxin_mea.analysis import electrode_selection_qc as Q

    t0 = time.time()
    uid = job["well_uid"]
    sample_id, plate_id, well_id = Q.split_well_uid(uid)
    try:
        # Read only this well's slice; the whole index is millions of rows.
        idx = pd.read_parquet(job["index_path"],
                              filters=[("well_uid", "==", uid)],
                              columns=["recording_key", "electrode"])
        sets = {k: g["electrode"].to_numpy(dtype=np.int64)
                for k, g in idx.groupby("recording_key", sort=False)}

        all_scans = job["scans"]
        spec = Q.WindowSpec(job["window_kind"], job["window_lo"], job["window_hi"])
        tmap = Q.load_treatment_map(job["treatment_map"]) if job["treatment_map"] else {}
        raw_group, _seen = Q.well_groupname(all_scans)
        plating_meta = next((s.get("plating_date_meta") for s in all_scans
                             if s.get("plating_date_meta")), None)
        window = Q.resolve_window(spec, tmap, sample_id, plate_id, raw_group,
                                  plating_fallback=plating_meta)

        selected = Q.select_window(all_scans, window)
        selected, sets = Q.collapse_same_day(selected, job["collapse"], sets)
        if not selected:
            return {"ok": False, "well_uid": uid, "elapsed_s": 0.0,
                    "error": "no scans fall inside the requested window"}

        res = Q.compute_stability(uid, selected, sets, window=window,
                                  window_spec=spec, min_scans=job["min_scans"],
                                  n_scans_total=len(all_scans))
        out_dir = Q.well_dir(job["out_root"], sample_id, plate_id, well_id)
        Q.write_well_qc(res, out_dir)
        payload = Q.load_well_qc(out_dir)

        modes = set(job["modes"])
        map_rel = anim_rel = None
        if "map" in modes:
            VI.render_count_map(payload, out_dir / MAP_STEM, cmap=job["cmap"],
                                formats=tuple(job["map_formats"]))
            map_rel = f"{MAP_STEM}.png"
        if "anim" in modes and job["animate"]:
            written = Q.render_history_animation(
                res, out_dir / f"{ANIM_STEM}.mp4", decay_days=job["decay_days"],
                fps=job["fps"], hold_frames=job["hold_frames"], fmt=job["format"])
            anim_rel = written.name
        if "report" in modes:
            hint = job["anim_hint"].format(sample=sample_id, plate=plate_id,
                                           well=well_id)
            (out_dir / REPORT_FILE).write_text(VI.report_markdown(
                payload, map_rel=map_rel, anim_rel=anim_rel,
                anim_hint=None if anim_rel else hint))

        s = res.stability
        return {
            "ok": True, "well_uid": uid, "elapsed_s": round(time.time() - t0, 1),
            "out_dir": str(out_dir), "n_scans": s["n_scans"],
            "verdict": s["verdict"], "anim": anim_rel,
            "stability": s, "window": res.window.as_dict(),
        }
    except Exception as exc:  # noqa: BLE001 — one bad well must not stop the sweep
        import traceback
        return {"ok": False, "well_uid": uid, "elapsed_s": round(time.time() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- #
# Roll-ups
# --------------------------------------------------------------------------- #
_SUMMARY_COLUMNS = [
    "well_uid", "sample_id", "plate_id", "well_id", "verdict", "n_scans",
    "n_scans_total", "n_routed_median", "union_electrodes", "union_fraction",
    "n_never_selected", "gini", "mean_jaccard_dedup", "mean_jaccard_raw",
    "adjacent_jaccard_dedup", "n_identical_pairs", "n_declared_base_pairs",
    "centroid_drift_um",
]


def _summary_row(result: dict, window: dict) -> dict:
    s = result["stability"]
    row = {k: s.get(k) for k in _SUMMARY_COLUMNS}
    row["core_50"] = (s.get("core_counts") or {}).get("0.5")
    row["core_80"] = (s.get("core_counts") or {}).get("0.8")
    row["core_100"] = (s.get("core_counts") or {}).get("1")
    row["core_80_fraction"] = (s.get("core_fraction") or {}).get("0.8")
    row["half_life_days_dedup"] = (s.get("lag_decay_dedup") or {}).get("half_life_days")
    row["half_life_days_raw"] = (s.get("lag_decay_raw") or {}).get("half_life_days")
    row["shift_null_enrichment"] = (s.get("shift_null") or {}).get("median_enrichment")
    row["shift_null_median_p"] = (s.get("shift_null") or {}).get("median_p")
    row["window_kind"] = window.get("kind")
    row["window_anchor"] = window.get("anchor")
    return row


def write_summary(out_root: Path, rows: list[dict]) -> tuple[Path, "object"] | None:
    """Merge this run's wells into the roll-up; return ``(path, merged_df)``.

    Merged, not replaced: a run scoped to ``--sample``/``--well``, or one resumed
    after a failure, must not drop the 143 wells an earlier full run recorded.
    Rows are keyed on ``well_uid`` so a recomputed well replaces its old row.
    """
    if not rows:
        return None
    import pandas as pd

    fresh = pd.DataFrame(rows)
    path = out_root / SUMMARY_FILE
    if path.exists():
        try:
            prior = pd.read_csv(path)
        except (OSError, ValueError) as exc:
            logger.warning("could not merge %s (%s); replacing it", path, exc)
        else:
            kept = prior[~prior["well_uid"].isin(set(fresh["well_uid"]))]
            fresh = pd.concat([kept, fresh], ignore_index=True)

    merged = fresh.sort_values(["sample_id", "plate_id", "well_id"])
    merged.to_csv(path, index=False)
    # The file can legitimately straddle windows — wells whose treatment date is
    # unknown fall back to "all" while their anchored neighbours stay on "tau" —
    # so report the composition rather than warning about a mix that is expected.
    composition = merged["window_kind"].value_counts().to_dict()
    if len(composition) > 1:
        logger.info("summary spans window kinds %s (see the window_kind and "
                    "window_anchor columns)", composition)
    return path, merged


def write_index_md(out_root: Path, rows: list, window_label: str) -> Path:
    """Plate-by-plate roll-up — the only view that shows whether one well is an
    outlier or the whole chip behaves the same way."""
    from yuxin_mea.analysis import electrode_selection_inspector as VI

    lines = ["# Electrode-selection QC", "",
             f"Window: **{window_label}** · {len(rows)} well(s).", ""]
    by_plate: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_plate.setdefault((r["sample_id"], r["plate_id"]), []).append(r)

    for (sample_id, plate_id), group in sorted(by_plate.items()):
        lines += [f"## {sample_id} · {plate_id}", ""]
        lines.append("| well | name | verdict | scans | routed (med) | union | "
                     "core ≥80% | mean J | half-life (d) | report |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(group, key=lambda x: x["well_id"]):
            rel = f"{sample_id}/{plate_id}/{r['well_id']}/{REPORT_FILE}"

            def _f(v, d=3):
                # Values round-tripped through the merged CSV arrive as float
                # NaN, not None, so both have to be caught.
                if v is None or (isinstance(v, float) and v != v):
                    return "—"
                return f"{v:.{d}f}" if isinstance(v, float) else str(v)

            lines.append(
                f"| {r['well_id']} | {VI.well_name(r['well_id'])} | "
                f"{r['verdict']} | {r['n_scans']} | {_f(r['n_routed_median'], 0)} | "
                f"{r['union_electrodes']} | {r['core_80']} | "
                f"{_f(r['mean_jaccard_dedup'])} | "
                f"{_f(r['half_life_days_dedup'], 1)} | [report]({rel}) |")
        lines.append("")

    path = out_root / INDEX_MD
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QC Network electrode selection across time, per well.")
    p.add_argument("--config", type=Path, default=Path("pipeline_config_local.json"),
                   help="pipeline config json (globals). Default pipeline_config_local.json")
    p.add_argument("--out-root", default=None,
                   help="Override <analysis_root>/electrode_selection_qc.")
    p.add_argument("--mode", default="map,report",
                   help="Comma list of map,anim,report. Default 'map,report'.")

    p.add_argument("--sample", default=None, help="Comma-separated sample ids.")
    p.add_argument("--plate", default=None, help="Comma-separated plate ids.")
    p.add_argument("--well", default=None,
                   help="Comma-separated well indices or ids ('0,1' or 'well000').")

    w = p.add_mutually_exclusive_group()
    w.add_argument("--tau-range", nargs=2, type=float, metavar=("LO", "HI"),
                   default=None,
                   help="Treatment-day window. Default 0 21.")
    w.add_argument("--date-range", nargs=2, type=int, metavar=("LO", "HI"),
                   default=None, help="Recording-date window, YYMMDD.")
    w.add_argument("--div-range", nargs=2, type=float, metavar=("LO", "HI"),
                   default=None, help="Days-in-vitro window.")
    w.add_argument("--all-scans", action="store_true",
                   help="Use every Network scan, ignoring the treatment window.")

    p.add_argument("--treatment-map", type=Path, default=None,
                   help="CSV carrying treatment/plating dates. Default: "
                        "treatment_map.csv beside the config, else in the cwd.")
    p.add_argument("--collapse-same-day", choices=["no", "first", "union"],
                   default="no",
                   help="Fold repeat runs on one date. Default 'no' (each run is a scan).")
    p.add_argument("--min-scans", type=int, default=3,
                   help="Below this many in-window scans the verdict is withheld.")

    p.add_argument("--cmap", default="RdYlGn",
                   help="Count-map colormap. Default RdYlGn (red->green, as "
                        "requested); 'viridis' is the colour-blind-safe swap.")
    p.add_argument("--format", choices=["mp4", "gif"], default="mp4",
                   help="Animation container. Default mp4.")
    p.add_argument("--decay-days", type=float, default=7.0,
                   help="Fade time-constant for past scans, in days. Default 7.")
    p.add_argument("--fps", type=int, default=12, help="Animation fps. Default 12.")
    p.add_argument("--hold-frames", type=int, default=12,
                   help="Frames each scan is held. Default 12 (~1 s at 12 fps).")
    p.add_argument("--anim-all", action="store_true",
                   help="Render animations for every well, not just the selected ones.")

    p.add_argument("--jobs", "-j", type=int, default=1, help="Worker processes.")
    p.add_argument("--limit", type=int, default=None, help="Stop after N wells.")
    p.add_argument("--refresh", action="store_true",
                   help="Re-extract the selection index instead of reusing it.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would run; write nothing.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def _split_csv(value):
    if not value:
        return None
    items = {s.strip() for s in str(value).split(",") if s.strip()}
    return items or None


def _well_filter(value):
    raw = _split_csv(value)
    if not raw:
        return None
    return {int(s.replace("well", "")) for s in raw}


def resolve_treatment_map(explicit: Path | None, config: Path) -> Path | None:
    """Locate ``treatment_map.csv``.

    It is gitignored, so it exists only in the checkout it was created in and is
    absent from every git worktree — resolving it purely from the cwd would
    silently leave every well unanchored and quietly turn a ``--tau-range`` run
    into an ``--all-scans`` one. Looked up beside the config first, since that is
    where the real one lives.
    """
    if explicit is not None:
        return explicit if explicit.exists() else None
    for candidate in (config.resolve().parent / "treatment_map.csv",
                      Path("treatment_map.csv")):
        if candidate.exists():
            return candidate
    return None


def _window_spec(args):
    from yuxin_mea.analysis.electrode_selection_qc import EXPERIMENT_RANGE, WindowSpec

    if args.all_scans:
        return WindowSpec("all", None, None)
    if args.date_range:
        return WindowSpec("date", args.date_range[0], args.date_range[1])
    if args.div_range:
        return WindowSpec("div", args.div_range[0], args.div_range[1])
    lo, hi = args.tau_range if args.tau_range else EXPERIMENT_RANGE
    return WindowSpec("tau", lo, hi)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    from yuxin_mea.analysis import electrode_selection_qc as Q
    from yuxin_mea.config import ConfigManager

    if not args.config.exists():
        logger.error("config not found: %s", args.config)
        return 2
    cm = ConfigManager()
    cm.load(args.config)
    data_root = cm.get_global("data_root")
    analysis_root = cm.get_global("analysis_root")
    if not data_root or not analysis_root:
        logger.error("data_root and analysis_root must be set in %s", args.config)
        return 2

    modes = {m.strip() for m in args.mode.split(",") if m.strip()}
    unknown = modes - {"map", "anim", "report"}
    if unknown:
        logger.error("unknown --mode value(s): %s", ", ".join(sorted(unknown)))
        return 2

    tmap_path = resolve_treatment_map(args.treatment_map, args.config)
    if tmap_path:
        logger.info("treatment map: %s", tmap_path)
    else:
        logger.warning("no treatment_map.csv found — treatment-day (tau) and DIV "
                       "windows cannot be anchored and will fall back to all "
                       "scans. Pass --treatment-map to point at it.")

    out_root = Q.output_root(analysis_root, args.out_root)
    samples = _split_csv(args.sample)
    plates = _split_csv(args.plate)
    wells = _well_filter(args.well)
    spec = _window_spec(args)

    entries = load_experiment_cache(Path(analysis_root))
    index_df, scans_df = build_index(entries, Path(data_root), out_root,
                                     samples=samples, plates=plates, wells=wells,
                                     refresh=args.refresh)
    if not len(scans_df):
        logger.warning("no Network scans in scope")
        return 0

    # The cached index may be broader than this run's filters.
    sel = scans_df
    if samples:
        sel = sel[sel["sample_id"].isin(samples)]
    if plates:
        sel = sel[sel["plate_id"].isin(plates)]
    if wells:
        sel = sel[sel["well_id"].isin({f"well{w:03d}" for w in wells})]
    if not len(sel):
        logger.warning("no wells match the requested filters")
        return 0

    # An animation is rendered only for explicitly named wells unless --anim-all:
    # the maps and reports are cheap, the MP4s are not.
    explicit = bool(samples or plates or wells)
    animate = "anim" in modes and (args.anim_all or explicit)
    if "anim" in modes and not animate:
        logger.info("no --sample/--plate/--well filter given: skipping animations "
                    "(pass --anim-all to render them for every well)")

    hint = ("python scripts/run_electrode_selection_qc.py --config "
            f"{args.config} --sample {{sample}} --plate {{plate}} "
            "--well {well} --mode anim")

    jobs = []
    for uid, group in sel.groupby("well_uid", sort=True):
        rows = group.sort_values(["date", "run_id"]).to_dict("records")
        jobs.append({
            "well_uid": uid,
            "scans": rows,
            "index_path": str(out_root / INDEX_FILE),
            "out_root": str(out_root),
            "window_kind": spec.kind, "window_lo": spec.lo, "window_hi": spec.hi,
            "treatment_map": str(tmap_path) if tmap_path else "",
            "collapse": args.collapse_same_day,
            "min_scans": args.min_scans,
            "modes": sorted(modes),
            "animate": animate,
            "cmap": args.cmap,
            "map_formats": ["png", "pdf"],
            "format": args.format,
            "decay_days": args.decay_days,
            "fps": args.fps,
            "hold_frames": args.hold_frames,
            "anim_hint": hint,
        })
    if args.limit:
        jobs = jobs[:args.limit]

    logger.info("%d well(s) in scope · window %s · modes %s%s",
                len(jobs), spec.label(), ",".join(sorted(modes)),
                "" if animate else " (no animations)")
    if args.dry_run:
        for j in jobs[:20]:
            print(f"[dry-run] would process {j['well_uid']} "
                  f"({len(j['scans'])} scan(s))")
        if len(jobs) > 20:
            print(f"[dry-run] … and {len(jobs) - 20} more")
        return 0

    counts: Counter[str] = Counter()
    failures: list[dict] = []
    summary: list[dict] = []
    t0 = time.time()

    def _record(r: dict) -> None:
        counts["ok" if r["ok"] else "failed"] += 1
        if r["ok"]:
            counts[f"verdict:{r['verdict']}"] += 1
            summary.append(_summary_row(r, r.get("window", {})))
        else:
            failures.append(r)
            logger.warning("FAILED %s: %s", r["well_uid"], r.get("error"))

    if args.jobs <= 1:
        for i, job in enumerate(jobs, 1):
            _record(run_one(job))
            if i % 10 == 0 or i == len(jobs):
                logger.info("%d/%d wells in %.0fs", i, len(jobs), time.time() - t0)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(run_one, job): job for job in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                job = futures[fut]
                try:
                    _record(fut.result())
                except Exception as exc:  # noqa: BLE001 — worker died outright
                    _record({"ok": False, "well_uid": job["well_uid"],
                             "elapsed_s": 0.0,
                             "error": f"worker died: {type(exc).__name__}: {exc}"})
                if i % 10 == 0 or i == len(jobs):
                    logger.info("%d/%d wells in %.0fs", i, len(jobs), time.time() - t0)

    if summary:
        path, merged = write_summary(out_root, summary)
        # Rebuilt from the merged set so a scoped run does not shrink the index.
        write_index_md(out_root, merged.to_dict("records"), spec.label())
        logger.info("summary: %s (%d well(s))", path, len(merged))

    logger.info("done in %.0fs: %s", time.time() - t0, dict(counts))
    for f in failures[:10]:
        logger.error("FAILED %s\n%s", f["well_uid"], f.get("traceback", f.get("error")))
    if len(failures) > 10:
        logger.error("… and %d more failures", len(failures) - 10)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
