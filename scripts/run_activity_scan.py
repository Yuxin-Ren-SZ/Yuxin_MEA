#!/usr/bin/env python
"""Compute per-electrode activity maps for ActivityScan recordings.

The 96 ActivityScan recordings in this dataset have never been processed: the
pipeline queue holds Network wells only. This script fills that gap *outside*
the pipeline, deliberately:

* ``PipelineManager.register_computation_task`` patches a NOT_RUN record into
  **every** existing entry, so registering an activity-scan task would attach one
  to all 1898+ Network wells; and ``cli/run.py``'s auto-queue enumerates
  ``h5_recordings`` (56 recs x 6 wells for an ActivityScan) and attaches *every*
  task, i.e. preprocessing + kilosort on checkerboard configurations.
* The unit is wrong for the pipeline anyway: one result covers a whole well
  *assembled across its 14-29 configurations*, not one (rec_name, well) pair.

So this writes to ``activity_scan_data/<recording_key>/<well_id>/`` and never
touches ``pipeline_cache.json``. It reads ``experiment_cache.json`` for discovery
(no NAS rescan) and stamps each well with the standard provenance sidecar, so the
outputs verify like any other artifact.

    python scripts/run_activity_scan.py --config pipeline_config_local.json --dry-run
    python scripts/run_activity_scan.py --config pipeline_config_local.json \
        --samples CX169 --jobs 11

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
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("run_activity_scan")

SCAN_TYPE = "ActivityScan"
DEFAULT_OUTPUT_SUBDIR = "activity_scan_data"
STATS_FILE = "stats.json"


@dataclass(frozen=True)
class WellJob:
    recording_key: str
    data_path: str
    well_index: int
    well_id: str
    out_dir: str


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def load_experiment_cache(analysis_root: Path) -> dict:
    path = Path(analysis_root) / "experiment_cache.json"
    try:
        with path.open() as fh:
            cache = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}")
    return cache.get("entries", cache)


def discover_jobs(
    entries: dict,
    data_root: Path,
    output_root: Path,
    *,
    recordings: set[str] | None = None,
    samples: set[str] | None = None,
    wells: set[int] | None = None,
) -> list[WellJob]:
    """One job per (ActivityScan recording, physical well) with a readable file."""
    jobs: list[WellJob] = []
    for key, entry in sorted(entries.items()):
        if not isinstance(entry, dict) or entry.get("scan_type") != SCAN_TYPE:
            continue
        if recordings and key not in recordings:
            continue
        if samples and entry.get("sample_id") not in samples:
            continue

        path = Path(data_root) / str(entry.get("data_path", ""))
        # One ActivityScan in the dataset is a 0-byte stub; opening it would kill
        # a worker mid-drain, so it is skipped here with a warning instead.
        try:
            size = path.stat().st_size
        except OSError as exc:
            logger.warning("skipping %s: %s", key, exc)
            continue
        if size == 0:
            logger.warning("skipping %s: zero-size %s", key, path)
            continue

        well_ids: set[int] = set()
        for well_list in (entry.get("h5_recordings") or {}).values():
            for w in well_list:
                try:
                    well_ids.add(int(str(w).replace("well", "")))
                except ValueError:
                    continue
        if not well_ids:
            logger.warning("skipping %s: no wells in h5_recordings", key)
            continue

        for w in sorted(well_ids):
            if wells and w not in wells:
                continue
            well_id = f"well{w:03d}"
            jobs.append(WellJob(
                recording_key=key,
                data_path=str(path),
                well_index=w,
                well_id=well_id,
                out_dir=str(Path(output_root) / key / well_id),
            ))
    return jobs


# --------------------------------------------------------------------------- #
# Freshness
# --------------------------------------------------------------------------- #
def _stamp_for(job: WellJob, params_hash_value: str) -> dict:
    from yuxin_mea.provenance import stat_sig

    sig = stat_sig(Path(job.data_path))
    return {
        "h5": {"method": "stat", "file_size": sig["size"], "mtime_ns": sig["mtime_ns"]},
        "config_hash": params_hash_value,
        "stamped_at": time.time(),
    }


def is_fresh(job: WellJob, params_hash_value: str) -> bool:
    """True when the well was already computed from this raw file and params."""
    from yuxin_mea.provenance import read_sidecar, sidecar_dir, stat_sig

    out = Path(job.out_dir)
    if not (out / STATS_FILE).exists():
        return False
    sidecar = read_sidecar(sidecar_dir(out))
    if not sidecar:
        return False
    if sidecar.get("config_hash") != params_hash_value:
        return False
    try:
        cur = stat_sig(Path(job.data_path))
    except OSError:
        return False
    h5 = sidecar.get("h5") or {}
    return (h5.get("file_size") == cur["size"]
            and h5.get("mtime_ns") == cur["mtime_ns"])


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def run_one(job: WellJob, params: dict, stamp: dict) -> dict:
    """Compute + write one well. Pickle-safe (module-level, plain args)."""
    from yuxin_mea.analysis.activity_scan import (
        ActivityScanConfig,
        compute_well_activity,
        write_activity_scan,
    )
    from yuxin_mea.provenance import sidecar_dir, write_sidecar

    t0 = time.time()
    try:
        cfg = ActivityScanConfig.from_task_params(params)
        results = compute_well_activity(job.data_path, job.well_index, cfg)
        out = write_activity_scan(results, job.out_dir)
        write_sidecar(sidecar_dir(out), {
            "recording_key": job.recording_key,
            "well_id": job.well_id,
            "task": "activity_scan",
            "output_path": str(out),
            **stamp,
        })
        return {
            "ok": True,
            "recording_key": job.recording_key,
            "well_id": job.well_id,
            "elapsed_s": round(time.time() - t0, 1),
            "n_scanned": results.stats.get("n_scanned"),
            "n_active": results.stats.get("n_active"),
        }
    except Exception as exc:  # noqa: BLE001 — one bad well must not stop the sweep
        import traceback
        return {
            "ok": False,
            "recording_key": job.recording_key,
            "well_id": job.well_id,
            "elapsed_s": round(time.time() - t0, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def write_manifest(output_root: Path, recording_key: str, results: list[dict],
                   params: dict) -> None:
    """Per-recording roll-up so the viewer can list wells without a NAS walk.

    Merged, not replaced: a run scoped to ``--wells`` or resumed after a failure
    must not drop the wells an earlier run recorded.
    """
    from yuxin_mea.provenance import params_hash

    out = Path(output_root) / recording_key
    out.mkdir(parents=True, exist_ok=True)
    target = out / "manifest.json"
    wells: dict = {}
    if target.exists():
        try:
            wells = dict((json.loads(target.read_text()).get("wells") or {}))
        except (OSError, ValueError):
            wells = {}
    for r in results:
        wells[r["well_id"]] = {k: v for k, v in r.items() if k != "traceback"}
    # anything already on disk that no run in this process touched
    for d in sorted(out.glob("well*")):
        if (d / STATS_FILE).exists() and d.name not in wells:
            wells[d.name] = {"ok": True, "well_id": d.name,
                             "recording_key": recording_key}

    payload = {
        "recording_key": recording_key,
        "written_at": time.time(),
        "config_hash": params_hash(params),
        "params": params,
        "wells": dict(sorted(wells.items())),
    }
    target.write_text(json.dumps(payload, indent=1, default=str))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-electrode activity maps for ActivityScan recordings.",
    )
    p.add_argument("--config", required=True, type=Path,
                   help="pipeline config json (globals + optional activity_scan block).")
    p.add_argument("--recordings", default=None,
                   help="Comma-separated recording keys to limit the scope.")
    p.add_argument("--samples", default=None,
                   help="Comma-separated sample ids (e.g. 'CX169').")
    p.add_argument("--wells", default=None,
                   help="Comma-separated well indices or ids (e.g. '0,1' or 'well000').")
    p.add_argument("--jobs", "-j", type=int, default=1,
                   help="Worker processes (one well each). Default 1.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many wells are computed.")
    p.add_argument("--force", action="store_true",
                   help="Recompute wells that are already up to date.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would run; compute and write nothing.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def _split_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    items = {s.strip() for s in value.split(",") if s.strip()}
    return items or None


def _well_filter(value: str | None) -> set[int] | None:
    raw = _split_csv(value)
    if not raw:
        return None
    return {int(s.replace("well", "")) for s in raw}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    from yuxin_mea.analysis.activity_scan import ActivityScanConfig
    from yuxin_mea.config import ConfigManager
    from yuxin_mea.provenance import params_hash

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

    # The activity_scan block is a plain config section, NOT a registered task —
    # ConfigManager reads any section, and staying unregistered is what keeps
    # this off the pipeline queue.
    file_params = cm.get_task_params("activity_scan")
    output_root = Path(file_params.get("output_root")
                       or Path(analysis_root) / DEFAULT_OUTPUT_SUBDIR)
    cfg = ActivityScanConfig.from_task_params(file_params)
    params = cfg.as_dict()
    phash = params_hash(params)

    entries = load_experiment_cache(Path(analysis_root))
    jobs = discover_jobs(
        entries, Path(data_root), output_root,
        recordings=_split_csv(args.recordings),
        samples=_split_csv(args.samples),
        wells=_well_filter(args.wells),
    )
    n_recordings = len({j.recording_key for j in jobs})
    logger.info("%d well(s) across %d ActivityScan recording(s) in scope",
                len(jobs), n_recordings)
    if not jobs:
        return 0

    todo = jobs if args.force else [j for j in jobs if not is_fresh(j, phash)]
    n_fresh = len(jobs) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    logger.info("%d up to date, %d to compute (params %s)", n_fresh, len(todo), phash[:12])

    if args.dry_run:
        for j in todo[:20]:
            print(f"[dry-run] would compute {j.recording_key}/{j.well_id}")
        if len(todo) > 20:
            print(f"[dry-run] … and {len(todo) - 20} more")
        print(f"Done. {len(todo)} well(s) would be computed; "
              f"{n_fresh} already up to date (dry-run; nothing written).")
        return 0

    counts: Counter[str] = Counter()
    failures: list[dict] = []
    by_recording: dict[str, list[dict]] = {}
    t0 = time.time()

    def _record(result: dict) -> None:
        counts["ok" if result["ok"] else "failed"] += 1
        by_recording.setdefault(result["recording_key"], []).append(result)
        if not result["ok"]:
            failures.append(result)
            logger.warning("FAILED %s/%s: %s", result["recording_key"],
                           result["well_id"], result.get("error"))

    if args.jobs <= 1:
        for i, job in enumerate(todo, 1):
            _record(run_one(job, params, _stamp_for(job, phash)))
            if i % 10 == 0 or i == len(todo):
                logger.info("%d/%d wells (%s) in %.0fs", i, len(todo),
                            dict(counts), time.time() - t0)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(run_one, job, params, _stamp_for(job, phash)): job
                       for job in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                job = futures[fut]
                try:
                    _record(fut.result())
                except Exception as exc:  # noqa: BLE001 — worker died outright
                    _record({"ok": False, "recording_key": job.recording_key,
                             "well_id": job.well_id, "elapsed_s": 0.0,
                             "error": f"worker died: {type(exc).__name__}: {exc}"})
                if i % 10 == 0 or i == len(todo):
                    logger.info("%d/%d wells (%s) in %.0fs", i, len(todo),
                                dict(counts), time.time() - t0)

    for recording_key, results in by_recording.items():
        try:
            write_manifest(output_root, recording_key, results, params)
        except OSError as exc:
            logger.warning("could not write manifest for %s: %s", recording_key, exc)

    logger.info("done in %.0fs: %s", time.time() - t0, dict(counts))
    for f in failures[:10]:
        logger.error("FAILED %s/%s\n%s", f["recording_key"], f["well_id"],
                     f.get("traceback", f.get("error")))
    if len(failures) > 10:
        logger.error("… and %d more failures", len(failures) - 10)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
