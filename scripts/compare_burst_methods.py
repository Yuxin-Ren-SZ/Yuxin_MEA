#!/usr/bin/env python3
"""Compare ML vs traditional network-burst detection, well by well.

The pipeline runs two independent burst detectors on the *same* curated spikes:

  * traditional -- ``compute_network_bursts`` (task ``burst_detection``), output
    tree ``analysis_root/burst_detection_data/.../burst_detection/``;
  * ML          -- ``compute_ml_bursts`` (task ``ml_burst_detection``), output
    tree ``analysis_root/ml_burst_data_umap/.../ml_burst_detection/``.

Nothing quantifies how their results agree or diverge. This script does, across
every well where *both* detectors completed. Discovery is authoritative: a
single ``pipeline_cache.json`` carries both tasks' status + ``output_path`` per
well, so we join on that rather than globbing two directory trees.

Comparable tier = ``network_bursts`` only. The ML detector emits a single event
level parked in the ``network_bursts`` field; the traditional detector's
``network_bursts`` is its gap-merged main-burst tier. (ML burstlets/superbursts
are empty by construction -- not compared.)

Two families of comparison, per well:

  1. Metric-level -- burst count, rate, mean duration, spikes-per-burst,
     participation, peak-synchrony for each method, plus paired delta / log2
     ratio. All derived straight from the ``network_bursts.pkl`` event table
     (authoritative; ``metrics.json`` is ``{}`` when a method finds no bursts).

  2. Temporal divergence -- greedy one-to-one interval pairing by IoU. Because
     two different algorithms essentially never emit bit-identical intervals,
     the divergence signal lives in *continuous* metrics (per-pair IoU
     distribution, boundary-offset distributions, time-coverage Jaccard); a
     strict "same burst" count at ``--overlap-threshold`` (default 0.9;
     1.0 = literal exact) is reported alongside as the near-identical rate.

Aggregate across wells: paired Wilcoxon signed-rank (ml vs trad), Cliff's delta,
median delta + bootstrap CI, Spearman correlation, Benjamini-Hochberg FDR.

Run::

    python scripts/compare_burst_methods.py --config pipeline_config.json

Outputs land under ``<figure_root>/burst_method_comparison/``: per_well.csv,
summary_stats.csv, report.html (self-contained; PNGs embedded as base64).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("compare_burst_methods")

TRAD_TASK = "burst_detection"
ML_TASK = "ml_burst_detection"
CUR_TASK = "auto_curation"  # not required, but both detectors depend on it

# Per-well scalar metrics compared between methods. Each: display -> reducer over
# the network_bursts event table (list of dicts). "rate_per_min" is special-cased
# because it needs the shared recording duration.
EVENT_METRICS: dict[str, str] = {
    "duration_s": "duration_s",
    "spikes_per_burst": "total_spikes",
    "participation": "participation",
    "peak_synchrony": "peak_synchrony",
}
# Order of metrics reported in the aggregate table / figures.
COMPARE_METRICS = ["count", "rate_per_min", "duration_s", "spikes_per_burst",
                   "participation", "peak_synchrony"]


# --------------------------------------------------------------------------- #
# Config / cache loading (mirrors scripts/visualize_bursts.py)
# --------------------------------------------------------------------------- #
def _load_config(config_path: Path) -> tuple[Path, Path]:
    with config_path.open() as fh:
        cfg = json.load(fh)
    g = cfg.get("global", cfg)
    analysis_root = Path(g["analysis_root"])
    figure_root = Path(g.get("figure_root") or analysis_root)
    return analysis_root, figure_root


def _load_pipeline_cache(analysis_root: Path) -> dict[str, Any]:
    path = analysis_root / "pipeline_cache.json"
    if not path.exists():
        raise SystemExit(f"pipeline_cache.json not found at {path}")
    with path.open() as fh:
        return json.load(fh)


def _load_experiment_cache(analysis_root: Path) -> dict[str, Any]:
    path = analysis_root / "experiment_cache.json"
    if not path.exists():
        logger.warning("experiment_cache.json not found at %s; groupnames -> '?'", path)
        return {}
    with path.open() as fh:
        return json.load(fh)


def _parse_plating_date(s: Any) -> datetime | None:
    """Parse a day-first dot/slash/dash date (e.g. '13.3.2026', '30.01.2026').

    Copied from scripts/well_recording_umap_overlay.py so this script stays
    standalone."""
    if s is None:
        return None
    s = str(s).strip()
    for sep in (".", "/", "-"):
        if sep in s:
            parts = s.split(sep)
            break
    else:
        return None
    if len(parts) != 3:
        return None
    try:
        d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if y < 100:
        y += 2000
    try:
        return datetime(y, m, d)
    except ValueError:
        return None


def _parse_yymmdd(s: str) -> datetime | None:
    """Parse a 6-digit recording date (e.g. '260327'). Copied from the overlay."""
    s = str(s).strip()
    if len(s) != 6 or not s.isdigit():
        return None
    try:
        return datetime.strptime("20" + s, "%Y%m%d")
    except ValueError:
        return None


def _compute_div(rec_date: str, plating_date: str | None) -> float:
    """Days in vitro = recording date - plating date, else NaN."""
    rec_dt = _parse_yymmdd(rec_date)
    plating = _parse_plating_date(plating_date)
    if rec_dt is None or plating is None:
        return np.nan
    return float((rec_dt - plating).days)


def _group_lookup(experiment_cache: dict) -> dict[tuple[str, str], dict[str, str | None]]:
    """{(recording_key, bare_well_id): {well_name, groupname, plating_date}}.

    experiment_cache wells are keyed by bare id ("well000"); groupname/well_name/
    Plating Date live under the per-well ``metadata`` sub-dict.
    """
    out: dict[tuple[str, str], dict[str, str | None]] = {}
    for rec_key, rec in experiment_cache.items():
        wells = rec.get("wells", {}) if isinstance(rec, dict) else {}
        for well_id, meta in wells.items():
            if not isinstance(meta, dict):
                continue
            wmeta = meta.get("metadata", {}) if isinstance(meta.get("metadata"), dict) else {}
            out[(rec_key, well_id)] = {
                "well_name": str(wmeta.get("well_name", "?")),
                "groupname": str(wmeta.get("groupname", "?")),
                "plating_date": str(wmeta.get("Plating Date", "")) or None,
            }
    return out


def _make_resolver(search_roots: list[Path]):
    def _resolve(rel: str | None) -> Path | None:
        if not rel:
            return None
        p = Path(rel)
        if p.is_absolute():
            return p if p.exists() else None
        for base in search_roots:
            cand = (base / p).resolve()
            if cand.exists():
                return cand
        return None
    return _resolve


def _read_event_table(pkl_path: Path) -> list[dict[str, Any]]:
    """Load one network_bursts.pkl into a list of clean scalar dicts.

    Copied from scripts/visualize_bursts.py so this script stays standalone.
    Returns [] for a missing / empty / malformed table.
    """
    if not pkl_path.exists():
        return []
    try:
        df = pd.read_pickle(pkl_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", pkl_path, exc)
        return []
    if df is None or df.empty or "start" not in df or "end" not in df:
        return []
    rows: list[dict[str, Any]] = []
    for r in df.to_dict(orient="records"):
        try:
            s = float(r["start"])
            e = float(r["end"])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(s) and np.isfinite(e)) or e <= s:
            continue
        clean: dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, (np.generic,)):
                v = v.item()
            if isinstance(v, float) and not np.isfinite(v):
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
        clean["start"], clean["end"] = s, e
        rows.append(clean)
    return rows


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
@dataclass
class WellRef:
    recording_key: str
    well_compound: str          # "rec0000/well000"
    rec_name: str
    well_id: str
    well_name: str
    groupname: str
    trad_dir: Path
    ml_dir: Path
    sample_id: str = "?"        # recording_key = sample/date/plate/scan/run
    rec_date: str = ""          # 6-digit YYMMDD recording date
    scan_type: str = "?"
    run_id: str = "?"           # "assay ID" -- no dedicated assay field exists
    plating_date: str | None = None

    @property
    def div(self) -> float:
        """Days in vitro = recording date - plating date, else NaN."""
        return _compute_div(self.rec_date, self.plating_date)


def discover_wells(
    pipeline_cache: dict[str, Any],
    group_lut: dict[tuple[str, str], dict[str, str]],
    resolve,
    limit: int | None = None,
) -> list[WellRef]:
    refs: list[WellRef] = []
    n_incomplete = n_unresolved = 0
    for entry in pipeline_cache.values():
        tasks = entry.get("tasks", {})
        if tasks.get(TRAD_TASK, {}).get("status") != "complete":
            n_incomplete += 1
            continue
        if tasks.get(ML_TASK, {}).get("status") != "complete":
            n_incomplete += 1
            continue
        compound = entry.get("well_id", "")
        if "/" not in compound:
            continue
        rec_name, well_id = compound.split("/", 1)
        trad_dir = resolve(tasks[TRAD_TASK].get("output_path"))
        ml_dir = resolve(tasks[ML_TASK].get("output_path"))
        if trad_dir is None or ml_dir is None:
            n_unresolved += 1
            continue
        meta = group_lut.get((entry["recording_key"], well_id), {})
        # recording_key = sample_id/date/plate_id/scan_type/run_id
        rk_parts = entry["recording_key"].split("/")
        sample_id = rk_parts[0] if len(rk_parts) > 0 else "?"
        rec_date = rk_parts[1] if len(rk_parts) > 1 else ""
        scan_type = rk_parts[3] if len(rk_parts) > 3 else "?"
        run_id = rk_parts[4] if len(rk_parts) > 4 else "?"
        refs.append(WellRef(
            recording_key=entry["recording_key"],
            well_compound=compound,
            rec_name=rec_name,
            well_id=well_id,
            well_name=meta.get("well_name", "?"),
            groupname=meta.get("groupname", "?"),
            trad_dir=trad_dir,
            ml_dir=ml_dir,
            sample_id=sample_id,
            rec_date=rec_date,
            scan_type=scan_type,
            run_id=run_id,
            plating_date=meta.get("plating_date"),
        ))
    logger.info("both-complete wells: %d (skipped %d incomplete, %d unresolvable paths)",
                len(refs), n_incomplete, n_unresolved)
    refs.sort(key=lambda r: (r.recording_key, r.well_compound))
    if limit:
        refs = refs[:limit]
    return refs


# --------------------------------------------------------------------------- #
# Per-well metrics + temporal divergence
# --------------------------------------------------------------------------- #
def _intervals(events: list[dict]) -> list[tuple[float, float]]:
    return [(float(e["start"]), float(e["end"])) for e in events]


def _recording_duration(trad_ev: list[dict], ml_ev: list[dict],
                         trad_dir: Path, ml_dir: Path) -> float | None:
    """Shared recording length (seconds). Prefer metrics.json (rate = count/dur);
    fall back to the span of observed events. Same recording feeds both methods,
    so either detector's estimate is valid."""
    for d in (trad_dir, ml_dir):
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        try:
            nb = json.loads(mp.read_text()).get("network_bursts", {})
        except Exception:  # noqa: BLE001
            continue
        cnt, rate = nb.get("count"), nb.get("rate")
        if cnt and rate:  # both truthy and rate > 0
            return float(cnt) / float(rate)
    ends = [e for _, e in _intervals(trad_ev) + _intervals(ml_ev)]
    return max(ends) if ends else None


def _method_summary(events: list[dict], total_dur: float | None) -> dict[str, float]:
    n = len(events)
    out: dict[str, float] = {"count": float(n)}
    out["rate_per_min"] = (n / total_dur * 60.0) if total_dur else np.nan
    for name, col in EVENT_METRICS.items():
        vals = [float(e[col]) for e in events if col in e and np.isfinite(e[col])]
        out[name] = float(np.mean(vals)) if vals else np.nan
    return out


def _pair_intervals(a: list[tuple[float, float]], b: list[tuple[float, float]]
                    ) -> list[tuple[int, int, float]]:
    """Greedy one-to-one pairing of overlapping intervals, highest IoU first.

    Returns [(idx_a, idx_b, iou), ...] for every overlapping best-pair (each
    index used at most once). n bursts per well ~ tens, so O(n_a*n_b) is fine.
    """
    cands: list[tuple[float, int, int]] = []
    for i, (as_, ae) in enumerate(a):
        for j, (bs, be) in enumerate(b):
            ov = min(ae, be) - max(as_, bs)
            if ov <= 0:
                continue
            union = max(ae, be) - min(as_, bs)
            iou = ov / union if union > 0 else 0.0
            cands.append((iou, i, j))
    cands.sort(reverse=True, key=lambda x: x[0])
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for iou, i, j in cands:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j, iou))
    return matched


def _coverage_jaccard(a: list[tuple[float, float]], b: list[tuple[float, float]]
                      ) -> float:
    """Time-coverage Jaccard = overlap-time / union-time over the two interval
    sets. Intervals within a method are disjoint (event tables are sorted,
    non-overlapping), so pairwise overlaps sum without double counting."""
    total_a = sum(e - s for s, e in a)
    total_b = sum(e - s for s, e in b)
    overlap = 0.0
    for as_, ae in a:
        for bs, be in b:
            ov = min(ae, be) - max(as_, bs)
            if ov > 0:
                overlap += ov
    union = total_a + total_b - overlap
    return (overlap / union) if union > 0 else np.nan


@dataclass
class MatchStats:
    n_overlapping: int
    iou_mean: float
    iou_median: float
    n_same: int
    n_trad_only: int
    n_ml_only: int
    event_jaccard: float
    coverage_jaccard: float
    start_offset_median: float
    end_offset_median: float
    matched_pairs: list[tuple[int, int, float]] = field(default_factory=list)


def temporal_divergence(trad_ev: list[dict], ml_ev: list[dict],
                        threshold: float) -> MatchStats:
    a = _intervals(trad_ev)   # trad
    b = _intervals(ml_ev)     # ml
    n_a, n_b = len(a), len(b)
    matched = _pair_intervals(a, b)
    ious = np.array([m[2] for m in matched], dtype=float)
    n_same = int(np.sum(ious >= threshold)) if ious.size else 0

    # boundary offsets over overlapping pairs (ml - trad)
    start_off = [b[j][0] - a[i][0] for i, j, _ in matched]
    end_off = [b[j][1] - a[i][1] for i, j, _ in matched]

    denom = n_a + n_b - n_same
    return MatchStats(
        n_overlapping=len(matched),
        iou_mean=float(np.mean(ious)) if ious.size else np.nan,
        iou_median=float(np.median(ious)) if ious.size else np.nan,
        n_same=n_same,
        n_trad_only=n_a - n_same,
        n_ml_only=n_b - n_same,
        event_jaccard=(n_same / denom) if denom > 0 else np.nan,
        coverage_jaccard=_coverage_jaccard(a, b),
        start_offset_median=float(np.median(start_off)) if start_off else np.nan,
        end_offset_median=float(np.median(end_off)) if end_off else np.nan,
        matched_pairs=matched,
    )


def build_per_well(refs: list[WellRef], threshold: float, min_bursts: int,
                   dump_matched: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    for k, ref in enumerate(refs):
        trad_ev = _read_event_table(ref.trad_dir / "network_bursts.pkl")
        ml_ev = _read_event_table(ref.ml_dir / "network_bursts.pkl")
        total_dur = _recording_duration(trad_ev, ml_ev, ref.trad_dir, ref.ml_dir)
        t_sum = _method_summary(trad_ev, total_dur)
        m_sum = _method_summary(ml_ev, total_dur)

        row: dict[str, Any] = {
            "recording_key": ref.recording_key,
            "rec_name": ref.rec_name,
            "well_id": ref.well_id,
            "well_name": ref.well_name,
            "sample_id": ref.sample_id,
            "assay_id": ref.run_id,
            "scan_type": ref.scan_type,
            "groupname": ref.groupname,
            "div": ref.div,
            "recording_dur_s": total_dur if total_dur else np.nan,
        }
        for m in COMPARE_METRICS:
            tv, mv = t_sum[m], m_sum[m]
            row[f"trad_{m}"] = tv
            row[f"ml_{m}"] = mv
            row[f"delta_{m}"] = mv - tv
            row[f"log2ratio_{m}"] = _safe_log2ratio(mv, tv)

        # Temporal divergence only meaningful when at least one method fires
        # enough events; still emit the row so metric deltas (incl. 0-vs-N) count.
        enough = (len(trad_ev) >= min_bursts) or (len(ml_ev) >= min_bursts)
        if enough:
            ms = temporal_divergence(trad_ev, ml_ev, threshold)
            row.update({
                "n_trad": len(trad_ev), "n_ml": len(ml_ev),
                "n_overlapping": ms.n_overlapping,
                "iou_mean": ms.iou_mean, "iou_median": ms.iou_median,
                "n_same": ms.n_same,
                "n_trad_only": ms.n_trad_only, "n_ml_only": ms.n_ml_only,
                "event_jaccard": ms.event_jaccard,
                "coverage_jaccard": ms.coverage_jaccard,
                "start_offset_median": ms.start_offset_median,
                "end_offset_median": ms.end_offset_median,
            })
            if dump_matched:
                for i, j, iou in ms.matched_pairs:
                    matched_rows.append({
                        "recording_key": ref.recording_key,
                        "well_id": ref.well_id, "well_name": ref.well_name,
                        "groupname": ref.groupname,
                        "trad_start": trad_ev[i]["start"], "trad_end": trad_ev[i]["end"],
                        "ml_start": ml_ev[j]["start"], "ml_end": ml_ev[j]["end"],
                        "iou": iou,
                        "start_offset": ml_ev[j]["start"] - trad_ev[i]["start"],
                        "end_offset": ml_ev[j]["end"] - trad_ev[i]["end"],
                    })
        else:
            for c in ("n_trad", "n_ml", "n_overlapping", "iou_mean", "iou_median",
                      "n_same", "n_trad_only", "n_ml_only", "event_jaccard",
                      "coverage_jaccard", "start_offset_median", "end_offset_median"):
                row[c] = np.nan
        rows.append(row)
        if (k + 1) % 200 == 0:
            logger.info("  processed %d/%d wells", k + 1, len(refs))
    return pd.DataFrame(rows), pd.DataFrame(matched_rows)


def _safe_log2ratio(mv: float, tv: float) -> float:
    eps = 1e-9
    if not (np.isfinite(mv) and np.isfinite(tv)):
        return np.nan
    return float(np.log2((mv + eps) / (tv + eps)))


# --------------------------------------------------------------------------- #
# Aggregate statistics
# --------------------------------------------------------------------------- #
def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(x>y) - P(x<y). x=ml, y=trad. +1 => ml always larger."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size == 0 or y.size == 0:
        return np.nan
    gt = lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return (gt - lt) / (x.size * y.size)


def _bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return q.tolist()
    ps = p[idx]
    order = np.argsort(ps)
    ranked = ps[order]
    n = ranked.size
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    q[idx] = out
    return q.tolist()


def _bootstrap_median_ci(diffs: np.ndarray, rng: np.random.Generator,
                         n_boot: int = 2000) -> tuple[float, float]:
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 3:
        return (np.nan, np.nan)
    boots = np.empty(n_boot)
    n = diffs.size
    for b in range(n_boot):
        boots[b] = np.median(diffs[rng.integers(0, n, n)])
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def aggregate_stats(per_well: pd.DataFrame, rng: np.random.Generator,
                    group: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pvals: list[float] = []
    for m in COMPARE_METRICS:
        tcol, mcol = f"trad_{m}", f"ml_{m}"
        sub = per_well[[tcol, mcol]].dropna()
        t = sub[tcol].to_numpy(float)
        ml = sub[mcol].to_numpy(float)
        n = t.size
        diffs = ml - t
        # Wilcoxon needs at least one nonzero difference
        try:
            if n >= 1 and np.any(diffs != 0):
                w_p = float(stats.wilcoxon(ml, t, zero_method="wilcox").pvalue)
            else:
                w_p = np.nan
        except ValueError:
            w_p = np.nan
        try:
            rho = float(stats.spearmanr(t, ml).correlation) if n >= 3 else np.nan
        except Exception:  # noqa: BLE001
            rho = np.nan
        lo, hi = _bootstrap_median_ci(diffs, rng)
        rows.append({
            "group": group or "ALL",
            "metric": m,
            "n_wells": n,
            "trad_median": float(np.median(t)) if n else np.nan,
            "ml_median": float(np.median(ml)) if n else np.nan,
            "median_delta": float(np.median(diffs)) if n else np.nan,
            "delta_ci_lo": lo, "delta_ci_hi": hi,
            "cliffs_delta": _cliffs_delta(ml, t),
            "spearman_rho": rho,
            "wilcoxon_p": w_p,
        })
        pvals.append(w_p)
    q = _bh_fdr(pvals)
    for r, qv in zip(rows, q):
        r["wilcoxon_q_bh"] = qv
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures + HTML report
# --------------------------------------------------------------------------- #
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("ascii")


def build_figures(per_well: pd.DataFrame) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: dict[str, str] = {}

    # 1. count scatter trad vs ml
    d = per_well[["trad_count", "ml_count"]].dropna()
    if not d.empty:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(d["trad_count"], d["ml_count"], s=10, alpha=0.35,
                   edgecolors="none", color="#1f77b4")
        hi = float(max(d.max().max(), 1))
        ax.plot([0, hi], [0, hi], "--", color="0.5", lw=1, label="y = x")
        ax.set_xlabel("traditional burst count")
        ax.set_ylabel("ML burst count")
        ax.set_title("Per-well burst count: ML vs traditional")
        ax.legend(loc="upper left", fontsize=8)
        figs["count_scatter"] = _fig_to_b64(fig)

    # 2. Bland-Altman of mean duration
    d = per_well[["trad_duration_s", "ml_duration_s"]].dropna()
    if not d.empty:
        mean = (d["trad_duration_s"] + d["ml_duration_s"]) / 2
        diff = d["ml_duration_s"] - d["trad_duration_s"]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(mean, diff, s=10, alpha=0.35, edgecolors="none", color="#2ca02c")
        md, sd = float(diff.mean()), float(diff.std())
        for y, ls, lab in [(md, "-", f"mean {md:+.3f}s"),
                           (md + 1.96 * sd, "--", "+1.96 SD"),
                           (md - 1.96 * sd, "--", "-1.96 SD")]:
            ax.axhline(y, ls=ls, color="0.4", lw=1)
        ax.text(0.99, 0.02, f"mean {md:+.3f}s", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)
        ax.set_xlabel("mean of methods -- mean burst duration (s)")
        ax.set_ylabel("ML - traditional (s)")
        ax.set_title("Bland-Altman: mean burst duration")
        figs["duration_bland_altman"] = _fig_to_b64(fig)

    # 3. IoU + coverage-Jaccard histograms
    if "iou_mean" in per_well:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        iou = per_well["iou_mean"].dropna()
        if not iou.empty:
            axes[0].hist(iou, bins=30, color="#9467bd", alpha=0.85)
            axes[0].axvline(float(iou.median()), color="k", ls="--", lw=1,
                            label=f"median {iou.median():.2f}")
            axes[0].legend(fontsize=8)
        axes[0].set_xlabel("per-well mean IoU of paired bursts")
        axes[0].set_ylabel("# wells")
        axes[0].set_title("Interval agreement (IoU)")
        cov = per_well["coverage_jaccard"].dropna()
        if not cov.empty:
            axes[1].hist(cov, bins=30, color="#ff7f0e", alpha=0.85)
            axes[1].axvline(float(cov.median()), color="k", ls="--", lw=1,
                            label=f"median {cov.median():.2f}")
            axes[1].legend(fontsize=8)
        axes[1].set_xlabel("time-coverage Jaccard")
        axes[1].set_title("Burst-time overlap")
        fig.tight_layout()
        figs["agreement_hist"] = _fig_to_b64(fig)

    # 4. paired strip of log2 ratios per metric
    ratio_cols = [f"log2ratio_{m}" for m in COMPARE_METRICS]
    present = [c for c in ratio_cols if c in per_well]
    if present:
        fig, ax = plt.subplots(figsize=(7, 4))
        data = [per_well[c].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                for c in present]
        ax.axhline(0, color="0.6", lw=1)
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor("#1f77b4")
            pc.set_alpha(0.4)
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels([c.replace("log2ratio_", "") for c in present],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("log2(ML / traditional)")
        ax.set_title("Per-metric fold-change (0 = identical)")
        fig.tight_layout()
        figs["log2ratio_violin"] = _fig_to_b64(fig)

    return figs


def write_report(path: Path, per_well: pd.DataFrame, summary: pd.DataFrame,
                 figs: dict[str, str], meta: dict[str, Any]) -> None:
    def img(key: str, alt: str) -> str:
        if key not in figs:
            return ""
        return (f'<figure><img alt="{alt}" '
                f'src="data:image/png;base64,{figs[key]}"/></figure>')

    sum_tbl = summary.round(4).to_html(index=False, border=0,
                                       classes="tbl", na_rep="")
    # headline numbers
    n_wells = len(per_well)
    n_trad0 = int((per_well["trad_count"] == 0).sum())
    n_ml0 = int((per_well["ml_count"] == 0).sum())
    med_iou = per_well["iou_mean"].median() if "iou_mean" in per_well else np.nan
    med_cov = per_well["coverage_jaccard"].median() if "coverage_jaccard" in per_well else np.nan

    # Debug: ML-overdetection wells excluded from the comparison above.
    excluded = meta.get("excluded")
    excluded_kpi = ""
    excluded_section = ""
    if meta.get("exclude_enabled") and excluded is not None:
        n_excl = len(excluded)
        crit = (f"ML &ge; {meta.get('overdetect_ratio')}&times; traditional "
                f"AND (ML - traditional) &ge; {meta.get('overdetect_min_extra')}")
        excluded_kpi = f' <div><b>{n_excl}</b>wells excluded (ML overdetection)</div>'
        excl_tbl = (excluded.round(3).to_html(index=False, border=0,
                                              classes="tbl", na_rep="")
                    if n_excl else "<p class='note'>none</p>")
        excluded_section = (
            f'<h2>Excluded: ML over-detection</h2>'
            f'<p class="note">Debug filter removed <b>{n_excl}</b> wells before the '
            f'comparison above (criterion: {crit}). Full list in '
            f'excluded_ml_overdetection.csv.</p>{excl_tbl}')

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Burst method comparison: ML vs traditional</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
        max-width: 1000px; color: #1a1a1a; line-height: 1.45; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem;
        border-bottom: 1px solid #ddd; padding-bottom: .25rem; }}
 figure {{ margin: 1rem 0; text-align: center; }}
 img {{ max-width: 100%; height: auto; }}
 .tbl {{ border-collapse: collapse; font-size: .82rem; width: 100%; }}
 .tbl th, .tbl td {{ border: 1px solid #ddd; padding: 4px 7px; text-align: right; }}
 .tbl th {{ background: #f4f4f4; }}
 .kpi {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0; }}
 .kpi div {{ background: #f7f7f9; border-radius: 8px; padding: .7rem 1.1rem; }}
 .kpi b {{ display: block; font-size: 1.35rem; }}
 .note {{ color: #555; font-size: .85rem; }}
</style></head><body>
<h1>Burst detection: ML vs traditional</h1>
<p class="note">analysis_root: {meta['analysis_root']}<br>
 wells compared: <b>{n_wells}</b> (both detectors complete) &middot;
 strict same-burst IoU threshold: {meta['threshold']}</p>
<div class="kpi">
 <div><b>{n_wells}</b>wells</div>
 <div><b>{n_trad0}</b>wells trad found 0 bursts</div>
 <div><b>{n_ml0}</b>wells ML found 0 bursts</div>
 <div><b>{med_iou:.2f}</b>median IoU (paired)</div>
 <div><b>{med_cov:.2f}</b>median coverage Jaccard</div>
{excluded_kpi}
</div>

<h2>Aggregate comparison (paired, well = unit)</h2>
<p class="note">delta = ML - traditional. Cliff's delta &gt; 0 &rArr; ML larger.
 Wilcoxon signed-rank p, Benjamini-Hochberg FDR q across the {len(summary)} metrics.</p>
{sum_tbl}

<h2>Burst count</h2>
{img('count_scatter', 'count scatter')}

<h2>Burst duration</h2>
{img('duration_bland_altman', 'duration Bland-Altman')}

<h2>Temporal agreement</h2>
{img('agreement_hist', 'IoU and coverage histograms')}

<h2>Per-metric fold-change</h2>
{img('log2ratio_violin', 'log2 ratio violin')}

{excluded_section}

<p class="note">Generated by scripts/compare_burst_methods.py. Per-well detail in
 per_well.csv; aggregate stats in summary_stats.csv.</p>
</body></html>"""
    path.write_text(html)


# --------------------------------------------------------------------------- #
# ML over-detection exclusion (debug)
# --------------------------------------------------------------------------- #
EXCLUDED_COLS = ["sample_id", "assay_id", "scan_type", "groupname", "div",
                 "ml_count", "trad_count", "overdetect_ratio",
                 "recording_key", "well_id", "well_name"]


def split_ml_overdetection(per_well: pd.DataFrame, ratio: float, min_extra: int
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition per_well into (kept, excluded).

    A well is excluded when ML significantly over-detects: both
    ``ml_count >= ratio * trad_count`` and ``(ml_count - trad_count) >= min_extra``.
    Wells with non-finite counts are never excluded. ``excluded`` carries the
    debug columns the user asked to inspect (Sample ID, Assay ID, Group, DIV, ML
    count, Threshold count) plus traceability + the observed ratio.
    """
    ml = per_well["ml_count"].to_numpy(float)
    trad = per_well["trad_count"].to_numpy(float)
    finite = np.isfinite(ml) & np.isfinite(trad)
    mask = finite & (ml >= ratio * trad) & ((ml - trad) >= min_extra)
    excluded = per_well[mask].copy()
    kept = per_well[~mask].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        excluded["overdetect_ratio"] = (excluded["ml_count"]
                                        / excluded["trad_count"].replace(0, np.nan))
    excluded = excluded[[c for c in EXCLUDED_COLS if c in excluded.columns]]
    excluded = excluded.sort_values("overdetect_ratio", ascending=False)
    return kept, excluded


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True,
                   help="pipeline_config.json (read for analysis_root + figure_root).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output dir (default <figure_root>/burst_method_comparison).")
    p.add_argument("--overlap-threshold", type=float, default=0.9,
                   help="IoU cut for the strict 'same-burst' count (1.0 = literal exact).")
    p.add_argument("--min-bursts", type=int, default=1,
                   help="Skip temporal divergence when both methods have fewer events.")
    p.add_argument("--group-by", default=None, choices=[None, "groupname"],
                   help="Also emit aggregate stats split by this column.")
    p.add_argument("--limit", type=int, default=None, help="Cap #wells (dry run).")
    p.add_argument("--dump-matched-events", action="store_true",
                   help="Also write matched_events.csv (one row per paired burst; large).")
    p.add_argument("--exclude-ml-overdetect", action="store_true",
                   help="Debug: pull wells where ML significantly over-detects bursts "
                        "into excluded_ml_overdetection.csv and run the rest of the "
                        "comparison on the cleaned set.")
    p.add_argument("--ml-overdetect-ratio", type=float, default=2.0,
                   help="ML over-detection when ml_count >= this * trad_count (default 2.0).")
    p.add_argument("--ml-overdetect-min-extra", type=int, default=3,
                   help="...and (ml_count - trad_count) >= this absolute floor (default 3).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    analysis_root, figure_root = _load_config(args.config)
    pipeline_cache = _load_pipeline_cache(analysis_root)
    experiment_cache = _load_experiment_cache(analysis_root)
    group_lut = _group_lookup(experiment_cache)
    resolve = _make_resolver([analysis_root])

    refs = discover_wells(pipeline_cache, group_lut, resolve, limit=args.limit)
    if not refs:
        raise SystemExit("no wells with both detectors complete; check analysis_root")

    logger.info("building per-well comparison over %d wells ...", len(refs))
    per_well, matched = build_per_well(
        refs, threshold=args.overlap_threshold,
        min_bursts=args.min_bursts, dump_matched=args.dump_matched_events)

    out_dir = args.output_dir or (figure_root / "burst_method_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Debug: pull ML-overdetection wells out; comparison continues on cleaned set.
    excluded = pd.DataFrame()
    if args.exclude_ml_overdetect:
        n_before = len(per_well)
        per_well, excluded = split_ml_overdetection(
            per_well, args.ml_overdetect_ratio, args.ml_overdetect_min_extra)
        logger.info("excluded %d/%d wells (ML over-detection: ml >= %.2f*trad AND "
                    "ml-trad >= %d); %d remain",
                    len(excluded), n_before, args.ml_overdetect_ratio,
                    args.ml_overdetect_min_extra, len(per_well))
        excluded.to_csv(out_dir / "excluded_ml_overdetection.csv", index=False)
        logger.info("wrote %s", out_dir / "excluded_ml_overdetection.csv")

    rng = np.random.default_rng(args.seed)
    summary = aggregate_stats(per_well, rng)
    if args.group_by == "groupname":
        parts = [summary]
        for gname, gdf in per_well.groupby("groupname"):
            if len(gdf) >= 3:
                parts.append(aggregate_stats(gdf, rng, group=str(gname)))
        summary = pd.concat(parts, ignore_index=True)

    per_well.to_csv(out_dir / "per_well.csv", index=False)
    summary.to_csv(out_dir / "summary_stats.csv", index=False)
    if args.dump_matched_events:
        matched.to_csv(out_dir / "matched_events.csv", index=False)

    figs = build_figures(per_well)
    write_report(out_dir / "report.html", per_well, summary, figs,
                 meta={"analysis_root": str(analysis_root),
                       "threshold": args.overlap_threshold,
                       "excluded": excluded,
                       "exclude_enabled": args.exclude_ml_overdetect,
                       "overdetect_ratio": args.ml_overdetect_ratio,
                       "overdetect_min_extra": args.ml_overdetect_min_extra})

    logger.info("wrote %s", out_dir / "per_well.csv")
    logger.info("wrote %s", out_dir / "summary_stats.csv")
    logger.info("wrote %s", out_dir / "report.html")
    # one-line takeaway to stdout
    med = summary[summary["group"] == "ALL"].set_index("metric")["median_delta"]
    logger.info("median ML-trad delta -- count: %.2f  duration_s: %.4f",
                med.get("count", float("nan")), med.get("duration_s", float("nan")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
