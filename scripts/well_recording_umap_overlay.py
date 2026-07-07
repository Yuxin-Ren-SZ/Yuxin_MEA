"""Shared-axis bin-level UMAP/PCA overlay across one physical well's recordings.

Pools the per-bin 26-D feature matrices from *all* recordings of one physical
well, applies a single shared normalization, and fits ONE embedding so every
recording shares the same axes/loadings — you can then overlay the recordings and
see whether the resting/bursting clusters drift across days.

The figure-building core now lives in ``yuxin_mea.analysis.cluster_overlay`` (so
the same code backs the in-pipeline cluster-overlay tasks); this script is the
standalone CLI that discovers one-or-more wells' recordings by globbing the ml
output tree.

Data consumed per recording (must have run with ``debug=True``):
  <ml_root>/<sample>/<date>/<plate>/<scan>/<run_id>/<rec_name>/<well_id>/
      ml_burst_detection/{debug_trace.pkl, diagnostics.json}

Examples:
    python scripts/well_recording_umap_overlay.py \
        --config pipeline_config.json --well "CX118|T003346|well000"
    # structural-only view (remove per-recording baseline via background z-norm):
    python scripts/well_recording_umap_overlay.py \
        --config pipeline_config.json --well "CX118|T003346|well000" --norm per-recording
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from yuxin_mea.analysis.cluster_overlay import (
    PALETTE_CHOICES,
    SeriesTrace,
    build_overlay_figure,
    load_series_trace,
    pool_and_fit,
)

logger = logging.getLogger("well_recording_umap_overlay")

ML_TERMINAL = "ml_burst_detection"


# --------------------------------------------------------------------------- #
# Config / discovery
# --------------------------------------------------------------------------- #
def _load_config(config_path: Path) -> tuple[Path, Path]:
    with config_path.open() as fh:
        cfg = json.load(fh)
    g = cfg.get("global", {})
    analysis_root = Path(g["analysis_root"])
    figure_root = Path(g.get("figure_root") or analysis_root)
    return analysis_root, figure_root


def _parse_well_uid(s: str) -> tuple[str, str, str]:
    parts = [p for p in s.replace("/", "|").split("|") if p]
    if len(parts) != 3:
        raise SystemExit(
            f"--well must be 'sample|plate|well_id' (e.g. 'CX118|T003346|well000'); got {s!r}"
        )
    return parts[0], parts[1], parts[2]


def _parse_plating_date(s: Any) -> datetime | None:
    """Parse a day-first dot/slash/dash date (e.g. '13.3.2026', '30.01.2026')."""
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
    s = str(s).strip()
    if len(s) != 6 or not s.isdigit():
        return None
    try:
        return datetime.strptime("20" + s, "%Y%m%d")
    except ValueError:
        return None


def _plating_lookup(experiment_cache: dict, sample: str, plate: str, well_id: str) -> datetime | None:
    for rec_key, rec in experiment_cache.items():
        kp = rec_key.split("/")
        if not (len(kp) >= 3 and kp[0] == sample and kp[2] == plate):
            continue
        for _wid, w in (rec.get("wells") or {}).items():
            md = w.get("metadata") or {}
            if str(w.get("well_id")) == well_id or str(_wid) == well_id:
                pd_ = _parse_plating_date(md.get("Plating Date"))
                if pd_ is not None:
                    return pd_
    return None


def _discover_recordings(ml_root: Path, sample: str, plate: str, well_id: str, scan_type: str) -> list[Path]:
    """Sorted ``debug_trace.pkl`` paths for one physical well (sorted glob is already
    date → run_id → rec_name order)."""
    pattern = f"{sample}/*/{plate}/{scan_type}/*/*/{well_id}/{ML_TERMINAL}/debug_trace.pkl"
    return sorted(ml_root.glob(pattern))


def _label_for_trace(trace_path: Path, ml_root: Path, plating: datetime | None) -> str:
    rel = trace_path.relative_to(ml_root).parts  # (sample, date, plate, scan, run_id, rec_name, well, term, file)
    date, run_id = rel[1], rel[4]
    div = None
    rec_dt = _parse_yymmdd(date)
    if plating is not None and rec_dt is not None:
        div = (rec_dt - plating).days
    return f"{date}" + (f" DIV{div}" if div is not None else "") + f" {run_id}"


def _embed_one_well(ml_root: Path, experiment_cache: dict, well_str: str, args):
    """Discover, load, pool, fit one physical well → (title, series, method_coords) or None."""
    sample, plate, well_id = _parse_well_uid(well_str)
    well_uid = f"{sample}|{plate}|{well_id}"

    traces = _discover_recordings(ml_root, sample, plate, well_id, args.scan_type)
    if not traces:
        logger.warning("no debug_trace.pkl found for %s under %s — skipping", well_uid, ml_root)
        return None
    logger.info("[%s] found %d recording traces", well_uid, len(traces))

    plating = _plating_lookup(experiment_cache, sample, plate, well_id)
    logger.info("[%s] plating date: %s", well_uid, plating.date() if plating else "unknown")

    series: list[SeriesTrace] = []
    feat: list[str] | None = None
    for tp in traces:
        label = _label_for_trace(tp, ml_root, plating)
        st, feat = load_series_trace(tp.parent, label, args.max_bins_per_rec, feat)
        if st is not None:
            series.append(st)
    if not series:
        logger.warning("[%s] no usable recordings after loading — skipping", well_uid)
        return None
    if len(series) < 2:
        logger.warning("[%s] only %d usable recording — overlay needs >=2 to show drift", well_uid, len(series))

    total_bins = sum(int(s.X_raw.shape[0]) for s in series)
    logger.info("[%s] pooled %d recordings, %d bins (max %d/rec)",
                well_uid, len(series), total_bins, args.max_bins_per_rec)
    method_coords = pool_and_fit(
        series, args.norm, feat, args.method, args.n_neighbors, args.min_dist, args.seed
    )
    return well_uid, series, method_coords


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--well", type=str, required=True, nargs="+",
                    help="one or more physical wells 'sample|plate|well_id'")
    ap.add_argument("--scan-type", default="Network")
    ap.add_argument("--ml-dirname", default="ml_burst_data_umap")
    ap.add_argument("--norm", choices=["shared", "per-recording"], default="shared")
    ap.add_argument("--method", choices=["umap", "pca"], default="umap")
    ap.add_argument("--palette", choices=PALETTE_CHOICES, default="dark24")
    ap.add_argument("--max-bins-per-rec", type=int, default=1500)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--animate", dest="animate", action="store_true", default=True,
                    help="single-well only: add a time-lapse over recordings (default on)")
    ap.add_argument("--no-animate", dest="animate", action="store_false",
                    help="static overlay (no time-lapse); the only mode for multi-well")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(message)s")

    analysis_root, figure_root = _load_config(args.config)
    ml_root = analysis_root / args.ml_dirname
    try:
        with (analysis_root / "experiment_cache.json").open() as fh:
            experiment_cache = json.load(fh)
    except Exception:  # noqa: BLE001
        experiment_cache = {}

    blocks = []
    for well_str in args.well:
        block = _embed_one_well(ml_root, experiment_cache, well_str, args)
        if block is not None:
            blocks.append(block)
    if not blocks:
        raise SystemExit("no wells produced a usable embedding")

    grand_bins = sum(sum(int(s.X_raw.shape[0]) for s in series) for _, series, _ in blocks)
    if grand_bins > 120_000:
        logger.warning("%d total plotted points across %d wells — HTML will be large; "
                       "consider fewer wells or a smaller --max-bins-per-rec", grand_bins, len(blocks))
    if args.animate and len(blocks) > 1:
        logger.info("animation is single-well only; rendering static overlay for %d wells", len(blocks))

    fig = build_overlay_figure(
        blocks, color_by="recording", animate=args.animate, palette=args.palette, norm=args.norm
    )

    if args.output:
        out = args.output
    elif len(blocks) == 1:
        s, p, w = blocks[0][0].split("|")
        out = figure_root / "well_umap_overlay" / f"{s}_{p}_{w}.html"
    else:
        tag = "-".join(b[0].split("|")[-1] for b in blocks)[:60]
        out = figure_root / "well_umap_overlay" / f"overlay_{len(blocks)}wells_{tag}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="inline", full_html=True)
    logger.info("wrote %s", out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
