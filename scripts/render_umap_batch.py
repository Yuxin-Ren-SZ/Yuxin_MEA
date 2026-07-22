#!/usr/bin/env python3
"""Batch-render per-well UMAP figures (PNG) and/or animations (MP4), N-way parallel.

Neither ``render_slim_png.py`` (dim-sweep only) nor ``animate_ml_umap.py`` (serial)
renders a plain per-well UMAP set for a finished analysis in parallel. This driver
does: it discovers every well with a completed ``ml_burst_detection`` from
``pipeline_cache.json`` (same discovery as the two scripts) and fans the render out
over a process pool, reusing the *tested* renderers verbatim:

  * PNG  -> ``render_slim_png._render_png``  (3-panel: raster + burst-events + UMAP)
  * MP4  -> ``animate_ml_umap._animate_well`` (UMAP trajectory ball + fading trail)

Sharding is recording-aware (each job is a full ``(rec_key, rec_name, well_id)``),
so it is safe across recordings that reuse ``well000``-style ids — unlike the
``--well`` name filter, which would match the same id in every recording.

Output layout (rec_key already encodes the nested recording path, so it is unique):
  {png-dir}/{rec_key}/{well_name}_{rec_name}_{well_id}.png
  {anim-dir}/{rec_key}/{well_name}_{rec_name}_{well_id}.mp4

Run — PNG first (fast), then animations (slow, background it):
    python scripts/render_umap_batch.py --config pipeline_config_run.json \
        --mode png  --jobs 8 --png-dir  /mnt/.../analysis/figures
    python scripts/render_umap_batch.py --config pipeline_config_run.json \
        --mode anim --jobs 8 --anim-dir /mnt/.../analysis/animations

Smoke-test first with ``--limit 3`` (and ``--jobs 1`` to time one animation).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

# Reuse discovery/loaders + both renderers from the sibling scripts. Importing
# these sets matplotlib's Agg backend but creates no figure state, so it is safe
# to do in the parent before forking workers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspect_ml_bursts as iml  # noqa: E402
import render_slim_png as rsp  # noqa: E402
import animate_ml_umap as anim  # noqa: E402

logger = logging.getLogger("render_umap_batch")

# Per-worker state, built once per process by _init_worker (the resolver is a
# closure and cannot be pickled across the pool, so each worker rebuilds it).
_G: dict = {}


def _init_worker(config_path: str, analysis_root: str,
                 png_dir: str | None, anim_dir: str | None, opts: dict) -> None:
    global _G
    ar = Path(analysis_root)
    resolve = iml._make_resolver(
        [Path(config_path).resolve().parent, ar, Path.cwd().resolve()]
    )
    _G = {
        "resolve": resolve,
        "png_dir": Path(png_dir) if png_dir else None,
        "anim_dir": Path(anim_dir) if anim_dir else None,
        **opts,
    }


def _work(job: tuple) -> tuple:
    """Render PNG and/or MP4 for one well. Returns (rec_key, well_id, status, detail)."""
    rec_key, rec_name, well_id, entry, well_meta = job
    try:
        bundle = iml._load_well_bundle(
            rec_key, rec_name, well_id, entry, well_meta, _G["resolve"]
        )
    except Exception as exc:  # noqa: BLE001
        return (rec_key, well_id, "error-load", repr(exc))
    if bundle is None:
        return (rec_key, well_id, "skip-nobundle", "")

    stem = f"{bundle.well_name}_{rec_name}_{well_id}"
    done: list[str] = []

    if _G["png_dir"] is not None:
        out = _G["png_dir"] / rec_key / f"{stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        if _G["skip_existing"] and out.exists():
            done.append("png-skip")
        else:
            try:
                rsp._render_png(bundle, out, _G["dpi_png"])
                done.append("png")
            except Exception as exc:  # noqa: BLE001
                return (rec_key, well_id, "error-png", repr(exc))

    if _G["anim_dir"] is not None:
        out = _G["anim_dir"] / rec_key / f"{stem}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        if _G["skip_existing"] and out.exists() and anim._valid_mp4(out):
            done.append("anim-skip")
        else:
            try:
                ok = anim._animate_well(
                    bundle, out, _G["duration"], _G["history"], _G["fps"], _G["dpi_anim"]
                )
                done.append("anim" if ok else "anim-noumap")
            except Exception as exc:  # noqa: BLE001
                return (rec_key, well_id, "error-anim", repr(exc))

    return (rec_key, well_id, "ok", "+".join(done) or "nothing")


def _want(well_id: str, well_meta: dict, filters: list[str] | None) -> bool:
    if not filters:
        return True
    name = str(well_meta.get("well_name", "")).lower()
    keys = {well_id.lower(), name}
    return any(f.lower() in keys or f.lower() in well_id.lower() for f in filters)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="pipeline_config*.json (read for analysis_root/figure_root).")
    ap.add_argument("--mode", choices=["png", "anim", "both"], default="both")
    ap.add_argument("--png-dir", type=Path, default=None,
                    help="PNG output root. Default: global.figure_root.")
    ap.add_argument("--anim-dir", type=Path, default=None,
                    help="MP4 output root. Default: {analysis_root}/animations.")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="Render only the first N wells (smoke test). 0 = all.")
    ap.add_argument("--well", action="append", default=None,
                    help="Only wells matching well_id/well_name (repeatable).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a well whose output already exists (mp4 re-validated).")
    ap.add_argument("--duration", type=float, default=60.0, help="Anim: target seconds.")
    ap.add_argument("--history", type=float, default=5.0, help="Anim: trail seconds.")
    ap.add_argument("--fps", type=int, default=None, help="Anim: override fps.")
    ap.add_argument("--dpi-anim", type=int, default=120)
    ap.add_argument("--dpi-png", type=int, default=300)
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s — %(message)s")

    cfg = json.loads(args.config.read_text())
    analysis_root = Path(cfg["global"]["analysis_root"])

    png_dir = anim_dir = None
    if args.mode in ("png", "both"):
        png_dir = args.png_dir or Path(cfg["global"].get("figure_root", analysis_root))
        png_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("anim", "both"):
        anim_dir = args.anim_dir or (analysis_root / "animations")
        anim_dir.mkdir(parents=True, exist_ok=True)

    pipeline_cache = iml._load_pipeline_cache(analysis_root)
    experiment_cache = iml._load_experiment_cache(analysis_root)
    wells = iml._collect_ml_wells(pipeline_cache)
    if not wells:
        logger.warning("No completed ml_burst_detection wells in %s", analysis_root)
        return 0

    jobs: list[tuple] = []
    for rec_key, rec_name, well_id, entry in wells:
        well_meta = (experiment_cache.get(rec_key, {})
                     .get("wells", {}).get(well_id, {}).get("metadata", {}))
        if _want(well_id, well_meta, args.well):
            jobs.append((rec_key, rec_name, well_id, entry, well_meta))
    if args.limit > 0:
        jobs = jobs[:args.limit]

    logger.info("Rendering %d well(s), mode=%s, jobs=%d  png=%s anim=%s",
                len(jobs), args.mode, args.jobs, png_dir, anim_dir)

    opts = dict(dpi_png=args.dpi_png, dpi_anim=args.dpi_anim, duration=args.duration,
                history=args.history, fps=args.fps, skip_existing=args.skip_existing)
    init_args = (str(args.config), str(analysis_root),
                 str(png_dir) if png_dir else None,
                 str(anim_dir) if anim_dir else None, opts)

    ctx = mp.get_context("fork")
    t0 = time.monotonic()
    n_ok = n_err = 0
    with ctx.Pool(max(1, args.jobs), initializer=_init_worker, initargs=init_args) as pool:
        for i, (rec_key, well_id, status, detail) in enumerate(
                pool.imap_unordered(_work, jobs, chunksize=1), 1):
            if status == "ok":
                n_ok += 1
            else:
                n_err += 1
                logger.warning("%s/%s -> %s %s", rec_key, well_id, status, detail)
            if i % 25 == 0 or i == len(jobs):
                el = time.monotonic() - t0
                rate = i / el if el > 0 else 0.0
                eta = (len(jobs) - i) / rate if rate > 0 else 0.0
                logger.info("progress %d/%d  ok=%d err=%d  %.2f well/s  ETA %.0f min",
                            i, len(jobs), n_ok, n_err, rate, eta / 60.0)

    logger.info("Done. ok=%d err=%d in %.1f min", n_ok, n_err, (time.monotonic() - t0) / 60.0)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
