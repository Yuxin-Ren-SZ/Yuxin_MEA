"""Pre-build per-recording viewer bundles for the plate viewer (Caching guide Tier 3).

Collapses each recording's per-well outputs into ONE pickled ``list[WellRecord]``
under ``<analysis_root>/viewer_bundles/<recording_key>/<source>.viewer.pkl``. The
dashboard reads a fresh bundle in a single sequential file read instead of
walking 24 wells' ``.npy``/``.pkl`` files, and re-checks each bundle's stored
staleness signature so a pipeline re-run transparently invalidates it.

This is the OPT-IN alternative to a DAG-registered pipeline step (which would need
plate-level task enumeration the pipeline doesn't have). Run it whenever you like;
nothing else changes if you don't. No source files are modified.

Example:
    python scripts/build_viewer_bundles.py --config pipeline_config_run.json
    python scripts/build_viewer_bundles.py --config cfg.json --source ml --recording CX118/260205/T003346/Network/000004
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("build_viewer_bundles")

# Method -> (per-well terminal dir == burst task name). Mirrors burst_inspector.
_SOURCE_TERMINAL = {"traditional": "burst_detection", "ml": "ml_burst_detection"}


def _recordings_with(cache: dict, burst_task: str) -> list[str]:
    """Recording keys that have ≥1 well complete for both burst_task + auto_curation."""
    keys: set[str] = set()
    for entry in cache.values():
        if not isinstance(entry, dict):
            continue
        tasks = entry.get("tasks") or {}
        if (tasks.get(burst_task, {}).get("status") == "complete"
                and tasks.get("auto_curation", {}).get("status") == "complete"):
            keys.add(entry["recording_key"])
    return sorted(keys)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="Path to pipeline config JSON (read for analysis_root).")
    p.add_argument("--source", choices=sorted(_SOURCE_TERMINAL), default="traditional",
                   help="Which detector's outputs to bundle (default: traditional).")
    p.add_argument("--recording", default=None,
                   help="Only build this recording_key (default: all complete ones).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Import here so --help works without the analysis stack loaded.
    from yuxin_mea.analysis.burst_inspector import well_output_dirs_from_cache
    from yuxin_mea.analysis.plate_raster_synchrony import load_plate_data
    from yuxin_mea.analysis.viewer_bundle import bundle_path, write_viewer_bundle
    from yuxin_mea.dashboard.data_cache import manifest_signature

    cfg = json.loads(Path(args.config).read_text())
    analysis_root = Path(cfg["global"]["analysis_root"]).resolve()
    burst_terminal = _SOURCE_TERMINAL[args.source]
    exp_cache = analysis_root / "experiment_cache.json"
    exp_cache = exp_cache if exp_cache.exists() else None
    bundle_root = analysis_root / "viewer_bundles"

    pc_path = analysis_root / "pipeline_cache.json"
    if not pc_path.exists():
        logger.error("no pipeline_cache.json at %s", analysis_root)
        return 1
    cache = json.loads(pc_path.read_text())

    if args.recording:
        recordings = [args.recording]
    else:
        recordings = _recordings_with(cache, burst_terminal)
    logger.info("building %d bundle(s) [source=%s] -> %s",
                len(recordings), args.source, bundle_root)

    burst_root = analysis_root / ("burst_detection_data" if args.source == "traditional"
                                  else "ml_burst_data")
    curation_root = analysis_root / "curation_data"

    written = 0
    for rk in recordings:
        bwd = well_output_dirs_from_cache(analysis_root, rk, burst_terminal) or None
        cwd = well_output_dirs_from_cache(analysis_root, rk, "auto_curation") or None
        if not bwd and not cwd:
            logger.warning("skip %s: no manifest entries", rk)
            continue
        records = load_plate_data(
            burst_detection_root=burst_root, curation_output_root=curation_root,
            recording_key=rk, experiment_cache_path=exp_cache,
            burst_terminal=burst_terminal,
            burst_well_dirs=bwd, curation_well_dirs=cwd,
        )
        n_ok = sum(w.status == "ok" for w in records)
        sig = manifest_signature(bwd, cwd, exp_cache)
        out = write_viewer_bundle(records, bundle_path(bundle_root, rk, args.source), signature=sig)
        written += 1
        logger.info("wrote %s (%d/24 ok)", out, n_ok)

    logger.info("done: %d bundle(s) written", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
