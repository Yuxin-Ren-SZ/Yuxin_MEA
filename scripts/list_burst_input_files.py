"""Write a manifest of every file the burst detectors consume.

Both the traditional (BurstDetectionTask) and ML (MLBurstDetectionTask)
detectors read exactly one file per well — ``curated_spike_times.npy`` produced
by AutoCurationTask. This script walks pipeline_cache.json next to
analysis_root, collects the absolute path of every such file from wells whose
``auto_curation`` task is COMPLETE, and writes them one per line to a text file.

No files are copied or moved.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("list_burst_input_files")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="Path to pipeline_config.json (read for analysis_root).")
    p.add_argument("--output", type=Path, required=True,
                   help="Text file to write the manifest to (one absolute path per line).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s — %(message)s")

    with args.config.open() as fh:
        cfg = json.load(fh)
    analysis_root = Path(cfg["global"]["analysis_root"]).resolve()
    # Tasks store relative output_paths resolved against the runner's CWD,
    # which is typically the directory containing pipeline_config.json.
    config_dir = args.config.resolve().parent
    search_roots: list[Path] = [config_dir, analysis_root, Path.cwd().resolve()]

    def _resolve(rel: str) -> Path | None:
        p = Path(rel)
        if p.is_absolute():
            return p if p.exists() else None
        for base in search_roots:
            cand = (base / p).resolve()
            if cand.exists():
                return cand
        return None

    cache_path = analysis_root / "pipeline_cache.json"
    if not cache_path.exists():
        raise SystemExit(f"pipeline_cache.json not found at {cache_path}")
    with cache_path.open() as fh:
        cache = json.load(fh)

    paths: list[Path] = []
    skipped_not_complete = 0
    skipped_missing_file = 0

    for compound_key, entry in cache.items():
        tasks = entry.get("tasks", {})
        cur = tasks.get("auto_curation", {})
        if cur.get("status") != "complete":
            skipped_not_complete += 1
            continue
        rel = cur.get("output_path")
        if not rel:
            skipped_missing_file += 1
            continue
        out_dir = _resolve(rel)
        if out_dir is None:
            logger.debug("output dir unresolved for %s (rel=%s)", compound_key, rel)
            skipped_missing_file += 1
            continue
        candidate = (out_dir / "curated_spike_times.npy").resolve()
        if not candidate.is_file():
            logger.debug("Missing on disk: %s (entry %s)", candidate, compound_key)
            skipped_missing_file += 1
            continue
        paths.append(candidate)

    paths.sort()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    with args.output.open("w") as fh:
        fh.write(f"# Burst-detector input manifest\n")
        fh.write(f"# generated: {timestamp}\n")
        fh.write(f"# config:    {args.config.resolve()}\n")
        fh.write(f"# analysis_root: {analysis_root}\n")
        fh.write(f"# count:     {len(paths)}\n")
        fh.write(f"# each line below is an absolute path to a curated_spike_times.npy file\n")
        for p_abs in paths:
            fh.write(f"{p_abs}\n")

    logger.info(
        "Wrote %d path(s) to %s. Skipped %d (auto_curation not complete), %d (file missing on disk).",
        len(paths), args.output, skipped_not_complete, skipped_missing_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
