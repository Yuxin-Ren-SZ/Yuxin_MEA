#!/usr/bin/env python
"""Force a full rescan of the recording cache (experiment_cache.json).

    conda run -n yuxin_mea python scripts/refresh_cache.py --config pipeline_config.json

DOES NOT run any analysis. `DatasetManager` and `PipelineManager` are
independent — this only rebuilds the dataset cache. Task completion status in
pipeline_cache.json is untouched, so refreshing will NOT rerun preprocessing,
sorting, or any downstream task. Run yuxin-mea-run separately when you want to
(re)compute analysis.

Use this over the incremental scan (which runs automatically on every
DatasetManager construct) when new recordings were added under an already-cached
Date directory — the incremental scan is date-granular and misses those.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from yuxin_mea.config import ConfigManager
from yuxin_mea.dataset import DatasetManager


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Full rescan of experiment_cache.json (no analysis rerun).")
    p.add_argument("--config", required=True, type=Path,
                   help="pipeline_config.json (for data_root + analysis_root).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2

    cm = ConfigManager()
    cm.load(args.config)
    data_root = Path(cm.get_global("data_root"))
    analysis_root = Path(cm.get_global("analysis_root"))

    dm = DatasetManager(data_root, analysis_root)
    before = len(dm._cache)          # count after the auto incremental scan on construct
    dm.refresh()                     # clear + full rescan from scratch
    after = len(dm._cache)

    print(f"\nrecordings before refresh : {before}")
    print(f"recordings after  refresh : {after}")
    delta = after - before
    if delta > 0:
        print(f"+{delta} recording(s) the incremental scan had missed.")
    elif delta < 0:
        print(f"{delta} recording(s) removed (no longer on disk).")
    else:
        print("No change — cache was already complete.")
    print("\nDone. Analysis NOT rerun (dataset cache only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
