#!/usr/bin/env python
"""Read-only cache correctness check — disk vs experiment_cache.json.

Does NOT modify the cache and does NOT run any analysis. It walks data_root
for `data.raw.h5` files, derives the expected recording keys, and diffs them
against what's already cached.

    conda run -n yuxin_mea python scripts/check_cache.py --config pipeline_config.json

Exit code 0 = cache in sync, 1 = disk has recordings the cache is missing
(run scripts/refresh_cache.py to fix), 2 = usage/IO error.

Why a standalone walk instead of DatasetManager? Constructing DatasetManager
triggers an incremental scan that *writes* the cache. This script is a pure
observer — no side effects — so you can trust the diff it reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yuxin_mea.config import ConfigManager
from yuxin_mea.dataset.cache import CACHE_FILENAME


def _load_config(config_path: Path) -> ConfigManager:
    cm = ConfigManager()
    cm.load(config_path)
    return cm


def _cache_path(analysis_root: Path) -> Path:
    return analysis_root / CACHE_FILENAME


def _cached_keys(cache_file: Path) -> set[str]:
    if not cache_file.is_file():
        return set()
    with open(cache_file) as fh:
        return set(json.load(fh).keys())


def _disk_keys(data_root: Path) -> set[str]:
    """Derive recording keys (sample/date/plate/scan/run) from data.raw.h5 paths.

    Handles both root-level data_root (children are Sample dirs) and
    sample-level data_root (children are Date dirs) by relative-path length.
    """
    keys: set[str] = set()
    for h5 in data_root.rglob("data.raw.h5"):
        parts = h5.relative_to(data_root).parts
        if len(parts) == 6:          # Sample/Date/Plate/Scan/Run/data.raw.h5
            keys.add("/".join(parts[:5]))
        elif len(parts) == 5:        # data_root IS the sample: Date/Plate/Scan/Run/data.raw.h5
            keys.add(data_root.name + "/" + "/".join(parts[:4]))
        # anything else = unexpected layout, skip silently
    return keys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Read-only disk-vs-cache correctness check.")
    p.add_argument("--config", required=True, type=Path,
                   help="pipeline_config.json (for data_root + analysis_root).")
    p.add_argument("--network-only", action="store_true",
                   help="Only compare '/Network/' recordings.")
    args = p.parse_args(argv)

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2

    cm = _load_config(args.config)
    data_root = Path(cm.get_global("data_root"))
    analysis_root = Path(cm.get_global("analysis_root"))
    cache_file = _cache_path(analysis_root)

    print(f"data_root : {data_root}")
    print(f"cache     : {cache_file}")
    print("Walking disk for data.raw.h5 (slow on NAS)...", flush=True)

    disk = _disk_keys(data_root)
    cached = _cached_keys(cache_file)

    if args.network_only:
        disk = {k for k in disk if "/Network/" in k}
        cached = {k for k in cached if "/Network/" in k}

    missing = sorted(disk - cached)   # on disk, not cached -> refresh needed
    stale = sorted(cached - disk)     # cached, no longer on disk

    print(f"\ndisk recordings   : {len(disk)}")
    print(f"cached recordings : {len(cached)}")
    print(f"missing (disk not cached) : {len(missing)}")
    print(f"stale  (cached not disk)  : {len(stale)}")

    if missing:
        print("\n-- MISSING (run scripts/refresh_cache.py) --")
        for k in missing:
            print(f"  {k}")
    if stale:
        print("\n-- STALE (on disk removed/renamed) --")
        for k in stale:
            print(f"  {k}")

    if missing:
        print("\nCache OUT OF SYNC — disk has recordings not in cache.")
        return 1
    print("\nCache in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
