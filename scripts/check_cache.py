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


def _print_verify(cm, data_root: Path, analysis_root: Path, *,
                  hash_mode: str | None, network_only: bool, mark_stale: bool) -> int:
    """Run + print the provenance verification. Returns 1 on computational drift.

    With ``mark_stale`` the RAW/CONFIG-drifted tasks are reset to NOT_RUN (and
    their dependents cascaded) so the next run recomputes them — the only
    mutating path here.
    """
    from yuxin_mea.provenance.verify import verify_provenance

    print("\n== Provenance verification "
          f"({'re-hash raw: ' + hash_mode if hash_mode else 'against cached fingerprints'}) ==",
          flush=True)
    rep = verify_provenance(cm, data_root, analysis_root,
                            hash_mode=hash_mode, network_only=network_only)
    c = rep.counts
    print(f"OK              : {c['OK']}")
    print(f"RAW-CHANGED     : {c['RAW-CHANGED']}   (re-run analysis)")
    print(f"CONFIG-CHANGED  : {c['CONFIG-CHANGED']}   (re-run analysis)")
    print(f"METADATA-CHANGED: {c['METADATA-CHANGED']}   (labels only; refresh groups)")
    print(f"UNKNOWN         : {c['UNKNOWN']}   (no stamp — predates provenance)")
    if rep.errors:
        print("\n-- fingerprint errors --")
        for e in rep.errors[:50]:
            print(f"  ! {e}")
    if rep.details:
        print("\n-- drift detail --")
        for labels, where in rep.details[:200]:
            print(f"  {labels:40s} {where}")
        if len(rep.details) > 200:
            print(f"  … and {len(rep.details) - 200} more")
    print("\nProvenance: computational drift found." if rep.computational_drift
          else "\nProvenance: no computational drift.")

    if mark_stale and rep.drifted:
        n = _mark_stale(analysis_root, cm, rep.drifted)
        print(f"\n--mark-stale: reset {n} task record(s) (incl. dependent cascade) "
              "to NOT_RUN. Re-run yuxin-mea-run to recompute.")
    elif mark_stale:
        print("\n--mark-stale: nothing to reset.")
    return 1 if rep.computational_drift else 0


def _mark_stale(analysis_root: Path, cm, drifted: list[dict]) -> int:
    """Reset each RAW/CONFIG-drifted task (+ dependents) via PipelineManager."""
    from yuxin_mea.pipeline import PipelineManager
    from yuxin_mea.tasks import TASK_CLASSES

    pm = PipelineManager(analysis_root, config_provider=cm)
    for cls in TASK_CLASSES:
        try:
            pm.register_task(cls)
        except ValueError:
            pass
    n = 0
    for d in drifted:
        try:
            n += pm.refresh(d["task"], recording_key=d["recording_key"],
                            well_ids=[d["well_id"]])
        except ValueError:
            # Unknown/unregistered task — skip (e.g. a renamed task in old cache).
            pass
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Read-only disk-vs-cache correctness check.")
    p.add_argument("--config", required=True, type=Path,
                   help="pipeline_config.json (for data_root + analysis_root).")
    p.add_argument("--network-only", action="store_true",
                   help="Only compare '/Network/' recordings.")
    p.add_argument("--verify-provenance", action="store_true",
                   help="Also verify each COMPLETE task's provenance stamp against "
                        "the current raw fingerprint + config (reproducibility check).")
    p.add_argument("--hash-raw", choices=["struct", "content", "full"], default=None,
                   help="With --verify-provenance, re-fingerprint each h5 from disk at "
                        "this level instead of trusting the cached fingerprint. "
                        "struct = small analysis-critical datasets only (no bulk raw "
                        "reads); content = + sampled raw; full = + entire raw. All are "
                        "NAS-costly; omit to compare cached fingerprints (free).")
    p.add_argument("--full-hash", action="store_true",
                   help="Deprecated alias for --hash-raw full.")
    p.add_argument("--mark-stale", action="store_true",
                   help="With --verify-provenance, reset RAW/CONFIG-drifted tasks "
                        "(and their dependents) to NOT_RUN so the next run recomputes "
                        "them. Mutates pipeline_cache.json. Metadata drift is left "
                        "alone (labels only).")
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

    key_rc = 1 if missing else 0
    if not missing:
        print("\nCache in sync.")
    else:
        print("\nCache OUT OF SYNC — disk has recordings not in cache.")

    if args.verify_provenance:
        prov_rc = _print_verify(
            cm, data_root, analysis_root,
            hash_mode="full" if args.full_hash else args.hash_raw,
            network_only=args.network_only,
            mark_stale=args.mark_stale,
        )
        return max(key_rc, prov_rc)
    return key_rc


if __name__ == "__main__":
    raise SystemExit(main())
