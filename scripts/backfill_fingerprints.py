#!/usr/bin/env python
"""Backfill raw-data fingerprints into experiment_cache.json (provenance).

Recordings scanned before provenance existed (or scanned with the default cheap
``fingerprint_mode="stat"``) carry no content fingerprint, so their outputs can
only ever verify as UNKNOWN. This walks the *already-cached* recordings and fills
in ``RecordingEntry.raw_fingerprint`` at the tier you choose.

    # recommended: cheap tier, Network recordings only, see the plan first
    python scripts/backfill_fingerprints.py --config pipeline_config_local.json \
        --mode struct --network-only --dry-run

    python scripts/backfill_fingerprints.py --config pipeline_config_local.json \
        --mode struct --network-only

Design notes (learned the hard way — see doc/caching.md):

- **NAS-bound, not CPU-bound.** Hashing a 30-100 GB h5 is dominated by the HDF5
  tree walk + scattered chunk reads; under load a single file can stall for many
  minutes. Hence: incremental saves, per-file timing, ETA, and ``--limit`` so you
  can do it in bites. Run off-peak.
- **Resumable / idempotent.** A recording whose fingerprint is already at the
  requested tier *and* whose h5 size+mtime are unchanged is skipped, so just
  re-run after an interrupt. ``Ctrl-C`` saves progress before exiting.
- **Reads the dataset cache directly** (not ``DatasetManager``) because
  constructing the manager triggers an incremental disk scan of ``data_root`` —
  an unwanted NAS walk here. Same rationale as ``dashboard/data.py``.
- **Does not run analysis** and never touches ``pipeline_cache.json``. Existing
  task stamps are untouched; this only makes *future* stamps (and
  ``--verify-provenance``) content-aware.

Do not run while a "Scan disk" / refresh is writing experiment_cache.json — last
writer wins.

Exit: 0 = all requested recordings fingerprinted (or already fresh), 1 = some
failed, 2 = usage/config error.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from yuxin_mea.config import ConfigManager
from yuxin_mea.dataset.cache import JsonCacheStore
from yuxin_mea.provenance import file_hash, h5_fingerprint, stat_sig
from yuxin_mea.provenance.fingerprint import H5_HASH_MODES, h5_kwargs_for, h5_method_for

logger = logging.getLogger("backfill_fingerprints")


def _is_fresh(entry, data_file: Path, want_method: str) -> bool:
    """True when the cached h5 fingerprint is already at want_method and current."""
    h5 = (entry.raw_fingerprint or {}).get("h5") or {}
    if h5.get("method") != want_method:
        return False
    cur = stat_sig(data_file)
    return (h5.get("file_size") == cur["size"]
            and h5.get("mtime_ns") == cur["mtime_ns"])


def _select(entries: dict, args) -> list:
    """Apply the recording filters, returning entries sorted for stable progress."""
    keys = {k.strip() for k in args.recordings.split(",")} if args.recordings else None
    out = []
    for e in entries.values():
        if keys is not None and e.cache_key not in keys:
            continue
        if args.sample and e.sample_id not in {s.strip() for s in args.sample.split(",")}:
            continue
        if args.scan_type and e.scan_type not in {s.strip() for s in args.scan_type.split(",")}:
            continue
        if args.network_only and e.scan_type != "Network":
            continue
        out.append(e)
    out.sort(key=lambda e: (e.sample_id, e.date, e.run_id))
    return out


def _fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backfill raw-data fingerprints into experiment_cache.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Tiers: struct = small analysis-critical datasets + raw shapes (no bulk "
               "reads, recommended); content = + sampled raw; full = + entire raw "
               "(very slow on 30-100 GB). See doc/caching.md.",
    )
    p.add_argument("--config", required=True, type=Path,
                   help="pipeline config json (for data_root + analysis_root).")
    p.add_argument("--mode", choices=list(H5_HASH_MODES), default="struct",
                   help="Fingerprint tier to compute (default: struct).")
    p.add_argument("--force", action="store_true",
                   help="Re-hash even when the cached fingerprint is already current.")
    # scope
    p.add_argument("--recordings", default=None,
                   help="Comma-separated recording keys (Sample/Date/Plate/Scan/Run).")
    p.add_argument("--sample", default=None, help="Comma-separated sample_ids (e.g. CX118).")
    p.add_argument("--scan-type", default=None, help="Comma-separated scan types.")
    p.add_argument("--network-only", action="store_true", help="Only Network recordings.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many recordings are hashed (do it in bites).")
    # behaviour
    p.add_argument("--save-every", type=int, default=5,
                   help="Persist experiment_cache.json every N recordings (default 5). "
                        "Keeps progress if the NAS stalls and you interrupt.")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be hashed (and what's already fresh); write nothing.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s — %(message)s")

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2
    cm = ConfigManager()
    cm.load(args.config)
    data_root, analysis_root = cm.get_global("data_root"), cm.get_global("analysis_root")
    if not data_root or not analysis_root:
        print("ERROR: data_root and analysis_root must be set in the config.", file=sys.stderr)
        return 2
    data_root, analysis_root = Path(data_root), Path(analysis_root)

    store = JsonCacheStore(analysis_root)
    entries = store.load()
    if not entries:
        print(f"No cached recordings in {analysis_root}/experiment_cache.json — "
              "run a scan first (Recordings → Scan disk, or scripts/refresh_cache.py).")
        return 2

    want_method = h5_method_for(args.mode)
    kwargs = h5_kwargs_for(args.mode)
    selected = _select(entries, args)

    todo, fresh = [], 0
    for e in selected:
        if not args.force and _is_fresh(e, data_root / e.data_path, want_method):
            fresh += 1
        else:
            todo.append(e)
    if args.limit is not None:
        todo = todo[:args.limit]

    print(f"config        : {args.config}")
    print(f"data_root     : {data_root}")
    print(f"tier          : {args.mode}  (method {want_method})")
    print(f"cached recs   : {len(entries)}   selected: {len(selected)}")
    print(f"already fresh : {fresh}   to hash: {len(todo)}"
          + (f"  (--limit {args.limit})" if args.limit is not None else ""))
    if args.dry_run:
        for e in todo:
            print(f"  would hash  {e.cache_key}")
        print("\n(dry-run — nothing written)")
        return 0
    if not todo:
        print("\nNothing to do — all selected recordings already fingerprinted at this tier.")
        return 0
    print("\nHashing (NAS-bound; Ctrl-C saves progress and exits)...\n", flush=True)

    done = failed = 0
    t0 = time.time()
    unsaved = 0

    def _save() -> None:
        nonlocal unsaved
        if unsaved:
            store.save(entries)
            logger.debug("saved experiment_cache.json (%d new)", unsaved)
            unsaved = 0

    try:
        for i, e in enumerate(todo, 1):
            data_file = data_root / e.data_path
            t = time.time()
            try:
                e.raw_fingerprint["h5"] = h5_fingerprint(data_file, **kwargs)
                meta = data_file.parent / "mxassay.metadata"
                e.raw_fingerprint["metadata"] = file_hash(meta) if meta.is_file() else None
                done += 1
                unsaved += 1
            except Exception as exc:  # noqa: BLE001 — one bad file must not abort the run
                failed += 1
                logger.warning("FAILED %s: %s", e.cache_key, exc)
                continue
            finally:
                dt = time.time() - t
                elapsed = time.time() - t0
                rate = elapsed / max(1, i)
                eta = rate * (len(todo) - i)
                print(f"[{i}/{len(todo)}] {_fmt(dt):>6} {e.cache_key}   "
                      f"(elapsed {_fmt(elapsed)}, eta {_fmt(eta)})", flush=True)
            if unsaved >= args.save_every:
                _save()
    except KeyboardInterrupt:
        _save()
        print(f"\nInterrupted — saved {done} fingerprint(s). Re-run to resume "
              "(already-hashed recordings are skipped).")
        return 1
    finally:
        _save()

    print(f"\nDone in {_fmt(time.time() - t0)}: {done} hashed, {fresh} already fresh, "
          f"{failed} failed.")
    if failed:
        print("Some recordings failed — re-run to retry just those.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
