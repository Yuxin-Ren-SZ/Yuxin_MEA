#!/usr/bin/env python
"""One-time: stamp provenance onto pipeline outputs that predate provenance.

Tasks completed before provenance existed carry ``provenance=None`` and can only
ever verify as UNKNOWN. This retro-stamps them so they become verifiable going
forward. Two very different things happen, and the distinction matters:

- **config_hash — a true measurement, not an assumption.** ``TaskRecord.config``
  is the config snapshot frozen when the task transitioned to RUNNING, i.e. the
  params the task *actually ran with*. We hash that. So a task that ran with old
  params correctly reports CONFIG-CHANGED against today's config — nothing is
  papered over.

- **h5 / metadata — an *adopted baseline*, i.e. an assumption.** We cannot know
  what bytes were read months ago; we copy the raw fingerprint as it is *now* and
  assert "this output came from the file currently on disk". That is only honest
  when the file has not been touched since the task ran, so it is **guarded**:
  a task is skipped unless ``raw mtime <= task.last_updated``. Adopted stamps are
  marked ``"adopted": true`` so nobody later mistakes them for measured provenance.

    python scripts/adopt_provenance_baseline.py --config pipeline_config_local.json --dry-run
    python scripts/adopt_provenance_baseline.py --config pipeline_config_local.json

Run *after* scripts/backfill_fingerprints.py, so recordings have a content
fingerprint to adopt; without one the stamp records config only (raw stays
unverifiable, which is the truthful outcome).

Only ever touches tasks with no stamp — real stamps written by a run are never
overwritten. Writes pipeline_cache.json; do not run while yuxin-mea-run is
draining the same analysis_root (last writer wins).

Exit: 0 ok, 1 nothing adopted / some skipped as unsafe, 2 usage error.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from yuxin_mea.config import ConfigManager
from yuxin_mea.dataset.cache import JsonCacheStore
from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
from yuxin_mea.provenance import params_hash

logger = logging.getLogger("adopt_provenance_baseline")

_ADOPT_NOTE = ("raw fingerprint adopted from the file on disk at backfill time, not "
               "measured during the original run; guarded by raw mtime <= last_updated")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retro-stamp provenance onto pre-provenance pipeline outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be stamped; write nothing.")
    p.add_argument("--force", action="store_true",
                   help="Adopt even when the raw file is NEWER than the task's completion "
                        "(i.e. the file changed after the analysis ran). Strongly "
                        "discouraged — it stamps a claim known to be false.")
    p.add_argument("--upgrade", action="store_true",
                   help="Also re-stamp earlier *adopted, config-only* stamps whose "
                        "recording has since been fingerprinted (e.g. you ran this "
                        "before the backfill finished). Never touches stamps measured "
                        "by a real run.")
    p.add_argument("--recordings", default=None, help="Comma-separated recording keys.")
    p.add_argument("--network-only", action="store_true")
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
    analysis_root = cm.get_global("analysis_root")
    if not analysis_root:
        print("ERROR: analysis_root not set in config.", file=sys.stderr)
        return 2
    analysis_root = Path(analysis_root)

    recs = JsonCacheStore(analysis_root).load()
    pstore = JsonPipelineCacheStore(analysis_root)
    pipe = pstore.load()
    keys = {k.strip() for k in args.recordings.split(",")} if args.recordings else None

    now = time.time()
    stats = Counter()
    adopted_cfg_state = Counter()

    for pkey, entry in sorted(pipe.items()):
        rkey = entry.recording_key
        if keys is not None and rkey not in keys:
            continue
        rec = recs.get(rkey)
        if args.network_only and (rec is None or rec.scan_type != "Network"):
            continue

        for tname, tr in entry.tasks.items():
            if tr.status != "complete":
                continue
            if tr.provenance is not None:
                prov = tr.provenance
                # Only an *adopted, config-only* stamp may be upgraded, and only once
                # its recording actually has a fingerprint. A stamp measured by a real
                # run is never touched.
                upgradable = (
                    args.upgrade
                    and prov.get("adopted") is True
                    and prov.get("h5") is None
                    and str((((rec.raw_fingerprint if rec else {}) or {}).get("h5")
                             or {}).get("method", "")).startswith("h5")
                )
                if not upgradable:
                    stats["already stamped (left alone)"] += 1
                    continue
            if not tr.config:
                stats["skipped: no frozen config snapshot"] += 1
                continue
            if tr.last_updated is None:
                stats["skipped: no last_updated (cannot time-guard)"] += 1
                continue

            rf = (rec.raw_fingerprint if rec else {}) or {}
            h5 = rf.get("h5") or {}
            has_content = str(h5.get("method", "")).startswith("h5")

            # Time guard: only adopt the raw fingerprint if the file has not been
            # touched since this task completed.
            raw_ok = rec is not None and rec.mtime <= tr.last_updated
            if not raw_ok and not args.force:
                stats["skipped: raw NEWER than task (unsafe to adopt)"] += 1
                continue

            stamp = {
                "config_hash": params_hash(tr.config),   # measured (frozen snapshot)
                "config_file": Path(args.config).name,
                "stamped_at": now,
                "adopted": True,
                "adopted_note": _ADOPT_NOTE,
            }
            if has_content:
                stamp["h5"] = dict(h5)
                stamp["metadata"] = rf.get("metadata")
                stats["adopted (raw + config)"] += 1
            else:
                # No fingerprint to adopt → config-only stamp. Raw stays
                # unverifiable, which is the truthful result.
                stamp["h5"] = None
                stamp["metadata"] = None
                stats["adopted (config only — recording not fingerprinted)"] += 1
            if not raw_ok:
                stamp["adopted_unsafe"] = True  # --force was used

            # What will this stamp verify as, against the current config?
            cur = params_hash(cm.get_task_params(tname))
            adopted_cfg_state["CONFIG-CHANGED" if stamp["config_hash"] != cur else "OK"] += 1

            if not args.dry_run:
                tr.provenance = stamp

    print(f"config        : {args.config}")
    print(f"analysis_root : {analysis_root}\n")
    for k, v in sorted(stats.items()):
        print(f"  {v:6d}  {k}")
    print("\nHow the newly-stamped tasks will verify against the CURRENT config:")
    for k, v in sorted(adopted_cfg_state.items()):
        print(f"  {v:6d}  {k}")
    n = sum(v for k, v in stats.items() if k.startswith("adopted"))
    if args.dry_run:
        print("\n(dry-run — nothing written)")
        return 0
    if n:
        pstore.save(pipe)
        print(f"\nWrote {n} adopted stamp(s) to pipeline_cache.json.")
        print("NOTE: these are marked 'adopted' — the raw fingerprint is a baseline "
              "assertion (guarded by mtime), not provenance measured during the run.")
    else:
        print("\nNothing adopted.")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
