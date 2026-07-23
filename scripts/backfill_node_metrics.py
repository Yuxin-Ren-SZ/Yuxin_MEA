#!/usr/bin/env python
"""One-time: write node_metrics.parquet / node_scalars.json for wells that lack them.

``ConnectivityTask`` now produces node metrics as part of its normal output, but
the wells that completed before that change do not have them: the files present
on the NAS were written by uncommitted throwaway code, and 382 of 1898 wells were
never covered at all.

Re-running ``connectivity`` to fill the gap is the wrong tool — it would redo the
STTC matrix and its 100-shuffle null for every well, hours of work to obtain
something derivable in a second. The node pass only needs the thresholded edge
table the task already wrote, so this script recomputes it in place:

    edges.parquet  +  sttc_sweep.npz["unit_ids"]  ->  compute_node_metrics()

Both inputs are existing per-well artifacts; no spike times are read, no STTC is
recomputed, and ``pipeline_cache.json`` is never touched (a well's ``connectivity``
record stays COMPLETE and keeps its provenance stamp — the artifacts this fills in
are derived from that same output, not a new run of it).

    python scripts/backfill_node_metrics.py --config pipeline_config_local.json --dry-run
    python scripts/backfill_node_metrics.py --config pipeline_config_local.json

Wells that already have node metrics are skipped unless ``--force``. Verified
before writing this: the committed ``compute_node_metrics`` reproduces the
existing files exactly (24/24 random wells, frame and scalars identical), so a
backfill of only the missing wells leaves one uniform dataset rather than a
mixture of two code paths.

Exit: 0 ok (or dry-run), 1 some wells failed, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger("backfill_node_metrics")

NODE_METRICS_FILE = "node_metrics.parquet"
NODE_SCALARS_FILE = "node_scalars.json"


def connectivity_dirs(analysis_root: Path, recordings: set[str] | None) -> list[Path]:
    """Per-well ``connectivity`` output dirs, from the pipeline cache manifest.

    Uses the recorded ``output_path`` of every COMPLETE ``connectivity`` task
    rather than walking the NAS — the same Tier-0 manifest discipline the viewers
    use. Only dirs that actually carry an ``edges.parquet`` are returned.
    """
    cache_path = analysis_root / "pipeline_cache.json"
    try:
        with cache_path.open() as fh:
            cache = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {cache_path}: {exc}")
    entries = cache.get("entries", cache)

    out: list[Path] = []
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        if recordings and entry.get("recording_key") not in recordings:
            continue
        task = (entry.get("tasks") or {}).get("connectivity")
        if not task or task.get("status") != "complete":
            continue
        path = task.get("output_path")
        if not path:
            continue
        d = Path(path)
        if (d / "edges.parquet").exists():
            out.append(d)
    return sorted(set(out))


def backfill_one(d: Path, *, force: bool, dry_run: bool) -> str:
    """Return an outcome tag for one well: skipped / would-write / written / failed."""
    import numpy as np
    import pandas as pd

    from yuxin_mea.analysis.graph_nodes import compute_node_metrics

    if (d / NODE_METRICS_FILE).exists() and (d / NODE_SCALARS_FILE).exists() and not force:
        return "skipped"
    if dry_run:
        return "would-write"

    edges = pd.read_parquet(d / "edges.parquet")
    # The full unit set (isolated nodes included) so role fractions use the right
    # denominator — same source the connectivity writer stores it in.
    with np.load(d / "sttc_sweep.npz") as z:
        unit_ids = z["unit_ids"]

    res = compute_node_metrics(edges, node_ids=unit_ids)
    res.node_metrics.to_parquet(d / NODE_METRICS_FILE)
    tmp = d / (NODE_SCALARS_FILE + ".tmp")
    tmp.write_text(json.dumps(res.scalars))
    tmp.replace(d / NODE_SCALARS_FILE)
    return "written"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backfill node_metrics.parquet / node_scalars.json into "
                    "existing connectivity outputs.",
    )
    p.add_argument("--config", required=True, type=Path,
                   help="pipeline config json (for analysis_root).")
    p.add_argument("--recordings", default=None,
                   help="Comma-separated recording keys to limit the scope.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many wells are written (do it in bites).")
    p.add_argument("--force", action="store_true",
                   help="Recompute even when node metrics already exist.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be written; write nothing.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    from yuxin_mea.config import ConfigManager

    cm = ConfigManager()
    if not args.config.exists():
        logger.error("config not found: %s", args.config)
        return 2
    cm.load(args.config)
    analysis_root = cm.get_global("analysis_root")
    if not analysis_root:
        logger.error("analysis_root is not set in %s", args.config)
        return 2

    recordings = ({r.strip() for r in args.recordings.split(",") if r.strip()}
                  if args.recordings else None)
    dirs = connectivity_dirs(Path(analysis_root), recordings)
    logger.info("%d connectivity wells in scope", len(dirs))

    counts: Counter[str] = Counter()
    failures: list[tuple[Path, str]] = []
    t0 = time.time()
    for i, d in enumerate(dirs, 1):
        try:
            outcome = backfill_one(d, force=args.force, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001 — one bad well must not stop the sweep
            outcome = "failed"
            failures.append((d, f"{type(exc).__name__}: {exc}"))
            logger.warning("%s: %s", d, exc)
        counts[outcome] += 1
        if outcome in ("written", "would-write") and args.limit \
                and counts["written"] + counts["would-write"] >= args.limit:
            logger.info("--limit %d reached", args.limit)
            break
        if i % 200 == 0:
            logger.info("%d/%d … %s", i, len(dirs), dict(counts))

    logger.info("done in %.1fs: %s", time.time() - t0, dict(counts))
    for d, err in failures[:20]:
        logger.error("FAILED %s: %s", d, err)
    if len(failures) > 20:
        logger.error("… and %d more failures", len(failures) - 20)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
