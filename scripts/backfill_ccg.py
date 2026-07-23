#!/usr/bin/env python
"""Backfill cross-correlograms into existing connectivity outputs.

Wells computed before CCGs landed have ``edges.parquet`` but no ``ccg.npz`` and
no ``ccg_*`` columns. Re-running :class:`ConnectivityTask` would rebuild them,
but the expensive part of that stage is the ``n_shuffle=100`` circular-shuffle
null — which the correlograms do not need. This walks existing connectivity
output dirs and adds, in place:

* ``ccg.npz``            — counts + hollow-Gaussian baseline, keyed by ``pairs``
* ``edges.parquet``      — the ``ccg_*`` columns joined on ``(u, v)``
* ``graph_metrics.json`` — the ``ccg_*`` well-level summary
* ``diagnostics.json``   — the ``ccg_*`` provenance of this run

The STTC matrix, the adjacency and every pre-existing metric are left untouched.

    python scripts/backfill_ccg.py --config pipeline_config_run.json --dry-run
    python scripts/backfill_ccg.py --config pipeline_config_run.json \
        --recording-filter Network --limit 50

Idempotent: a well that already has ``ccg.npz`` is skipped unless ``--force``.
CCG params come from the config's ``connectivity`` section (falling back to the
task defaults), so a backfilled well matches what a fresh run would produce.

Wells are independent (separate output dirs, atomic writes), so ``--jobs N``
fans them across processes. The work is NAS-read-bound (loading each well's
``curated_spike_times.npy``), so processes overlap I/O rather than saturating a
CPU — 8-11 is a reasonable range. Keep the per-well ``ConnectivityConfig``
``n_jobs`` at 1: this is the outer parallelism.

Note: adding the ``ccg_*`` params to the connectivity config changes that task's
params hash, so ``--verify-provenance`` will report CONFIG-CHANGED for wells that
predate them. Backfilling does not clear that flag — it only makes the artifacts
current.

Exit: 0 = every requested well backfilled (or already current), 1 = some failed,
2 = usage/config error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from yuxin_mea.analysis.connectivity import (
    CCG_COLUMNS,
    ConnectivityConfig,
    ConnectivityResults,
    _atomic_json_write,
    attach_ccg,
    ccg_summary,
    compute_ccg,
    write_ccg_npz,
)
from yuxin_mea.config import ConfigManager
from yuxin_mea.tasks.connectivity import ConnectivityTask

logger = logging.getLogger("backfill_ccg")


def _resolve_root(analysis_root: Path, value: str | None, default: str) -> Path:
    """Config roots are written relative to analysis_root (or absolute)."""
    p = Path(value) if value else Path(default)
    return p if p.is_absolute() else analysis_root / p


def _well_dirs(conn_root: Path) -> list[Path]:
    """Every ``.../connectivity`` dir that carries an edges table."""
    return sorted(p.parent for p in conn_root.glob("**/connectivity/edges.parquet"))


def _curation_dir(conn_dir: Path, conn_root: Path, curation_root: Path) -> Path:
    """`<conn_root>/<rel>/connectivity` -> `<curation_root>/<rel>/auto_curation`."""
    rel = conn_dir.parent.relative_to(conn_root)
    return curation_root / rel / "auto_curation"


def _load_spike_times(curation_dir: Path) -> dict | None:
    path = curation_dir / "curated_spike_times.npy"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True).item()


def backfill_well(
    conn_dir: Path, curation_dir: Path, cfg: ConnectivityConfig, dry_run: bool = False,
) -> str:
    """Backfill one well. Returns a short status word for the run summary."""
    edges = pd.read_parquet(conn_dir / "edges.parquet")
    if not len(edges):
        # A sparse well legitimately has no edges — still stamp an empty block so
        # the artifact set matches a fresh run.
        spike_times: dict = {}
    else:
        spike_times = _load_spike_times(curation_dir)  # type: ignore[assignment]
        if spike_times is None:
            return "no-spikes"

    if dry_run:
        return "would-backfill"

    # Drop any stale ccg_* columns so a re-run cannot end up with _x/_y suffixes.
    edges = edges.drop(columns=[c for c in CCG_COLUMNS if c in edges.columns])
    ccg = compute_ccg(spike_times, edges, config=cfg)
    edges = attach_ccg(edges, ccg["per_pair"])

    gm_path = conn_dir / "graph_metrics.json"
    gm = json.loads(gm_path.read_text()) if gm_path.exists() else {}
    gm.update(ccg_summary(ccg["per_pair"]))

    diag_path = conn_dir / "diagnostics.json"
    diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}
    diag.update(ccg["meta"])

    results = ConnectivityResults(
        sttc_matrix=np.zeros((0, 0)), sttc_sweep={}, adjacency=np.zeros((0, 0), bool),
        graph_metrics=gm, edges=edges, unit_ids=[], diagnostics=diag,
        ccg_counts=ccg["counts"], ccg_baseline=ccg["baseline"],
        ccg_lags_ms=ccg["lags_ms"], ccg_pairs=ccg["pairs"],
        ccg_n_ref_spikes=ccg["n_ref_spikes"], ccg_n_tgt_spikes=ccg["n_tgt_spikes"],
        ccg_meta=ccg["meta"],
    )
    # This rewrites live analysis outputs, so every write lands atomically: a
    # crash (or a full disk) must never leave a half-written edges.parquet where
    # a complete one used to be.
    _replace(conn_dir / "ccg.npz", lambda p: write_ccg_npz(results, p))
    _replace(conn_dir / "edges.parquet", lambda p: edges.to_parquet(p))
    _atomic_json_write(gm, gm_path)
    _atomic_json_write(diag, diag_path)
    return "ok"


def _replace(dest: Path, write) -> None:
    """Write via ``write(tmp)`` next to ``dest``, then atomically rename over it."""
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        write(tmp)
        # np.savez appends .npz to a path that lacks it — find what was written.
        if not tmp.exists() and tmp.with_suffix(tmp.suffix + ".npz").exists():
            tmp = tmp.with_suffix(tmp.suffix + ".npz")
        os.replace(tmp, dest)
    except Exception:
        for stale in (tmp, tmp.with_suffix(tmp.suffix + ".npz")):
            try:
                stale.unlink()
            except OSError:
                pass
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="pipeline config JSON")
    ap.add_argument("--recording-filter", default="",
                    help="only wells whose path contains this substring")
    ap.add_argument("--limit", type=int, default=0, help="stop after N wells")
    ap.add_argument("--force", action="store_true",
                    help="recompute wells that already have ccg.npz")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be backfilled, write nothing")
    ap.add_argument("--jobs", type=int, default=1,
                    help="processes fanning wells in parallel (NAS-read-bound; "
                    "8-11 is reasonable). 1 = serial.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    cm = ConfigManager()
    try:
        cm.load(args.config)
    except Exception as exc:  # noqa: BLE001
        logger.error("cannot load config %s: %s", args.config, exc)
        return 2

    analysis_root = Path(cm.get_global("analysis_root", "."))
    params = {**ConnectivityTask.default_params(), **cm.get_task_params("connectivity")}
    cfg = ConnectivityConfig.from_task_params(params)
    if not cfg.ccg_enable:
        logger.error("ccg_enable is False in %s — nothing to backfill", args.config)
        return 2

    conn_root = _resolve_root(analysis_root, params.get("output_root"),
                              "connectivity_data")
    curation_root = _resolve_root(analysis_root, params.get("curation_output_root"),
                                  "curation_data")
    if not conn_root.exists():
        logger.error("connectivity root not found: %s", conn_root)
        return 2
    logger.info("connectivity root %s", conn_root)
    logger.info("curation root     %s", curation_root)

    dirs = _well_dirs(conn_root)
    if args.recording_filter:
        dirs = [d for d in dirs if args.recording_filter in str(d)]
    logger.info("%d wells with an edges table", len(dirs))

    # Skip already-current wells up front (cheap stat) so the work list — and any
    # --limit — counts only wells that actually need computing.
    counts: dict[str, int] = {}
    todo: list[Path] = []
    for conn_dir in dirs:
        if (conn_dir / "ccg.npz").exists() and not args.force:
            counts["skip-current"] = counts.get("skip-current", 0) + 1
        else:
            todo.append(conn_dir)
    if args.limit:
        todo = todo[: args.limit]
    logger.info("%d wells to backfill (%d already current)",
                len(todo), counts.get("skip-current", 0))

    def _one(conn_dir: Path) -> tuple[str, Path, str]:
        try:
            status = backfill_well(
                conn_dir, _curation_dir(conn_dir, conn_root, curation_root),
                cfg, dry_run=args.dry_run,
            )
            return status, conn_dir, ""
        except Exception as exc:  # noqa: BLE001 — one bad well must not stop the run
            return "failed", conn_dir, str(exc)

    if args.jobs and args.jobs != 1 and len(todo) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(_one)(d) for d in todo)
    else:
        results = [_one(d) for d in todo]

    failed = 0
    for status, conn_dir, msg in results:
        counts[status] = counts.get(status, 0) + 1
        if status == "failed":
            failed += 1
            logger.warning("%s: %s", conn_dir, msg)
        else:
            logger.debug("%s %s", status, conn_dir)

    logger.info("done: %s", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
