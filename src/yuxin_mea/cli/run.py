"""`yuxin-mea-run` worker CLI — drains the pipeline queue for a config.

Runs the S2 `UnifiedScheduler` over the layered store: the finest (per-well) layer
lives in `pipeline_cache.json` (dashboard-native), the aggregate layer in
`instance_store.json`. The drain runs the finest phase first (serial or, with
`--jobs N`, a ProcessPool where the parent owns all store writes), then aggregate
instances serially.

The dashboard's Run page generates the exact invocation string for this CLI;
it never spawns long-running work itself.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path

import json

from yuxin_mea.config import ConfigManager
from yuxin_mea.dataset import DatasetManager
from yuxin_mea.pipeline import (
    LayeredInstanceStore,
    UnifiedScheduler,
    WorkItem,
)
from yuxin_mea.pipeline import unified_scheduler as _uni
from yuxin_mea.pipeline.task_record import TaskStatus
from yuxin_mea.tasks import TASK_CLASSES, UNIFIED_AGGREGATE_TASK_CLASSES


logger = logging.getLogger("yuxin_mea.run")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yuxin-mea-run",
        description=(
            "Drain the per-well pipeline queue for a yuxin_mea config. "
            "Loops get_next_task → task.run → update_status until empty."
        ),
    )
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to pipeline_config.json (same file the dashboard reads).",
    )
    p.add_argument(
        "--tasks",
        default=None,
        help=(
            "Comma-separated task allowlist (e.g. 'preprocessing,sorting'). "
            "Defaults to all registered tasks."
        ),
    )
    p.add_argument(
        "--recordings",
        default=None,
        help=(
            "Comma-separated recording-key allowlist (e.g. "
            "'Sample/2025-01-01/Plate/Network/000001,...'). "
            "Defaults to every queued recording."
        ),
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Also re-run tasks currently in FAILED state.",
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Stop after this many tasks complete (useful for smoke tests). "
             "This is a cap on total work — distinct from --jobs which sets "
             "how many workers run concurrently.",
    )
    p.add_argument(
        "--jobs", "-j",
        type=int,
        default=1,
        help="Concurrent worker processes (default 1 = sequential). "
             "Each worker re-loads its own ConfigManager/DatasetManager; the "
             "parent owns all pipeline cache writes.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the queue plan and exit without executing.",
    )
    p.add_argument(
        "--aggregate",
        dest="aggregate",
        action="store_true",
        default=True,
        help="After the per-well drain, run scope-level aggregate tasks (default on).",
    )
    p.add_argument(
        "--no-aggregate",
        dest="aggregate",
        action="store_false",
        help="Skip the aggregate-task pass (per-well drain only).",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip the per-well drain and run ONLY the aggregate-task pass "
             "(useful for regenerating figures without recomputing wells).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default INFO).",
    )
    return p


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [s.strip() for s in value.split(",") if s.strip()]
    return items or None


def _setup_pipeline(config_path: Path) -> tuple[ConfigManager, DatasetManager]:
    cm = ConfigManager()
    for cls in TASK_CLASSES:
        cm.register_task(cls)
    cm.load(config_path)

    data_root = cm.get_global("data_root")
    analysis_root = cm.get_global("analysis_root")
    if not data_root or not analysis_root:
        raise SystemExit(
            "data_root and analysis_root must be set in the config (use "
            "`yuxin-mea-dashboard --config ...` Settings page to edit)."
        )

    dataset_mgr = DatasetManager(Path(data_root), Path(analysis_root))
    return cm, dataset_mgr


def _resolve_recording_path(dataset_mgr: DatasetManager, recording_key: str) -> Path:
    matches = dataset_mgr.get_recording_by([("cache_key", "==", recording_key)])
    if not matches:
        raise KeyError(
            f"Recording {recording_key!r} not found in DatasetManager. "
            "Has it been scanned (Recordings page → Scan disk)?"
        )
    return dataset_mgr.get_path(matches[0])


def _run_one_worker(work_item: WorkItem, config_path: Path) -> dict:
    """Run a single task in a child process. Pickle-safe (top-level fn).

    The worker rebuilds ConfigManager + DatasetManager from disk — it does
    NOT touch the pipeline cache. The parent process owns all status writes
    so concurrent workers can't race on pipeline_cache.json.
    """
    cm = ConfigManager()
    for cls in TASK_CLASSES:
        cm.register_task(cls)
    cm.load(config_path)
    data_root = cm.get_global("data_root")
    analysis_root = cm.get_global("analysis_root")
    if not data_root or not analysis_root:
        return {
            "status": TaskStatus.FAILED,
            "output_path": None,
            "error": "data_root or analysis_root unset in config",
            "elapsed": 0.0,
        }
    dataset_mgr = DatasetManager(Path(data_root), Path(analysis_root))

    task_cls = next(
        (c for c in TASK_CLASSES if c.task_name == work_item.task_name),
        None,
    )
    if task_cls is None:
        return {
            "status": TaskStatus.FAILED,
            "output_path": None,
            "error": f"unknown task {work_item.task_name!r}",
            "elapsed": 0.0,
        }

    params = cm.get_task_params(work_item.task_name)
    try:
        data_path = _resolve_recording_path(dataset_mgr, work_item.recording_key)
    except KeyError as exc:
        return {
            "status": TaskStatus.FAILED,
            "output_path": None,
            "error": str(exc),
            "elapsed": 0.0,
        }

    task = task_cls()
    t0 = time.time()
    try:
        output_path = task.run(
            work_item.recording_key,
            work_item.well_id,
            data_path,
            params,
        )
    except Exception:
        return {
            "status": TaskStatus.FAILED,
            "output_path": None,
            "error": traceback.format_exc(),
            "elapsed": time.time() - t0,
        }
    return {
        "status": TaskStatus.COMPLETE,
        "output_path": output_path,
        "error": None,
        "elapsed": time.time() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if not args.config.exists():
        print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
        return 2

    cm, dataset_mgr = _setup_pipeline(args.config)

    task_allow = _split_csv(args.tasks)
    rec_allow = _split_csv(args.recordings)

    if task_allow:
        registered = (
            {cls.task_name for cls in TASK_CLASSES}
            | {cls.task_name for cls in UNIFIED_AGGREGATE_TASK_CLASSES}
        )
        bad = [t for t in task_allow if t not in registered]
        if bad:
            print(
                f"ERROR: unknown --tasks values: {bad}. "
                f"Known tasks: {sorted(registered)}",
                file=sys.stderr,
            )
            return 2

    return _main_unified(args, cm, dataset_mgr, task_allow, rec_allow)


def _load_experiment_cache(analysis_root: Path) -> dict:
    exp_path = analysis_root / "experiment_cache.json"
    try:
        with exp_path.open() as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        logger.warning(
            "experiment_cache.json not readable at %s; groupname-scoped tasks "
            "bucket under '?'.", exp_path,
        )
        return {}


def _build_unified_scheduler(cm: ConfigManager) -> UnifiedScheduler:
    """Build the S2 scheduler over the layered store: the finest layer reads/writes
    the dashboard-native ``pipeline_cache.json``, the aggregate layer
    ``instance_store.json`` (migrating S1's ``aggregate_cache.json`` on first use)."""
    analysis_root = Path(cm.get_global("analysis_root"))
    store = LayeredInstanceStore(analysis_root)
    exp = _load_experiment_cache(analysis_root)
    return UnifiedScheduler(store, UNIFIED_AGGREGATE_TASK_CLASSES, cm, exp)


def _main_unified(
    args, cm: ConfigManager, dataset_mgr: DatasetManager,
    task_allow: list[str] | None, rec_allow: list[str] | None,
) -> int:
    sched = _build_unified_scheduler(cm)

    if rec_allow and not args.aggregate_only:
        well_specs = [(c.task_name, list(c.dependencies)) for c in TASK_CLASSES]
        n = 0
        for rec_key in rec_allow:
            matches = dataset_mgr.get_recording_by([("cache_key", "==", rec_key)])
            if not matches or not matches[0].h5_recordings:
                logger.warning("Recording %r not scanned / has no wells — skipping.", rec_key)
                continue
            for rec_name, well_ids in matches[0].h5_recordings.items():
                for well_id in well_ids:
                    sched.register_finest(rec_key, f"{rec_name}/{well_id}", well_specs)
                    n += 1
        if n:
            logger.info("Registered %d finest instance(s) [unified].", n)

    if args.dry_run:
        if not args.aggregate_only:
            finest = sched.finest_queue(
                n=10_000, retry_failed=args.retry_failed,
                recording_keys=rec_allow, task_names=task_allow,
            )
            for wu in finest:
                rk = wu.scope_key["recording_key"]
                cw = f"{wu.scope_key['rec_name']}/{wu.scope_key['well_id']}"
                print(f"[dry-run] would run {wu.task_name} on {rk}/{cw}")
            print(f"{len(finest)} per-well task(s) eligible now [unified].")
        for dec in sched.aggregate_plan(task_allow):
            print(
                f"[dry-run agg] {dec.decision.upper():>10}  {dec.task_name}  "
                f"{dec.scope_key}  (complete {len(dec.members_complete)}/{dec.n_total})"
            )
        print("Done (dry-run; unified; no state changes).")
        return 0

    if not args.aggregate_only:
        n_rec = sched.recover_from_crash()
        if n_rec:
            logger.warning("Recovered %d task(s) left RUNNING by a previous run.", n_rec)

    n_ran, n_failed = _drain_unified(
        sched, cm, dataset_mgr,
        task_allow=task_allow, rec_allow=rec_allow,
        retry_failed=args.retry_failed, max_tasks=args.max_tasks,
        aggregate=(args.aggregate or args.aggregate_only),
        finest=not args.aggregate_only,
        jobs=args.jobs, config_path=args.config,
    )
    print(f"Done [unified]. Ran {n_ran} task(s); {n_failed} failed.")
    return 0 if n_failed == 0 else 1


def _run_finest_unit(wu, cm, dataset_mgr, sched, task_instances) -> str:
    task = task_instances[wu.task_name]
    rk = wu.scope_key["recording_key"]
    compound = f"{wu.scope_key['rec_name']}/{wu.scope_key['well_id']}"
    params = cm.get_task_params(wu.task_name)
    try:
        data_path = _resolve_recording_path(dataset_mgr, rk)
    except KeyError as exc:
        sched.update_status("well", wu.scope_key, wu.task_name, TaskStatus.FAILED, error=str(exc))
        logger.error("Task FAILED %s/%s/%s: %s", rk, compound, wu.task_name, exc)
        return TaskStatus.FAILED
    config = cm.get_config(wu.task_name, rk, compound)
    sched.update_status("well", wu.scope_key, wu.task_name, TaskStatus.RUNNING, config=config)
    t0 = time.time()
    try:
        out = task.run(rk, compound, data_path, params)
    except Exception:
        tb = traceback.format_exc()
        sched.update_status("well", wu.scope_key, wu.task_name, TaskStatus.FAILED, error=tb)
        logger.error("Task FAILED %s/%s/%s", rk, compound, wu.task_name)
        return TaskStatus.FAILED
    sched.update_status("well", wu.scope_key, wu.task_name, TaskStatus.COMPLETE, output_path=out)
    logger.info("Task COMPLETE %s/%s/%s in %.1fs → %s", rk, compound, wu.task_name,
                time.time() - t0, out)
    return TaskStatus.COMPLETE


def _run_agg_unit(dec, cm, sched, agg_instances, figure_root) -> str:
    task = agg_instances[dec.task_name]
    cfg = cm.get_task_params(dec.task_name)
    params = dict(cfg)
    params.setdefault("output_root", figure_root)
    n_complete = len(dec.members_complete)
    sched.update_status(
        dec.scope_name, dec.scope_key, dec.task_name, TaskStatus.RUNNING,
        config=cfg, dependencies=list(task.dependencies),
        member_hash=dec.member_hash, n_members_total=dec.n_total,
        n_members_complete=n_complete,
    )
    t0 = time.time()
    try:
        out = task.run(dec.scope_key, dec.members_complete, params)
    except Exception:
        tb = traceback.format_exc()
        logger.error("Aggregate FAILED %s %s: %s", dec.task_name, dec.scope_key,
                     tb.strip().splitlines()[-1])
        sched.update_status(
            dec.scope_name, dec.scope_key, dec.task_name, TaskStatus.FAILED, error=tb,
            config=cfg, dependencies=list(task.dependencies),
            member_hash=dec.member_hash, n_members_total=dec.n_total,
            n_members_complete=n_complete,
        )
        return TaskStatus.FAILED
    sched.update_status(
        dec.scope_name, dec.scope_key, dec.task_name, TaskStatus.COMPLETE, output_path=out,
        config=cfg, dependencies=list(task.dependencies),
        member_hash=dec.member_hash, n_members_total=dec.n_total,
        n_members_complete=n_complete,
    )
    logger.info("Aggregate %s %s %s in %.1fs → %s (%d/%d members)",
                dec.decision.upper(), dec.task_name, dec.scope_key,
                time.time() - t0, out, n_complete, dec.n_total)
    return TaskStatus.COMPLETE


def _drain_unified(
    sched: UnifiedScheduler, cm: ConfigManager, dataset_mgr: DatasetManager, *,
    task_allow, rec_allow, retry_failed, max_tasks, aggregate, finest,
    jobs: int = 1, config_path: Path | None = None,
    well_instances=None, agg_instances=None, finest_runner=None,
) -> tuple[int, int]:
    """Drain the unified store: the finest (per-well) phase first, then aggregates.

    Finest before aggregates means the barrier sees no eligible-now blockers, so
    aggregates fire on their complete subsets (as in S1). The finest phase runs
    in parallel when ``jobs > 1`` (parent owns all store writes, mirroring the
    legacy `_drain_parallel`); aggregates run serially (cheap, and containment
    ordering is easiest to reason about serially — parallel aggregates are a
    non-blocking follow-up).

    ``well_instances``/``agg_instances`` default to the real registries; tests
    inject fakes for the serial path. ``finest_runner`` (a callable
    ``(task_name, recording_key, compound_well_id) -> result-dict``) lets tests
    drive the PARALLEL path in-process (ThreadPool) without spawning subprocesses.
    """
    if well_instances is None:
        well_instances = {c.task_name: c() for c in TASK_CLASSES}
    if agg_instances is None:
        agg_instances = {c.task_name: c() for c in UNIFIED_AGGREGATE_TASK_CLASSES}
    figure_root = cm.get_global("figure_root") or cm.get_global("analysis_root")

    n_ran = n_failed = 0
    if finest:
        if (jobs and jobs > 1) or finest_runner is not None:
            fr, ff = _drain_finest_parallel(
                sched, cm, dataset_mgr,
                task_allow=task_allow, rec_allow=rec_allow, retry_failed=retry_failed,
                max_tasks=max_tasks, jobs=max(2, jobs or 2),
                config_path=config_path, finest_runner=finest_runner,
            )
        else:
            fr, ff = _drain_finest_serial(
                sched, cm, dataset_mgr,
                task_allow=task_allow, rec_allow=rec_allow, retry_failed=retry_failed,
                max_tasks=max_tasks, well_instances=well_instances,
            )
        n_ran += fr
        n_failed += ff

    if aggregate:
        remaining = None if max_tasks is None else max(0, max_tasks - n_ran)
        if remaining is None or remaining > 0:
            ar, af = _drain_aggregates_unified(
                sched, cm, task_allow=task_allow, max_tasks=remaining,
                agg_instances=agg_instances, figure_root=figure_root,
            )
            n_ran += ar
            n_failed += af
    return n_ran, n_failed


def _drain_finest_serial(
    sched, cm, dataset_mgr, *, task_allow, rec_allow, retry_failed, max_tasks,
    well_instances,
) -> tuple[int, int]:
    n_ran = n_failed = 0
    attempted: set = set()
    while True:
        if max_tasks is not None and n_ran >= max_tasks:
            break
        queue = sched.finest_queue(
            n=10_000, retry_failed=retry_failed,
            recording_keys=rec_allow, task_names=task_allow,
        )
        wu = next(
            (w for w in queue if (w.instance_key, w.task_name) not in attempted), None
        )
        if wu is None:
            break
        attempted.add((wu.instance_key, wu.task_name))
        status = _run_finest_unit(wu, cm, dataset_mgr, sched, well_instances)
        n_ran += 1
        n_failed += (status == TaskStatus.FAILED)
    return n_ran, n_failed


def _drain_finest_parallel(
    sched, cm, dataset_mgr, *, task_allow, rec_allow, retry_failed, max_tasks,
    jobs, config_path, finest_runner,
) -> tuple[int, int]:
    """Parallel finest drain. The parent marks RUNNING before submit and writes
    COMPLETE/FAILED on each future — workers never touch the instance store, so
    concurrent workers cannot race on it (same discipline as `_drain_parallel`).

    Real runs use a ProcessPool + the top-level `_run_one_worker`; tests inject
    `finest_runner` and get a ThreadPool (in-process, no pickling)."""
    use_threads = finest_runner is not None
    executor_factory = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    n_ran = n_failed = 0
    attempted: set = set()

    def _submit(ex, wu):
        rk = wu.scope_key["recording_key"]
        compound = f"{wu.scope_key['rec_name']}/{wu.scope_key['well_id']}"
        if use_threads:
            return ex.submit(finest_runner, wu.task_name, rk, compound)
        return ex.submit(_run_one_worker, WorkItem(rk, compound, wu.task_name), config_path)

    with executor_factory(max_workers=jobs) as ex:
        inflight: dict[Future, object] = {}

        def _refill():
            while True:
                if max_tasks is not None and (n_ran + len(inflight)) >= max_tasks:
                    return
                slots = jobs - len(inflight)
                if slots <= 0:
                    return
                queue = sched.finest_queue(
                    n=10_000, retry_failed=retry_failed,
                    recording_keys=rec_allow, task_names=task_allow,
                )
                picked = []
                for wu in queue:
                    if (wu.instance_key, wu.task_name) in attempted:
                        continue
                    picked.append(wu)
                    if len(picked) >= slots:
                        break
                if not picked:
                    return
                for wu in picked:
                    if max_tasks is not None and (n_ran + len(inflight)) >= max_tasks:
                        return
                    attempted.add((wu.instance_key, wu.task_name))
                    config = cm.get_config(
                        wu.task_name, wu.scope_key["recording_key"],
                        f"{wu.scope_key['rec_name']}/{wu.scope_key['well_id']}",
                    )
                    sched.update_status(
                        "well", wu.scope_key, wu.task_name, TaskStatus.RUNNING, config=config
                    )
                    inflight[_submit(ex, wu)] = wu

        _refill()
        while inflight:
            done, _pending = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                wu = inflight.pop(fut)
                try:
                    result = fut.result()
                except Exception:
                    result = {"status": TaskStatus.FAILED, "output_path": None,
                              "error": traceback.format_exc()}
                if result["status"] == TaskStatus.COMPLETE:
                    sched.update_status("well", wu.scope_key, wu.task_name,
                                        TaskStatus.COMPLETE, output_path=result["output_path"])
                else:
                    sched.update_status("well", wu.scope_key, wu.task_name,
                                        TaskStatus.FAILED, error=result["error"])
                    n_failed += 1
                    logger.error("Task FAILED %s/%s", wu.scope_key["recording_key"],
                                 wu.task_name)
                n_ran += 1
            _refill()
    return n_ran, n_failed


def _drain_aggregates_unified(
    sched, cm, *, task_allow, max_tasks, agg_instances, figure_root,
) -> tuple[int, int]:
    n_ran = n_failed = 0
    attempted: set = set()
    while True:
        if max_tasks is not None and n_ran >= max_tasks:
            break
        acted = False
        for dec in sched.aggregate_plan(task_allow):
            if dec.decision in (_uni.RUN, _uni.RERUN):
                key = (dec.instance_key, dec.task_name)
                if key in attempted:
                    continue
                attempted.add(key)
                status = _run_agg_unit(dec, cm, sched, agg_instances, figure_root)
                n_ran += 1
                n_failed += (status == TaskStatus.FAILED)
                acted = True
                break
            if dec.decision == _uni.SKIP_EMPTY:
                key = (dec.instance_key, dec.task_name, "skip")
                if key in attempted:
                    continue
                attempted.add(key)
                logger.info("Aggregate SKIP-empty %s %s (0 complete members).",
                            dec.task_name, dec.scope_key)
                sched.update_status(
                    dec.scope_name, dec.scope_key, dec.task_name, TaskStatus.COMPLETE,
                    member_hash=dec.member_hash, n_members_total=0,
                    n_members_complete=0, config=cm.get_task_params(dec.task_name),
                )
                acted = True
                break
        if not acted:
            break
    return n_ran, n_failed


if __name__ == "__main__":
    sys.exit(main())
