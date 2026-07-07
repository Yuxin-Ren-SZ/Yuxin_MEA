"""`yuxin-mea-run` worker CLI — drains the pipeline queue for a config.

Workflow mirrors `notebooks/v2/00_full_pipeline.ipynb`:

    1. Load ConfigManager + DatasetManager + PipelineManager.
    2. Loop pm.get_next_task() → task.run() → pm.update_status().
    3. Stop when the queue is empty or --max-tasks is reached.

The dashboard's Run page generates the exact invocation string for this CLI;
it never spawns long-running work itself.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path

import json

from yuxin_mea.config import ConfigManager
from yuxin_mea.dataset import DatasetManager
from yuxin_mea.pipeline import (
    AggregateScheduler,
    JsonAggregateCacheStore,
    PipelineManager,
    WorkItem,
)
from yuxin_mea.pipeline import aggregate_scheduler as _agg
from yuxin_mea.pipeline.task_record import TaskStatus
from yuxin_mea.tasks import AGGREGATE_TASK_CLASSES, TASK_CLASSES


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


def _setup_pipeline(config_path: Path) -> tuple[ConfigManager, DatasetManager, PipelineManager]:
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
    pipeline_mgr = PipelineManager(Path(analysis_root), config_provider=cm)
    for cls in TASK_CLASSES:
        try:
            pipeline_mgr.register_task(cls)
        except ValueError:
            # Already registered from a previous invocation against the same cache.
            pass
    return cm, dataset_mgr, pipeline_mgr


def _resolve_recording_path(dataset_mgr: DatasetManager, recording_key: str) -> Path:
    matches = dataset_mgr.get_recording_by([("cache_key", "==", recording_key)])
    if not matches:
        raise KeyError(
            f"Recording {recording_key!r} not found in DatasetManager. "
            "Has it been scanned (Recordings page → Scan disk)?"
        )
    return dataset_mgr.get_path(matches[0])


def _auto_queue_wells(
    dataset_mgr: DatasetManager,
    pipeline_mgr: PipelineManager,
    recording_keys: list[str],
) -> int:
    """Auto-queue wells for recordings present in the dataset but not the pipeline.

    Queuing is idempotent (add_well skips existing entries) and always writes
    to the cache so that a subsequent dry-run can preview eligible tasks.
    """
    n_queued = 0
    for rec_key in recording_keys:
        if pipeline_mgr.get_entries_for_recording(rec_key):
            continue

        matches = dataset_mgr.get_recording_by([("cache_key", "==", rec_key)])
        if not matches:
            logger.warning(
                "Recording %r not found in dataset cache — cannot auto-queue. "
                "Has it been scanned (Recordings page → Scan disk)?",
                rec_key,
            )
            continue

        entry = matches[0]
        if not entry.h5_recordings:
            logger.warning("Recording %r has no h5_recordings — nothing to queue.", rec_key)
            continue

        wells = 0
        for rec_name, well_ids in entry.h5_recordings.items():
            for well_id in well_ids:
                pipeline_mgr.add_well(rec_key, f"{rec_name}/{well_id}")
                wells += 1
        n_queued += wells
        logger.info("Auto-queued %d well(s) for recording %s.", wells, rec_key)
    return n_queued


def _run_one(
    work_item: WorkItem,
    cm: ConfigManager,
    dataset_mgr: DatasetManager,
    pipeline_mgr: PipelineManager,
    task_instances: dict[str, object],
) -> None:
    task = task_instances[work_item.task_name]
    params = cm.get_task_params(work_item.task_name)
    data_path = _resolve_recording_path(dataset_mgr, work_item.recording_key)

    pipeline_mgr.update_status(work_item, TaskStatus.RUNNING)
    t0 = time.time()
    try:
        output_path = task.run(  # type: ignore[attr-defined]
            work_item.recording_key,
            work_item.well_id,
            data_path,
            params,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(
            "Task FAILED %s/%s/%s: %s",
            work_item.recording_key, work_item.well_id, work_item.task_name, exc,
        )
        pipeline_mgr.update_status(
            work_item,
            TaskStatus.FAILED,
            error=tb,
        )
        return

    elapsed = time.time() - t0
    pipeline_mgr.update_status(
        work_item,
        TaskStatus.COMPLETE,
        output_path=output_path,
    )
    logger.info(
        "Task COMPLETE %s/%s/%s in %.1fs → %s",
        work_item.recording_key, work_item.well_id, work_item.task_name,
        elapsed, output_path,
    )


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

    cm, dataset_mgr, pipeline_mgr = _setup_pipeline(args.config)

    task_allow = _split_csv(args.tasks)
    rec_allow = _split_csv(args.recordings)

    if task_allow:
        registered = (
            {cls.task_name for cls in TASK_CLASSES}
            | {cls.task_name for cls in AGGREGATE_TASK_CLASSES}
        )
        bad = [t for t in task_allow if t not in registered]
        if bad:
            print(
                f"ERROR: unknown --tasks values: {bad}. "
                f"Known tasks: {sorted(registered)}",
                file=sys.stderr,
            )
            return 2

    if rec_allow:
        n_auto = _auto_queue_wells(dataset_mgr, pipeline_mgr, rec_allow)
        if n_auto:
            logger.info("Auto-queued %d well(s) for unqueued recordings.", n_auto)

    if args.dry_run:
        # Dry-run is best-effort only — it can't simulate dependency unlocking
        # because nothing transitions to COMPLETE. It enumerates everything
        # that's *currently* eligible and lists it.
        if not args.aggregate_only:
            eligible_now = pipeline_mgr.get_next_task(
                n=10_000,
                retry_failed=args.retry_failed,
                recording_keys=rec_allow,
                task_names=task_allow,
            )
            for w in eligible_now:
                print(f"[dry-run] would run {w.task_name} on {w.recording_key}/{w.well_id}")
            print(f"{len(eligible_now)} per-well task(s) eligible now.")
        if args.aggregate or args.aggregate_only:
            sched = _build_scheduler(cm, pipeline_mgr)
            for pi in sched.plan(task_allow):
                print(
                    f"[dry-run agg] {pi.decision.upper():>10}  {pi.task.task_name}  "
                    f"{pi.scope_key}  (complete {len(pi.members_complete)}/{pi.n_total})"
                )
        print("Done (dry-run; no state changes; downstream unlocks not modelled).")
        return 0

    n_ran = n_failed = 0
    if not args.aggregate_only:
        # Recover from any prior worker crash before executing.
        n_recovered = pipeline_mgr.recover_from_crash()
        if n_recovered:
            logger.warning(
                "Recovered %d task(s) left RUNNING/FAILED by a previous run.",
                n_recovered,
            )
        if args.jobs is None or args.jobs <= 1:
            n_ran, n_failed = _drain_serial(
                cm, dataset_mgr, pipeline_mgr,
                task_allow=task_allow,
                rec_allow=rec_allow,
                retry_failed=args.retry_failed,
                max_tasks=args.max_tasks,
            )
        else:
            n_ran, n_failed = _drain_parallel(
                pipeline_mgr,
                config_path=args.config,
                jobs=args.jobs,
                task_allow=task_allow,
                rec_allow=rec_allow,
                retry_failed=args.retry_failed,
                max_tasks=args.max_tasks,
            )

    n_agg = n_agg_failed = 0
    if args.aggregate or args.aggregate_only:
        sched = _build_scheduler(cm, pipeline_mgr)
        n_agg, n_agg_failed = _drain_aggregate(
            sched, cm, task_allow=task_allow, max_tasks=args.max_tasks
        )

    print(
        f"Done. Ran {n_ran} per-well + {n_agg} aggregate task(s); "
        f"{n_failed + n_agg_failed} failed."
    )
    return 0 if (n_failed + n_agg_failed) == 0 else 1


def _build_scheduler(cm: ConfigManager, pipeline_mgr: PipelineManager) -> AggregateScheduler:
    analysis_root = Path(cm.get_global("analysis_root"))
    exp_path = analysis_root / "experiment_cache.json"
    try:
        with exp_path.open() as fh:
            experiment_cache = json.load(fh)
    except Exception:  # noqa: BLE001
        logger.warning(
            "experiment_cache.json not readable at %s; groupname-scoped tasks bucket under '?'.",
            exp_path,
        )
        experiment_cache = {}
    store = JsonAggregateCacheStore(analysis_root)
    return AggregateScheduler(
        pipeline_mgr, cm, experiment_cache, AGGREGATE_TASK_CLASSES, store
    )


def _drain_aggregate(
    sched: AggregateScheduler,
    cm: ConfigManager,
    *,
    task_allow: list[str] | None,
    max_tasks: int | None,
) -> tuple[int, int]:
    """Run ready aggregate instances (serial). Reads per-well statuses; owns only
    the aggregate cache. The per-well drain functions are untouched."""
    figure_root = cm.get_global("figure_root") or cm.get_global("analysis_root")
    n_ran = 0
    n_failed = 0
    n_wait = n_uptodate = n_skip = 0
    for pi in sched.plan(task_allow):
        if max_tasks is not None and n_ran >= max_tasks:
            logger.info("Reached --max-tasks=%d (aggregate), stopping.", max_tasks)
            break
        task = pi.task
        if pi.decision == _agg.WAIT:
            n_wait += 1
            continue
        if pi.decision == _agg.UPTODATE:
            n_uptodate += 1
            continue
        if pi.decision == _agg.SKIP_EMPTY:
            n_skip += 1
            logger.info("Aggregate SKIP-empty %s %s (0 complete members).",
                        task.task_name, pi.scope_key)
            sched.mark_skipped_empty(task, pi.scope_key, pi.member_hash)
            continue

        params = dict(cm.get_task_params(task.task_name))
        params.setdefault("output_root", figure_root)
        sched.mark_running(task, pi.scope_key)
        t0 = time.time()
        try:
            out = task.run(pi.scope_key, pi.members_complete, params)
        except Exception:
            tb = traceback.format_exc()
            logger.error("Aggregate FAILED %s %s: %s",
                         task.task_name, pi.scope_key, tb.strip().splitlines()[-1])
            sched.mark_failed(task, pi.scope_key, tb, pi.member_hash,
                              pi.n_total, len(pi.members_complete))
            n_failed += 1
            continue
        sched.mark_complete(task, pi.scope_key, out, pi.member_hash,
                            pi.n_total, len(pi.members_complete))
        n_ran += 1
        logger.info(
            "Aggregate %s %s %s in %.1fs → %s (%d/%d members)",
            pi.decision.upper(), task.task_name, pi.scope_key,
            time.time() - t0, out, len(pi.members_complete), pi.n_total,
        )
    if n_wait or n_uptodate or n_skip:
        logger.info(
            "Aggregate pass: ran %d, failed %d · %d waiting (deps not attempted), "
            "%d up-to-date, %d skipped-empty.",
            n_ran, n_failed, n_wait, n_uptodate, n_skip,
        )
    return n_ran, n_failed


def _drain_serial(
    cm: ConfigManager,
    dataset_mgr: DatasetManager,
    pipeline_mgr: PipelineManager,
    *,
    task_allow: list[str] | None,
    rec_allow: list[str] | None,
    retry_failed: bool,
    max_tasks: int | None,
) -> tuple[int, int]:
    """Single-worker drain — preserves the pre-concurrency behavior verbatim."""
    task_instances: dict[str, object] = {cls.task_name: cls() for cls in TASK_CLASSES}

    n_ran = 0
    n_failed = 0
    # Tracks (recording_key, well_id, task_name) we've already attempted in this
    # invocation. With --retry-failed, a hard-failing task would otherwise be
    # re-eligible and loop forever; this gate makes each work item attempted at
    # most once per invocation, while still letting future invocations retry.
    attempted: set[tuple[str, str, str]] = set()
    while True:
        if max_tasks is not None and n_ran >= max_tasks:
            logger.info("Reached --max-tasks=%d, stopping.", max_tasks)
            break

        batch = pipeline_mgr.get_next_task(
            n=max(1, len(attempted) + 1),
            retry_failed=retry_failed,
            recording_keys=rec_allow,
            task_names=task_allow,
        )
        work_item = next(
            (
                w for w in batch
                if (w.recording_key, w.well_id, w.task_name) not in attempted
            ),
            None,
        )
        if work_item is None:
            break

        attempted.add((work_item.recording_key, work_item.well_id, work_item.task_name))
        _run_one(work_item, cm, dataset_mgr, pipeline_mgr, task_instances)
        n_ran += 1
        record = pipeline_mgr.get_entry(
            work_item.recording_key, work_item.well_id
        ).tasks[work_item.task_name]
        if record.status == TaskStatus.FAILED:
            n_failed += 1
    return n_ran, n_failed


def _drain_parallel(
    pipeline_mgr: PipelineManager,
    *,
    config_path: Path,
    jobs: int,
    task_allow: list[str] | None,
    rec_allow: list[str] | None,
    retry_failed: bool,
    max_tasks: int | None,
) -> tuple[int, int]:
    """Parallel drain via ProcessPoolExecutor.

    The parent owns the pipeline cache: it marks RUNNING before submit and
    writes COMPLETE/FAILED on each future's resolution. Workers re-load their
    own ConfigManager/DatasetManager and return a plain dict — they never
    touch pipeline_cache.json.
    """
    n_ran = 0
    n_failed = 0
    attempted: set[tuple[str, str, str]] = set()

    def _fetch_eligible(slots: int) -> list[WorkItem]:
        # Over-fetch so we can skip in-flight + already-attempted items.
        n_need = slots + len(attempted) + 1
        batch = pipeline_mgr.get_next_task(
            n=n_need,
            retry_failed=retry_failed,
            recording_keys=rec_allow,
            task_names=task_allow,
        )
        out: list[WorkItem] = []
        for w in batch:
            key = (w.recording_key, w.well_id, w.task_name)
            if key in attempted:
                continue
            out.append(w)
            if len(out) >= slots:
                break
        return out

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures: dict[Future, WorkItem] = {}

        def _refill() -> None:
            while True:
                if max_tasks is not None and (n_ran + len(futures)) >= max_tasks:
                    return
                slots = jobs - len(futures)
                if slots <= 0:
                    return
                next_items = _fetch_eligible(slots)
                if not next_items:
                    return
                for w in next_items:
                    if max_tasks is not None and (n_ran + len(futures)) >= max_tasks:
                        return
                    key = (w.recording_key, w.well_id, w.task_name)
                    attempted.add(key)
                    pipeline_mgr.update_status(w, TaskStatus.RUNNING)
                    fut = ex.submit(_run_one_worker, w, config_path)
                    futures[fut] = w

        _refill()
        while futures:
            done, _pending = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                w = futures.pop(fut)
                try:
                    result = fut.result()
                except Exception:
                    result = {
                        "status": TaskStatus.FAILED,
                        "output_path": None,
                        "error": traceback.format_exc(),
                        "elapsed": 0.0,
                    }
                if result["status"] == TaskStatus.COMPLETE:
                    pipeline_mgr.update_status(
                        w, TaskStatus.COMPLETE, output_path=result["output_path"]
                    )
                    logger.info(
                        "Task COMPLETE %s/%s/%s in %.1fs → %s",
                        w.recording_key, w.well_id, w.task_name,
                        result["elapsed"], result["output_path"],
                    )
                else:
                    pipeline_mgr.update_status(
                        w, TaskStatus.FAILED, error=result["error"]
                    )
                    logger.error(
                        "Task FAILED %s/%s/%s",
                        w.recording_key, w.well_id, w.task_name,
                    )
                    n_failed += 1
                n_ran += 1
            _refill()

    return n_ran, n_failed


if __name__ == "__main__":
    sys.exit(main())
