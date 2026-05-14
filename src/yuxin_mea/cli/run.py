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
from pathlib import Path

from yuxin_mea.config import ConfigManager
from yuxin_mea.dataset import DatasetManager
from yuxin_mea.pipeline import PipelineManager, WorkItem
from yuxin_mea.pipeline.task_record import TaskStatus
from yuxin_mea.tasks import TASK_CLASSES


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
        help="Stop after this many tasks (useful for smoke tests).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the queue plan and exit without executing.",
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
        registered = {cls.task_name for cls in TASK_CLASSES}
        bad = [t for t in task_allow if t not in registered]
        if bad:
            print(
                f"ERROR: unknown --tasks values: {bad}. "
                f"Known tasks: {sorted(registered)}",
                file=sys.stderr,
            )
            return 2

    if args.dry_run:
        # Dry-run is best-effort only — it can't simulate dependency unlocking
        # because nothing transitions to COMPLETE. It enumerates everything
        # that's *currently* eligible (status NOT_RUN/FAILED with deps already
        # complete) and lists it. Tasks that would only unlock after an earlier
        # task ran are not shown.
        eligible_now = pipeline_mgr.get_next_task(
            n=10_000,
            retry_failed=args.retry_failed,
            recording_keys=rec_allow,
            task_names=task_allow,
        )
        for w in eligible_now:
            print(f"[dry-run] would run {w.task_name} on {w.recording_key}/{w.well_id}")
        print(
            f"Done. {len(eligible_now)} task(s) eligible right now "
            "(dry-run; no state changes; downstream unlocks not modelled)."
        )
        return 0

    # Recover from any prior worker crash before executing. Dry-run intentionally
    # skips this — previewing the queue must not mutate state.
    n_recovered = pipeline_mgr.recover_from_crash()
    if n_recovered:
        logger.warning(
            "Recovered %d task(s) left RUNNING/FAILED by a previous run.",
            n_recovered,
        )

    task_instances: dict[str, object] = {cls.task_name: cls() for cls in TASK_CLASSES}

    n_ran = 0
    n_failed = 0
    # Tracks (recording_key, well_id, task_name) we've already attempted in this
    # invocation. With --retry-failed, a hard-failing task would otherwise be
    # re-eligible and loop forever; this gate makes each work item attempted at
    # most once per invocation, while still letting future invocations retry.
    attempted: set[tuple[str, str, str]] = set()
    while True:
        if args.max_tasks is not None and n_ran >= args.max_tasks:
            logger.info("Reached --max-tasks=%d, stopping.", args.max_tasks)
            break

        # Ask for several at once so we can skip already-attempted work items.
        batch = pipeline_mgr.get_next_task(
            n=max(1, len(attempted) + 1),
            retry_failed=args.retry_failed,
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

    print(f"Done. Ran {n_ran} task(s); {n_failed} failed.")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
