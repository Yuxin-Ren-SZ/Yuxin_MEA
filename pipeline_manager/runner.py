from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .base_task import BaseAnalysisTask
from .manager import PipelineManager
from .task_record import TaskStatus
from .work_item import WorkItem


class _DatasetLike(Protocol):
    def get_path(self, entry: Any) -> Path:
        ...


class _TaskParamsProvider(Protocol):
    def get_task_params(self, task_name: str) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class PipelineRunOptions:
    retry_failed_tasks: set[str] = field(default_factory=set)
    stop_after_task: str | None = None
    plate_viewer_only: bool = False
    plate_recording_keys: set[str] = field(default_factory=set)
    run_plate_viewer: bool = True


@dataclass(frozen=True)
class PipelineRunResult:
    completed_work_items: list[WorkItem] = field(default_factory=list)
    failed_work_items: list[WorkItem] = field(default_factory=list)
    plate_outputs: dict[str, Path] = field(default_factory=dict)


def run_pipeline_session(
    *,
    pipeline_mgr: PipelineManager,
    dataset_mgr: _DatasetLike,
    config_provider: _TaskParamsProvider,
    recordings: list[Any],
    well_task_classes: list[type[BaseAnalysisTask]],
    plate_task_class: type[BaseAnalysisTask] | None = None,
    options: PipelineRunOptions | None = None,
) -> PipelineRunResult:
    """Run pending work for the currently loaded recordings.

    This is the notebook-friendly orchestration layer around PipelineManager:
    it scopes cached work to recordings loaded in this session, runs ready
    per-well tasks in dependency order, records status transitions, and
    optionally generates a plate-level viewer after burst detection completes.
    """
    opts = options or PipelineRunOptions()
    task_instances: dict[str, BaseAnalysisTask] = {
        cls.task_name: cls() for cls in well_task_classes
    }
    rec_lookup: dict[str, Any] = {r.cache_key: r for r in recordings}
    current_recording_keys = set(rec_lookup.keys())
    completed: list[WorkItem] = []
    failed: list[WorkItem] = []

    if opts.plate_viewer_only:
        print("\nSkipping well-level tasks; refreshing plate viewer only.")

    while not opts.plate_viewer_only:
        work_items = pipeline_mgr.get_next_task(
            n=1,
            retry_failed=False,
            recording_keys=current_recording_keys,
        )

        if not work_items and opts.retry_failed_tasks:
            retry_pool_size = max(
                1,
                sum(
                    1
                    for entry in pipeline_mgr.entries
                    if entry.recording_key in current_recording_keys
                )
                * len(task_instances),
            )
            work_items = [
                item
                for item in pipeline_mgr.get_next_task(
                    n=retry_pool_size,
                    retry_failed=True,
                    recording_keys=current_recording_keys,
                )
                if item.task_name in opts.retry_failed_tasks
            ]

        if not work_items:
            break

        item = work_items[0]
        task = task_instances[item.task_name]
        rec_entry = rec_lookup[item.recording_key]
        rec_name, well_id = item.well_id.split("/", 1)
        params = config_provider.get_task_params(item.task_name)

        print(f"\n[{item.task_name}]  {item.recording_key} / {rec_name} / {well_id}")
        pipeline_mgr.update_status(item, TaskStatus.RUNNING)

        t0 = time.perf_counter()
        try:
            output_path = task.run(
                item.recording_key,
                item.well_id,
                dataset_mgr.get_path(rec_entry),
                params,
            )
            elapsed = time.perf_counter() - t0
            pipeline_mgr.update_status(item, TaskStatus.COMPLETE, output_path=output_path)
            print(f"  OK  {elapsed:.1f}s  -> {output_path}")
            completed.append(item)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            pipeline_mgr.update_status(item, TaskStatus.FAILED, error=str(exc))
            traceback.print_exc()
            print(f"  FAILED  {elapsed:.1f}s  failed: {exc}")
            failed.append(item)

        if opts.stop_after_task and item.task_name == opts.stop_after_task:
            print(f"\nPaused after {opts.stop_after_task!r} as requested.")
            break

    if not opts.plate_viewer_only:
        print("\n-- No more pending well-level tasks. --")

    plate_outputs: dict[str, Path] = {}
    if opts.run_plate_viewer and plate_task_class is not None:
        plate_outputs = _run_plate_viewer(
            pipeline_mgr=pipeline_mgr,
            dataset_mgr=dataset_mgr,
            config_provider=config_provider,
            rec_lookup=rec_lookup,
            current_recording_keys=current_recording_keys,
            plate_task_class=plate_task_class,
            plate_recording_keys=opts.plate_recording_keys,
            stop_after_task=opts.stop_after_task,
            plate_viewer_only=opts.plate_viewer_only,
        )

    return PipelineRunResult(
        completed_work_items=completed,
        failed_work_items=failed,
        plate_outputs=plate_outputs,
    )


def _run_plate_viewer(
    *,
    pipeline_mgr: PipelineManager,
    dataset_mgr: _DatasetLike,
    config_provider: _TaskParamsProvider,
    rec_lookup: dict[str, Any],
    current_recording_keys: set[str],
    plate_task_class: type[BaseAnalysisTask],
    plate_recording_keys: set[str],
    stop_after_task: str | None,
    plate_viewer_only: bool,
) -> dict[str, Path]:
    plate_viewer = plate_task_class()
    plate_params = config_provider.get_task_params(plate_task_class.task_name)
    plate_outputs: dict[str, Path] = {}

    requested = set(plate_recording_keys or current_recording_keys)
    selected = requested & current_recording_keys
    missing = requested - current_recording_keys
    for recording_key in sorted(missing):
        print(
            f"\n[{plate_task_class.task_name}]  {recording_key}  "
            "skipped: not loaded in this session"
        )

    if stop_after_task is not None and not plate_viewer_only:
        print("\nPlate viewer skipped because STOP_AFTER_TASK is set.")
        return plate_outputs

    for recording_key in sorted(selected):
        rec_entry = rec_lookup[recording_key]
        if not _task_complete_for_loaded_wells(
            pipeline_mgr,
            recording_key,
            plate_task_class.dependencies,
        ):
            print(
                f"\n[{plate_task_class.task_name}]  {recording_key}  "
                f"skipped: {', '.join(plate_task_class.dependencies)} incomplete"
            )
            continue

        print(f"\n[{plate_task_class.task_name}]  {recording_key}")
        t0 = time.perf_counter()
        try:
            output_path = plate_viewer.run(
                recording_key,
                "__plate__",
                dataset_mgr.get_path(rec_entry),
                plate_params,
            )
            elapsed = time.perf_counter() - t0
            plate_outputs[recording_key] = output_path
            print(f"  OK  {elapsed:.1f}s  -> {output_path}")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            traceback.print_exc()
            print(f"  FAILED  {elapsed:.1f}s  failed: {exc}")

    return plate_outputs


def _task_complete_for_loaded_wells(
    pipeline_mgr: PipelineManager,
    recording_key: str,
    task_names: list[str],
) -> bool:
    entries = [
        entry
        for entry in pipeline_mgr.get_entries_for_recording(recording_key)
        if "/" in entry.well_id
    ]
    if not entries:
        return False

    return all(
        all(
            (task := entry.tasks.get(task_name)) is not None
            and task.status == TaskStatus.COMPLETE
            for task_name in task_names
        )
        for entry in entries
    )
