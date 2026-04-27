from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from .cache_store import BasePipelineCacheStore, JsonPipelineCacheStore
from .pipeline_entry import PipelineEntry
from .task_record import TaskRecord, TaskStatus
from .work_item import WorkItem

logger = logging.getLogger(__name__)

_VALID_UPDATE_STATUSES = {TaskStatus.RUNNING, TaskStatus.COMPLETE, TaskStatus.FAILED}


class PipelineManager:
    """Active computing manager for per-well analysis tasks.

    Workflow:
        1. Register task types and their dependency names once at startup.
        2. Register (recording, well) pairs via add_well().
        3. Call get_next_task() to receive WorkItems that are ready to run.
        4. Call update_status() with the WorkItem and the new status.
        5. Repeat until is_all_complete() is True.
        6. Use refresh() to reset a task (and its dependents) when re-runs are needed.
    """

    def __init__(
        self,
        analysis_dir: Path,
        cache_store:  BasePipelineCacheStore | None = None,
    ) -> None:
        self._analysis_dir = Path(analysis_dir)
        self._store        = cache_store or JsonPipelineCacheStore(self._analysis_dir)

        # task_name → ordered list of immediate dep names
        self._forward_deps: dict[str, list[str]] = {}
        # task_name → set of task names that directly or transitively depend on it
        self._reverse_deps: dict[str, set[str]]  = {}

        self._cache: dict[str, PipelineEntry] = self._store.load()
        logger.info("Loaded %d pipeline entries from cache.", len(self._cache))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def register_computation_task(self, name: str, dependencies: list[str]) -> None:
        """Register a task type and its immediate upstream dependencies.

        Must be called before any add_well() calls for correct initialisation,
        but is also safe to call afterwards — existing entries are patched.
        Raises ValueError on duplicate names or unknown dependency names.
        """
        if name in self._forward_deps:
            raise ValueError(f"Task {name!r} is already registered.")
        for dep in dependencies:
            if dep not in self._forward_deps:
                raise ValueError(
                    f"Unknown dependency {dep!r} for task {name!r}. "
                    "Register dependencies before the tasks that depend on them."
                )

        self._forward_deps[name] = list(dependencies)
        self._reverse_deps[name] = set()
        for dep in dependencies:
            self._reverse_deps[dep].add(name)

        # Patch any wells registered before this task was defined.
        for entry in self._cache.values():
            if name not in entry.tasks:
                entry.tasks[name] = self._make_task_record(name)

        self._store.save(self._cache)
        logger.debug("Registered task %r with deps %s.", name, dependencies)

    def add_well(self, recording_key: str, well_id: str) -> None:
        """Register a (recording, well) pair and initialise all registered tasks."""
        key = f"{recording_key}/{well_id}"
        if key in self._cache:
            return

        tasks = {
            name: self._make_task_record(name)
            for name in self._forward_deps
        }
        self._cache[key] = PipelineEntry(
            recording_key=recording_key,
            well_id=well_id,
            created_at=time.time(),
            tasks=tasks,
        )
        self._store.save(self._cache)
        logger.debug("Added well %s.", key)

    # ------------------------------------------------------------------
    # Work loop
    # ------------------------------------------------------------------

    def get_next_task(self, n: int = 1, type: str | None = None) -> list[WorkItem]:
        """Return up to n WorkItems whose dependencies are complete and status is NOT_RUN.

        FAILED tasks are not returned — use refresh() to reset them first.
        type must be None (reserved for future parallelisation).
        """
        if type is not None:
            raise ValueError("type parameter is reserved and must be None.")

        results: list[WorkItem] = []
        for entry in self._cache.values():
            if len(results) >= n:
                break
            for task_name, record in entry.tasks.items():
                if len(results) >= n:
                    break
                if record.status != TaskStatus.NOT_RUN:
                    continue
                if self._deps_complete(entry, task_name):
                    results.append(
                        WorkItem(
                            recording_key=entry.recording_key,
                            well_id=entry.well_id,
                            task_name=task_name,
                        )
                    )
        return results

    def update_status(
        self,
        work_item:   WorkItem,
        status:      str,
        output_path: Path | None = None,
        error:       str | None  = None,
    ) -> None:
        """Update the status of a task identified by a WorkItem.

        status must be one of: "running", "complete", "failed".
        output_path is stored when status == "complete".
        error is stored when status == "failed".
        """
        if status not in _VALID_UPDATE_STATUSES:
            raise ValueError(
                f"Invalid status {status!r}. "
                f"Must be one of: {sorted(_VALID_UPDATE_STATUSES)}"
            )

        entry = self._require_entry(work_item.recording_key, work_item.well_id)
        record = self._require_task(entry, work_item.task_name)

        record.status       = status
        record.last_updated = time.time()
        record.error        = error if status == TaskStatus.FAILED else None
        record.output_path  = Path(output_path) if (
            status == TaskStatus.COMPLETE and output_path is not None
        ) else record.output_path

        self._store.save(self._cache)
        logger.info(
            "Task %s/%s/%s → %s",
            work_item.recording_key, work_item.well_id, work_item.task_name, status,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_all_complete(self) -> bool:
        """True iff every (recording, well) × registered task is COMPLETE."""
        for entry in self._cache.values():
            for task_name in self._forward_deps:
                record = entry.tasks.get(task_name)
                if record is None or record.status != TaskStatus.COMPLETE:
                    return False
        return True

    @property
    def entries(self) -> list[PipelineEntry]:
        return list(self._cache.values())

    def get_entry(self, recording_key: str, well_id: str) -> PipelineEntry | None:
        return self._cache.get(f"{recording_key}/{well_id}")

    def get_entries_for_recording(self, recording_key: str) -> list[PipelineEntry]:
        prefix = f"{recording_key}/"
        return [e for k, e in self._cache.items() if k.startswith(prefix)]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def refresh(
        self,
        task_name:     str,
        recording_key: str | None = None,
        well_id:       str | None = None,
    ) -> None:
        """Reset task_name (and all transitive dependents) to NOT_RUN.

        Scope:
            refresh("sorting")                          → all wells
            refresh("sorting", recording_key=...)       → all wells for that recording
            refresh("sorting", recording_key=..., well_id=...)  → one well only
        """
        if task_name not in self._forward_deps:
            raise ValueError(f"Unknown task {task_name!r}.")

        tasks_to_reset = self._cascade_tasks(task_name)

        if recording_key is not None and well_id is not None:
            entries = [e for e in [self.get_entry(recording_key, well_id)] if e]
        elif recording_key is not None:
            entries = self.get_entries_for_recording(recording_key)
        else:
            entries = list(self._cache.values())

        for entry in entries:
            for t in tasks_to_reset:
                if t in entry.tasks:
                    self._reset_task_record(entry.tasks[t])

        self._store.save(self._cache)
        logger.info(
            "Refreshed %s (cascade: %s) for %d entries.",
            task_name, tasks_to_reset, len(entries),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_task_record(self, task_name: str) -> TaskRecord:
        return TaskRecord(
            status=TaskStatus.NOT_RUN,
            dependencies=list(self._forward_deps[task_name]),
            output_path=None,
            last_updated=None,
            error=None,
        )

    def _deps_complete(self, entry: PipelineEntry, task_name: str) -> bool:
        for dep in self._forward_deps.get(task_name, []):
            dep_record = entry.tasks.get(dep)
            if dep_record is None or dep_record.status != TaskStatus.COMPLETE:
                return False
        return True

    def _cascade_tasks(self, task_name: str) -> list[str]:
        """Return task_name + all transitively dependent task names (BFS)."""
        result: list[str] = []
        queue: deque[str] = deque([task_name])
        visited: set[str] = {task_name}
        while queue:
            current = queue.popleft()
            result.append(current)
            for dependent in self._reverse_deps.get(current, set()):
                if dependent not in visited:
                    visited.add(dependent)
                    queue.append(dependent)
        return result

    @staticmethod
    def _reset_task_record(record: TaskRecord) -> None:
        record.status       = TaskStatus.NOT_RUN
        record.output_path  = None
        record.last_updated = time.time()
        record.error        = None

    def _require_entry(self, recording_key: str, well_id: str) -> PipelineEntry:
        entry = self.get_entry(recording_key, well_id)
        if entry is None:
            raise KeyError(
                f"No entry for recording={recording_key!r}, well={well_id!r}. "
                "Call add_well() first."
            )
        return entry

    @staticmethod
    def _require_task(entry: PipelineEntry, task_name: str) -> TaskRecord:
        record = entry.tasks.get(task_name)
        if record is None:
            raise KeyError(
                f"Task {task_name!r} not found in entry {entry.pipeline_key!r}. "
                "Call register_computation_task() first."
            )
        return record
