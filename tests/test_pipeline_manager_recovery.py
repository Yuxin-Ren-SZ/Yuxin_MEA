"""`PipelineManager.__init__` is a side-effect-free read; only
`recover_from_crash()` mutates the cache to reset stale RUNNING / FAILED
tasks. This is load-bearing for the dashboard, which builds a transient
PipelineManager per callback to compute previews — those constructions
must not race a live worker writing the same cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from yuxin_mea.pipeline import PipelineManager
from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
from yuxin_mea.pipeline.pipeline_entry import PipelineEntry
from yuxin_mea.pipeline.task_record import TaskRecord, TaskStatus
from yuxin_mea.tasks import PreprocessingTask


def _write_cache_with_running_task(analysis_dir: Path) -> JsonPipelineCacheStore:
    store = JsonPipelineCacheStore(analysis_dir)
    entry = PipelineEntry(
        recording_key="SampleA/240415/PlateX/Network/001",
        well_id="rec0000/well000",
        created_at=0.0,
        tasks={
            "preprocessing": TaskRecord(
                status=TaskStatus.RUNNING,
                dependencies=[],
                output_path=None,
                last_updated=1.0,
                error=None,
            ),
        },
    )
    store.save({entry.pipeline_key: entry})
    return store


def test_init_does_not_reset_running_tasks():
    """Constructing a PipelineManager must NOT flip RUNNING → NOT_RUN on
    disk or in memory. register_task() may rewrite the cache to back-fill
    task records, but the existing statuses survive."""
    with TemporaryDirectory() as tmp:
        analysis_dir = Path(tmp)
        store = _write_cache_with_running_task(analysis_dir)
        cache_path = analysis_dir / "pipeline_cache.json"

        manager = PipelineManager(analysis_dir, cache_store=store)
        # Read-only assertion BEFORE register_task — purest construction check.
        entry = manager.get_entry("SampleA/240415/PlateX/Network/001", "rec0000/well000")
        assert entry is not None
        assert entry.tasks["preprocessing"].status == TaskStatus.RUNNING

        # Disk preserved through both __init__ and register_task.
        manager.register_task(PreprocessingTask)
        on_disk = json.loads(cache_path.read_text())
        only_entry = next(iter(on_disk.values()))
        assert only_entry["tasks"]["preprocessing"]["status"] == TaskStatus.RUNNING


def test_recover_from_crash_resets_running_tasks():
    with TemporaryDirectory() as tmp:
        analysis_dir = Path(tmp)
        store = _write_cache_with_running_task(analysis_dir)

        manager = PipelineManager(analysis_dir, cache_store=store)
        manager.register_task(PreprocessingTask)
        n = manager.recover_from_crash()
        assert n == 1

        entry = manager.get_entry("SampleA/240415/PlateX/Network/001", "rec0000/well000")
        assert entry is not None
        assert entry.tasks["preprocessing"].status == TaskStatus.NOT_RUN
