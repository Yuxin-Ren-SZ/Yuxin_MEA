from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def test_pipeline_tasks_import_does_not_require_viewer_runtime():
    import pipeline_tasks

    assert pipeline_tasks.PreprocessingTask.task_name == "preprocessing"


def test_dataset_manager_root_exports_public_collaborators():
    import dataset_manager

    assert dataset_manager.JsonCacheStore is not None
    assert dataset_manager.BaseCacheStore is not None
    assert dataset_manager.MxassayMetadataExtractor is not None
    assert dataset_manager.DummyMetadataExtractor is not None


def test_json_cache_store_accepts_string_analysis_dir():
    from dataset_manager import JsonCacheStore

    with TemporaryDirectory() as tmp:
        store = JsonCacheStore(tmp)
        store.save({})
        assert (Path(tmp) / "experiment_cache.json").exists()


def test_analysis_exports_public_error_and_writer_alias():
    from pipeline_tasks.analysis import (
        BurstDetectorError,
        ParquetBurstOutputWriter,
        PickleBurstOutputWriter,
    )

    assert issubclass(BurstDetectorError, Exception)
    assert ParquetBurstOutputWriter is PickleBurstOutputWriter


def test_config_template_creates_parent_directory():
    from config_manager import ConfigManager
    from pipeline_tasks import PreprocessingTask

    with TemporaryDirectory() as tmp:
        output = Path(tmp) / "nested" / "pipeline_config.json"
        cm = ConfigManager()
        cm.set_global("analysis_root", "/path/to/analysis")
        cm.register_task(PreprocessingTask)
        cm.generate_template(output)

        assert output.exists()


def test_pipeline_manager_filters_next_task_by_recording_key():
    from pipeline_manager import PipelineManager
    from pipeline_manager.task_record import TaskStatus
    from pipeline_tasks import PreprocessingTask

    with TemporaryDirectory() as tmp:
        manager = PipelineManager(Path(tmp))
        manager.register_task(PreprocessingTask)
        manager.add_well("SampleA/240415/PlateX/Network/001", "rec0000/well000")
        manager.add_well("SampleB/240416/PlateX/Network/001", "rec0000/well000")

        scoped = manager.get_next_task(
            n=10,
            recording_keys={"SampleB/240416/PlateX/Network/001"},
        )

        assert [item.recording_key for item in scoped] == [
            "SampleB/240416/PlateX/Network/001"
        ]

        manager.update_status(scoped[0], TaskStatus.RUNNING)
        manager.update_status(scoped[0], TaskStatus.COMPLETE, output_path=Path(tmp))

        remaining = manager.get_next_task(
            n=10,
            recording_keys={"SampleA/240415/PlateX/Network/001"},
        )
        assert [item.recording_key for item in remaining] == [
            "SampleA/240415/PlateX/Network/001"
        ]


def test_pipeline_manager_ignores_cached_unregistered_tasks():
    from pipeline_manager import PipelineManager
    from pipeline_manager.cache_store import JsonPipelineCacheStore
    from pipeline_manager.task_record import TaskRecord, TaskStatus
    from pipeline_tasks import PreprocessingTask

    with TemporaryDirectory() as tmp:
        analysis_dir = Path(tmp)
        store = JsonPipelineCacheStore(analysis_dir)
        manager = PipelineManager(analysis_dir, cache_store=store)
        manager.register_task(PreprocessingTask)
        manager.add_well("SampleA/240415/PlateX/Network/001", "rec0000/well000")
        entry = manager.get_entry("SampleA/240415/PlateX/Network/001", "rec0000/well000")
        assert entry is not None
        entry.tasks["old_task"] = TaskRecord(
            status=TaskStatus.NOT_RUN,
            dependencies=[],
            output_path=None,
            last_updated=None,
            error=None,
        )
        store.save({entry.pipeline_key: entry})

        reloaded = PipelineManager(analysis_dir, cache_store=store)
        reloaded.register_task(PreprocessingTask)

        assert [item.task_name for item in reloaded.get_next_task(n=10)] == [
            PreprocessingTask.task_name
        ]


def test_plate_viewer_loads_compound_upstream_paths():
    from pipeline_tasks.plate_viewer import PlateViewerTask

    @dataclass
    class StubWellRecord:
        well_id: str
        well_name: str
        groupname: str
        plot_signals: dict | None = None
        spike_times: dict | None = None
        status: str = "ok"

    recording_key = "SampleA/240415/PlateX/Network/001"
    rec_name = "rec0000"
    well_id = "well000"

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        burst_dir = root / "burst" / recording_key / rec_name / well_id / "burst_detection"
        curation_dir = root / "curation" / recording_key / rec_name / well_id / "auto_curation"
        burst_dir.mkdir(parents=True)
        curation_dir.mkdir(parents=True)
        np.save(
            burst_dir / "plot_signals.npy",
            {"t": np.array([0.0]), "rate_signal": np.array([1.0])},
        )
        np.save(curation_dir / "curated_spike_times.npy", {"unit_0": np.array([0.1])})

        record = PlateViewerTask()._load_well_record(
            well_id,
            recording_key,
            rec_name,
            root / "burst",
            root / "curation",
            {well_id: {"well_name": "A1", "groupname": "control"}},
            StubWellRecord,
        )

    assert record.status == "ok"
    assert record.well_name == "A1"
    assert record.plot_signals is not None
    assert record.spike_times is not None
