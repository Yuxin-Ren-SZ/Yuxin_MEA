from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


def test_dataset_manager_root_exports_public_collaborators():
    from yuxin_mea import dataset as dataset_manager

    assert dataset_manager.JsonCacheStore is not None
    assert dataset_manager.BaseCacheStore is not None
    assert dataset_manager.MxassayMetadataExtractor is not None
    assert dataset_manager.DummyMetadataExtractor is not None


def test_json_cache_store_accepts_string_analysis_dir():
    from yuxin_mea.dataset import JsonCacheStore

    with TemporaryDirectory() as tmp:
        store = JsonCacheStore(tmp)
        store.save({})
        assert (Path(tmp) / "experiment_cache.json").exists()


def test_analysis_exports_public_error_and_writer_alias():
    from yuxin_mea.analysis import (
        BurstDetectorError,
        ParquetBurstOutputWriter,
        PickleBurstOutputWriter,
    )

    assert issubclass(BurstDetectorError, Exception)
    assert ParquetBurstOutputWriter is PickleBurstOutputWriter


def test_config_template_creates_parent_directory():
    from yuxin_mea.config import ConfigManager
    from yuxin_mea.tasks import PreprocessingTask

    with TemporaryDirectory() as tmp:
        output = Path(tmp) / "nested" / "pipeline_config.json"
        cm = ConfigManager()
        cm.set_global("analysis_root", "/path/to/analysis")
        cm.register_task(PreprocessingTask)
        cm.generate_template(output)

        assert output.exists()


def test_pipeline_manager_filters_next_task_by_recording_key():
    from yuxin_mea.pipeline import PipelineManager
    from yuxin_mea.pipeline.task_record import TaskStatus
    from yuxin_mea.tasks import PreprocessingTask

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
    from yuxin_mea.pipeline import PipelineManager
    from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
    from yuxin_mea.pipeline.task_record import TaskRecord, TaskStatus
    from yuxin_mea.tasks import PreprocessingTask

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

