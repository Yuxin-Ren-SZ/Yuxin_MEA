from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pipeline_manager import PipelineManager, PipelineRunOptions, run_pipeline_session
from pipeline_manager.base_task import BaseAnalysisTask
from pipeline_manager.task_record import TaskStatus


ORDER_CALLS: list[str] = []


@dataclass(frozen=True)
class DummyRecording:
    cache_key: str


class DummyDataset:
    def __init__(self, root: Path) -> None:
        self.root = root

    def get_path(self, entry: DummyRecording) -> Path:
        return self.root / entry.cache_key.replace("/", "_") / "data.raw.h5"


class DummyConfig:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def get_config(self, task_name: str, recording_key: str, well_id: str) -> dict:
        return {"task_name": task_name}

    def get_task_params(self, task_name: str) -> dict[str, Any]:
        return {"output_root": self.output_root}


def _output_path(task_name: str, recording_key: str, well_id: str, params: dict) -> Path:
    return (
        Path(params["output_root"])
        / recording_key.replace("/", "_")
        / well_id.replace("/", "_")
        / task_name
    )


class OrderFirstTask(BaseAnalysisTask):
    task_name = "order_first"
    dependencies: list[str] = []

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        ORDER_CALLS.append(self.task_name)
        return _output_path(self.task_name, recording_key, well_id, params)


class OrderSecondTask(BaseAnalysisTask):
    task_name = "order_second"
    dependencies = ["order_first"]

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        ORDER_CALLS.append(self.task_name)
        return _output_path(self.task_name, recording_key, well_id, params)


class FailingTask(BaseAnalysisTask):
    task_name = "failing"
    dependencies: list[str] = []

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        raise RuntimeError("boom")


class FlakyTask(BaseAnalysisTask):
    task_name = "flaky"
    dependencies: list[str] = []
    attempts = 0

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        type(self).attempts += 1
        if type(self).attempts == 1:
            raise RuntimeError("try again")
        return _output_path(self.task_name, recording_key, well_id, params)


class ScopeTask(BaseAnalysisTask):
    task_name = "scope"
    dependencies: list[str] = []

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        return _output_path(self.task_name, recording_key, well_id, params)


def _make_runner_parts(tmp: str, task_classes: list[type[BaseAnalysisTask]]):
    root = Path(tmp)
    config = DummyConfig(root / "outputs")
    manager = PipelineManager(root, config_provider=config)
    for task_class in task_classes:
        manager.register_task(task_class)
    dataset = DummyDataset(root / "raw")
    return manager, dataset, config


def test_runner_runs_ready_tasks_in_dependency_order_and_marks_complete():
    ORDER_CALLS.clear()
    recording = DummyRecording("SampleA/240415/PlateX/Network/001")

    with TemporaryDirectory() as tmp:
        manager, dataset, config = _make_runner_parts(
            tmp,
            [OrderFirstTask, OrderSecondTask],
        )
        manager.add_well(recording.cache_key, "rec0000/well000")

        result = run_pipeline_session(
            pipeline_mgr=manager,
            dataset_mgr=dataset,
            config_provider=config,
            recordings=[recording],
            well_task_classes=[OrderFirstTask, OrderSecondTask],
            options=PipelineRunOptions(run_plate_viewer=False),
        )

        entry = manager.get_entry(recording.cache_key, "rec0000/well000")

    assert entry is not None
    assert ORDER_CALLS == ["order_first", "order_second"]
    assert [item.task_name for item in result.completed_work_items] == [
        "order_first",
        "order_second",
    ]
    assert entry.tasks["order_first"].status == TaskStatus.COMPLETE
    assert entry.tasks["order_second"].status == TaskStatus.COMPLETE
    assert entry.tasks["order_second"].output_path is not None


def test_runner_marks_exceptions_failed_and_records_error():
    recording = DummyRecording("SampleA/240415/PlateX/Network/001")

    with TemporaryDirectory() as tmp:
        manager, dataset, config = _make_runner_parts(tmp, [FailingTask])
        manager.add_well(recording.cache_key, "rec0000/well000")

        result = run_pipeline_session(
            pipeline_mgr=manager,
            dataset_mgr=dataset,
            config_provider=config,
            recordings=[recording],
            well_task_classes=[FailingTask],
            options=PipelineRunOptions(run_plate_viewer=False),
        )

        entry = manager.get_entry(recording.cache_key, "rec0000/well000")

    assert entry is not None
    assert [item.task_name for item in result.failed_work_items] == ["failing"]
    assert entry.tasks["failing"].status == TaskStatus.FAILED
    assert entry.tasks["failing"].error == "boom"


def test_runner_retries_only_failed_tasks_named_in_options():
    FlakyTask.attempts = 0
    recording = DummyRecording("SampleA/240415/PlateX/Network/001")

    with TemporaryDirectory() as tmp:
        manager, dataset, config = _make_runner_parts(tmp, [FlakyTask])
        manager.add_well(recording.cache_key, "rec0000/well000")

        run_pipeline_session(
            pipeline_mgr=manager,
            dataset_mgr=dataset,
            config_provider=config,
            recordings=[recording],
            well_task_classes=[FlakyTask],
            options=PipelineRunOptions(run_plate_viewer=False),
        )
        result = run_pipeline_session(
            pipeline_mgr=manager,
            dataset_mgr=dataset,
            config_provider=config,
            recordings=[recording],
            well_task_classes=[FlakyTask],
            options=PipelineRunOptions(
                retry_failed_tasks={"flaky"},
                run_plate_viewer=False,
            ),
        )

        entry = manager.get_entry(recording.cache_key, "rec0000/well000")

    assert entry is not None
    assert FlakyTask.attempts == 2
    assert [item.task_name for item in result.completed_work_items] == ["flaky"]
    assert entry.tasks["flaky"].status == TaskStatus.COMPLETE
    assert entry.tasks["flaky"].error is None


def test_runner_honors_stop_after_task():
    recording = DummyRecording("SampleA/240415/PlateX/Network/001")

    with TemporaryDirectory() as tmp:
        manager, dataset, config = _make_runner_parts(
            tmp,
            [OrderFirstTask, OrderSecondTask],
        )
        manager.add_well(recording.cache_key, "rec0000/well000")

        result = run_pipeline_session(
            pipeline_mgr=manager,
            dataset_mgr=dataset,
            config_provider=config,
            recordings=[recording],
            well_task_classes=[OrderFirstTask, OrderSecondTask],
            options=PipelineRunOptions(
                stop_after_task="order_first",
                run_plate_viewer=False,
            ),
        )

        entry = manager.get_entry(recording.cache_key, "rec0000/well000")

    assert entry is not None
    assert [item.task_name for item in result.completed_work_items] == ["order_first"]
    assert entry.tasks["order_first"].status == TaskStatus.COMPLETE
    assert entry.tasks["order_second"].status == TaskStatus.NOT_RUN


def test_runner_scopes_work_to_loaded_recordings():
    loaded = DummyRecording("SampleB/240416/PlateX/Network/001")
    cached_only = DummyRecording("SampleA/240415/PlateX/Network/001")

    with TemporaryDirectory() as tmp:
        manager, dataset, config = _make_runner_parts(tmp, [ScopeTask])
        manager.add_well(cached_only.cache_key, "rec0000/well000")
        manager.add_well(loaded.cache_key, "rec0000/well000")

        result = run_pipeline_session(
            pipeline_mgr=manager,
            dataset_mgr=dataset,
            config_provider=config,
            recordings=[loaded],
            well_task_classes=[ScopeTask],
            options=PipelineRunOptions(run_plate_viewer=False),
        )

        cached_entry = manager.get_entry(cached_only.cache_key, "rec0000/well000")
        loaded_entry = manager.get_entry(loaded.cache_key, "rec0000/well000")

    assert cached_entry is not None
    assert loaded_entry is not None
    assert [item.recording_key for item in result.completed_work_items] == [
        loaded.cache_key
    ]
    assert cached_entry.tasks["scope"].status == TaskStatus.NOT_RUN
    assert loaded_entry.tasks["scope"].status == TaskStatus.COMPLETE
