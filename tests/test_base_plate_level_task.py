"""Tests for `BasePlateLevelTask` — the abstract scaffolding for plate-level tasks.

Phase 5 renamed `BasePlateViewer` → `BasePlateLevelTask` and moved it from
`tasks/` to `pipeline/` (it's generic infrastructure, not a viewer thing).
The abstract method `build_figure` also renamed to `aggregate_records` so
the seam doesn't bake "viewer" semantics into a generalized base class.

There are no concrete subclasses in the codebase post-Phase-5 — these tests
exercise the abstract orchestrator via a local `_FakePlateTask` fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yuxin_mea.pipeline import BasePlateLevelTask


class _FakePlateTask(BasePlateLevelTask):
    """Minimal concrete subclass for testing the ABC template."""

    task_name = "fake_plate_task"
    dependencies: list[str] = []

    def run(self, recording_key, well_id, data_path, params):
        return self._run_template(recording_key, well_id, data_path, params)

    def aggregate_records(self, well_records, params):
        self.captured_well_records = well_records
        self.captured_params = params
        return "FAKE_RESULT"

    def write_output(self, result, recording_key, params):
        self.captured_result = result
        self.captured_recording_key = recording_key
        return Path("/tmp/fake_output.html")


def _empty_params(tmp_path: Path) -> dict:
    """Params that point everything at an empty tmpdir — all 24 wells will be 'missing'."""
    return {
        "burst_detection_root": str(tmp_path),
        "curation_output_root": str(tmp_path),
        "experiment_cache_path": str(tmp_path / "missing_cache.json"),
        "rec_name": "auto",
    }


def test_run_template_assembles_24_well_records(tmp_path):
    """`_run_template` must produce exactly 24 WellRecords (all missing with no data)."""
    task = _FakePlateTask()
    task._run_template(
        recording_key="test/recording",
        well_id="__plate__",
        data_path=tmp_path,
        params=_empty_params(tmp_path),
    )
    assert len(task.captured_well_records) == 24
    assert all(wr.status == "missing" for wr in task.captured_well_records)


def test_run_template_passes_resolved_params_to_hooks(tmp_path):
    """`_run_template` must call `aggregate_records` with resolved params."""
    task = _FakePlateTask()
    task._run_template(
        recording_key="test/recording",
        well_id="__plate__",
        data_path=tmp_path,
        params=_empty_params(tmp_path),
    )
    assert "burst_detection_root" in task.captured_params


def test_run_template_threads_result_from_aggregate_to_write(tmp_path):
    """`_run_template` must pass `aggregate_records` return value to `write_output`."""
    task = _FakePlateTask()
    task._run_template(
        recording_key="test/recording",
        well_id="__plate__",
        data_path=tmp_path,
        params=_empty_params(tmp_path),
    )
    assert task.captured_result == "FAKE_RESULT"


def test_run_template_returns_write_output_path(tmp_path):
    """`_run_template` must return exactly what `write_output` returns."""
    task = _FakePlateTask()
    result = task._run_template(
        recording_key="test/recording",
        well_id="__plate__",
        data_path=tmp_path,
        params=_empty_params(tmp_path),
    )
    assert result == Path("/tmp/fake_output.html")


def test_base_plate_level_task_cannot_be_instantiated_without_hooks():
    """`BasePlateLevelTask` must be abstract — direct instantiation is forbidden."""
    with pytest.raises(TypeError):
        BasePlateLevelTask()  # type: ignore[abstract]
