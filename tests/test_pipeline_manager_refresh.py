"""Tests for PipelineManager.refresh / bulk_refresh.

The dashboard's bulk-reset card calls these directly; they're load-bearing
for "reset all FAILED `sorting` tasks across these 3 recordings". Coverage
must include status_filter (per-record), well_ids (subset), and the
cartesian product behavior of bulk_refresh — none of which the existing
recovery test exercises.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from yuxin_mea.pipeline import PipelineManager
from yuxin_mea.pipeline.task_record import TaskStatus
from yuxin_mea.tasks import PreprocessingTask, SortingTask


REC_A = "SampleA/240101/PlateX/Network/001"
REC_B = "SampleB/240101/PlateY/Network/001"


def _make_manager(tmp: Path) -> PipelineManager:
    m = PipelineManager(tmp)
    m.register_task(PreprocessingTask)
    m.register_task(SortingTask)
    return m


def _set_status(manager: PipelineManager, rec: str, well: str, task: str, status: TaskStatus) -> None:
    entry = manager.get_entry(rec, well)
    assert entry is not None
    entry.tasks[task].status = status


def test_refresh_returns_reset_count():
    with TemporaryDirectory() as tmp:
        m = _make_manager(Path(tmp))
        m.add_well(REC_A, "rec0/well000")
        m.add_well(REC_A, "rec0/well001")
        _set_status(m, REC_A, "rec0/well000", "sorting", TaskStatus.COMPLETE)
        _set_status(m, REC_A, "rec0/well001", "sorting", TaskStatus.COMPLETE)

        n = m.refresh("sorting", recording_key=REC_A)
        # sorting has no downstream deps in this two-task setup, so cascade == {sorting}.
        assert n == 2
        assert m.get_entry(REC_A, "rec0/well000").tasks["sorting"].status == TaskStatus.NOT_RUN


def test_refresh_status_filter_only_failed():
    with TemporaryDirectory() as tmp:
        m = _make_manager(Path(tmp))
        m.add_well(REC_A, "rec0/well000")
        m.add_well(REC_A, "rec0/well001")
        _set_status(m, REC_A, "rec0/well000", "sorting", TaskStatus.FAILED)
        _set_status(m, REC_A, "rec0/well001", "sorting", TaskStatus.COMPLETE)

        n = m.refresh("sorting", status_filter={TaskStatus.FAILED})
        assert n == 1
        assert m.get_entry(REC_A, "rec0/well000").tasks["sorting"].status == TaskStatus.NOT_RUN
        # COMPLETE record left alone.
        assert m.get_entry(REC_A, "rec0/well001").tasks["sorting"].status == TaskStatus.COMPLETE


def test_refresh_well_ids_subset():
    with TemporaryDirectory() as tmp:
        m = _make_manager(Path(tmp))
        m.add_well(REC_A, "rec0/well000")
        m.add_well(REC_A, "rec0/well001")
        m.add_well(REC_A, "rec0/well002")
        for w in ("rec0/well000", "rec0/well001", "rec0/well002"):
            _set_status(m, REC_A, w, "sorting", TaskStatus.COMPLETE)

        n = m.refresh(
            "sorting",
            recording_key=REC_A,
            well_ids=["rec0/well000", "rec0/well002"],
        )
        assert n == 2
        assert m.get_entry(REC_A, "rec0/well001").tasks["sorting"].status == TaskStatus.COMPLETE


def test_refresh_cascades_to_dependents():
    with TemporaryDirectory() as tmp:
        m = _make_manager(Path(tmp))
        m.add_well(REC_A, "rec0/well000")
        _set_status(m, REC_A, "rec0/well000", "preprocessing", TaskStatus.COMPLETE)
        _set_status(m, REC_A, "rec0/well000", "sorting", TaskStatus.COMPLETE)

        # Resetting preprocessing must also flip sorting (its dependent).
        n = m.refresh("preprocessing")
        assert n == 2
        entry = m.get_entry(REC_A, "rec0/well000")
        assert entry.tasks["preprocessing"].status == TaskStatus.NOT_RUN
        assert entry.tasks["sorting"].status == TaskStatus.NOT_RUN


def test_bulk_refresh_per_task_counts():
    with TemporaryDirectory() as tmp:
        m = _make_manager(Path(tmp))
        m.add_well(REC_A, "rec0/well000")
        m.add_well(REC_B, "rec0/well000")
        # REC_A: sorting FAILED, REC_B: sorting COMPLETE.
        _set_status(m, REC_A, "rec0/well000", "sorting", TaskStatus.FAILED)
        _set_status(m, REC_B, "rec0/well000", "sorting", TaskStatus.COMPLETE)

        # Scope to REC_A only, FAILED only — REC_B sorting stays COMPLETE.
        counts = m.bulk_refresh(
            task_names=["sorting"],
            recording_keys=[REC_A],
            status_filter={TaskStatus.FAILED},
        )
        assert counts == {"sorting": 1}
        assert m.get_entry(REC_B, "rec0/well000").tasks["sorting"].status == TaskStatus.COMPLETE
        assert m.get_entry(REC_A, "rec0/well000").tasks["sorting"].status == TaskStatus.NOT_RUN


def test_bulk_refresh_cartesian_across_recordings():
    with TemporaryDirectory() as tmp:
        m = _make_manager(Path(tmp))
        for rec in (REC_A, REC_B):
            m.add_well(rec, "rec0/well000")
            _set_status(m, rec, "rec0/well000", "sorting", TaskStatus.COMPLETE)

        counts = m.bulk_refresh(
            task_names=["sorting"],
            recording_keys=[REC_A, REC_B],
        )
        assert counts == {"sorting": 2}
        for rec in (REC_A, REC_B):
            assert m.get_entry(rec, "rec0/well000").tasks["sorting"].status == TaskStatus.NOT_RUN
