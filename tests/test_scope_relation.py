"""Scope-containment lattice: relation_to / contains, and the TaskRecord round-trip
that carries the new S2 aggregate-bookkeeping fields."""

import pytest

from yuxin_mea.pipeline.cache import _decode, _entry_to_dict
from yuxin_mea.pipeline.pipeline_entry import PipelineEntry
from yuxin_mea.pipeline.scope import (
    FINEST_GROUP_BY,
    SCOPE_RECORDING,
    SCOPE_SAMPLE,
    SCOPE_SAMPLE_GROUP,
    SCOPE_WELL,
    SCOPE_WELL_UID,
    Scope,
    ScopeRelation,
    _closure,
)
from yuxin_mea.pipeline.task_record import TaskRecord, TaskStatus
from yuxin_mea.pipeline.well_dims import WellDims

RK = "CX169/260602/T003346/Network/000004"


# --------------------------------------------------------------------------- #
# is_finest / _closure
# --------------------------------------------------------------------------- #

def test_is_finest():
    assert SCOPE_WELL.is_finest is True
    assert SCOPE_RECORDING.is_finest is False
    assert SCOPE_SAMPLE.is_finest is False


def test_closure_finest_is_everything():
    # A fully-identified well determines every dimension (incl. composite well_uid
    # and LUT-derived groupname), not just the pairwise structural splits.
    from yuxin_mea.pipeline.scope import DIMENSION_NAMES
    assert _closure(FINEST_GROUP_BY) == frozenset(DIMENSION_NAMES)


def test_closure_recording_key():
    assert _closure(("recording_key",)) == frozenset(
        {"recording_key", "sample_id", "date", "plate_id", "scan_type", "run_id"}
    )


def test_closure_well_uid():
    assert _closure(("well_uid",)) == frozenset(
        {"well_uid", "sample_id", "plate_id", "well_id"}
    )


# --------------------------------------------------------------------------- #
# relation_to  (downstream.relation_to(dependency))
# --------------------------------------------------------------------------- #

def test_same_relation():
    assert SCOPE_WELL.relation_to(SCOPE_WELL) is ScopeRelation.SAME
    assert SCOPE_SAMPLE.relation_to(Scope("s2", ("sample_id",))) is ScopeRelation.SAME


def test_well_contains_recording_dep():
    # recording task depends on the finest (per-well) task
    assert SCOPE_RECORDING.relation_to(SCOPE_WELL) is ScopeRelation.CONTAINS


def test_sample_contains_well_dep():
    assert SCOPE_SAMPLE.relation_to(SCOPE_WELL) is ScopeRelation.CONTAINS


def test_sample_contains_recording_dep():
    # sample_id and recording_key share NO literal dimension name, yet a recording
    # refines a sample — the lattice, not name-intersection, gets this right.
    assert SCOPE_SAMPLE.relation_to(SCOPE_RECORDING) is ScopeRelation.CONTAINS


def test_sample_contains_well_uid_dep():
    # the real SampleOverlaySummaryTask edge: SCOPE_SAMPLE depends on SCOPE_WELL_UID
    assert SCOPE_SAMPLE.relation_to(SCOPE_WELL_UID) is ScopeRelation.CONTAINS


def test_sample_group_contains_well_dep():
    assert SCOPE_SAMPLE_GROUP.relation_to(SCOPE_WELL) is ScopeRelation.CONTAINS


def test_reverse_is_rejected():
    # a recording does NOT refine a well; a sample does NOT refine a recording
    with pytest.raises(ValueError):
        SCOPE_WELL.relation_to(SCOPE_RECORDING)
    with pytest.raises(ValueError):
        SCOPE_RECORDING.relation_to(SCOPE_SAMPLE)


def test_disjoint_is_rejected():
    grp = Scope("g", ("groupname",))
    run = Scope("r", ("run_id",))
    with pytest.raises(ValueError):
        run.relation_to(grp)


def test_coarse_on_coarse_rejected():
    by_date = Scope("d", ("date",))
    by_plate = Scope("p", ("plate_id",))
    with pytest.raises(ValueError):
        by_plate.relation_to(by_date)


# --------------------------------------------------------------------------- #
# contains  (instance-level key check)
# --------------------------------------------------------------------------- #

def test_contains_well_under_sample():
    dims = WellDims.from_entry(RK, "rec0000/well002", "Control")
    sample_key = SCOPE_SAMPLE.scope_key_dict(dims)          # {"sample_id": "CX169"}
    well_key = SCOPE_WELL.scope_key_dict(dims)
    assert SCOPE_SAMPLE.contains(sample_key, well_key, SCOPE_WELL) is True


def test_contains_well_uid_under_sample():
    dims = WellDims.from_entry(RK, "rec0000/well002", "Control")
    sample_key = SCOPE_SAMPLE.scope_key_dict(dims)
    wuid_key = SCOPE_WELL_UID.scope_key_dict(dims)
    assert SCOPE_SAMPLE.contains(sample_key, wuid_key, SCOPE_WELL_UID) is True


def test_contains_rejects_other_sample():
    dims_a = WellDims.from_entry(RK, "rec0000/well002", "Control")
    dims_b = WellDims.from_entry("CX999/260602/T003346/Network/000004", "rec0000/well002", "Control")
    sample_key = SCOPE_SAMPLE.scope_key_dict(dims_a)        # CX169
    other_well = SCOPE_WELL.scope_key_dict(dims_b)          # CX999
    assert SCOPE_SAMPLE.contains(sample_key, other_well, SCOPE_WELL) is False


def test_contains_false_when_no_relation():
    grp = Scope("g", ("groupname",))
    run = Scope("r", ("run_id",))
    assert run.contains({"run_id": "x"}, {"groupname": "y"}, grp) is False


# --------------------------------------------------------------------------- #
# TaskRecord round-trip with new S2 fields (back-compat + forward)
# --------------------------------------------------------------------------- #

def test_taskrecord_defaults():
    t = TaskRecord(TaskStatus.NOT_RUN, [], None, None, None)
    assert t.member_hash == ""
    assert t.n_members_total == 0
    assert t.n_members_complete == 0


def test_taskrecord_roundtrip_new_fields():
    entry = PipelineEntry(
        recording_key=RK, well_id="rec0000/well002", created_at=1.0,
        tasks={
            "ml_burst_detection": TaskRecord(
                TaskStatus.COMPLETE, [], None, 2.0, None, {"seed": 42},
                member_hash="abc123", n_members_total=24, n_members_complete=23,
            )
        },
    )
    d = _entry_to_dict(entry)
    # decode task dict back through the object_hook path
    task_dict = d["tasks"]["ml_burst_detection"]
    rt = _decode(task_dict)
    assert isinstance(rt, TaskRecord)
    assert rt.member_hash == "abc123"
    assert rt.n_members_total == 24
    assert rt.n_members_complete == 23


def test_taskrecord_decode_legacy_missing_fields():
    # old cache entry without the new keys must still decode (subset gate)
    legacy = {
        "status": "complete", "dependencies": [], "output_path": None,
        "last_updated": 1.0, "error": None, "config": {},
    }
    rt = _decode(legacy)
    assert isinstance(rt, TaskRecord)
    assert rt.member_hash == ""
    assert rt.n_members_total == 0
