"""UnifiedScheduler: parity with the two legacy schedulers, the finest-semantics
oracle, and the aggregate→aggregate containment behaviour (no legacy equivalent)."""

from pathlib import Path

import pytest

from yuxin_mea.pipeline import JsonInstanceStore, ScopedInstance, UnifiedScheduler
from yuxin_mea.pipeline import unified_scheduler as us
from yuxin_mea.pipeline.task_record import TaskRecord, TaskStatus
from yuxin_mea.tasks.cluster_overlay_tasks import (
    ClusterOverlayByRecordingTask,
    SampleOverlaySummaryTask,
)

from tests._parity_fixtures import (
    RK2,
    FakeCfg,
    build_unified,
    freeze_get_next_order,
    freeze_plan,
    stage,
    unified_finest_order,
    unified_plan,
)


# --------------------------------------------------------------------------- #
# Parity: finest queue == legacy get_next_task order
# --------------------------------------------------------------------------- #

def test_finest_queue_parity(tmp_path):
    pm, _sched = stage(tmp_path)
    legacy = freeze_get_next_order(pm)
    unified = unified_finest_order(build_unified(tmp_path))
    assert unified == legacy
    # sanity: the seed leaves real eligible per-well work
    assert len(legacy) > 0


def test_finest_queue_never_returns_complete(tmp_path):
    """Finest semantics oracle (advisor pt #2): a COMPLETE per-well task is never
    returned by EITHER queue — config drift / a newer upstream cannot auto-rerun it
    (that path is an explicit refresh, unchanged by S2)."""
    pm, _sched = stage(tmp_path)
    legacy = set(freeze_get_next_order(pm))
    unified = set(unified_finest_order(build_unified(tmp_path)))
    # RK2 well000 ml_burst_detection is COMPLETE in the seed
    complete_unit = (RK2, "rec0000/well000", "ml_burst_detection")
    assert complete_unit not in legacy
    assert complete_unit not in unified
    assert unified == legacy


# --------------------------------------------------------------------------- #
# Parity: aggregate decisions == S1 plan()
# --------------------------------------------------------------------------- #

def test_aggregate_plan_parity(tmp_path):
    pm, sched = stage(tmp_path)
    legacy = freeze_plan(sched)               # S1 AggregateScheduler.plan()
    unified = unified_plan(build_unified(tmp_path))
    assert unified == legacy
    # the fixture exercises WAIT, RUN (2/3 with a hole) and SKIP_EMPTY
    decisions = {v[0] for v in legacy.values()}
    assert {"wait", "run", "skip_empty"} <= decisions


def test_aggregate_member_hash_matches_s1(tmp_path):
    pm, sched = stage(tmp_path)
    legacy = freeze_plan(sched)
    unified = unified_plan(build_unified(tmp_path))
    # RK2 is the RUN case with 2 complete members — the fingerprint must be identical
    key = next(k for k in legacy if RK2.replace("/", "-") in k and legacy[k][0] == "run")
    assert unified[key][3] == legacy[key][3] != ""   # member_hash equal & non-empty


# --------------------------------------------------------------------------- #
# Containment: sample_overlay_summary (aggregate→aggregate) — own behaviour
# --------------------------------------------------------------------------- #

WUID0 = "CXA|PL|well000"
WUID1 = "CXA|PL|well001"


def _make_sample_store(tmp_path, overlay_states):
    """overlay_states: {well_uid: (status, output_path|None)} for the upstream
    cluster_overlay_by_recording task. Builds 4 finest wells (2 dates × 2 wells)
    of sample CXA plus the given well_uid overlay instances."""
    insts: dict[str, ScopedInstance] = {}
    for date in ("260101", "260115"):
        rk = f"CXA/{date}/PL/Network/000001"
        for wid in ("well000", "well001"):
            fi = ScopedInstance(
                "well", {"recording_key": rk, "rec_name": "rec0000", "well_id": wid}, {}
            )
            insts[fi.instance_key] = fi
    for wuid, (status, outp) in overlay_states.items():
        inst = ScopedInstance(
            "longitudinal_well", {"well_uid": wuid},
            {"cluster_overlay_by_recording": TaskRecord(status, [], outp, 1.0, None)},
        )
        insts[inst.instance_key] = inst
    store = JsonInstanceStore(tmp_path)
    store.save(insts)
    return UnifiedScheduler(
        store, (ClusterOverlayByRecordingTask, SampleOverlaySummaryTask), FakeCfg(), {}
    )


def _sample_decision(sched):
    plan = sched.aggregate_plan(["sample_overlay_summary"])
    assert len(plan) == 1
    return plan[0]


def test_sample_runs_on_complete_overlays(tmp_path):
    sched = _make_sample_store(tmp_path, {
        WUID0: (TaskStatus.COMPLETE, tmp_path / "w0" / "overlay.html"),
        WUID1: (TaskStatus.COMPLETE, tmp_path / "w1" / "overlay.html"),
    })
    d = _sample_decision(sched)
    assert d.decision == us.RUN
    assert d.n_total == 2
    assert len(d.members_complete) == 2
    assert d.scope_key == {"sample_id": "CXA"}
    # members carry the upstream aggregate instance identity
    assert {m.scope_key["well_uid"] for m in d.members_complete} == {WUID0, WUID1}


def test_sample_waits_on_running_upstream(tmp_path):
    sched = _make_sample_store(tmp_path, {
        WUID0: (TaskStatus.COMPLETE, tmp_path / "w0" / "overlay.html"),
        WUID1: (TaskStatus.RUNNING, None),
    })
    assert _sample_decision(sched).decision == us.WAIT


def test_sample_skip_empty_when_all_upstream_empty(tmp_path):
    """Advisor-critical: upstream COMPLETE-without-output (SKIP_EMPTY) is a HOLE,
    so an all-empty sample is SKIP_EMPTY — NOT a RUN that then fails."""
    sched = _make_sample_store(tmp_path, {
        WUID0: (TaskStatus.COMPLETE, None),
        WUID1: (TaskStatus.COMPLETE, None),
    })
    d = _sample_decision(sched)
    assert d.decision == us.SKIP_EMPTY
    assert len(d.members_complete) == 0


def test_sample_tolerates_failed_hole(tmp_path):
    sched = _make_sample_store(tmp_path, {
        WUID0: (TaskStatus.COMPLETE, tmp_path / "w0" / "overlay.html"),
        WUID1: (TaskStatus.FAILED, None),
    })
    d = _sample_decision(sched)
    assert d.decision == us.RUN
    assert len(d.members_complete) == 1
    assert d.members_complete[0].scope_key["well_uid"] == WUID0


def test_sample_task_runs_end_to_end(tmp_path):
    # real overlay files so SampleOverlaySummaryTask.run writes a linking index
    o0 = tmp_path / "w0" / "overlay.html"
    o1 = tmp_path / "w1" / "overlay.html"
    for o in (o0, o1):
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text("<html>overlay</html>")
    sched = _make_sample_store(tmp_path, {
        WUID0: (TaskStatus.COMPLETE, o0),
        WUID1: (TaskStatus.COMPLETE, o1),
    })
    d = _sample_decision(sched)
    task = SampleOverlaySummaryTask()
    out = task.run(d.scope_key, d.members_complete, {"output_root": str(tmp_path / "fig")})
    assert out.exists()
    body = out.read_text()
    assert "CXA" in body and "overlay.html" in body


def test_relation_error_surfaces_for_bad_dependency(tmp_path):
    # a sample task depending on a COARSER scope than itself must raise, not silently
    # tolerate — guards the containment validation path.
    from yuxin_mea.pipeline.scope import SCOPE_SAMPLE, Scope
    assert SCOPE_SAMPLE.relation_to  # sanity
    with pytest.raises(ValueError):
        SCOPE_SAMPLE.relation_to(Scope("coarser", ("date",)))
