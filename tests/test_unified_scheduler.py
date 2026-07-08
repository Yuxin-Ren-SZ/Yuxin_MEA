"""UnifiedScheduler: parity with the two legacy schedulers, the finest-semantics
oracle, and the aggregate→aggregate containment behaviour (no legacy equivalent)."""

from pathlib import Path

import pytest

from yuxin_mea.pipeline import LayeredInstanceStore, ScopedInstance, UnifiedScheduler
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
    stage,
    unified_finest_order,
    unified_plan,
)


# --------------------------------------------------------------------------- #
# Parity: finest queue == legacy get_next_task order
# --------------------------------------------------------------------------- #

def test_finest_queue_parity(tmp_path):
    pm = stage(tmp_path)
    legacy = freeze_get_next_order(pm)
    unified = unified_finest_order(build_unified(tmp_path))
    assert unified == legacy
    # sanity: the seed leaves real eligible per-well work
    assert len(legacy) > 0


def test_finest_queue_never_returns_complete(tmp_path):
    """Finest semantics oracle (advisor pt #2): a COMPLETE per-well task is never
    returned by EITHER queue — config drift / a newer upstream cannot auto-rerun it
    (that path is an explicit refresh, unchanged by S2)."""
    pm = stage(tmp_path)
    legacy = set(freeze_get_next_order(pm))
    unified = set(unified_finest_order(build_unified(tmp_path)))
    # RK2 well000 ml_burst_detection is COMPLETE in the seed
    complete_unit = (RK2, "rec0000/well000", "ml_burst_detection")
    assert complete_unit not in legacy
    assert complete_unit not in unified
    assert unified == legacy


# --------------------------------------------------------------------------- #
# Parity: aggregate decisions == the S1-AUTHORED golden
#
# The golden below was generated from S1's AggregateScheduler.plan() on the shared
# fixture (member_hash omitted — it hashes upstream timestamps and isn't static),
# and confirmed live (`unified == S1` decisions + member_hash byte-equivalence) one
# last time before S1 was deleted in Phase 4.3. It is S1-AUTHORED, not unified —
# regenerating it from unified's own output would make the test prove nothing.
# --------------------------------------------------------------------------- #

GOLDEN_S1_AGG_PLAN = {
    "dummy_overlay::recording_key=SAMP-260101-PLATE-Network-000001": ("wait", 6, 0),
    "dummy_overlay::recording_key=SAMP-260102-PLATE-Network-000002": ("run", 3, 2),
    "dummy_overlay::recording_key=SAMP-260103-PLATE-Network-000003": ("skip_empty", 2, 0),
}


def _decisions_only(plan: dict) -> dict:
    return {k: (v[0], v[1], v[2]) for k, v in plan.items()}


def test_aggregate_plan_matches_golden(tmp_path):
    """Unified aggregate decisions reproduce the S1-authored golden."""
    stage(tmp_path)   # writes pipeline_cache.json (finest layer)
    unified = _decisions_only(unified_plan(build_unified(tmp_path)))
    assert unified == GOLDEN_S1_AGG_PLAN


def test_aggregate_member_hash_populated_and_recomputes(tmp_path):
    """Unified RUN instances carry a non-empty member_hash equal to the fingerprint
    of their complete members (the re-run trigger). No S1 dependency."""
    stage(tmp_path)
    sched = build_unified(tmp_path)
    run = next(d for d in sched.aggregate_plan() if d.decision == us.RUN)
    assert run.member_hash != ""
    assert us._member_fingerprint(run.members_complete) == run.member_hash


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
    store = LayeredInstanceStore(tmp_path)
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
