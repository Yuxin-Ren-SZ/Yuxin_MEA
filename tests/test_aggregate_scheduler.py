"""AggregateScheduler: bucketing, barrier-with-holes, membership re-run, warnings."""

from pathlib import Path

from yuxin_mea.pipeline import PipelineManager, WorkItem
from yuxin_mea.pipeline.aggregate_cache import JsonAggregateCacheStore
from yuxin_mea.pipeline.aggregate_scheduler import (
    RERUN,
    RUN,
    SKIP_EMPTY,
    UPTODATE,
    WAIT,
    AggregateScheduler,
)
from yuxin_mea.pipeline.aggregate_task import BaseAggregateTask
from yuxin_mea.pipeline.groupname import detect_groupname_issues
from yuxin_mea.pipeline.scope import Scope
from yuxin_mea.pipeline.task_record import TaskStatus
from yuxin_mea.pipeline.well_dims import WellDims

RK = "SAMP/260101/PLATE/Network/000001"    # has a blocker -> WAIT
RK2 = "SAMP/260102/PLATE/Network/000002"   # no blocker    -> RUN on complete subset
RK3 = "SAMP/260103/PLATE/Network/000003"   # all failed    -> SKIP_EMPTY


class _FakeCfg:
    def __init__(self, params=None):
        self._p = params or {}

    def get_config(self, task_name, recording_key, well_id):
        return dict(self._p.get(task_name, {}))

    def get_task_params(self, task_name):
        return dict(self._p.get(task_name, {}))

    def get_global(self, key, default=None):
        return default


class _DummyAgg(BaseAggregateTask):
    task_name = "dummy_overlay"
    dependencies = ["ml_burst_detection"]
    scope = Scope("rec", ("recording_key",), where=lambda d: d.scan_type == "Network")

    def run(self, scope_key, members, params):
        out = Path(params["output_root"]) / "out.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(str(len(members)))
        return out


def _set(pm, rk, cw, task, status, out=None):
    pm.update_status(WorkItem(rk, cw, task), status, output_path=out)


def _seed(pm, tmp):
    o = Path(tmp)
    # --- RK: 6 wells; w4 ml is eligible-now -> the whole instance WAITs ---
    for i in range(6):
        pm.add_well(RK, f"rec0000/well00{i}")
    for i in (0, 1, 2, 4, 5):                     # curation complete (NOT w3 -> cascade hole)
        _set(pm, RK, f"rec0000/well00{i}", "curation", TaskStatus.COMPLETE, o)
    _set(pm, RK, "rec0000/well000", "ml_burst_detection", TaskStatus.COMPLETE, o / "w0")
    _set(pm, RK, "rec0000/well001", "ml_burst_detection", TaskStatus.COMPLETE, o / "w1")
    _set(pm, RK, "rec0000/well002", "ml_burst_detection", TaskStatus.FAILED)   # hole
    # w3 curation NOT_RUN -> ml NOT_RUN & not eligible -> cascade hole (leave)
    # w4 curation COMPLETE, ml NOT_RUN -> eligible-now -> BLOCKER (leave ml NOT_RUN)
    _set(pm, RK, "rec0000/well005", "ml_burst_detection", TaskStatus.COMPLETE, o / "w5")

    # --- RK2: 3 wells; no blocker; 2 complete + 1 failed hole -> RUN 2/3 ---
    for i in range(3):
        pm.add_well(RK2, f"rec0000/well00{i}")
        _set(pm, RK2, f"rec0000/well00{i}", "curation", TaskStatus.COMPLETE, o)
    _set(pm, RK2, "rec0000/well000", "ml_burst_detection", TaskStatus.COMPLETE, o / "a")
    _set(pm, RK2, "rec0000/well001", "ml_burst_detection", TaskStatus.COMPLETE, o / "b")
    _set(pm, RK2, "rec0000/well002", "ml_burst_detection", TaskStatus.FAILED)

    # --- RK3: 2 wells; both ml failed; no complete -> SKIP_EMPTY ---
    for i in range(2):
        pm.add_well(RK3, f"rec0000/well00{i}")
        _set(pm, RK3, f"rec0000/well00{i}", "curation", TaskStatus.COMPLETE, o)
        _set(pm, RK3, f"rec0000/well00{i}", "ml_burst_detection", TaskStatus.FAILED)


def _mk(tmp):
    pm = PipelineManager(Path(tmp), config_provider=_FakeCfg())
    pm.register_computation_task("curation", [])
    pm.register_computation_task("ml_burst_detection", ["curation"])
    _seed(pm, tmp)
    store = JsonAggregateCacheStore(Path(tmp))
    sched = AggregateScheduler(pm, _FakeCfg(), {}, (_DummyAgg,), store)
    return pm, sched


def _by_rk(plan):
    return {pi.scope_key["recording_key"]: pi for pi in plan}


def test_bucketing_and_barrier(tmp_path):
    _pm, sched = _mk(tmp_path)
    plan = _by_rk(sched.plan())
    assert set(plan) == {RK, RK2, RK3}          # all Network recordings bucketed
    assert plan[RK].decision == WAIT            # w4 ml eligible-now blocks
    assert plan[RK2].decision == RUN
    assert plan[RK2].n_total == 3
    assert len(plan[RK2].members_complete) == 2   # 2 complete, 1 failed hole
    assert plan[RK3].decision == SKIP_EMPTY


def test_membership_hash_rerun(tmp_path):
    pm, sched = _mk(tmp_path)
    pi = _by_rk(sched.plan())[RK2]
    # simulate running it
    sched.mark_complete(pi.task, pi.scope_key, tmp_path / "out.txt", pi.member_hash,
                        pi.n_total, len(pi.members_complete))
    assert _by_rk(sched.plan())[RK2].decision == UPTODATE

    # a member's ml re-ran (new output/timestamp) -> fingerprint changes -> RERUN
    _set(pm, RK2, "rec0000/well000", "ml_burst_detection", TaskStatus.COMPLETE, tmp_path / "a2")
    assert _by_rk(sched.plan())[RK2].decision == RERUN

    # a newly-completed member also triggers RERUN
    pm2, sched2 = _mk(tmp_path / "b")
    pi2 = _by_rk(sched2.plan())[RK2]
    sched2.mark_complete(pi2.task, pi2.scope_key, tmp_path / "o2", pi2.member_hash,
                         pi2.n_total, len(pi2.members_complete))
    _set(pm2, RK2, "rec0000/well002", "ml_burst_detection", TaskStatus.COMPLETE, tmp_path / "b" / "c")
    assert _by_rk(sched2.plan())[RK2].decision == RERUN


def test_groupname_issue_detection():
    dims = [
        WellDims.from_entry("CX1/260101/P/Network/000001", "rec0000/well000", "Control"),
        WellDims.from_entry("CX2/260101/P/Network/000001", "rec0000/well000", "Control"),   # cross-sample
        WellDims.from_entry("CX1/260101/P/Network/000001", "rec0000/well001", "10uM_H2O2"),
        WellDims.from_entry("CX1/260101/P/Network/000001", "rec0000/well002", "10 uM H2O2"),  # variant
    ]
    issues = detect_groupname_issues(dims)
    joined = "\n".join(issues)
    assert "spans" in joined            # cross-sample collision (Control)
    assert "variants" in joined         # 10uM_H2O2 vs 10 uM H2O2
