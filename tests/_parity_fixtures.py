"""Shared deterministic fixture for testing the S2 unified scheduler.

Stages a ``pipeline_cache.json`` (the finest layer) via ``PipelineManager`` and
exposes the legacy finest-queue order so the unified scheduler can be asserted
equal to it. The aggregate parity oracle is a static S1-authored golden in
``test_unified_scheduler.py`` (S1 itself is gone).

Not a test module itself (leading underscore); imported by
``test_instance_store.py`` and ``test_unified_scheduler.py``.
"""

from __future__ import annotations

from pathlib import Path

from yuxin_mea.pipeline import (
    LayeredInstanceStore,
    PipelineManager,
    UnifiedScheduler,
    WorkItem,
)
from yuxin_mea.pipeline.aggregate_task import BaseAggregateTask
from yuxin_mea.pipeline.scope import SCOPE_RECORDING
from yuxin_mea.pipeline.task_record import TaskStatus

RK = "SAMP/260101/PLATE/Network/000001"    # has a blocker -> WAIT
RK2 = "SAMP/260102/PLATE/Network/000002"   # no blocker    -> RUN on complete subset
RK3 = "SAMP/260103/PLATE/Network/000003"   # all failed    -> SKIP_EMPTY


class FakeCfg:
    def __init__(self, params=None):
        self._p = params or {}

    def get_config(self, task_name, recording_key, well_id):
        return dict(self._p.get(task_name, {}))

    def get_task_params(self, task_name):
        return dict(self._p.get(task_name, {}))

    def get_global(self, key, default=None):
        return default


class DummyOverlay(BaseAggregateTask):
    task_name = "dummy_overlay"
    dependencies = ["ml_burst_detection"]
    scope = SCOPE_RECORDING

    def run(self, scope_key, members, params):
        out = Path(params["output_root"]) / "out.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(str(len(members)))
        return out


def _set(pm, rk, cw, task, status, out=None):
    pm.update_status(WorkItem(rk, cw, task), status, output_path=out)


def _seed(pm, analysis_dir):
    o = Path(analysis_dir)
    # --- RK: 6 wells; w4 ml eligible-now -> whole instance WAITs ---
    for i in range(6):
        pm.add_well(RK, f"rec0000/well00{i}")
    for i in (0, 1, 2, 4, 5):                      # curation complete (NOT w3 -> cascade hole)
        _set(pm, RK, f"rec0000/well00{i}", "curation", TaskStatus.COMPLETE, o)
    _set(pm, RK, "rec0000/well000", "ml_burst_detection", TaskStatus.COMPLETE, o / "w0")
    _set(pm, RK, "rec0000/well001", "ml_burst_detection", TaskStatus.COMPLETE, o / "w1")
    _set(pm, RK, "rec0000/well002", "ml_burst_detection", TaskStatus.FAILED)   # hole
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


def stage(analysis_dir) -> PipelineManager:
    """Seed a PipelineManager (persists the finest layer to pipeline_cache.json).
    Returns the manager (for the live finest-queue oracle)."""
    pm = PipelineManager(Path(analysis_dir), config_provider=FakeCfg())
    pm.register_computation_task("curation", [])
    pm.register_computation_task("ml_burst_detection", ["curation"])
    _seed(pm, analysis_dir)
    return pm


def build_unified(analysis_dir, agg_task_classes=(DummyOverlay,)) -> UnifiedScheduler:
    """UnifiedScheduler over the LayeredInstanceStore: finest is read live from the
    ``pipeline_cache.json`` that ``stage()``'s PipelineManager wrote; aggregate state
    migrates from ``aggregate_cache.json`` on load (same DummyOverlay task as S1)."""
    store = LayeredInstanceStore(Path(analysis_dir))
    return UnifiedScheduler(store, agg_task_classes, FakeCfg(), {})


def freeze_get_next_order(pm: PipelineManager) -> list[tuple[str, str, str]]:
    """The legacy eligible-now per-well queue as ``(recording_key, well_id, task)``."""
    return [
        (wi.recording_key, wi.well_id, wi.task_name)
        for wi in pm.get_next_task(n=10_000)
    ]


def unified_finest_order(sched: UnifiedScheduler) -> list[tuple[str, str, str]]:
    """The unified finest queue in the SAME ``(recording_key, compound_well, task)``
    shape as :func:`freeze_get_next_order`, for direct parity comparison."""
    out = []
    for wu in sched.finest_queue(n=10_000):
        rk = wu.scope_key["recording_key"]
        compound = f"{wu.scope_key['rec_name']}/{wu.scope_key['well_id']}"
        out.append((rk, compound, wu.task_name))
    return out


def unified_plan(sched: UnifiedScheduler) -> dict[str, tuple]:
    """Unified aggregate decisions keyed the same way, for parity comparison."""
    from yuxin_mea.pipeline.scope import Scope
    return {
        f"{d.task_name}::{Scope.encode_path(d.scope_key)}":
            (d.decision, d.n_total, len(d.members_complete), d.member_hash)
        for d in sched.aggregate_plan()
    }
