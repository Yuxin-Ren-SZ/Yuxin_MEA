"""End-to-end unified drain: finest → aggregate → aggregate→aggregate, exercising
_drain_unified / _run_finest_unit / _run_agg_unit with injected fake tasks (no
real spike data)."""

from pathlib import Path

from yuxin_mea.cli.run import _drain_unified
from yuxin_mea.pipeline import LayeredInstanceStore, UnifiedScheduler
from yuxin_mea.pipeline.aggregate_task import BaseAggregateTask
from yuxin_mea.pipeline.scope import SCOPE_RECORDING, SCOPE_SAMPLE, Scope
from yuxin_mea.pipeline.task_record import TaskStatus

RK = "S/260101/P/Network/000001"


class FakeCM:
    def __init__(self, figure_root):
        self.fig = str(figure_root)

    def get_task_params(self, task_name):
        return {}

    def get_config(self, task_name, recording_key, well_id):
        return {}

    def get_global(self, key, default=None):
        return self.fig if key in ("figure_root", "analysis_root") else default


class FakeDataset:
    def __init__(self, recdir):
        self.recdir = recdir

    def get_recording_by(self, filters):
        return [object()]

    def get_path(self, entry):
        return self.recdir


class FakeCuration:
    task_name = "curation"
    dependencies: list[str] = []

    def run(self, rk, cw, data_path, params):
        out = Path(data_path) / "curation" / cw.replace("/", "_")
        out.mkdir(parents=True, exist_ok=True)
        (out / "c").write_text("x")
        return out


class FakeMl:
    task_name = "ml"
    dependencies = ["curation"]

    def run(self, rk, cw, data_path, params):
        out = Path(data_path) / "ml" / cw.replace("/", "_")
        out.mkdir(parents=True, exist_ok=True)
        (out / "m").write_text("x")
        return out


class FakeOverlay(BaseAggregateTask):
    task_name = "ov"
    dependencies = ["ml"]
    scope = SCOPE_RECORDING

    def run(self, scope_key, members, params):
        out = Path(params["output_root"]) / "ov" / Scope.encode_path(scope_key)
        out.mkdir(parents=True, exist_ok=True)
        f = out / "o.html"
        f.write_text(f"ov:{len(members)}")
        return f


class FakeSummary(BaseAggregateTask):
    task_name = "sm"
    dependencies = ["ov"]           # aggregate→aggregate: recording ⊂ sample
    scope = SCOPE_SAMPLE

    def run(self, scope_key, members, params):
        out = Path(params["output_root"]) / "sm" / Scope.encode_path(scope_key)
        out.mkdir(parents=True, exist_ok=True)
        f = out / "s.html"
        f.write_text(f"sm:{len(members)}")
        return f


def _build(base):
    base.mkdir(parents=True, exist_ok=True)
    store = LayeredInstanceStore(base)
    sched = UnifiedScheduler(store, (FakeOverlay, FakeSummary), FakeCM(base / "fig"), {})
    for w in ("well000", "well001"):
        sched.register_finest(RK, f"rec0000/{w}", [("curation", []), ("ml", ["curation"])])
    return store, sched


def _make_finest_runner(recdir):
    """A ThreadPool-friendly finest runner over the fake per-well tasks."""
    tasks = {"curation": FakeCuration(), "ml": FakeMl()}

    def run(task_name, rk, compound):
        try:
            out = tasks[task_name].run(rk, compound, recdir, {})
            return {"status": TaskStatus.COMPLETE, "output_path": out, "error": None}
        except Exception:
            import traceback
            return {"status": TaskStatus.FAILED, "output_path": None,
                    "error": traceback.format_exc()}

    return run


def _drive(base, sched, *, max_tasks=None, jobs=1):
    recdir = base / "rec"
    recdir.mkdir(parents=True, exist_ok=True)
    return _drain_unified(
        sched, FakeCM(base / "fig"), FakeDataset(recdir),
        task_allow=None, rec_allow=None, retry_failed=False, max_tasks=max_tasks,
        aggregate=True, finest=True, jobs=jobs,
        well_instances={"curation": FakeCuration(), "ml": FakeMl()},
        agg_instances={"ov": FakeOverlay(), "sm": FakeSummary()},
        finest_runner=(_make_finest_runner(recdir) if jobs > 1 else None),
    )


def test_full_chain_completes(tmp_path):
    (tmp_path / "rec").mkdir()
    _store, sched = _build(tmp_path)
    n_ran, n_failed = _drive(tmp_path, sched)
    assert n_failed == 0
    # 4 finest (2 wells × curation+ml) + 1 overlay + 1 summary
    assert n_ran == 6

    reloaded = LayeredInstanceStore(tmp_path).load()
    # every finest task COMPLETE
    finest = [i for i in reloaded.values() if i.scope_name == "well"]
    assert len(finest) == 2
    for inst in finest:
        assert inst.tasks["curation"].status == TaskStatus.COMPLETE
        assert inst.tasks["ml"].status == TaskStatus.COMPLETE
    # the recording overlay ran on 2 members
    ov = next(i for i in reloaded.values() if i.scope_name == "recording")
    assert ov.tasks["ov"].status == TaskStatus.COMPLETE
    assert ov.tasks["ov"].output_path.read_text() == "ov:2"
    # the sample summary (aggregate→aggregate) ran on the 1 overlay member
    sm = next(i for i in reloaded.values() if i.scope_name == "sample")
    assert sm.tasks["sm"].status == TaskStatus.COMPLETE
    assert sm.tasks["sm"].output_path.read_text() == "sm:1"
    assert sm.tasks["sm"].n_members_complete == 1


def test_rerun_is_idempotent(tmp_path):
    (tmp_path / "rec").mkdir()
    _store, sched = _build(tmp_path)
    _drive(tmp_path, sched)
    # a second drain over the completed store does nothing (all UPTODATE/COMPLETE)
    store2 = LayeredInstanceStore(tmp_path)
    sched2 = UnifiedScheduler(store2, (FakeOverlay, FakeSummary), FakeCM(tmp_path / "fig"), {})
    n_ran, n_failed = _drive(tmp_path, sched2)
    assert (n_ran, n_failed) == (0, 0)


def test_max_tasks_caps_drain(tmp_path):
    (tmp_path / "rec").mkdir()
    _store, sched = _build(tmp_path)
    n_ran, _ = _drive(tmp_path, sched, max_tasks=2)
    assert n_ran == 2


def _fresh_sched(base):
    return UnifiedScheduler(
        LayeredInstanceStore(base), (FakeOverlay, FakeSummary), FakeCM(base / "fig"), {}
    )


def test_finest_completion_stamps_last_updated_and_retriggers_aggregate(tmp_path):
    """Regression guard: the unified drain must stamp TaskRecord.last_updated on
    finest completion (legacy did). It is load-bearing — the aggregate re-run
    fingerprint hashes upstream last_updated, and per-well output paths are stable,
    so without the stamp a re-run upstream never re-triggers its aggregate."""
    _s, sched = _build(tmp_path)
    _drive(tmp_path, sched)

    loaded = LayeredInstanceStore(tmp_path).load()
    fin = next(i for i in loaded.values() if i.scope_name == "well")
    assert fin.tasks["ml"].last_updated is not None          # the fix
    ov_hash_before = next(
        i for i in loaded.values() if i.scope_name == "recording"
    ).tasks["ov"].member_hash
    assert ov_hash_before != ""

    # reset one well's ml → NOT_RUN, then re-drive: ml re-runs (fresh last_updated,
    # SAME output path), so the overlay's fingerprint changes → it RERUNs.
    reset = _fresh_sched(tmp_path)
    well = next(i for i in reset.instances.values() if i.scope_name == "well")
    reset.update_status("well", well.scope_key, "ml", TaskStatus.NOT_RUN)

    _drive(tmp_path, _fresh_sched(tmp_path))

    after = LayeredInstanceStore(tmp_path).load()
    ov_after = next(i for i in after.values() if i.scope_name == "recording").tasks["ov"]
    assert ov_after.status == TaskStatus.COMPLETE
    assert ov_after.member_hash != ov_hash_before            # re-triggered via last_updated
    assert ov_after.last_updated is not None


def _store_statuses(base):
    insts = LayeredInstanceStore(base).load()
    return {
        (i.scope_name, i.instance_key.split("::", 1)[1], tn): t.status
        for i in insts.values() for tn, t in i.tasks.items()
    }


def test_parallel_drain_matches_serial(tmp_path):
    """Parallel finest drain reaches the SAME final store state as serial —
    order-independence of the parent-owned-writes discipline."""
    _s, s_sched = _build(tmp_path / "serial")
    _p, p_sched = _build(tmp_path / "par")
    serial = _drive(tmp_path / "serial", s_sched, jobs=1)
    par = _drive(tmp_path / "par", p_sched, jobs=4)

    assert serial == par == (6, 0)
    # identical scope instances + identical per-task statuses, regardless of order
    assert _store_statuses(tmp_path / "serial") == _store_statuses(tmp_path / "par")
    # the aggregate→aggregate summary consumed the same member counts both ways
    for base in (tmp_path / "serial", tmp_path / "par"):
        insts = LayeredInstanceStore(base).load()
        ov = next(i for i in insts.values() if i.scope_name == "recording")
        sm = next(i for i in insts.values() if i.scope_name == "sample")
        assert ov.tasks["ov"].n_members_complete == 2
        assert sm.tasks["sm"].n_members_complete == 1
