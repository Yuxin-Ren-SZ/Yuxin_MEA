"""InstanceStore round-trip + migration loaders from the two legacy caches."""

from pathlib import Path

from yuxin_mea.pipeline import (
    JsonInstanceStore,
    ScopedInstance,
    load_legacy_aggregates,
    load_legacy_wells,
)
from yuxin_mea.pipeline.aggregate_cache import (
    AggregateInstanceRecord,
    JsonAggregateCacheStore,
)
from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
from yuxin_mea.pipeline.task_record import TaskRecord, TaskStatus

from tests._parity_fixtures import RK, RK2, stage


def test_load_legacy_wells_count_and_order(tmp_path):
    pm, _sched = stage(tmp_path)
    entries = JsonPipelineCacheStore(tmp_path).load()
    wells = load_legacy_wells(entries)
    # 6 + 3 + 2 = 11 wells, all finest scope
    assert len(wells) == 11
    assert all(inst.scope_name == "well" for inst in wells.values())
    # order preserved from the legacy cache (parity constraint)
    legacy_order = [
        (e.recording_key, e.well_id) for e in entries.values()
    ]
    well_order = [
        (i.scope_key["recording_key"], f"{i.scope_key['rec_name']}/{i.scope_key['well_id']}")
        for i in wells.values()
    ]
    assert well_order == legacy_order


def test_well_scope_key_shape(tmp_path):
    pm, _sched = stage(tmp_path)
    entries = JsonPipelineCacheStore(tmp_path).load()
    wells = load_legacy_wells(entries)
    inst = next(iter(wells.values()))
    assert set(inst.scope_key) == {"recording_key", "rec_name", "well_id"}
    # the finest instance carries the per-well tasks verbatim
    assert "ml_burst_detection" in inst.tasks


def test_load_legacy_aggregates_from_marked_run(tmp_path):
    pm, sched = stage(tmp_path)
    plan = {pi.scope_key["recording_key"]: pi for pi in sched.plan()}
    pi = plan[RK2]
    sched.mark_complete(pi.task, pi.scope_key, tmp_path / "out.txt",
                        pi.member_hash, pi.n_total, len(pi.members_complete))
    records = JsonAggregateCacheStore(tmp_path).load()
    aggs = load_legacy_aggregates(records)
    assert len(aggs) == 1
    inst = next(iter(aggs.values()))
    assert inst.scope_name == "recording"
    assert inst.scope_key == {"recording_key": RK2}
    assert "dummy_overlay" in inst.tasks
    assert inst.tasks["dummy_overlay"].status == TaskStatus.COMPLETE
    assert inst.tasks["dummy_overlay"].n_members_complete == 2


def test_aggregate_siblings_group_into_one_instance():
    # two tasks at the SAME scope_key must reduce to one ScopedInstance
    key = {"recording_key": RK}
    records = {
        "a": AggregateInstanceRecord("task_a", "recording", key, TaskStatus.COMPLETE),
        "b": AggregateInstanceRecord("task_b", "recording", key, TaskStatus.FAILED),
    }
    aggs = load_legacy_aggregates(records)
    assert len(aggs) == 1
    inst = next(iter(aggs.values()))
    assert set(inst.tasks) == {"task_a", "task_b"}


def test_json_instance_store_roundtrip(tmp_path):
    pm, sched = stage(tmp_path)
    entries = JsonPipelineCacheStore(tmp_path).load()
    instances = load_legacy_wells(entries)
    store = JsonInstanceStore(tmp_path)
    store.save(instances)
    reloaded = store.load()
    assert set(reloaded) == set(instances)
    assert len(reloaded) == 11
    # spot-check a task record survived incl. the new S2 fields
    any_key = next(iter(reloaded))
    assert isinstance(reloaded[any_key], ScopedInstance)
    tr = next(iter(reloaded[any_key].tasks.values()))
    assert isinstance(tr, TaskRecord)
    assert tr.member_hash == ""      # finest instance: default


def test_roundtrip_preserves_aggregate_bookkeeping(tmp_path):
    inst = ScopedInstance(
        "recording", {"recording_key": RK},
        tasks={
            "dummy_overlay": TaskRecord(
                TaskStatus.COMPLETE, [], tmp_path / "o", 5.0, None, {"seed": 1},
                member_hash="deadbeef", n_members_total=6, n_members_complete=4,
            )
        },
        created_at=1.0,
    )
    store = JsonInstanceStore(tmp_path)
    store.save({inst.instance_key: inst})
    rt = store.load()[inst.instance_key]
    t = rt.tasks["dummy_overlay"]
    assert t.member_hash == "deadbeef"
    assert t.n_members_total == 6
    assert t.n_members_complete == 4
    assert t.output_path == tmp_path / "o"
