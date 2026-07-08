"""InstanceStore round-trip + migration loaders from the two legacy caches."""

from pathlib import Path

from yuxin_mea.pipeline import (
    JsonInstanceStore,
    LayeredInstanceStore,
    ScopedInstance,
    load_legacy_aggregates,
    load_legacy_wells,
    project_wells_to_entries,
)
from yuxin_mea.pipeline.aggregate_cache import (
    AggregateInstanceRecord,
    JsonAggregateCacheStore,
)
from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
from yuxin_mea.pipeline.task_record import TaskRecord, TaskStatus

from tests._parity_fixtures import RK, RK2, stage


def _write_legacy_agg(tmp_path, task_name, recording_key, *, n_total, n_complete,
                      status=TaskStatus.COMPLETE, output="out.txt"):
    """Write an aggregate_cache.json record directly (S1's writer path, kept for
    migration + tests)."""
    store = JsonAggregateCacheStore(tmp_path)
    rec = AggregateInstanceRecord(
        task_name=task_name, scope_name="recording",
        scope_key={"recording_key": recording_key}, status=status,
        output_path=(tmp_path / output) if output else None,
        member_hash="fp", n_members_total=n_total, n_members_complete=n_complete,
    )
    store.save({rec.instance_key: rec})


def test_load_legacy_wells_count_and_order(tmp_path):
    stage(tmp_path)
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
    stage(tmp_path)
    entries = JsonPipelineCacheStore(tmp_path).load()
    wells = load_legacy_wells(entries)
    inst = next(iter(wells.values()))
    assert set(inst.scope_key) == {"recording_key", "rec_name", "well_id"}
    # the finest instance carries the per-well tasks verbatim
    assert "ml_burst_detection" in inst.tasks


def test_load_legacy_aggregates_from_cache(tmp_path):
    _write_legacy_agg(tmp_path, "dummy_overlay", RK2, n_total=3, n_complete=2)
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
    stage(tmp_path)
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


def test_project_wells_roundtrip_identity(tmp_path):
    """project_wells_to_entries ∘ load_legacy_wells == identity on the finest layer —
    the load-bearing seam for keeping the dashboard on pipeline_cache.json. Guards
    against a silently dropped field (config / member_hash / n_members_* / created_at /
    the rec_name/well_id compound split-join)."""
    orig = ScopedInstance(
        "well",
        {"recording_key": RK, "rec_name": "rec0000", "well_id": "well003"},
        tasks={
            "preprocessing": TaskRecord(
                TaskStatus.COMPLETE, [], tmp_path / "pp", 3.0, None, {"freq": 300},
            ),
            "ml_burst_detection": TaskRecord(
                TaskStatus.COMPLETE, ["auto_curation"], tmp_path / "ml", 9.0, None,
                {"seed": 42}, member_hash="cafef00d", n_members_total=1,
                n_members_complete=1,
            ),
        },
        created_at=123.5,
    )
    key = orig.instance_key
    # round-trip through the projection and back
    entries = project_wells_to_entries({key: orig})
    back = load_legacy_wells(entries)[key]

    assert back.scope_name == "well"
    assert back.scope_key == orig.scope_key
    assert back.created_at == 123.5
    assert set(back.tasks) == set(orig.tasks)
    for tn, tr in orig.tasks.items():
        b = back.tasks[tn]
        assert (b.status, b.dependencies, b.output_path, b.last_updated, b.error,
                b.config, b.member_hash, b.n_members_total, b.n_members_complete) == (
                tr.status, tr.dependencies, tr.output_path, tr.last_updated, tr.error,
                tr.config, tr.member_hash, tr.n_members_total, tr.n_members_complete)


def test_layered_store_splits_and_merges(tmp_path):
    """LayeredInstanceStore writes finest → pipeline_cache.json and aggregate →
    instance_store.json, and load() merges them back into one view."""
    finest = ScopedInstance(
        "well", {"recording_key": RK, "rec_name": "rec0000", "well_id": "well000"},
        tasks={"ml_burst_detection": TaskRecord(TaskStatus.COMPLETE, [], tmp_path / "o", 1.0, None)},
        created_at=1.0,
    )
    agg = ScopedInstance(
        "recording", {"recording_key": RK},
        tasks={"cluster_overlay_by_well": TaskRecord(
            TaskStatus.COMPLETE, [], tmp_path / "ov", 2.0, None,
            member_hash="h", n_members_total=6, n_members_complete=5)},
    )
    store = LayeredInstanceStore(tmp_path)
    store.save({finest.instance_key: finest, agg.instance_key: agg})

    # finest landed in pipeline_cache.json, aggregate in instance_store.json
    assert (tmp_path / "pipeline_cache.json").exists()
    assert (tmp_path / "instance_store.json").exists()
    from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
    pipe = JsonPipelineCacheStore(tmp_path).load()
    assert f"{RK}/rec0000/well000" in pipe
    inst_only = JsonInstanceStore(tmp_path).load()
    assert all(i.scope_name != "well" for i in inst_only.values())

    merged = LayeredInstanceStore(tmp_path).load()
    assert set(merged) == {finest.instance_key, agg.instance_key}
    assert merged[agg.instance_key].tasks["cluster_overlay_by_well"].n_members_complete == 5


def test_layered_store_migrates_legacy_aggregate_cache(tmp_path):
    """First load with no instance_store.json migrates S1's aggregate_cache.json."""
    stage(tmp_path)                                          # finest → pipeline_cache.json
    _write_legacy_agg(tmp_path, "dummy_overlay", RK2, n_total=3, n_complete=2)
    assert not (tmp_path / "instance_store.json").exists()
    merged = LayeredInstanceStore(tmp_path).load()
    # finest (from pipeline_cache.json) + the one migrated aggregate instance
    agg = [i for i in merged.values() if i.scope_name == "recording"]
    assert len(agg) == 1
    assert "dummy_overlay" in agg[0].tasks


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
