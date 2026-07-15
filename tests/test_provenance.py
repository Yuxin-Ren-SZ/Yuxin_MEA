"""Provenance: fingerprints, serialization round-trips, and drift verification."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from yuxin_mea.provenance import (
    file_hash,
    h5_fingerprint,
    params_hash,
    raw_fingerprint,
    read_sidecar,
    write_sidecar,
)
from yuxin_mea.provenance.verify import classify_task, h5_match, meta_match


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_h5(path: Path, *, gain: float = 512.0, nframes: int = 4000,
             raw: np.ndarray | None = None) -> None:
    import h5py
    with h5py.File(path, "w") as h:
        h.create_dataset("version", data=np.bytes_([b"20000"]))
        w = h.create_group("recordings/rec0000/well000")
        w.create_dataset("settings/gain", data=np.array([gain]))
        w.create_dataset("settings/mapping", data=np.arange(20, dtype="i4"))
        arr = raw if raw is not None else np.arange(1008 * nframes, dtype="u2").reshape(1008, nframes)
        w.create_dataset("groups/routed/raw", data=arr, chunks=(1008, 200), compression="gzip")
        w.attrs["note"] = "orig"


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------

def test_h5_fingerprint_deterministic_and_versioned():
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.h5"
        _make_h5(p)
        fp1, fp2 = h5_fingerprint(p), h5_fingerprint(p)
        assert fp1["sha256"] == fp2["sha256"]
        assert fp1["method"] == "h5struct-v1"
        assert fp1["file_size"] > 0
        assert h5_fingerprint(p, full=True)["method"] == "h5full-v1"


def test_h5_fingerprint_detects_changes():
    with TemporaryDirectory() as tmp:
        t = Path(tmp)
        _make_h5(t / "base.h5")
        base = h5_fingerprint(t / "base.h5")["sha256"]

        _make_h5(t / "gain.h5", gain=2.0)  # small dataset (full-hashed) change
        assert h5_fingerprint(t / "gain.h5")["sha256"] != base

        raw = np.arange(1008 * 4000, dtype="u2").reshape(1008, 4000)
        raw[0, 0] += 1  # a sampled position in the large raw dataset
        _make_h5(t / "raw.h5", raw=raw)
        assert h5_fingerprint(t / "raw.h5")["sha256"] != base

        _make_h5(t / "shape.h5", nframes=5000)  # shape change
        assert h5_fingerprint(t / "shape.h5")["sha256"] != base


def test_h5_fingerprint_detects_attr_change():
    import h5py
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.h5"
        _make_h5(p)
        base = h5_fingerprint(p)["sha256"]
        with h5py.File(p, "a") as h:
            h["recordings/rec0000/well000"].attrs["note"] = "edited"
        assert h5_fingerprint(p)["sha256"] != base


def test_file_hash_and_params_hash():
    with TemporaryDirectory() as tmp:
        m = Path(tmp) / "mxassay.metadata"
        m.write_text("groupname=NPH\n")
        h1 = file_hash(m)
        m.write_text("groupname=IVH\n")
        assert file_hash(m)["sha256"] != h1["sha256"]

    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})
    assert params_hash({"a": 1}) != params_hash({"a": 2})


def test_raw_fingerprint_metadata_none_when_absent():
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.h5"
        _make_h5(p)
        assert raw_fingerprint(p)["metadata"] is None
        m = Path(tmp) / "mxassay.metadata"
        m.write_text("x=1\n")
        assert raw_fingerprint(p, m)["metadata"]["sha256"]


# ---------------------------------------------------------------------------
# sidecar
# ---------------------------------------------------------------------------

def test_sidecar_roundtrip():
    with TemporaryDirectory() as tmp:
        d = Path(tmp) / "out"
        write_sidecar(d, {"config_hash": "abc", "task": "preprocessing"})
        back = read_sidecar(d)
        assert back["config_hash"] == "abc" and back["task"] == "preprocessing"
        assert read_sidecar(Path(tmp) / "missing") is None


# ---------------------------------------------------------------------------
# serialization round-trips + backward compat
# ---------------------------------------------------------------------------

def test_recording_entry_raw_fingerprint_roundtrip_and_compat():
    from yuxin_mea.dataset.cache import JsonCacheStore
    from yuxin_mea.dataset.entries import RecordingEntry, WellEntry

    with TemporaryDirectory() as tmp:
        e = RecordingEntry(
            sample_id="S1", date="260101", plate_id="P", scan_type="Network",
            run_id="001", data_path=Path("S1/260101/P/Network/001/data.raw.h5"),
            file_size=100, mtime=1.0, discovered_at=2.0,
            wells={"well000": WellEntry("well000", {"groupname": "NPH"})},
        )
        e.raw_fingerprint["h5"] = {"method": "h5struct-v1", "file_size": 100,
                                   "mtime_ns": 123, "sha256": "abc"}
        e.raw_fingerprint["metadata"] = {"size": 10, "mtime_ns": 5, "sha256": "def"}
        st = JsonCacheStore(Path(tmp))
        st.save({e.cache_key: e})
        assert st.load()[e.cache_key].raw_fingerprint == e.raw_fingerprint

        # backward-compat: strip the field → decodes to {}
        f = Path(tmp) / "experiment_cache.json"
        raw = json.loads(f.read_text())
        for v in raw.values():
            v.pop("raw_fingerprint")
        f.write_text(json.dumps(raw))
        assert JsonCacheStore(Path(tmp)).load()[e.cache_key].raw_fingerprint == {}


def test_task_record_provenance_roundtrip_and_compat():
    from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
    from yuxin_mea.pipeline.pipeline_entry import PipelineEntry
    from yuxin_mea.pipeline.task_record import TaskRecord, TaskStatus

    with TemporaryDirectory() as tmp:
        tr = TaskRecord(status=TaskStatus.COMPLETE, dependencies=[], output_path=Path("/o"),
                        last_updated=1.0, error=None, config={"a": 1},
                        provenance={"h5": {"sha256": "abc"}, "config_hash": "c1"})
        pe = PipelineEntry(recording_key="S1/260101/P/Network/001",
                           well_id="rec0000/well000", created_at=0.0,
                           tasks={"preprocessing": tr})
        st = JsonPipelineCacheStore(Path(tmp))
        st.save({pe.pipeline_key: pe})
        assert st.load()[pe.pipeline_key].tasks["preprocessing"].provenance == tr.provenance

        f = Path(tmp) / "pipeline_cache.json"
        raw = json.loads(f.read_text())
        for v in raw.values():
            for t in v["tasks"].values():
                t.pop("provenance")
        f.write_text(json.dumps(raw))
        assert JsonPipelineCacheStore(Path(tmp)).load()[pe.pipeline_key].tasks[
            "preprocessing"].provenance is None


# ---------------------------------------------------------------------------
# verification matchers + classify
# ---------------------------------------------------------------------------

def test_match_helpers():
    a = {"method": "h5struct-v1", "sha256": "x", "file_size": 1, "mtime_ns": 2}
    assert h5_match(a, a) is True
    assert h5_match(a, {**a, "sha256": "y"}) is False
    assert h5_match(a, None) is None
    # stat fallback when one side has no content hash
    s = {"file_size": 1, "mtime_ns": 2}
    assert h5_match(s, {"file_size": 1, "mtime_ns": 2}) is True
    assert h5_match(s, {"file_size": 9, "mtime_ns": 2}) is False
    # Different fingerprint methods are incomparable → UNKNOWN, not RAW-CHANGED.
    assert h5_match({"method": "h5struct-v1", "sha256": "x"},
                    {"method": "h5full-v1", "sha256": "x"}) is None
    assert meta_match({"sha256": "m"}, {"sha256": "m"}) is True
    assert meta_match({"sha256": "m"}, {"sha256": "n"}) is False


def test_classify_task():
    prov = {"h5": {"method": "h5struct-v1", "sha256": "H"},
            "metadata": {"sha256": "M"}, "config_hash": "C"}
    h5, meta = {"method": "h5struct-v1", "sha256": "H"}, {"sha256": "M"}
    assert classify_task(prov, h5, meta, "C") == []                       # OK
    assert classify_task(None, h5, meta, "C") == ["UNKNOWN"]              # no stamp
    assert "CONFIG-CHANGED" in classify_task(prov, h5, meta, "C2")
    assert "RAW-CHANGED" in classify_task(prov, {"method": "h5struct-v1", "sha256": "H2"}, meta, "C")
    assert "METADATA-CHANGED" in classify_task(prov, h5, {"sha256": "M2"}, "C")


# ---------------------------------------------------------------------------
# end-to-end verify over synthetic caches
# ---------------------------------------------------------------------------

def _build_caches(tmp: Path, *, prov_h5_sha="H", prov_cfg="C",
                  cur_h5_sha="H", cur_meta_sha="M", prov_meta_sha="M",
                  with_prov=True):
    from yuxin_mea.dataset.cache import JsonCacheStore
    from yuxin_mea.dataset.entries import RecordingEntry
    from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
    from yuxin_mea.pipeline.pipeline_entry import PipelineEntry
    from yuxin_mea.pipeline.task_record import TaskRecord, TaskStatus

    rkey = "S1/260101/P/Network/001"
    e = RecordingEntry(sample_id="S1", date="260101", plate_id="P", scan_type="Network",
                       run_id="001", data_path=Path(f"{rkey}/data.raw.h5"),
                       file_size=1, mtime=1.0, discovered_at=1.0)
    e.raw_fingerprint["h5"] = {"method": "h5struct-v1", "sha256": cur_h5_sha,
                               "file_size": 1, "mtime_ns": 1}
    e.raw_fingerprint["metadata"] = {"sha256": cur_meta_sha, "size": 1, "mtime_ns": 1}
    JsonCacheStore(tmp).save({rkey: e})

    prov = None
    if with_prov:
        prov = {"h5": {"method": "h5struct-v1", "sha256": prov_h5_sha},
                "metadata": {"sha256": prov_meta_sha}, "config_hash": prov_cfg}
    tr = TaskRecord(status=TaskStatus.COMPLETE, dependencies=[], output_path=None,
                    last_updated=1.0, error=None, config={}, provenance=prov)
    pe = PipelineEntry(recording_key=rkey, well_id="rec0000/well000",
                       created_at=0.0, tasks={"preprocessing": tr})
    JsonPipelineCacheStore(tmp).save({pe.pipeline_key: pe})
    return rkey


class _FakeCM:
    """Minimal ConfigManager stand-in: get_task_params → fixed dict."""
    def __init__(self, params):
        self._p = params
    def get_task_params(self, name):
        return dict(self._p)


def test_verify_end_to_end():
    from yuxin_mea.provenance import params_hash
    from yuxin_mea.provenance.verify import verify_provenance

    params = {"sorter": "kilosort4"}
    cfg_hash = params_hash(params)
    cm = _FakeCM(params)

    with TemporaryDirectory() as tmp:
        tp = Path(tmp)
        # OK
        _build_caches(tp, prov_cfg=cfg_hash, prov_h5_sha="H", cur_h5_sha="H",
                      prov_meta_sha="M", cur_meta_sha="M")
        r = verify_provenance(cm, tp, tp)
        assert r.counts["OK"] == 1 and r.computational_drift == 0

        # RAW-CHANGED
        _build_caches(tp, prov_cfg=cfg_hash, prov_h5_sha="H", cur_h5_sha="H2")
        r = verify_provenance(cm, tp, tp)
        assert r.counts["RAW-CHANGED"] == 1 and r.computational_drift == 1

        # CONFIG-CHANGED
        _build_caches(tp, prov_cfg="STALE", cur_h5_sha="H", prov_h5_sha="H")
        r = verify_provenance(cm, tp, tp)
        assert r.counts["CONFIG-CHANGED"] == 1 and r.computational_drift == 1

        # METADATA-CHANGED (label only → not computational)
        _build_caches(tp, prov_cfg=cfg_hash, prov_meta_sha="M", cur_meta_sha="M2")
        r = verify_provenance(cm, tp, tp)
        assert r.counts["METADATA-CHANGED"] == 1 and r.computational_drift == 0

        # UNKNOWN (no stamp)
        _build_caches(tp, with_prov=False)
        r = verify_provenance(cm, tp, tp)
        assert r.counts["UNKNOWN"] == 1 and r.computational_drift == 0


def test_verify_drifted_structured_and_per_recording():
    from yuxin_mea.provenance import params_hash
    from yuxin_mea.provenance.verify import verify_provenance

    cm = _FakeCM({"sorter": "kilosort4"})
    cfg = params_hash({"sorter": "kilosort4"})
    with TemporaryDirectory() as tmp:
        tp = Path(tmp)
        rkey = _build_caches(tp, prov_cfg=cfg, prov_h5_sha="H", cur_h5_sha="H2")  # RAW
        r = verify_provenance(cm, tp, tp)
        assert len(r.drifted) == 1
        d = r.drifted[0]
        assert d["recording_key"] == rkey and d["well_id"] == "rec0000/well000"
        assert d["task"] == "preprocessing" and d["labels"] == ["RAW-CHANGED"]
        assert r.per_recording[rkey] == "RAW-CHANGED"
        # metadata-only drift is NOT in drifted (label-only, not computational)
        _build_caches(tp, prov_cfg=cfg, prov_meta_sha="M", cur_meta_sha="M2")
        r2 = verify_provenance(cm, tp, tp)
        assert r2.drifted == [] and r2.per_recording[rkey] == "METADATA-CHANGED"


def test_mark_stale_resets_drifted_task():
    from yuxin_mea.pipeline import PipelineManager
    from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
    from yuxin_mea.provenance import params_hash
    from yuxin_mea.provenance.verify import verify_provenance
    from yuxin_mea.tasks import TASK_CLASSES

    cm = _FakeCM({"sorter": "kilosort4"})
    cfg = params_hash({"sorter": "kilosort4"})
    with TemporaryDirectory() as tmp:
        tp = Path(tmp)
        rkey = _build_caches(tp, prov_cfg="STALE", cur_h5_sha="H", prov_h5_sha="H")  # CONFIG
        rep = verify_provenance(cm, tp, tp)
        assert rep.drifted and rep.drifted[0]["labels"] == ["CONFIG-CHANGED"]

        pm = PipelineManager(tp, config_provider=cm)
        for c in TASK_CLASSES:
            try:
                pm.register_task(c)
            except ValueError:
                pass
        total = 0
        for d in rep.drifted:
            total += pm.refresh(d["task"], recording_key=d["recording_key"],
                                well_ids=[d["well_id"]])
        assert total >= 1
        tr = JsonPipelineCacheStore(tp).load()[f"{rkey}/rec0000/well000"].tasks["preprocessing"]
        assert tr.status == "not_run"


# ---------------------------------------------------------------------------
# integration: the real _run_one completion path (stamp + sidecar placement)
# ---------------------------------------------------------------------------

class _StubCM:
    def get_task_params(self, name):
        return {"p": 1}
    def get_config(self, task, rec, well):
        return {"p": 1}


def test_run_one_stamps_and_places_sidecar_in_well_dir():
    """Drive the real completion path; sidecar must land in the well dir, not its
    parent (directory-returning tasks would otherwise collide across wells)."""
    from yuxin_mea.cli.run import _run_one
    from yuxin_mea.dataset import DatasetManager
    from yuxin_mea.pipeline import PipelineManager
    from yuxin_mea.pipeline.cache import JsonPipelineCacheStore

    with TemporaryDirectory() as tmp:
        t = Path(tmp)
        data_root, analysis = t / "data", t / "analysis"
        analysis.mkdir(parents=True)
        run_dir = data_root / "S1" / "260101" / "P" / "Network" / "001"
        run_dir.mkdir(parents=True)
        _make_h5(run_dir / "data.raw.h5")
        (run_dir / "mxassay.metadata").write_text("groupname=NPH\n")

        dm = DatasetManager(data_root, analysis, fingerprint_mode="content")
        rkey = dm.recordings[0].cache_key

        well_dir = analysis / "out" / rkey / "rec0000" / "well000"

        class _StubTask:
            task_name = "stub"
            dependencies: list[str] = []
            def run(self, rk, wid, data_path, params):
                well_dir.mkdir(parents=True, exist_ok=True)
                (well_dir / "result.npy").write_bytes(b"x")
                return well_dir  # a DIRECTORY (like burst_detection/auto_curation)

        pm = PipelineManager(analysis, config_provider=_StubCM())
        pm.register_computation_task("stub", [])
        pm.add_well(rkey, "rec0000/well000")
        wi = pm.get_next_task(n=1)[0]

        _run_one(wi, _StubCM(), dm, pm, {"stub": _StubTask()}, Path("cfg.json"))

        # cache copy
        entry = JsonPipelineCacheStore(analysis).load()[f"{rkey}/rec0000/well000"]
        tr = entry.tasks["stub"]
        assert tr.status == "complete"
        assert tr.provenance is not None
        assert tr.provenance["h5"]["sha256"]        # content fingerprint carried
        assert tr.provenance["metadata"]["sha256"]  # metadata fingerprint carried
        assert tr.provenance["config_file"] == "cfg.json"

        # durable sidecar: in the returned WELL dir, NOT the shared parent
        from yuxin_mea.provenance import PROVENANCE_FILENAME
        assert (well_dir / PROVENANCE_FILENAME).exists()
        assert not (well_dir.parent / PROVENANCE_FILENAME).exists()
        # filename must not collide with spikeinterface's provenance.json
        assert PROVENANCE_FILENAME != "provenance.json"
        back = read_sidecar(well_dir)
        assert back["task"] == "stub" and back["well_id"] == "rec0000/well000"


def test_read_sidecar_rejects_foreign_json():
    """A foreign json sharing the name (e.g. spikeinterface's) is not our stamp."""
    from yuxin_mea.provenance import PROVENANCE_FILENAME
    with TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / PROVENANCE_FILENAME).write_text('{"class": "spikeinterface", "kwargs": {}}')
        assert read_sidecar(d) is None  # no _schema marker → rejected
