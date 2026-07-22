"""ManualCurationTask — registration, pure helpers, skip paths, SI apply.

The full apply path builds a real (tiny) SortingAnalyzer via SpikeInterface so
``apply_curation`` runs end-to-end; the skip paths (no manifest / stale) and the
pure path/split helpers need no SI.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from yuxin_mea.analysis import manual_curation_store as mc
from yuxin_mea.tasks import TASK_CLASSES
from yuxin_mea.tasks.manual_curation import ManualCurationTask


def test_registered_after_analyzer():
    names = [c.task_name for c in TASK_CLASSES]
    assert "manual_curation" in names
    assert names.index("analyzer") < names.index("manual_curation")
    assert ManualCurationTask.dependencies == ["analyzer"]


def test_build_output_path_and_root_resolution():
    out = ManualCurationTask.build_output_path(
        "/a/manual_curation_applied", "S/260/T/Network/000", "rec0", "well0")
    assert out.parts[-1] == "manual_curation"
    assert "well0" in out.parts
    ar = Path("/x/analysis")
    assert ManualCurationTask._resolve_root("./manual_curation_data", ar) == \
        ar / "manual_curation_data"
    assert ManualCurationTask._resolve_root("/abs/here", ar) == Path("/abs/here")


def test_skip_when_no_manifest():
    with TemporaryDirectory() as tmp:
        ar = Path(tmp)
        (ar / "analyzer_data").mkdir()
        out = ManualCurationTask().run(
            "S/260/T/Network/000", "rec0/well0", Path("."),
            {"analyzer_output_root": str(ar / "analyzer_data"),
             "manual_curation_root": str(ar / "manual_curation_data"),
             "applied_output_root": str(ar / "manual_curation_applied")})
        res = json.loads((out / "result.json").read_text())
        assert res["skipped"] and res["skipped_reason"] == "no_manifest"
        assert not (out / "curated_spike_times.npy").exists()


def test_resolve_splits_partitions_by_amplitude():
    with TemporaryDirectory() as tmp:
        adir = Path(tmp) / "analyzer"
        (adir / "extensions" / "spike_amplitudes").mkdir(parents=True)
        (adir / "sorting").mkdir(parents=True)
        # unit 0: 4 spikes, unit 1: 2 spikes (global order interleaved).
        spikes = np.array([(10, 0, 0), (20, 1, 0), (30, 0, 0), (40, 0, 0),
                           (50, 1, 0), (60, 0, 0)],
                          dtype=[("sample_index", "<i8"), ("unit_index", "<i8"),
                                 ("segment_index", "<i8")])
        np.save(adir / "sorting" / "spikes.npy", spikes)
        # Global amps; unit-0 spikes are global rows 0,2,3,5 -> [-50,-10,-60,-5].
        amps = np.array([-50, -99, -10, -60, -99, -5], dtype=np.float32)
        np.save(adir / "extensions" / "spike_amplitudes" / "amplitudes.npy", amps)
        man = mc.default_manifest("", "", "", [0, 1], "fp")
        man = mc.set_split(man, 0, "amplitude", -30.0)
        got = ManualCurationTask._resolve_splits(man, adir, [0, 1], "amplitude")
        # within unit-0 order: <=-30 -> idx 0,2 ; >-30 -> idx 1,3
        assert got == {0: [[0, 2], [1, 3]]}


def _build_si_analyzer(folder: Path):
    si = pytest.importorskip("spikeinterface.full")
    from spikeinterface.core import NumpyRecording, NumpySorting
    fs = 10000.0
    rng = np.random.default_rng(0)
    rec = NumpyRecording([rng.normal(0, 1, (30_000, 4)).astype("float32")], fs)
    rec.set_dummy_probe_from_locations(
        np.column_stack([np.zeros(4), np.arange(4) * 20.0]))
    sorting = NumpySorting.from_unit_dict(
        {0: np.arange(100, 5000, 50), 1: np.arange(200, 3000, 70),
         2: np.arange(300, 8000, 40), 3: np.arange(150, 2000, 90)}, fs)
    # sparse=False avoids estimate_sparsity (no probe needed); the task only
    # needs analyzer.sorting for label/merge/remove, so no extensions required.
    analyzer = si.create_sorting_analyzer(
        sorting, rec, format="binary_folder", folder=str(folder), sparse=False)
    return analyzer


def test_full_apply_merge_remove_label():
    pytest.importorskip("spikeinterface.full")
    with TemporaryDirectory() as tmp:
        ar = Path(tmp)
        rk, rec, well = "S/260/T/Network/000", "rec0", "well0"
        analyzer_path = (ar / "analyzer_data" / rk / rec / well / "analyzer")
        analyzer_path.parent.mkdir(parents=True)
        _build_si_analyzer(analyzer_path)

        man = mc.default_manifest(rk, rec, well, [0, 1, 2, 3],
                                  mc.sorting_fingerprint(analyzer_path, [0, 1, 2, 3]))
        man = mc.set_label(man, 1, "noise")     # -> removed
        man = mc.add_merge_group(man, [0, 2])   # merge
        mc.save_manifest(mc.manifest_path(ar / "manual_curation_data", rk, rec, well),
                         man)

        out = ManualCurationTask().run(
            rk, f"{rec}/{well}", Path("."),
            {"analyzer_output_root": str(ar / "analyzer_data"),
             "manual_curation_root": str(ar / "manual_curation_data"),
             "applied_output_root": str(ar / "manual_curation_applied")})
        res = json.loads((out / "result.json").read_text())
        assert not res["skipped"]
        assert res["n_units_in"] == 4
        # 4 units - remove(1) - merge(2->1 net -1) = 2 out
        assert res["n_units_out"] == 2
        st = np.load(out / "curated_spike_times.npy", allow_pickle=True).item()
        assert len(st) == 2
        # applied curation dict is persisted + valid
        cur = json.loads((out / "applied_curation.json").read_text())
        from spikeinterface.curation import validate_curation_dict
        validate_curation_dict(cur)


def test_full_apply_stale_manifest_skips():
    pytest.importorskip("spikeinterface.full")
    with TemporaryDirectory() as tmp:
        ar = Path(tmp)
        rk, rec, well = "S/260/T/Network/000", "rec0", "well0"
        analyzer_path = (ar / "analyzer_data" / rk / rec / well / "analyzer")
        analyzer_path.parent.mkdir(parents=True)
        _build_si_analyzer(analyzer_path)
        man = mc.default_manifest(rk, rec, well, [0, 1, 2, 3], "stale-fingerprint")
        man = mc.set_label(man, 0, "good")
        mc.save_manifest(mc.manifest_path(ar / "manual_curation_data", rk, rec, well),
                         man)
        out = ManualCurationTask().run(
            rk, f"{rec}/{well}", Path("."),
            {"analyzer_output_root": str(ar / "analyzer_data"),
             "manual_curation_root": str(ar / "manual_curation_data"),
             "applied_output_root": str(ar / "manual_curation_applied")})
        res = json.loads((out / "result.json").read_text())
        assert res["skipped"] and res["skipped_reason"] == "stale"
