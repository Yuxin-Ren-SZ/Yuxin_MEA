"""Unit-inspector library + curation store + page registration.

The library reads a SortingAnalyzer binary-folder with plain ``np.load`` (no
SpikeInterface), so the analyzer is faked here from hand-written arrays — fast
and dependency-free. Figures are asserted to build with the expected trace
counts; the curation store is round-tripped and exported to a valid SI dict.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from yuxin_mea.analysis import manual_curation_store as mc
from yuxin_mea.analysis import unit_inspector as ui


@pytest.fixture(scope="module", autouse=True)
def _dash_app():
    """`dash.register_page` (run when the page module imports) needs an app to
    exist first — build one so page-importing tests work in any run order."""
    from yuxin_mea.dashboard import build_app
    with TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "pipeline_config.json"
        cfg.write_text(json.dumps({"global": {"analysis_root": tmp,
                                              "data_root": tmp}, "tasks": {}}))
        build_app(str(cfg))
    yield


# --------------------------------------------------------------------------- #
# Synthetic analyzer folder
# --------------------------------------------------------------------------- #
def _make_analyzer(base: Path, *, n_units: int = 3, n_samp: int = 30,
                   n_chan: int = 8, n_sparse: int = 4) -> Path:
    """Write a minimal but faithful analyzer binary-folder under ``base``."""
    adir = base / "analyzer"
    ext = adir / "extensions"
    (adir / "sorting" / "properties").mkdir(parents=True)
    for sub in ("templates", "waveforms", "random_spikes", "spike_amplitudes",
                "quality_metrics", "template_metrics", "unit_locations"):
        (ext / sub).mkdir(parents=True)

    rng = np.random.default_rng(0)
    # Templates: give each unit a distinct peak channel = unit index.
    templates = rng.normal(0, 1, (n_units, n_samp, n_chan)).astype(np.float64)
    for u in range(n_units):
        pk = u % n_chan
        templates[u, :, pk] *= 8.0  # widen p2p on the peak channel
    np.save(ext / "templates" / "average.npy", templates)
    np.save(ext / "templates" / "std.npy", np.abs(templates) * 0.1)
    (ext / "templates" / "params.json").write_text(
        json.dumps({"operators": ["average", "std"], "ms_before": 1.0,
                    "ms_after": 2.0}))

    # sparsity: each unit's peak channel + the next 3, wrapped.
    mask = np.zeros((n_units, n_chan), dtype=bool)
    for u in range(n_units):
        chans = [(u + k) % n_chan for k in range(n_sparse)]
        mask[u, sorted(chans)] = True
    np.save(adir / "sparsity_mask.npy", mask)

    # spikes: per-unit counts, globally sorted by sample_index.
    counts = [5, 3, 4][:n_units] + [4] * max(0, n_units - 3)
    recs = []
    for u, c in enumerate(counts):
        samples = np.sort(rng.integers(0, 300_000, c))
        for s in samples:
            recs.append((int(s), u, 0))
    recs.sort(key=lambda t: t[0])
    spikes = np.array(recs, dtype=[("sample_index", "<i8"),
                                   ("unit_index", "<i8"),
                                   ("segment_index", "<i8")])
    np.save(adir / "sorting" / "spikes.npy", spikes)
    (adir / "sorting" / "numpysorting_info.json").write_text(
        json.dumps({"sampling_frequency": 10000.0}))
    ks = np.array((["good", "mua", "good"] * n_units)[:n_units], dtype=object)
    np.save(adir / "sorting" / "properties" / "KSLabel.npy", ks)

    # amplitudes: one per spike (global order).
    np.save(ext / "spike_amplitudes" / "amplitudes.npy",
            rng.normal(-40, 10, len(spikes)).astype(np.float32))

    # waveforms: use ALL spikes as the random subset; left-align sparse cols.
    rand_idx = np.arange(len(spikes))
    waveforms = np.zeros((len(spikes), n_samp, n_sparse), dtype=np.float32)
    for i, sp in enumerate(spikes):
        u = int(sp["unit_index"])
        chans = np.nonzero(mask[u])[0]
        for c_col, ch in enumerate(chans):
            waveforms[i, :, c_col] = templates[u, :, ch] + rng.normal(0, 0.2, n_samp)
    np.save(ext / "waveforms" / "waveforms.npy", waveforms)
    np.save(ext / "random_spikes" / "random_spikes_indices.npy", rand_idx)

    # metrics + locations.
    idx = list(range(n_units))
    pd.DataFrame({
        "presence_ratio": [0.9] * n_units,
        "rp_contamination": [0.05] * n_units,
        "firing_rate": [1.5, 0.4, 0.9][:n_units],
        "amplitude_median": [-40.0, -12.0, -55.0][:n_units],
    }, index=idx).to_csv(ext / "quality_metrics" / "metrics.csv")
    pd.DataFrame({
        "peak_to_trough_duration": [0.0006, 0.0004, 0.0009][:n_units],
        "trough_half_width": [0.0003] * n_units,
    }, index=idx).to_csv(ext / "template_metrics" / "metrics.csv")
    np.save(ext / "unit_locations" / "unit_locations.npy",
            rng.normal(0, 50, (n_units, 3)))
    return adir


@pytest.fixture()
def analyzer_dir():
    with TemporaryDirectory() as tmp:
        yield _make_analyzer(Path(tmp))


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def test_load_bundle_shapes_and_metrics(analyzer_dir):
    b = ui.load_unit_bundle(analyzer_dir)
    assert b.unit_ids == [0, 1, 2]
    assert b.fs == 10000.0
    assert b.templates.shape == (3, 30, 8)
    for col in ("firing_rate", "amplitude_median", "ks_label", "quality",
                "n_spikes"):
        assert col in b.metrics.columns
    assert list(b.metrics["n_spikes"]) == [5, 3, 4]
    assert b.metrics.loc[0, "quality"] == "good"
    assert b.metrics.loc[1, "quality"] == "MUA"


def test_peak_channel_matches_construction(analyzer_dir):
    b = ui.load_unit_bundle(analyzer_dir)
    for u in b.unit_ids:
        assert b.peak_channel(b.unit_pos(u)) == u % 8


def test_peak_waveforms_align_to_template(analyzer_dir):
    b = ui.load_unit_bundle(analyzer_dir)
    got = b.unit_peak_waveforms(0)
    assert got is not None
    waves, template_peak = got
    assert waves.shape[1] == 30
    # Mean snippet on the peak channel tracks the template peak channel.
    corr = np.corrcoef(waves.mean(0), template_peak)[0, 1]
    assert corr > 0.95


def test_cached_loader_returns_same_object(analyzer_dir):
    ui.clear_cache()
    a = ui.load_unit_bundle_cached(analyzer_dir)
    b = ui.load_unit_bundle_cached(analyzer_dir)
    assert a is b


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_all_figures_build(analyzer_dir):
    b = ui.load_unit_bundle(analyzer_dir)
    for fn in (ui.fig_unit_template, ui.fig_unit_overlay, ui.fig_isi_hist,
               ui.fig_autocorrelogram, ui.fig_amplitude_time,
               ui.fig_probe_location):
        fig = fn(b, 0)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


def test_metrics_kv_and_table(analyzer_dir):
    b = ui.load_unit_bundle(analyzer_dir)
    kv = ui.unit_metrics_kv(b, 0)
    assert "firing rate (Hz)" in kv and "Kilosort label" in kv
    tbl = ui.unit_table(b)
    assert list(tbl["unit_id"]) == [0, 1, 2]


def test_discovery_fallback_walk(analyzer_dir):
    # Place the analyzer where the walk fallback expects it.
    with TemporaryDirectory() as tmp:
        ar = Path(tmp)
        dest = ar / "analyzer_data" / "S/260/T/Network/000" / "rec0" / "well0"
        dest.mkdir(parents=True)
        # copy the fixture tree
        import shutil
        shutil.copytree(analyzer_dir, dest / "analyzer")
        dirs = ui.analyzer_dirs_for_recording(ar, "S/260/T/Network/000")
        assert set(dirs) == {"well0"}
        assert (dirs["well0"] / "extensions" / "templates" / "average.npy").exists()


# --------------------------------------------------------------------------- #
# Curation store
# --------------------------------------------------------------------------- #
def test_manifest_roundtrip_and_staleness():
    with TemporaryDirectory() as tmp:
        p = mc.manifest_path(Path(tmp), "S/260/T/Network/000", "rec0", "well0")
        man = mc.default_manifest("S/260/T/Network/000", "rec0", "well0",
                                  [0, 1, 2, 3], "fp-A")
        man = mc.set_label(man, 0, "good")
        man = mc.set_note(man, 0, "clean")
        mc.save_manifest(p, man)
        loaded = mc.load_manifest(p)
        assert loaded is not None
        assert loaded["labels"] == {"0": "good"}
        assert loaded["updated_at"]
        assert not mc.is_stale(loaded, "fp-A")
        assert mc.is_stale(loaded, "fp-B")


def test_merge_group_coalesces_overlaps():
    man = mc.default_manifest("", "", "", [0, 1, 2, 3, 4], "fp")
    man = mc.add_merge_group(man, [0, 1])
    man = mc.add_merge_group(man, [1, 2])  # overlaps -> coalesce
    assert man["merges"] == [[0, 1, 2]]
    man = mc.add_merge_group(man, [3, 4])
    assert sorted(man["merges"]) == [[0, 1, 2], [3, 4]]


def test_toggle_removed_and_split():
    man = mc.default_manifest("", "", "", [0, 1, 2], "fp")
    man = mc.toggle_removed(man, 1)
    assert man["removed"] == [1]
    man = mc.toggle_removed(man, 1)
    assert man["removed"] == []
    man = mc.set_split(man, 2, "amplitude", -30.0)
    assert man["splits"][0]["unit_id"] == 2
    man = mc.clear_split(man, 2)
    assert man["splits"] == []


def test_set_label_rejects_bad_value():
    man = mc.default_manifest("", "", "", [0], "fp")
    with pytest.raises(ValueError):
        mc.set_label(man, 0, "great")


def test_to_si_curation_dict_noise_becomes_removed():
    man = mc.default_manifest("", "", "", [0, 1, 2, 3], "fp")
    man = mc.set_label(man, 0, "good")
    man = mc.set_label(man, 1, "noise")
    man = mc.add_merge_group(man, [2, 3])
    cur = mc.to_si_curation_dict(man, [0, 1, 2, 3])
    assert cur["format_version"] == "2"
    assert 1 in cur["removed"]                      # noise -> removed
    assert {"unit_ids": [2, 3]} in cur["merges"]
    assert {"unit_id": 0, "labels": {"quality": ["good"]}} in cur["manual_labels"]


def test_to_si_curation_dict_validates_with_spikeinterface():
    pytest.importorskip("spikeinterface")
    from spikeinterface.curation import validate_curation_dict
    man = mc.default_manifest("", "", "", [0, 1, 2], "fp")
    man = mc.set_label(man, 0, "MUA")
    man = mc.add_merge_group(man, [1, 2])
    cur = mc.to_si_curation_dict(man, [0, 1, 2])
    validate_curation_dict(cur)  # raises on malformed


# --------------------------------------------------------------------------- #
# Unit filter
# --------------------------------------------------------------------------- #
def _unit_tbl():
    return pd.DataFrame({
        "unit_id": [0, 1, 2, 3],
        "firing_rate": [0.1, 2.0, 0.5, 5.0],
        "n_spikes": [10, 400, 60, 900],
        "auto_curated": pd.array([True, False, True, False], dtype="boolean"),
    })


def test_filter_units_by_label():
    from yuxin_mea.dashboard.pages.unit_inspector import _filter_units
    labels = {"0": "good", "1": "noise"}  # 2,3 unlabeled
    got = _filter_units(_unit_tbl(), labels, ["good"], "all", None, None)
    assert list(got["unit_id"]) == [0]
    got = _filter_units(_unit_tbl(), labels, ["unlabeled"], "all", None, None)
    assert list(got["unit_id"]) == [2, 3]


def test_filter_units_by_curated_and_thresholds():
    from yuxin_mea.dashboard.pages.unit_inspector import _filter_units
    tbl = _unit_tbl()
    assert list(_filter_units(tbl, {}, [], "pass", None, None)["unit_id"]) == [0, 2]
    assert list(_filter_units(tbl, {}, [], "reject", None, None)["unit_id"]) == [1, 3]
    assert list(_filter_units(tbl, {}, [], "all", 1.0, None)["unit_id"]) == [1, 3]
    assert list(_filter_units(tbl, {}, [], "all", None, 100)["unit_id"]) == [1, 3]
    # combined
    assert list(_filter_units(tbl, {}, [], "reject", 3.0, None)["unit_id"]) == [3]


# --------------------------------------------------------------------------- #
# Page registration
# --------------------------------------------------------------------------- #
def test_page_registered_and_rail_section():
    import dash
    from yuxin_mea.dashboard import build_app
    from yuxin_mea.dashboard.components.layout import _NAV_META, _SECTION_ORDER

    with TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "pipeline_config.json"
        cfg.write_text(json.dumps({"global": {"analysis_root": tmp,
                                              "data_root": tmp}, "tasks": {}}))
        build_app(str(cfg))
    paths = {p["name"]: p["path"] for p in dash.page_registry.values()}
    assert paths.get("Unit inspector") == "/unit-inspector"
    assert _NAV_META["Unit inspector"][0] == "single-unit"
    assert "single-unit" in _SECTION_ORDER and "network" in _SECTION_ORDER
