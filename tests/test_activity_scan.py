"""ActivityScan detection + assembly on a synthetic h5 built to the real layout.

The fixture writes ``/data_store/dataNNNN`` nodes exactly as a MaxTwo
ActivityScan file does — ``well_id``, ``settings/{lsb,sampling,mapping}``,
``groups/routed/{channels,raw}`` — with spikes *planted* at known channels and
times, so detection can be checked against ground truth rather than against
another run of the same code.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from yuxin_mea.analysis.activity_scan import (
    GRID_COLS,
    ActivityScanConfig,
    compute_well_activity,
    grid_from_electrodes,
    load_activity_scan,
    well_store_nodes,
    write_activity_scan,
)

FS = 10_000.0
LSB_UV = 1.0                 # 1 count == 1 µV, so planted depths are in µV directly
OFFSET = 30_000             # DC offset (removed by the 300 Hz highpass)
N_SAMPLES = 30_000          # 3 s per config — enough for a stable MAD, fast to build


def _spike_wave(depth_uv: float) -> np.ndarray:
    """A short biphasic deflection: small positive shoulder, sharp negative trough."""
    return np.array([0.2, 0.5, 0.3, -0.4, -1.0, -0.6, 0.1, 0.3, 0.2]) * depth_uv


def _write_store(store: h5py.Group, name: str, *, well_id: int, rec_id: int,
                 electrodes: list[int], planted: dict[int, tuple[int, float]],
                 noise_uv: float, seed: int) -> None:
    """One (well, configuration) node. ``planted`` maps channel -> (n_spikes, depth)."""
    rng = np.random.default_rng(seed)
    n_ch = len(electrodes)
    node = store.create_group(name)
    node.create_dataset("well_id", data=np.array([well_id], dtype=np.int32))
    node.create_dataset("recording_id", data=np.array([rec_id], dtype=np.int32))

    settings = node.create_group("settings")
    settings.create_dataset("lsb", data=np.array([LSB_UV * 1e-6], dtype=np.float64))
    settings.create_dataset("sampling", data=np.array([FS], dtype=np.float64))
    settings.create_dataset("gain", data=np.array([512.0], dtype=np.float64))
    settings.create_dataset("hpf", data=np.array([1.0], dtype=np.float64))
    settings.create_dataset("spike_threshold", data=np.array([0.0], dtype=np.float64))
    mapping = np.zeros(n_ch, dtype=[("channel", "<i4"), ("electrode", "<i4"),
                                    ("x", "<f8"), ("y", "<f8")])
    for i, el in enumerate(electrodes):
        row, col = divmod(el, GRID_COLS)
        mapping[i] = (i, el, col * 17.5, row * 17.5)
    settings.create_dataset("mapping", data=mapping)

    raw = np.full((n_ch, N_SAMPLES), OFFSET, dtype=np.float64)
    raw += rng.normal(0.0, noise_uv, size=raw.shape)
    online_ch, online_frame, online_amp = [], [], []
    for ch, (n_spk, depth) in planted.items():
        wave = _spike_wave(depth)
        # evenly spaced, clear of the edges and each other
        gap = (N_SAMPLES - 400) // max(1, n_spk)
        for k in range(n_spk):
            s = 200 + k * gap
            raw[ch, s:s + wave.size] += wave
            online_ch.append(ch)
            online_frame.append(s + 4)
            online_amp.append(depth * 0.13)   # vendor's non-µV amplitude scale
    raw = np.clip(raw, 0, 65535).astype(np.uint16)

    routed = node.create_group("groups").create_group("routed")
    routed.create_dataset("channels", data=np.arange(n_ch, dtype=np.uint16))
    routed.create_dataset("raw", data=raw)
    routed.create_dataset("frame_nos",
                          data=np.arange(N_SAMPLES, dtype=np.uint64) + rec_id * 10**6)
    routed.create_dataset("triggered", data=np.array([0], dtype=np.int32))

    spikes = np.zeros(len(online_ch), dtype=[("frameno", "<i8"), ("channel", "<i4"),
                                             ("amplitude", "<f4")])
    if online_ch:
        spikes["frameno"] = np.array(online_frame) + rec_id * 10**6
        spikes["channel"] = online_ch
        spikes["amplitude"] = online_amp
    node.create_dataset("spikes", data=spikes)


def _build_h5(path: Path, well_configs: dict[int, list[dict]]) -> None:
    """well_configs: {well_id: [config_spec, …]}; each spec has electrodes/planted/noise."""
    with h5py.File(path, "w") as f:
        store = f.create_group("data_store")
        idx = 0
        for well_id, configs in well_configs.items():
            for c, spec in enumerate(configs):
                _write_store(store, f"data{idx:04d}", well_id=well_id, rec_id=idx,
                             electrodes=spec["electrodes"], planted=spec["planted"],
                             noise_uv=spec.get("noise_uv", 2.0),
                             seed=1000 + idx)
                idx += 1


@pytest.fixture
def simple_h5(tmp_path: Path) -> Path:
    """Well 0 across 3 configs; a handful of channels carry planted spikes."""
    p = tmp_path / "data.raw.h5"
    base = list(range(100, 140))                 # 40 electrodes
    _build_h5(p, {
        0: [
            {"electrodes": base,
             "planted": {5: (10, -60.0), 6: (4, -45.0), 7: (0, 0.0)},
             "noise_uv": 2.0},
            {"electrodes": base,      # same electrodes -> every one is dual-config
             "planted": {5: (10, -60.0)}, "noise_uv": 2.0},
            {"electrodes": list(range(140, 180)),   # a disjoint config
             "planted": {2: (6, -50.0)}, "noise_uv": 3.0},
        ],
    })
    return p


def test_store_nodes_found_by_well(simple_h5: Path):
    with h5py.File(simple_h5, "r") as f:
        assert well_store_nodes(f, 0) == ["data0000", "data0001", "data0002"]
        assert well_store_nodes(f, 7) == []


def test_detection_recovers_planted_counts(simple_h5: Path):
    res = compute_well_activity(simple_h5, 0, ActivityScanConfig())
    e = res.electrodes
    # config 0 channel 5 -> electrode 105, planted 10 spikes in each of two configs
    row = e.loc[105]
    assert row["n_configs"] == 2
    assert row["n_spikes"] == 20               # 10 + 10 across the two configs
    # channel 6 -> electrode 106, 4 spikes in config 0 only
    assert e.loc[106, "n_spikes"] == 4
    # a silent channel stays at zero, not NaN
    assert e.loc[107, "n_spikes"] == 0
    assert e.loc[107, "firing_rate_hz"] == pytest.approx(0.0)


def test_amplitude_is_negative_and_near_planted(simple_h5: Path):
    res = compute_well_activity(simple_h5, 0, ActivityScanConfig())
    amp = res.electrodes.loc[105, "amp_median_uv"]
    # zero-phase re-measure: sign is negative and the trough is within a
    # highpass-attenuation band of the −60 µV planted depth
    assert amp < 0
    assert -75 < amp < -30


def test_noise_estimate_tracks_injected_sigma(simple_h5: Path):
    res = compute_well_activity(simple_h5, 0, ActivityScanConfig())
    # config 0/1 electrodes were built at 2 µV; the median noise electrode should
    # sit near that (300 Hz highpass keeps most of white-noise variance)
    quiet = res.electrodes.loc[110:135, "noise_uv"]
    assert 1.0 < quiet.median() < 3.5


def test_electrode_geometry_round_trips(simple_h5: Path):
    res = compute_well_activity(simple_h5, 0, ActivityScanConfig())
    e = res.electrodes
    el = e.index.to_numpy()
    assert np.array_equal(el, e["row"].to_numpy() * GRID_COLS + e["col"].to_numpy())
    assert np.allclose(e["x_um"], e["col"] * 17.5)
    assert np.allclose(e["y_um"], e["row"] * 17.5)


def test_union_across_configs_dedups(simple_h5: Path):
    res = compute_well_activity(simple_h5, 0, ActivityScanConfig())
    e = res.electrodes
    # 40 shared (dual-config) + 40 disjoint = 80 unique electrodes
    assert len(e) == 80
    assert (e["n_configs"] == 2).sum() == 40
    assert (e["n_configs"] == 1).sum() == 40


def test_maps_are_nan_off_coverage(simple_h5: Path):
    res = compute_well_activity(simple_h5, 0, ActivityScanConfig())
    fr = res.maps["firing_rate"]
    cov = res.maps["coverage"]
    covered = cov > 0
    assert covered.sum() == len(res.electrodes)
    assert np.all(np.isfinite(fr[covered]))
    assert np.all(np.isnan(fr[~covered]))


def test_online_counts_carried(simple_h5: Path):
    res = compute_well_activity(simple_h5, 0, ActivityScanConfig())
    # electrode 105 had online spikes planted in both configs (10 each)
    assert res.electrodes.loc[105, "online_n_spikes"] == 20


def test_stats_report_both_areas(simple_h5: Path):
    res = compute_well_activity(simple_h5, 0, ActivityScanConfig())
    s = res.stats
    assert s["n_scanned"] == 80
    assert 0 < s["coverage_frac"] < 1
    assert s["n_active"] >= 3
    # observed fraction and array-extrapolated area are both present and distinct
    assert s["active_frac"] == pytest.approx(s["n_active"] / s["n_scanned"])
    assert s["active_area_mm2"] > 0
    assert np.isfinite(s["fr_gini"])
    cx, cy = s["activity_centroid_um"]
    assert np.isfinite(cx) and np.isfinite(cy)


def test_many_configs_supported(tmp_path: Path):
    p = tmp_path / "data.raw.h5"
    configs = [{"electrodes": list(range(200 + 5 * c, 205 + 5 * c)),
                "planted": {0: (3, -40.0)}, "noise_uv": 2.0} for c in range(29)]
    _build_h5(p, {3: configs})
    res = compute_well_activity(p, 3, ActivityScanConfig())
    assert res.diagnostics["n_configs"] == 29
    assert len(res.electrodes) == 29 * 5


def test_missing_well_raises(simple_h5: Path):
    from yuxin_mea.analysis.activity_scan import ActivityScanError

    with pytest.raises(ActivityScanError):
        compute_well_activity(simple_h5, 9, ActivityScanConfig())


def test_write_then_load_round_trips(simple_h5: Path, tmp_path: Path):
    res = compute_well_activity(simple_h5, 0, ActivityScanConfig())
    out = write_activity_scan(res, tmp_path / "out")
    assert (out / "electrodes.parquet").exists()
    assert (out / "maps.npz").exists()
    assert (out / "stats.json").exists()

    back = load_activity_scan(out)
    assert list(back.electrodes.columns) == list(res.electrodes.columns)
    assert back.stats["n_active"] == res.stats["n_active"]
    for k in res.maps:
        a, b = res.maps[k], back.maps[k]
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], atol=1e-4)


def test_grid_from_electrodes_places_values(simple_h5: Path):
    res = compute_well_activity(simple_h5, 0, ActivityScanConfig())
    g = grid_from_electrodes(res.electrodes, "n_spikes", fill=np.nan)
    row, col = divmod(105, GRID_COLS)
    assert g[row, col] == res.electrodes.loc[105, "n_spikes"]


def test_all_zero_channel_is_zero_rate_not_nan(tmp_path: Path):
    p = tmp_path / "data.raw.h5"
    _build_h5(p, {1: [{"electrodes": [300, 301, 302],
                       "planted": {}, "noise_uv": 2.0}]})
    res = compute_well_activity(p, 1, ActivityScanConfig())
    assert (res.electrodes["n_spikes"] == 0).all()
    assert (res.electrodes["firing_rate_hz"] == 0).all()
