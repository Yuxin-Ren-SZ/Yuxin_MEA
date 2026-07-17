"""Spatial activity map + burst propagation: field shape, planar-fit recovery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yuxin_mea.analysis.spatial_map import (
    SpatialMapConfig,
    SpatialMapError,
    activity_field,
    compute_spatial_map,
)


def _qm(units, xs, ys, rates):
    return pd.DataFrame(
        {"loc_x": xs, "loc_y": ys, "firing_rate": rates}, index=list(units)
    )


class TestActivityField:
    def test_normalised_and_shaped(self):
        pos = np.array([[10.0, 10.0], [90.0, 90.0]])
        fr = np.array([1.0, 1.0])
        field, extent = activity_field(pos, fr, bins=32, sigma_um=20.0)
        assert field.shape == (32, 32)
        assert field.sum() == pytest.approx(1.0, abs=1e-6)
        assert extent == (10.0, 90.0, 10.0, 90.0)

    def test_single_unit_peak_near_its_location(self):
        pos = np.array([[0.0, 0.0], [100.0, 100.0]])
        fr = np.array([0.0, 5.0])  # all weight on the second unit
        field, _ = activity_field(pos, fr, bins=16, sigma_um=5.0)
        r, c = np.unravel_index(int(np.argmax(field)), field.shape)
        # field[row=y, col=x]; unit at (100,100) → top-right (max row, max col)
        assert r > 8 and c > 8


class TestBurstPropagation:
    def test_recovers_planar_wave_speed(self):
        # 5 units along x at 0,100,200,300,400 µm; latency = 10 ms per 100 µm
        # → speed 100µm / 10ms = 10 µm/ms, planar r2 = 1.
        units = list(range(5))
        xs = [0.0, 100.0, 200.0, 300.0, 400.0]
        ys = [0.0, 0.0, 0.0, 0.0, 0.0]
        rates = [3.0] * 5
        qm = _qm(units, xs, ys, rates)
        start = 1.0
        spikes = {
            u: np.array([start + i * 0.01, start + i * 0.01 + 0.5])
            for i, u in enumerate(units)
        }
        bursts = pd.DataFrame({"start": [start], "end": [start + 0.6]})
        cfg = SpatialMapConfig(bins=16, sigma_um=30.0, min_participation=3)
        res = compute_spatial_map(spikes, qm, bursts, config=cfg)
        assert res.diagnostics["n_bursts_used"] == 1
        row = res.burst_propagation.iloc[0]
        assert row["speed_um_per_ms"] == pytest.approx(10.0, rel=1e-3)
        assert row["r2_planar_fit"] == pytest.approx(1.0, abs=1e-6)
        assert row["n_participating"] == 5
        assert res.metrics["mean_prop_speed_um_ms"] == pytest.approx(10.0, rel=1e-3)

    def test_missing_bursts_field_only(self):
        units = list(range(6))
        rng = np.random.default_rng(0)
        xs = rng.uniform(0, 400, 6)
        ys = rng.uniform(0, 400, 6)
        qm = _qm(units, xs, ys, [2.0] * 6)
        spikes = {u: np.sort(rng.uniform(0, 10, 20)) for u in units}
        res = compute_spatial_map(spikes, qm, bursts=None, config=SpatialMapConfig(bins=16))
        assert res.diagnostics["n_bursts_used"] == 0
        assert res.burst_propagation.empty
        assert res.activity_field.shape == (16, 16)
        assert np.isnan(res.metrics["mean_prop_speed_um_ms"])
        assert 0.0 <= res.metrics["activity_gini"] <= 1.0


class TestGuards:
    def test_no_spike_train_unit_overlap_raises(self):
        qm = _qm([100, 200], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0])
        spikes = {0: np.array([1.0, 2.0])}  # unit id not in qm.index
        with pytest.raises(SpatialMapError):
            compute_spatial_map(spikes, qm, None, SpatialMapConfig())

    def test_missing_loc_columns_raises(self):
        qm = pd.DataFrame({"firing_rate": [1.0, 2.0]}, index=[0, 1])
        spikes = {0: np.array([1.0]), 1: np.array([2.0])}
        with pytest.raises(SpatialMapError):
            compute_spatial_map(spikes, qm, None, SpatialMapConfig())
