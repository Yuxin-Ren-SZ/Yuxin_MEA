"""Tests for directed transfer-entropy connectivity (analysis/directed_connectivity.py)."""
import numpy as np
import pytest

from yuxin_mea.analysis import directed_connectivity as D


def test_te_direction():
    # y is x delayed by 1 bin -> TE(x->y) at delay 1 >> TE(y->x)
    rng = np.random.default_rng(0)
    x = (rng.random(4000) < 0.2).astype(int)
    y = np.zeros_like(x)
    y[1:] = x[:-1]
    flip = rng.random(4000) < 0.05
    y = np.abs(y - flip.astype(int))
    assert D._te_pair(x, y, 1) > 5 * D._te_pair(y, x, 1)


def test_te_zero_on_independent():
    rng = np.random.default_rng(1)
    x = (rng.random(4000) < 0.2).astype(int)
    y = (rng.random(4000) < 0.2).astype(int)
    assert D._te_pair(x, y, 1) < 0.02


def test_min_units_raises():
    with pytest.raises(D.DirectedError):
        D.compute_directed({0: np.array([0.1, 0.2, 0.3])},
                           D.DirectedConfig(min_units=5))


def test_compute_directed_end_to_end():
    # chain 0->1->2->... with delay, plus noise units
    rng = np.random.default_rng(2)
    n_bins = 6000
    bin_s = 0.005
    driver = (rng.random(n_bins) < 0.15).astype(int)
    spikes = {}
    prev = driver
    for u in range(8):
        cur = np.zeros(n_bins, int)
        cur[1:] = prev[:-1]
        cur = np.abs(cur - (rng.random(n_bins) < 0.05).astype(int))
        t = np.where(cur > 0)[0] * bin_s + rng.random((cur > 0).sum()) * bin_s
        spikes[u] = np.sort(t)
        prev = cur
    res = D.compute_directed(spikes, D.DirectedConfig(min_units=5, max_units=8,
                                                      min_spikes_per_unit=10,
                                                      n_shuffle=5))
    assert res.te_matrix.shape[0] >= 5
    assert set(("mean_te", "flow_hierarchy", "degree_asymmetry")).issubset(res.metrics)
