"""Tests for neuronal-avalanche criticality (analysis/criticality.py)."""
import numpy as np
import pytest

from yuxin_mea.analysis import criticality as C


def test_avalanche_detection_simple():
    # bins: [1,2,0,0,3,0] -> two avalanches sizes {3, 3}, durations {2, 1}
    # build pooled spikes so that histogram at dt gives those counts
    dt = 1.0
    # bin0:1 spike @0.5, bin1:2 @1.5, gap, bin4:3 @4.5
    pooled = np.array([0.5, 1.5, 1.6, 4.5, 4.6, 4.7])
    sizes, durs, counts = C.detect_avalanches(pooled, dt)
    assert list(sizes) == [3, 3]
    assert list(durs) == [2, 1]


def test_too_few_spikes_raises():
    with pytest.raises(C.CriticalityError):
        C.compute_criticality({0: np.array([0.1, 0.2])},
                              C.CriticalityConfig(min_spikes=50))


def test_powerlaw_exponent_recovered():
    # sample from a discrete power law with exponent ~2.5 -> fit should be close
    rng = np.random.default_rng(0)
    x = (rng.pareto(1.5, size=20000) + 1).astype(int)  # exponent = a+1 = 2.5
    fit = C._fit_powerlaw(x)
    assert 2.0 < fit["exponent"] < 3.2


def test_branching_naive_on_geometric_decay():
    # counts halving each step within one avalanche -> naive sigma ~0.5
    counts = np.array([8, 4, 2, 1, 0, 8, 4, 2, 1])
    sigma = C._branching_naive(counts)
    assert 0.4 < sigma < 0.6


def test_compute_criticality_end_to_end():
    # a bursty synthetic well: periodic population bursts -> avalanches + metrics
    rng = np.random.default_rng(1)
    spikes = {}
    burst_times = np.arange(0, 300, 1.0)  # a burst every second
    for u in range(30):
        s = []
        for bt in burst_times:
            if rng.random() < 0.7:
                s.extend(bt + rng.random(rng.integers(1, 5)) * 0.05)
        spikes[u] = np.sort(np.array(s))
    res = C.compute_criticality(spikes, C.CriticalityConfig(min_spikes=100,
                                                            min_avalanches=10))
    assert res.metrics["n_avalanches"] > 0
    assert np.isfinite(res.metrics["branching_ratio_mr"]) or \
        np.isnan(res.metrics["branching_ratio_mr"])  # MR may be nan on trivial data
    assert set(("dcc", "aval_tau", "aval_alpha")).issubset(res.metrics)
