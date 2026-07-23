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


class TestPowerlawDegenerateInput:
    """A well whose avalanche distribution defeats the powerlaw MLE must yield a
    NaN row, not sink the whole task.

    Found in production: a 2-unit well raised
    ``ValueError("No data points in defined range of the distribution")`` from
    inside scipy, via powerlaw's lazy distribution construction, and failed the
    well's entire criticality task. The exponent is genuinely unavailable there,
    but n_avalanches and the branching ratio still are.
    """

    @staticmethod
    def _good_data():
        rng = np.random.default_rng(0)
        return (rng.pareto(1.5, size=5000) + 1).astype(int)

    def test_all_nan_when_the_fit_itself_cannot_be_built(self, monkeypatch):
        class _Boom:
            def __init__(self, *a, **k):
                pass

            def __getattr__(self, name):
                # mirrors powerlaw.Fit.__getattr__ fitting lazily on access
                raise ValueError("No data points in defined range of the distribution.")

        import powerlaw
        monkeypatch.setattr(powerlaw, "Fit", _Boom)

        out = C._fit_powerlaw(self._good_data())
        assert np.isnan(out["exponent"]) and np.isnan(out["xmin"])
        assert np.isnan(out["ks"]) and np.isnan(out["R_lognormal"])
        assert out["n"] == 5000            # the count is still honest

    def test_exponent_survives_a_failing_lognormal_comparison(self, monkeypatch):
        """distribution_compare is a diagnostic — losing it must not lose tau."""
        import powerlaw
        real_fit = powerlaw.Fit

        class _NoCompare(real_fit):
            def distribution_compare(self, *a, **k):
                raise ValueError("No data points in defined range of the distribution.")

        monkeypatch.setattr(powerlaw, "Fit", _NoCompare)

        out = C._fit_powerlaw(self._good_data())
        assert 2.0 < out["exponent"] < 3.2      # exponent unaffected
        assert np.isfinite(out["xmin"])
        assert np.isnan(out["R_lognormal"]) and np.isnan(out["p_lognormal"])

    def test_healthy_data_is_unchanged_by_the_guards(self):
        out = C._fit_powerlaw(self._good_data())
        assert 2.0 < out["exponent"] < 3.2
        assert np.isfinite(out["R_lognormal"])

    def test_two_unit_well_completes_with_partial_metrics(self):
        """The production shape: enough pooled spikes, far too few units."""
        rng = np.random.default_rng(3)
        burst_times = np.arange(0, 200, 0.5)
        spikes = {}
        for u in range(2):
            s = []
            for bt in burst_times:
                s.extend(bt + rng.random(rng.integers(1, 4)) * 0.01)
            spikes[u] = np.sort(np.array(s))
        res = C.compute_criticality(spikes, C.CriticalityConfig())
        # completes rather than raising; exponents may or may not be fittable
        assert res.metrics["n_avalanches"] > 0
        for key in ("aval_tau", "aval_alpha", "dcc"):
            assert isinstance(res.metrics[key], float)
