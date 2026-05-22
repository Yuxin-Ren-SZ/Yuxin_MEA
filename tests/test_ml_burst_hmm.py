"""Unit tests for the per-unit 2-state Poisson HMM (ml_burst_hmm)."""
from __future__ import annotations

import unittest

import numpy as np

from yuxin_mea.analysis.ml_burst_hmm import (
    UnitHMMFit,
    fit_all_units,
    fit_unit_hmm,
    posterior_burst,
)


def _two_rate_counts(
    n_bins: int,
    bin_size: float,
    lam_bg_hz: float,
    lam_burst_hz: float,
    burst_intervals_bins: list[tuple[int, int]],
    seed: int,
) -> np.ndarray:
    """Generate synthetic bin counts with known λ_bg/λ_burst.

    Each bin's count is Poisson-distributed at the rate dictated by whether
    the bin index falls inside any of the burst intervals.
    """
    rng = np.random.default_rng(seed)
    state = np.zeros(n_bins, dtype=bool)
    for s, e in burst_intervals_bins:
        state[s:e] = True
    rates_per_bin = np.where(state, lam_burst_hz * bin_size, lam_bg_hz * bin_size)
    return rng.poisson(rates_per_bin).astype(float)


class FitUnitHMMTests(unittest.TestCase):
    def test_recovers_rates_within_tolerance(self):
        bin_size = 0.02
        counts = _two_rate_counts(
            n_bins=4000,
            bin_size=bin_size,
            lam_bg_hz=2.0,
            lam_burst_hz=40.0,
            burst_intervals_bins=[(200, 300), (1200, 1340), (2500, 2620), (3500, 3600)],
            seed=42,
        )
        fit = fit_unit_hmm(counts, bin_size, unit_id="u0", max_iter=100, tol=1e-4)
        self.assertIsNone(fit.skipped_reason, fit.skipped_reason)
        # ±50% tolerance on each rate — generous because EM on short bursts
        # has finite-sample noise, and we mainly care that they're separated.
        self.assertGreater(fit.lambda_burst, fit.lambda_bg)
        self.assertAlmostEqual(fit.lambda_bg, 2.0, delta=1.0)
        self.assertAlmostEqual(fit.lambda_burst, 40.0, delta=20.0)
        self.assertTrue(fit.converged or fit.n_iter == 100)

    def test_identifiability_flip_state1_is_burst(self):
        # Even with quantile init reversed, state 1 must end up as the burst.
        bin_size = 0.02
        counts = _two_rate_counts(
            n_bins=2000, bin_size=bin_size,
            lam_bg_hz=3.0, lam_burst_hz=30.0,
            burst_intervals_bins=[(100, 200), (800, 900), (1400, 1500)],
            seed=7,
        )
        fit = fit_unit_hmm(counts, bin_size, unit_id="u0")
        self.assertIsNone(fit.skipped_reason)
        self.assertGreater(fit.lambda_burst, fit.lambda_bg)

    def test_low_spike_count_skipped(self):
        # Sparse unit — should be skipped without raising
        counts = np.zeros(2000, dtype=float)
        counts[::500] = 1.0  # 4 total spikes
        fit = fit_unit_hmm(counts, 0.02, unit_id="u_sparse", min_spikes=50)
        self.assertEqual(fit.skipped_reason, "low_spike_count")

    def test_low_rate_ratio_marked(self):
        # Use a strict ratio threshold to ensure the gate is exercised. Even
        # well-separated synthetic data won't trivially clear ratio=50.
        bin_size = 0.02
        counts = _two_rate_counts(
            n_bins=2000, bin_size=bin_size,
            lam_bg_hz=5.0, lam_burst_hz=8.0,
            burst_intervals_bins=[(500, 600)],
            seed=2,
        )
        fit = fit_unit_hmm(counts, bin_size, unit_id="u_flat", min_rate_ratio=50.0)
        self.assertIn(fit.skipped_reason, ("low_rate_ratio", "low_spike_count"))


class PosteriorBurstTests(unittest.TestCase):
    def test_posterior_is_high_inside_burst_low_outside(self):
        bin_size = 0.02
        burst_bins = [(500, 700)]
        counts = _two_rate_counts(
            n_bins=2000, bin_size=bin_size,
            lam_bg_hz=2.0, lam_burst_hz=80.0,
            burst_intervals_bins=burst_bins,
            seed=11,
        )
        fit = fit_unit_hmm(counts, bin_size, unit_id="u0")
        self.assertIsNone(fit.skipped_reason)
        post = posterior_burst(counts, fit, bin_size)
        # Inside burst: mean posterior should be high
        self.assertGreater(post[500:700].mean(), 0.7)
        # Outside burst (a quiet region): mean posterior should be low
        self.assertLess(post[0:300].mean(), 0.3)

    def test_posterior_nan_for_skipped_fit(self):
        fit = UnitHMMFit(unit_id="u", skipped_reason="low_spike_count")
        post = posterior_burst(np.zeros(100), fit, 0.02)
        self.assertEqual(post.shape, (100,))
        self.assertTrue(np.isnan(post).all())


class FitAllUnitsTests(unittest.TestCase):
    def test_runs_serial_and_returns_aligned_outputs(self):
        bin_size = 0.02
        spike_matrix = np.stack([
            _two_rate_counts(1500, bin_size, 2.0, 30.0, [(300, 400)], seed=k)
            for k in range(4)
        ])
        unit_ids = [f"u_{k}" for k in range(4)]
        fits, post = fit_all_units(spike_matrix, unit_ids, bin_size, n_jobs=1)
        self.assertEqual(len(fits), 4)
        self.assertEqual(post.shape, (4, 1500))
        for i, fit in enumerate(fits):
            self.assertEqual(fit.unit_id, unit_ids[i])

    def test_skipped_units_have_nan_posterior_rows(self):
        bin_size = 0.02
        n_bins = 1500
        # Mix one valid unit and one sparse unit
        valid = _two_rate_counts(n_bins, bin_size, 2.0, 30.0, [(300, 400)], seed=3)
        sparse = np.zeros(n_bins)
        sparse[::500] = 1
        spike_matrix = np.stack([valid, sparse])
        fits, post = fit_all_units(
            spike_matrix, ["u_valid", "u_sparse"], bin_size, min_spikes=50, n_jobs=1
        )
        self.assertIsNone(fits[0].skipped_reason)
        self.assertEqual(fits[1].skipped_reason, "low_spike_count")
        self.assertFalse(np.isnan(post[0]).all())
        self.assertTrue(np.isnan(post[1]).all())


if __name__ == "__main__":
    unittest.main()
