"""Unit tests for bin-level feature assembly (ml_burst_features)."""
from __future__ import annotations

import unittest

import numpy as np

from yuxin_mea.analysis.burst_common import _compute_spike_matrix
from yuxin_mea.analysis.ml_burst_features import (
    build_feature_matrix,
    feature_names_for,
)
from yuxin_mea.analysis.ml_burst_hmm import UnitHMMFit


def _make_synthetic_well(
    n_units: int = 6,
    duration_s: float = 40.0,
    burst_centers: tuple = (10.0, 20.0, 30.0),
    burst_dur: float = 0.5,
    burst_rate: float = 80.0,
    bg_rate: float = 2.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    spike_times = {}
    for i in range(n_units):
        bg_n = rng.poisson(bg_rate * duration_s)
        bg = rng.uniform(0.0, duration_s, bg_n)
        burst_spikes = []
        for c in burst_centers:
            n = rng.poisson(burst_rate * burst_dur)
            burst_spikes.append(rng.normal(loc=c, scale=burst_dur / 6, size=n))
        spk = np.sort(np.concatenate([bg] + burst_spikes))
        spk = spk[(spk >= 0) & (spk < duration_s)]
        spike_times[f"u_{i:02d}"] = spk
    return spike_times


class FeatureMatrixShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bin_size = 0.02
        self.spike_times = _make_synthetic_well()
        self.units = list(self.spike_times.keys())
        self.bins = np.arange(0.0, 40.0 + self.bin_size, self.bin_size)
        self.t_centers = (self.bins[:-1] + self.bins[1:]) / 2
        self.spike_matrix = _compute_spike_matrix(
            self.spike_times, self.units, self.bins, len(self.t_centers)
        )
        # Fake fits: every unit gets a "valid" fit with known rates so we
        # don't have to actually run the HMM to test feature assembly.
        self.fits = [
            UnitHMMFit(
                unit_id=u,
                lambda_bg=2.0,
                lambda_burst=80.0,
                skipped_reason=None,
            )
            for u in self.units
        ]
        # Posteriors: synthetic high-during-burst, low-elsewhere
        n_units = len(self.units)
        n_bins = len(self.t_centers)
        rng = np.random.default_rng(7)
        post = rng.uniform(0.0, 0.2, size=(n_units, n_bins))
        for c in (10.0, 20.0, 30.0):
            in_burst = (self.t_centers >= c - 0.3) & (self.t_centers <= c + 0.3)
            post[:, in_burst] = rng.uniform(0.7, 1.0, size=(n_units, in_burst.sum()))
        self.posteriors = post

    def test_column_count_matches_feature_names_for(self):
        ff_mults = (0.5, 1.0, 2.0, 5.0)
        X, names = build_feature_matrix(
            self.spike_times,
            self.units,
            self.bins,
            self.t_centers,
            self.bin_size,
            self.fits,
            self.posteriors,
            ff_scale_multipliers=ff_mults,
            spike_matrix=self.spike_matrix,
        )
        self.assertEqual(X.shape[0], len(self.t_centers))
        self.assertEqual(X.shape[1], len(names))
        self.assertEqual(names, feature_names_for(ff_mults))

    def test_no_nan_or_inf_in_output(self):
        X, _ = build_feature_matrix(
            self.spike_times,
            self.units,
            self.bins,
            self.t_centers,
            self.bin_size,
            self.fits,
            self.posteriors,
            spike_matrix=self.spike_matrix,
        )
        self.assertTrue(np.isfinite(X).all())

    def test_post_frac_is_high_during_burst(self):
        X, names = build_feature_matrix(
            self.spike_times,
            self.units,
            self.bins,
            self.t_centers,
            self.bin_size,
            self.fits,
            self.posteriors,
            spike_matrix=self.spike_matrix,
        )
        idx = names.index("post_frac_gt_0_5")
        burst_mask = np.any(
            np.stack([(self.t_centers >= c - 0.3) & (self.t_centers <= c + 0.3) for c in (10.0, 20.0, 30.0)]),
            axis=0,
        )
        burst_mean = X[burst_mask, idx].mean()
        bg_mean = X[~burst_mask, idx].mean()
        self.assertGreater(burst_mean, bg_mean + 0.3)


class SkippedUnitsTests(unittest.TestCase):
    def test_skipped_rows_nan_dropped_via_aggregation(self):
        bin_size = 0.02
        spike_times = _make_synthetic_well(n_units=4)
        units = list(spike_times.keys())
        bins = np.arange(0.0, 40.0 + bin_size, bin_size)
        t_centers = (bins[:-1] + bins[1:]) / 2
        spike_matrix = _compute_spike_matrix(spike_times, units, bins, len(t_centers))
        # Mark two units skipped (representing the HMM's low-spike path)
        fits = [
            UnitHMMFit(unit_id=u, lambda_bg=2.0, lambda_burst=80.0, skipped_reason=None)
            for u in units
        ]
        fits[1].skipped_reason = "low_spike_count"
        fits[1].lambda_bg = float("nan")
        fits[1].lambda_burst = float("nan")
        post = np.zeros((4, len(t_centers)))
        post[1] = np.nan  # skipped row
        X, _ = build_feature_matrix(
            spike_times, units, bins, t_centers, bin_size, fits, post,
            spike_matrix=spike_matrix,
        )
        # Even with one NaN row in posteriors, the output is finite.
        self.assertTrue(np.isfinite(X).all())


if __name__ == "__main__":
    unittest.main()
