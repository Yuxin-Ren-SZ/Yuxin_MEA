"""Unit tests for second-stage burst typing (ml_burst_typing)."""
from __future__ import annotations

import importlib
import unittest

import numpy as np

from yuxin_mea.analysis.ml_burst_typing import (
    SCALAR_FEATURES,
    build_burst_feature_matrix,
    cluster_bursts,
)

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


def _event(start, end, **kw):
    ev = {
        "start": float(start),
        "end": float(end),
        "duration_s": float(end - start),
        "within_burst_fr": 0.0,
        "participation": 0.0,
        "llr_aggregate": 0.0,
        "posterior_peak": 0.0,
        "posterior_mean": 0.0,
        "ff_peak": 0.0,
        "n_distinct_clusters": 1,
    }
    ev.update(kw)
    return ev


class TestFeatureMatrix(unittest.TestCase):
    def setUp(self):
        # 30 bins at t = 0.5, 1.5, ... 29.5
        self.t_centers = np.arange(30) + 0.5
        # bins 0-9 cluster 0, 10-19 cluster 1, 20-29 noise (-1)
        self.bin_labels = np.array([0] * 10 + [1] * 10 + [-1] * 10)

    def test_columns_scalar_then_ratio(self):
        events = [_event(0, 10), _event(10, 20)]
        X, names, comps = build_burst_feature_matrix(
            events, self.bin_labels, self.t_centers
        )
        self.assertEqual(names[: len(SCALAR_FEATURES)], list(SCALAR_FEATURES))
        # union of cluster ids spanned = {-1?, 0, 1}; here only 0 and 1 spanned
        ratio_cols = names[len(SCALAR_FEATURES):]
        self.assertEqual(ratio_cols, ["cluster_ratio_0", "cluster_ratio_1"])
        self.assertEqual(X.shape, (2, len(SCALAR_FEATURES) + 2))

    def test_composition_sums_to_one(self):
        # event spanning a mix of cluster 0, 1, and noise
        events = [_event(5, 25)]  # bins 5..24 → 5 of cl0, 10 of cl1, 5 of noise
        X, names, comps = build_burst_feature_matrix(
            events, self.bin_labels, self.t_centers
        )
        self.assertAlmostEqual(sum(comps[0].values()), 1.0, places=9)
        n_scalar = len(SCALAR_FEATURES)
        self.assertAlmostEqual(float(X[0, n_scalar:].sum()), 1.0, places=9)
        # 5/20, 10/20, 5/20
        self.assertAlmostEqual(comps[0][0], 0.25, places=9)
        self.assertAlmostEqual(comps[0][1], 0.50, places=9)
        self.assertAlmostEqual(comps[0][-1], 0.25, places=9)

    def test_scalar_values_copied(self):
        events = [_event(0, 10, within_burst_fr=12.5, duration_s=10.0)]
        X, names, _ = build_burst_feature_matrix(
            events, self.bin_labels, self.t_centers
        )
        self.assertEqual(X[0, names.index("within_burst_fr")], 12.5)
        self.assertEqual(X[0, names.index("duration_s")], 10.0)


@unittest.skipUnless(SKLEARN_AVAILABLE, "sklearn required")
class TestClusterBursts(unittest.TestCase):
    def _two_blobs(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.normal(loc=[0.0, 0.0], scale=0.15, size=(n // 2, 2))
        b = rng.normal(loc=[5.0, 5.0], scale=0.15, size=(n // 2, 2))
        return np.vstack([a, b])

    def test_auto_selects_two_clusters(self):
        X = self._two_blobs()
        out = cluster_bursts(X, method="kmeans", max_k=6, k=0, min_bursts=8)
        self.assertIsNone(out.skipped_reason)
        self.assertEqual(out.k, 2)
        self.assertEqual(len(out.labels), X.shape[0])
        self.assertEqual(set(out.labels.tolist()), {0, 1})
        self.assertEqual(sum(out.per_type_counts.values()), X.shape[0])

    def test_too_few_bursts_skips(self):
        X = self._two_blobs(n=6)
        out = cluster_bursts(X, min_bursts=8)
        self.assertEqual(out.skipped_reason, "too_few_bursts")
        self.assertTrue(np.all(out.labels == 0))
        self.assertEqual(out.k, 1)

    def test_no_bursts(self):
        out = cluster_bursts(np.zeros((0, 5)), min_bursts=8)
        self.assertEqual(out.skipped_reason, "no_bursts")
        self.assertEqual(len(out.labels), 0)

    def test_degenerate_identical_rows(self):
        X = np.ones((20, 4))
        out = cluster_bursts(X, min_bursts=8)
        self.assertEqual(out.skipped_reason, "degenerate")
        self.assertTrue(np.all(out.labels == 0))

    def test_deterministic_seed(self):
        X = self._two_blobs(seed=1)
        a = cluster_bursts(X, random_state=42)
        b = cluster_bursts(X, random_state=42)
        np.testing.assert_array_equal(a.labels, b.labels)

    def test_fixed_k(self):
        X = self._two_blobs(n=40)
        out = cluster_bursts(X, k=3, min_bursts=8)
        self.assertEqual(out.k, 3)

    def test_gmm_method(self):
        X = self._two_blobs()
        out = cluster_bursts(X, method="gmm", k=0, min_bursts=8)
        self.assertIsNone(out.skipped_reason)
        self.assertEqual(out.method, "gmm")
        self.assertGreaterEqual(out.k, 2)


def _bursty_well(n_units=6, duration_s=60.0, seed=0):
    """Synthetic well with many well-separated bursts (enough to type)."""
    rng = np.random.default_rng(seed)
    centers = np.arange(3.0, duration_s - 3.0, 2.5)
    spike_times = {}
    for i in range(n_units):
        bg = rng.uniform(0.0, duration_s, rng.poisson(2.0 * duration_s))
        bursts = [
            rng.normal(loc=c, scale=0.08, size=rng.poisson(80 * 0.5))
            for c in centers
        ]
        spk = np.sort(np.concatenate([bg, *bursts]))
        spike_times[f"u_{i:02d}"] = spk[(spk >= 0) & (spk < duration_s)]
    return spike_times


@unittest.skipUnless(SKLEARN_AVAILABLE, "sklearn required")
class TestDetectorIntegration(unittest.TestCase):
    """The enabled/disabled branch inside compute_ml_bursts."""

    @classmethod
    def setUpClass(cls):
        from yuxin_mea.analysis.ml_burst_detector import MLBurstConfig
        cls.MLBurstConfig = MLBurstConfig
        cls.spikes = _bursty_well()

    def _run(self, **overrides):
        from yuxin_mea.analysis.ml_burst_detector import compute_ml_bursts
        cfg = self.MLBurstConfig(burst_typing_min_bursts=2, **overrides)
        return compute_ml_bursts(self.spikes, config=cfg)

    def test_enabled_adds_column_and_diagnostics(self):
        res = self._run(burst_typing_enabled=True)
        nb = res.network_bursts
        self.assertFalse(nb.empty)
        self.assertIn("burst_type", nb.columns)
        self.assertIn("within_burst_fr", nb.columns)
        self.assertIn("n_distinct_clusters", nb.columns)
        self.assertIn("burst_typing", res.diagnostics)
        # within_burst_fr == total_spikes / duration_s
        row = nb.iloc[0]
        self.assertAlmostEqual(
            row["within_burst_fr"], row["total_spikes"] / row["duration_s"], places=6
        )

    def test_disabled_omits_column(self):
        res = self._run(burst_typing_enabled=False)
        nb = res.network_bursts
        self.assertFalse(nb.empty)
        self.assertNotIn("burst_type", nb.columns)
        self.assertNotIn("burst_typing", res.diagnostics)


if __name__ == "__main__":
    unittest.main()
