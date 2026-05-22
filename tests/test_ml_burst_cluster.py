"""Unit tests for HDBSCAN-based clustering + temporal merge (ml_burst_cluster)."""
from __future__ import annotations

import importlib
import unittest

import numpy as np

from yuxin_mea.analysis.ml_burst_cluster import (
    burst_bin_mask,
    cluster_bins,
    temporal_merge,
)


HDBSCAN_AVAILABLE = importlib.util.find_spec("hdbscan") is not None


def _two_blob_features(
    n_bins_bg: int = 800,
    n_bins_burst: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Synthesize a feature matrix with two clearly separated regimes.

    Returns (X, feature_names, burst_index_mask) where burst_index_mask flags
    the bins drawn from the high-posterior regime.
    """
    rng = np.random.default_rng(seed)
    feat_names = [
        "post_frac_gt_0_5", "post_mean", "post_std", "post_q90", "post_entropy",
        "PFR", "participation", "FF0", "FF1", "llr_hmm_mean",
    ]
    n_features = len(feat_names)
    # Background: low values
    bg_block = rng.normal(loc=0.05, scale=0.05, size=(n_bins_bg, n_features))
    # Burst: high values on the post_* and llr_ columns
    burst_block = rng.normal(loc=0.8, scale=0.05, size=(n_bins_burst, n_features))
    # Interleave: keep bursts contiguous so temporal_merge has something to chew on
    X = np.concatenate([
        bg_block[: n_bins_bg // 2],
        burst_block,
        bg_block[n_bins_bg // 2 :],
    ], axis=0)
    n_total = X.shape[0]
    burst_mask = np.zeros(n_total, dtype=bool)
    burst_mask[n_bins_bg // 2 : n_bins_bg // 2 + n_bins_burst] = True
    return X, feat_names, burst_mask


class ClusterBinsTests(unittest.TestCase):
    @unittest.skipUnless(HDBSCAN_AVAILABLE, "hdbscan not installed")
    def test_two_blob_recovery(self):
        X, names, true_mask = _two_blob_features(seed=42)
        assignment = cluster_bins(
            X, names,
            ranking_feature="post_frac_gt_0_5",
            min_cluster_size=20,
            min_samples=5,
        )
        # Should find at least one cluster, and burst_label should match the
        # synthetic "burst" regime majority-wise.
        self.assertGreaterEqual(assignment.n_clusters, 1)
        recovered = burst_bin_mask(assignment)
        # Recovered burst bins should overlap heavily with the true burst bins.
        overlap = (recovered & true_mask).sum() / max(true_mask.sum(), 1)
        self.assertGreater(overlap, 0.7)

    def test_fallback_when_hdbscan_unavailable_or_all_noise(self):
        # Use a 1-feature blob with low contrast so even if hdbscan is present
        # it might not find structure; assert that the fallback path emits a
        # synthetic burst cluster from the ranking-feature threshold.
        names = ["post_frac_gt_0_5", "PFR"]
        rng = np.random.default_rng(0)
        n = 500
        post = rng.uniform(0.0, 0.2, size=n)
        post[100:120] = rng.uniform(0.7, 0.9, size=20)  # only 20 burst bins, below min_cluster_size
        X = np.column_stack([post, rng.normal(0, 1, size=n)])
        assignment = cluster_bins(
            X, names,
            ranking_feature="post_frac_gt_0_5",
            min_cluster_size=100,  # forces hdbscan to find no clusters → fallback
            min_samples=5,
            fallback_posterior_threshold=0.5,
        )
        self.assertIn(assignment.decision, {"hdbscan_all_noise", "fallback_threshold", "hdbscan_single"})
        if assignment.decision != "hdbscan_single":
            self.assertEqual(assignment.burst_label, 1)
            mask = burst_bin_mask(assignment)
            self.assertGreaterEqual(mask.sum(), 1)


class TemporalMergeTests(unittest.TestCase):
    def test_closes_short_gap_and_emits_candidates(self):
        # Build a mask with two burst regions separated by a 1-bin gap
        n_bins = 200
        mask = np.zeros(n_bins, dtype=bool)
        mask[50:70] = True
        mask[71:90] = True
        bin_size = 0.02
        bins = np.arange(0, (n_bins + 1) * bin_size, bin_size)
        t_centers = (bins[:-1] + bins[1:]) / 2
        ranking = np.zeros(n_bins)
        ranking[mask] = 0.9
        candidates, closed, threshold = temporal_merge(
            mask, t_centers, bins, bin_size, ranking,
            closing_bins=3, merge_mad_scale=0.75, merge_floor_frac=0.7,
            merge_gap_s=3.0 * bin_size,
        )
        # After closing + merge there should be one contiguous candidate
        self.assertEqual(len(candidates), 1)
        self.assertGreater(threshold, 0.0)

    def test_empty_mask_returns_no_candidates(self):
        n_bins = 100
        bin_size = 0.02
        bins = np.arange(0, (n_bins + 1) * bin_size, bin_size)
        t_centers = (bins[:-1] + bins[1:]) / 2
        candidates, _, _ = temporal_merge(
            np.zeros(n_bins, dtype=bool),
            t_centers, bins, bin_size, np.zeros(n_bins),
        )
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
