"""HDBSCAN clustering of bin-level features + temporal merge.

Pipeline:
  1. Density-based clustering on the bin-level feature matrix.
  2. Rank clusters by mean of a "ranking feature" (default post_frac_gt_0_5);
     the top cluster is labeled burst.
  3. All-noise fallback: if HDBSCAN labels everything −1, threshold on the
     ranking feature.
  4. Morphological closing fills small gaps in the burst mask.
  5. Gap-bridging using a relaxed valley condition (mirrors the iterative
     detector's ``_iter_merge``).

Why HDBSCAN over GMM?
---------------------
Burst-state bins do not form a Gaussian cluster — they're a long-tailed mode
in feature space adjacent to background. HDBSCAN handles non-convex shapes
without fixing k, and labels "in-between" bins as noise (−1) which we can then
decide what to do with via the temporal merge instead of forcing them into a
hard cluster.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .iterative_burst_detector import _iter_merge, _mask_to_candidates


@dataclass
class ClusterAssignment:
    """Output of ``cluster_bins``.

    labels
        (n_bins,) integer cluster id per bin. -1 is HDBSCAN noise.
    probabilities
        (n_bins,) cluster-membership probabilities from HDBSCAN. Zero for noise.
    burst_label
        Cluster id that ranked highest by ``ranking_feature``. ``-2`` denotes
        "fell back to thresholding" (HDBSCAN found no clusters).
    cluster_rank
        Mapping {cluster_id → ranking score}. Sorted descending in iteration
        order.
    scaler_mean, scaler_std
        Per-feature mean/std used to z-norm before clustering. Persisted so the
        debug trace can reconstruct the scaled space.
    decision
        Short tag describing the path taken: "hdbscan", "hdbscan_single",
        "hdbscan_all_noise", "fallback_threshold".
    """

    labels: np.ndarray
    probabilities: np.ndarray
    burst_label: int
    cluster_rank: dict
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    decision: str = "hdbscan"
    n_clusters: int = 0
    fallback_threshold: Optional[float] = None


def _background_mask_from_feature(
    values: np.ndarray,
    background_quantile: float,
) -> np.ndarray:
    """Bins whose ranking-feature value sits in the bottom ``background_quantile``
    fraction. Used as the bg mask for z-norm.
    """
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    cutoff = float(np.quantile(values, float(background_quantile)))
    mask = values <= cutoff
    # Avoid degenerate empty masks
    if mask.sum() < max(2, int(0.05 * values.size)):
        mask = values <= float(np.quantile(values, max(background_quantile, 0.5)))
    return mask


def _znorm_with_stats(X: np.ndarray, bg_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-norm relative to background bins; return the scaler stats for tracing."""
    bg = X[bg_mask] if bg_mask.any() else X
    mu = bg.mean(axis=0)
    std = bg.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mu) / std, mu, std


def cluster_bins(
    X: np.ndarray,
    feature_names: list[str],
    *,
    ranking_feature: str = "post_frac_gt_0_5",
    background_quantile: float = 0.5,
    min_cluster_size: int = 30,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    metric: str = "euclidean",
    fallback_posterior_threshold: float = 0.3,
    pca_n_components: int = 0,
) -> ClusterAssignment:
    """Cluster bins with HDBSCAN; identify the burst cluster by ranking_feature.

    Parameters mirror the ``hdbscan.HDBSCAN`` constructor. ``pca_n_components``
    of 0 disables PCA; any positive value applies sklearn's PCA after z-norm.
    """
    if ranking_feature not in feature_names:
        raise ValueError(
            f"ranking_feature {ranking_feature!r} not in feature_names; "
            f"got {feature_names}"
        )
    rank_idx = feature_names.index(ranking_feature)
    rank_values_raw = X[:, rank_idx]

    bg_mask = _background_mask_from_feature(rank_values_raw, background_quantile)
    X_norm, mu, std = _znorm_with_stats(X, bg_mask)

    if pca_n_components and int(pca_n_components) > 0:
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            X_to_cluster = X_norm
        else:
            n_comp = int(min(int(pca_n_components), X_norm.shape[1], X_norm.shape[0]))
            X_to_cluster = PCA(n_components=n_comp, random_state=42).fit_transform(X_norm)
    else:
        X_to_cluster = X_norm

    labels = np.full(X.shape[0], -1, dtype=int)
    probabilities = np.zeros(X.shape[0], dtype=float)
    cluster_rank: dict = {}
    decision = "hdbscan"
    burst_label = -2
    n_clusters = 0
    fallback_threshold: Optional[float] = None

    try:
        import hdbscan  # type: ignore
    except ImportError:
        decision = "fallback_threshold"
        fallback_threshold = float(fallback_posterior_threshold)
        labels = np.where(rank_values_raw > fallback_threshold, 1, -1).astype(int)
        probabilities = np.clip(rank_values_raw, 0.0, 1.0)
        # Treat the "above threshold" bin set as a single synthetic cluster.
        burst_label = 1
        if (labels == 1).any():
            cluster_rank = {1: float(rank_values_raw[labels == 1].mean())}
            n_clusters = 1
        return ClusterAssignment(
            labels=labels,
            probabilities=probabilities,
            burst_label=burst_label,
            cluster_rank=cluster_rank,
            scaler_mean=mu,
            scaler_std=std,
            decision=decision,
            n_clusters=n_clusters,
            fallback_threshold=fallback_threshold,
        )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_method=str(cluster_selection_method),
        metric=str(metric),
    )
    labels = clusterer.fit_predict(X_to_cluster).astype(int)
    probabilities = np.asarray(clusterer.probabilities_, dtype=float)

    unique = sorted(set(int(c) for c in labels if c >= 0))
    n_clusters = len(unique)

    if n_clusters == 0:
        # All noise — fall back to thresholding on the ranking feature.
        decision = "hdbscan_all_noise"
        fallback_threshold = float(fallback_posterior_threshold)
        labels = np.where(rank_values_raw > fallback_threshold, 1, -1).astype(int)
        probabilities = np.clip(rank_values_raw, 0.0, 1.0)
        burst_label = 1
        if (labels == 1).any():
            cluster_rank = {1: float(rank_values_raw[labels == 1].mean())}
            n_clusters = 1
    else:
        # Rank clusters by mean of the ranking feature in the original (raw)
        # space — that's the interpretable quantity (e.g. mean fraction of
        # units in burst-state across cluster members).
        for c in unique:
            mask = labels == c
            cluster_rank[int(c)] = float(rank_values_raw[mask].mean()) if mask.any() else float("-inf")
        # Sort by score descending so iteration order is stable
        cluster_rank = dict(sorted(cluster_rank.items(), key=lambda kv: -kv[1]))
        burst_label = int(next(iter(cluster_rank)))
        if n_clusters == 1:
            decision = "hdbscan_single"

    return ClusterAssignment(
        labels=labels,
        probabilities=probabilities,
        burst_label=burst_label,
        cluster_rank=cluster_rank,
        scaler_mean=mu,
        scaler_std=std,
        decision=decision,
        n_clusters=n_clusters,
        fallback_threshold=fallback_threshold,
    )


def burst_bin_mask(assignment: ClusterAssignment) -> np.ndarray:
    """Boolean mask selecting bins assigned to the burst cluster."""
    return assignment.labels == assignment.burst_label


def _binary_closing_1d(mask: np.ndarray, size: int) -> np.ndarray:
    """1D morphological closing (dilate then erode) with a flat structuring element.

    Closes gaps shorter than ``size`` bins, leaving longer gaps intact.
    """
    size = max(1, int(size))
    if size == 1 or mask.size == 0:
        return mask.astype(bool, copy=True)
    try:
        from scipy.ndimage import binary_closing
    except ImportError:
        # Fallback: manual dilation+erosion
        m = mask.astype(bool, copy=True)
        # Dilate
        d = m.copy()
        for k in range(1, size):
            d[:-k] |= m[k:]
            d[k:] |= m[:-k]
        # Erode
        e = d.copy()
        for k in range(1, size):
            e[:-k] &= d[k:]
            e[k:] &= d[:-k]
        return e
    return binary_closing(mask.astype(bool), structure=np.ones(size, dtype=bool)).astype(bool)


def temporal_merge(
    mask: np.ndarray,
    t_centers: np.ndarray,
    bins: np.ndarray,
    bin_size: float,
    ranking_signal: np.ndarray,
    *,
    closing_bins: int = 3,
    merge_mad_scale: float = 0.75,
    merge_floor_frac: float = 0.70,
    merge_gap_s: float = 0.3,
) -> tuple[list[dict], np.ndarray, float]:
    """Close small gaps in the burst mask and merge nearby candidates.

    Returns
    -------
    candidates : list[dict]
        Standard ``[{start, end, start_idx, end_idx}, ...]`` shape used by the
        rest of the burst pipeline.
    closed_mask : np.ndarray
        Post-closing boolean mask. Useful for debug overlays.
    threshold : float
        Effective ranking-signal threshold used by ``_iter_merge`` for the
        relaxed valley condition.

    Notes
    -----
    The ``ranking_signal`` (typically ``post_frac_gt_0_5``) is treated as the
    "composite" for valley-bridging logic. Threshold = median(bg) + mad_scale × MAD(bg)
    where bg = bins outside the closed mask.
    """
    closed = _binary_closing_1d(mask, int(closing_bins))
    candidates = _mask_to_candidates(closed, bins)

    if not candidates:
        return [], closed, 0.0

    bg = ranking_signal[~closed]
    if bg.size > 0:
        baseline = float(np.median(bg))
        mad = float(np.median(np.abs(bg - baseline)))
    else:
        baseline = float(np.median(ranking_signal))
        mad = float(np.median(np.abs(ranking_signal - baseline)))
    threshold = baseline + float(merge_mad_scale) * max(mad, 1e-6)

    merged = _iter_merge(
        candidates,
        ranking_signal,
        t_centers,
        bin_size,
        float(merge_gap_s),
        float(threshold),
        float(merge_floor_frac),
    )
    return merged, closed, float(threshold)
