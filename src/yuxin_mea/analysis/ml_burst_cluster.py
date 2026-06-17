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

from .burst_common import _iter_merge, _mask_to_candidates


@dataclass
class ClusterAssignment:
    """Output of ``cluster_bins``.

    labels
        (n_bins,) integer cluster id per bin. -1 is HDBSCAN noise.
    probabilities
        (n_bins,) cluster-membership probabilities from HDBSCAN. Zero for noise.
    burst_label
        Cluster id that ranked highest by ``ranking_feature``. ``-2`` denotes
        "fell back to thresholding" (HDBSCAN found no clusters). Kept for
        back-compat; ``burst_labels`` is the authoritative burst set.
    burst_labels
        All cluster ids selected as burst — the top-ranked cluster plus any
        cluster whose mean ranking exceeds the adaptive threshold. On a manifold
        the burst trajectory fragments across several clusters, so the burst is a
        *set*, not one cluster. ``burst_bin_mask`` keys off this.
    cluster_rank
        Mapping {cluster_id → ranking score}. Sorted descending in iteration
        order.
    scaler_mean, scaler_std
        Per-feature mean/std used to z-norm before clustering. Persisted so the
        debug trace can reconstruct the scaled space.
    decision
        Short tag describing the path taken: "hdbscan", "hdbscan_single",
        "hdbscan_all_noise", "fallback_threshold". UMAP embedding appends
        "+umap" (e.g. "hdbscan+umap").
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
    burst_labels: list = field(default_factory=list)


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


def _umap_embed(
    X_norm: np.ndarray,
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> Optional[np.ndarray]:
    """UMAP embedding of the z-normed features for density-based clustering.

    Returns None when ``umap-learn`` is unavailable so the caller can fall back
    to clustering the raw z-normed space.
    """
    try:
        import umap  # type: ignore
    except ImportError:
        return None
    n = X_norm.shape[0]
    n_comp = int(min(int(n_components), X_norm.shape[1], max(2, n - 1)))
    nn = max(2, min(int(n_neighbors), n - 1))
    reducer = umap.UMAP(
        n_components=n_comp, n_neighbors=nn, min_dist=float(min_dist),
        metric=str(metric), random_state=42,
    )
    return reducer.fit_transform(X_norm)


def _select_burst_clusters(
    labels: np.ndarray,
    rank_values_raw: np.ndarray,
    cluster_rank: dict,
    *,
    mad_scale: float,
) -> list[int]:
    """Burst = top-ranked cluster ∪ clusters whose mean ranking exceeds an
    adaptive threshold (median + mad_scale·MAD of the lower half).

    On a UMAP manifold the burst trajectory fragments across several clusters,
    so a single top cluster under-captures it. The adaptive threshold (computed
    on the lower half of the ranking signal = background) keeps the small outer
    non-burst clusters out, since their mean ranking stays low. Always keeping
    the top-ranked cluster preserves single-cluster / no-burst behavior.
    """
    if not cluster_rank:
        return []
    top = int(next(iter(cluster_rank)))
    top_score = float(cluster_rank[top])
    base = rank_values_raw[rank_values_raw <= np.median(rank_values_raw)]
    if base.size:
        med = float(np.median(base))
        mad = float(np.median(np.abs(base - med)))
    else:
        med, mad = float(np.median(rank_values_raw)), 0.0
    # Adaptive threshold from background spread, floored to at least 25% of the
    # gap up to the strongest cluster. The floor prevents over-selection when the
    # background MAD collapses (~0): without it a near-background cluster squeaks
    # over median+k*MAD and drags resting bins into the burst set.
    thr = med + float(mad_scale) * max(mad, 1e-6)
    thr = max(thr, med + 0.25 * (top_score - med))
    selected = {top}
    for c, score in cluster_rank.items():
        if int(c) >= 0 and float(score) > thr:
            selected.add(int(c))
    return sorted(selected)


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
    cluster_embedding_mode: str = "none",
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.0,
    umap_n_components: int = 5,
    burst_mad_scale: float = 3.0,
) -> ClusterAssignment:
    """Cluster bins with HDBSCAN; identify the burst cluster(s) by ranking_feature.

    Parameters mirror the ``hdbscan.HDBSCAN`` constructor. The clustering space
    is selected by ``cluster_embedding_mode``:

    - ``"none"``: cluster the z-normed features directly (optionally PCA-reduced
      when ``pca_n_components`` > 0 — kept for back-compat).
    - ``"umap"``: cluster a UMAP embedding of the z-normed features. The burst is
      a low-density trajectory that HDBSCAN discards as noise in the raw space;
      UMAP collapses it into a dense region HDBSCAN can recover.

    Burst selection is multi-cluster (see ``_select_burst_clusters``): the burst
    trajectory fragments across clusters on the manifold, so ``burst_labels`` is
    a set, while ``burst_label`` remains the top-ranked id for back-compat.
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

    embed_suffix = ""
    if str(cluster_embedding_mode) == "umap":
        emb = _umap_embed(
            X_norm,
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=metric,
        )
        if emb is not None:
            X_to_cluster = emb
            embed_suffix = "+umap"
        else:
            X_to_cluster = X_norm  # umap-learn unavailable; degrade to raw space
    elif pca_n_components and int(pca_n_components) > 0:
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
        burst_labels: list = []
        if (labels == 1).any():
            cluster_rank = {1: float(rank_values_raw[labels == 1].mean())}
            n_clusters = 1
            burst_labels = [1]
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
            burst_labels=burst_labels,
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

    burst_labels: list = []
    if n_clusters == 0:
        # All noise — fall back to thresholding on the ranking feature.
        decision = "hdbscan_all_noise" + embed_suffix
        fallback_threshold = float(fallback_posterior_threshold)
        labels = np.where(rank_values_raw > fallback_threshold, 1, -1).astype(int)
        probabilities = np.clip(rank_values_raw, 0.0, 1.0)
        burst_label = 1
        if (labels == 1).any():
            cluster_rank = {1: float(rank_values_raw[labels == 1].mean())}
            n_clusters = 1
            burst_labels = [1]
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
        # Multi-cluster burst selection: the trajectory fragments across clusters
        # on the manifold, so select every high-ranking cluster, not just the top.
        burst_labels = _select_burst_clusters(
            labels, rank_values_raw, cluster_rank, mad_scale=burst_mad_scale
        )
        decision = ("hdbscan_single" if n_clusters == 1 else "hdbscan") + embed_suffix

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
        burst_labels=burst_labels,
    )


def burst_bin_mask(assignment: ClusterAssignment) -> np.ndarray:
    """Boolean mask selecting bins assigned to any burst cluster."""
    bl = assignment.burst_labels or ([assignment.burst_label]
                                     if assignment.burst_label is not None else [])
    return np.isin(assignment.labels, list(bl))


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
