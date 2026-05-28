"""Clustering of bin-level features + temporal merge.

Two clustering algorithms are available, dispatched by ``cluster_bins(...,
algorithm=...)``:

* ``"diffmap_gmm"`` (default) — diffusion-map embedding (Coifman–Lafon
  α-normalisation) of the z-normed feature matrix, then GMM with BIC selection
  over ``gmm_k_range``. Right inductive bias for "two attractors connected by a
  sparse transition manifold": the embedding preserves connectivity through
  low-density bridges, so trajectory bins land on the manifold instead of
  being labelled noise. Yields BOTH a discrete cluster label per bin AND a
  continuous per-bin burst posterior.
* ``"hdbscan"`` — original density-based clustering on the z-normed (and
  optionally PCA-projected) matrix. Kept for backwards compatibility and A/B
  comparison; under the trajectory geometry it labels transition bins as noise.

Both paths return the same :class:`ClusterAssignment` dataclass; downstream
consumers (``compute_ml_bursts``, dashboard, inspector script) only see field
names, not the algorithm choice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .iterative_burst_detector import _iter_merge, _mask_to_candidates

logger = logging.getLogger(__name__)


@dataclass
class ClusterAssignment:
    """Output of ``cluster_bins``.

    labels
        (n_bins,) integer cluster id per bin. For HDBSCAN -1 means noise; for
        diffmap_gmm every bin has a non-negative label (GMM is exhaustive).
    probabilities
        (n_bins,) per-bin membership probability. HDBSCAN: cluster-membership
        probability. diffmap_gmm: max GMM posterior across components.
    burst_label
        Cluster id that ranked highest by ``ranking_feature``. ``-2`` denotes
        "fell back to thresholding".
    cluster_rank
        Mapping {cluster_id → ranking score}. Sorted descending in iteration
        order.
    scaler_mean, scaler_std
        Per-feature mean/std used to z-norm before clustering.
    decision
        Short tag describing the path taken: "diffmap_gmm",
        "diffmap_singleton", "hdbscan", "hdbscan_single",
        "hdbscan_all_noise", "fallback_threshold".
    embedding
        Diffusion-map coordinates (n_bins, n_components). None for the
        HDBSCAN paths and for fallback decisions.
    burst_posterior
        Per-bin posterior of the burst cluster from the GMM, shape (n_bins,).
        For HDBSCAN this is filled with 1.0 on burst bins and 0.0 elsewhere
        (a degenerate "posterior") so downstream code can read this field
        unconditionally.
    gmm_k_selected, gmm_bic, gmm_bic_table
        BIC bookkeeping for diffmap_gmm. None for other paths.
    """

    labels: np.ndarray
    probabilities: np.ndarray
    burst_label: int
    cluster_rank: dict
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    decision: str = "diffmap_gmm"
    n_clusters: int = 0
    fallback_threshold: Optional[float] = None
    embedding: Optional[np.ndarray] = None
    burst_posterior: Optional[np.ndarray] = None
    gmm_k_selected: Optional[int] = None
    gmm_bic: Optional[float] = None
    gmm_bic_table: Optional[dict] = None


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


# ---------------------------------------------------------------------------
# Diffusion-map embedding
# ---------------------------------------------------------------------------


def _diffusion_map_embed(
    X_norm: np.ndarray,
    n_components: int,
    k_neighbors: int,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Coifman–Lafon diffusion-map embedding.

    Pipeline:
      1. Symmetric k-NN graph with locally-adaptive Gaussian kernel
         (bandwidth = distance to the k-th neighbour).
      2. α-normalisation ``K_α = D^{-α} K D^{-α}`` (α=1 removes sampling-density
         bias — the key property for trajectory geometries).
      3. Row-stochastic Markov matrix ``P = D_α^{-1} K_α``; eigendecompose its
         symmetric conjugate ``T = D_α^{-1/2} K_α D_α^{-1/2}``.
      4. Top ``n_components`` non-trivial eigenvectors, weighted by their
         eigenvalues, form the embedding.

    Returns
    -------
    embedding : (n_bins, n_components) float
    eigvals : (n_components,) float
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    from sklearn.neighbors import NearestNeighbors

    n = X_norm.shape[0]
    if n <= n_components + 2:
        raise ValueError(f"too few samples ({n}) for n_components={n_components}")
    k = int(max(2, min(k_neighbors, n - 1)))

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_norm)
    dists, idxs = nn.kneighbors(X_norm)  # self at column 0

    # Local bandwidth = distance to k-th neighbour (excluding self)
    sigma = dists[:, -1].copy()
    sigma_floor = max(1e-12, float(np.median(sigma)) * 1e-6)
    sigma = np.where(sigma < sigma_floor, sigma_floor, sigma)

    rows = np.repeat(np.arange(n), k + 1)
    cols = idxs.flatten()
    sig_prod = sigma[rows] * sigma[cols]
    K_vals = np.exp(-(dists.flatten() ** 2) / sig_prod)
    K = csr_matrix((K_vals, (rows, cols)), shape=(n, n))
    K = (K + K.T).multiply(0.5)  # symmetrise
    K = K.tocsr()

    # α-normalisation
    d = np.asarray(K.sum(axis=1)).flatten()
    d_alpha = np.power(np.where(d > 1e-12, d, 1.0), float(alpha))
    inv_d_alpha = 1.0 / d_alpha
    # K_α = diag(1/d^α) K diag(1/d^α)
    K_alpha = K.multiply(inv_d_alpha[:, None]).multiply(inv_d_alpha[None, :])
    K_alpha = csr_matrix(K_alpha)

    # Symmetric conjugate of the row-stochastic P
    d2 = np.asarray(K_alpha.sum(axis=1)).flatten()
    d2_safe = np.where(d2 > 1e-12, d2, 1.0)
    inv_sqrt_d2 = 1.0 / np.sqrt(d2_safe)
    T = K_alpha.multiply(inv_sqrt_d2[:, None]).multiply(inv_sqrt_d2[None, :])
    T = csr_matrix(T)
    # Re-symmetrise to wash out any float-roundoff asymmetry from sparse mul.
    T = ((T + T.T) * 0.5).tocsr()

    n_eigs = int(min(n_components + 1, n - 1))
    eigvals, eigvecs = eigsh(T, k=n_eigs, which="LM")
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Drop the trivial eigenvector (λ≈1, constant in P's right basis)
    eigvecs = eigvecs[:, 1:n_components + 1]
    eigvals = eigvals[1:n_components + 1]

    # Symmetric → P-right-eigenvectors: φ = D^{-1/2} v
    phi = eigvecs * inv_sqrt_d2[:, None]
    embedding = phi * eigvals[None, :]
    return embedding, eigvals


# ---------------------------------------------------------------------------
# Fallback to threshold on the ranking feature
# ---------------------------------------------------------------------------


def _threshold_fallback_assignment(
    X: np.ndarray,
    feature_names: list[str],
    ranking_feature: str,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    fallback_posterior_threshold: float,
    decision: str,
) -> ClusterAssignment:
    rank_idx = feature_names.index(ranking_feature)
    rank_values_raw = X[:, rank_idx]
    threshold = float(fallback_posterior_threshold)
    labels = np.where(rank_values_raw > threshold, 1, -1).astype(int)
    probabilities = np.clip(rank_values_raw, 0.0, 1.0)
    burst_label = 1
    cluster_rank: dict = {}
    n_clusters = 0
    if (labels == 1).any():
        cluster_rank = {1: float(rank_values_raw[labels == 1].mean())}
        n_clusters = 1
    burst_posterior = np.where(labels == 1, 1.0, 0.0).astype(float)
    return ClusterAssignment(
        labels=labels,
        probabilities=probabilities,
        burst_label=burst_label,
        cluster_rank=cluster_rank,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        decision=decision,
        n_clusters=n_clusters,
        fallback_threshold=threshold,
        embedding=None,
        burst_posterior=burst_posterior,
        gmm_k_selected=None,
        gmm_bic=None,
        gmm_bic_table=None,
    )


# ---------------------------------------------------------------------------
# Diffusion-map + GMM clusterer
# ---------------------------------------------------------------------------


def cluster_bins_diffmap_gmm(
    X: np.ndarray,
    feature_names: list[str],
    *,
    ranking_feature: str = "post_frac_gt_0_5",
    background_quantile: float = 0.5,
    diffmap_n_components: int = 5,
    diffmap_k_neighbors: int = 30,
    diffmap_alpha: float = 1.0,
    gmm_k_range: tuple = (2, 3, 4),
    gmm_em_n_init: int = 5,
    gmm_em_reg_covar: float = 1e-4,
    fallback_posterior_threshold: float = 0.3,
    random_state: int = 42,
) -> ClusterAssignment:
    """Embed bins via diffusion map then cluster with GMM (BIC-selected).

    Trajectory bins receive intermediate burst-cluster posteriors instead of
    being labelled noise. The burst cluster is the GMM component whose mean
    raw ``ranking_feature`` is highest — preserves the existing
    ``cluster_burst_label`` semantics so downstream consumers are unaffected.
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

    try:
        embedding, _eigvals = _diffusion_map_embed(
            X_norm,
            n_components=int(diffmap_n_components),
            k_neighbors=int(diffmap_k_neighbors),
            alpha=float(diffmap_alpha),
        )
    except Exception as e:  # noqa: BLE001 — propagate as fallback, not crash
        logger.warning(
            "diffusion-map embedding failed (%s); falling back to ranking-feature threshold",
            e,
        )
        return _threshold_fallback_assignment(
            X, feature_names, ranking_feature, mu, std,
            fallback_posterior_threshold, decision="fallback_threshold",
        )

    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        logger.warning("sklearn.mixture unavailable; falling back to threshold")
        return _threshold_fallback_assignment(
            X, feature_names, ranking_feature, mu, std,
            fallback_posterior_threshold, decision="fallback_threshold",
        )

    best_gmm = None
    best_bic = float("inf")
    best_k = 0
    bic_table: dict[int, float] = {}
    for k in gmm_k_range:
        ki = int(k)
        if ki < 1 or ki >= embedding.shape[0]:
            continue
        try:
            gmm = GaussianMixture(
                n_components=ki,
                n_init=int(gmm_em_n_init),
                reg_covar=float(gmm_em_reg_covar),
                random_state=int(random_state),
                covariance_type="full",
            )
            gmm.fit(embedding)
            bic = float(gmm.bic(embedding))
        except Exception as e:  # noqa: BLE001
            logger.warning("GMM(k=%d) failed: %s", ki, e)
            continue
        bic_table[ki] = bic
        if bic < best_bic:
            best_gmm, best_bic, best_k = gmm, bic, ki

    if best_gmm is None:
        return _threshold_fallback_assignment(
            X, feature_names, ranking_feature, mu, std,
            fallback_posterior_threshold, decision="fallback_threshold",
        )

    if best_k <= 1:
        # GMM picked a singleton — diffusion map worked but no separable
        # components. Threshold on the ranking feature, but tag the decision
        # so the diagnostic surface tells the user why.
        assignment = _threshold_fallback_assignment(
            X, feature_names, ranking_feature, mu, std,
            fallback_posterior_threshold, decision="diffmap_singleton",
        )
        assignment.embedding = embedding
        assignment.gmm_k_selected = 1
        assignment.gmm_bic = best_bic
        assignment.gmm_bic_table = bic_table
        return assignment

    labels = best_gmm.predict(embedding).astype(int)
    posteriors = best_gmm.predict_proba(embedding)  # (n_bins, k)

    cluster_rank: dict = {}
    for c in range(best_k):
        mask = labels == c
        cluster_rank[int(c)] = float(rank_values_raw[mask].mean()) if mask.any() else float("-inf")
    cluster_rank = dict(sorted(cluster_rank.items(), key=lambda kv: -kv[1]))
    burst_label = int(next(iter(cluster_rank)))

    # Rank-weighted aggregate posterior: burst_posterior[i] = Σ_c P(c|i) * w_c,
    # where w_c ∈ [0, 1] scales each cluster's contribution by how burst-like its
    # mean raw ranking-feature value is. This makes trajectory bins receive
    # intermediate posteriors instead of near-zero (BIC may split the trajectory
    # into its own components, which would collapse a per-component posterior).
    rank_vals = np.array([cluster_rank[c] for c in range(best_k)], dtype=float)
    finite = np.isfinite(rank_vals)
    if finite.any():
        rmin = float(rank_vals[finite].min())
        rmax = float(rank_vals[finite].max())
        span = max(rmax - rmin, 1e-12)
        weights = np.where(finite, (rank_vals - rmin) / span, 0.0)
    else:
        weights = np.zeros(best_k)
    burst_posterior = (posteriors * weights[None, :]).sum(axis=1).astype(float)
    burst_posterior = np.clip(burst_posterior, 0.0, 1.0)

    probabilities = posteriors.max(axis=1).astype(float)

    return ClusterAssignment(
        labels=labels,
        probabilities=probabilities,
        burst_label=burst_label,
        cluster_rank=cluster_rank,
        scaler_mean=mu,
        scaler_std=std,
        decision="diffmap_gmm",
        n_clusters=best_k,
        fallback_threshold=None,
        embedding=embedding,
        burst_posterior=burst_posterior,
        gmm_k_selected=best_k,
        gmm_bic=best_bic,
        gmm_bic_table=bic_table,
    )


# ---------------------------------------------------------------------------
# HDBSCAN clusterer (preserved for A/B + back-compat)
# ---------------------------------------------------------------------------


def cluster_bins_hdbscan(
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

    Original density-based clusterer. Kept reachable via the
    ``algorithm="hdbscan"`` dispatch path.
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

    try:
        import hdbscan  # type: ignore
    except ImportError:
        logger.warning(
            "hdbscan package not installed; ml_burst clustering is silently "
            "degrading to a hard threshold on %s. Install with "
            "`conda install -n yuxin_mea -c conda-forge hdbscan` to restore.",
            ranking_feature,
        )
        return _threshold_fallback_assignment(
            X, feature_names, ranking_feature, mu, std,
            fallback_posterior_threshold, decision="fallback_threshold",
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
    cluster_rank: dict = {}
    decision = "hdbscan"
    burst_label = -2

    if n_clusters == 0:
        decision = "hdbscan_all_noise"
        assignment = _threshold_fallback_assignment(
            X, feature_names, ranking_feature, mu, std,
            fallback_posterior_threshold, decision=decision,
        )
        return assignment

    for c in unique:
        mask = labels == c
        cluster_rank[int(c)] = float(rank_values_raw[mask].mean()) if mask.any() else float("-inf")
    cluster_rank = dict(sorted(cluster_rank.items(), key=lambda kv: -kv[1]))
    burst_label = int(next(iter(cluster_rank)))
    if n_clusters == 1:
        decision = "hdbscan_single"

    burst_posterior = np.where(labels == burst_label, 1.0, 0.0).astype(float)

    return ClusterAssignment(
        labels=labels,
        probabilities=probabilities,
        burst_label=burst_label,
        cluster_rank=cluster_rank,
        scaler_mean=mu,
        scaler_std=std,
        decision=decision,
        n_clusters=n_clusters,
        fallback_threshold=None,
        embedding=None,
        burst_posterior=burst_posterior,
        gmm_k_selected=None,
        gmm_bic=None,
        gmm_bic_table=None,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def cluster_bins(
    X: np.ndarray,
    feature_names: list[str],
    *,
    algorithm: str = "diffmap_gmm",
    ranking_feature: str = "post_frac_gt_0_5",
    background_quantile: float = 0.5,
    fallback_posterior_threshold: float = 0.3,
    # HDBSCAN knobs
    min_cluster_size: int = 30,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    metric: str = "euclidean",
    pca_n_components: int = 0,
    # Diffusion-map + GMM knobs
    diffmap_n_components: int = 5,
    diffmap_k_neighbors: int = 30,
    diffmap_alpha: float = 1.0,
    gmm_k_range: tuple = (2, 3, 4),
    gmm_em_n_init: int = 5,
    gmm_em_reg_covar: float = 1e-4,
    random_state: int = 42,
) -> ClusterAssignment:
    """Dispatch to the requested clustering algorithm.

    ``algorithm`` ∈ {"diffmap_gmm", "hdbscan"}.
    """
    algo = str(algorithm).lower()
    if algo == "diffmap_gmm":
        return cluster_bins_diffmap_gmm(
            X, feature_names,
            ranking_feature=ranking_feature,
            background_quantile=background_quantile,
            diffmap_n_components=diffmap_n_components,
            diffmap_k_neighbors=diffmap_k_neighbors,
            diffmap_alpha=diffmap_alpha,
            gmm_k_range=tuple(int(k) for k in gmm_k_range),
            gmm_em_n_init=gmm_em_n_init,
            gmm_em_reg_covar=gmm_em_reg_covar,
            fallback_posterior_threshold=fallback_posterior_threshold,
            random_state=random_state,
        )
    if algo == "hdbscan":
        return cluster_bins_hdbscan(
            X, feature_names,
            ranking_feature=ranking_feature,
            background_quantile=background_quantile,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            metric=metric,
            fallback_posterior_threshold=fallback_posterior_threshold,
            pca_n_components=pca_n_components,
        )
    raise ValueError(f"unknown cluster_algorithm {algorithm!r}; expected 'diffmap_gmm' or 'hdbscan'")


def burst_bin_mask(
    assignment: ClusterAssignment,
    *,
    posterior_threshold: float = 0.3,
) -> np.ndarray:
    """Boolean mask selecting bins likely belonging to a burst event.

    For ``diffmap_gmm`` (and ``diffmap_singleton`` / ``fallback_threshold``
    paths, which populate ``burst_posterior``), this thresholds the
    rank-weighted aggregate posterior. The default 0.3 mirrors the legacy
    ``fallback_posterior_threshold`` and means "include the burst peak and
    any non-negligibly burst-leaning trajectory clusters".

    For ``hdbscan`` (no posterior, just labels), this falls back to the
    strict ``labels == burst_label`` semantic that the HDBSCAN code path
    has always used.
    """
    if assignment.burst_posterior is not None and assignment.decision != "hdbscan":
        return assignment.burst_posterior >= float(posterior_threshold)
    return assignment.labels == assignment.burst_label


def _binary_closing_1d(mask: np.ndarray, size: int) -> np.ndarray:
    """1D morphological closing (dilate then erode) with a flat structuring element."""
    size = max(1, int(size))
    if size == 1 or mask.size == 0:
        return mask.astype(bool, copy=True)
    try:
        from scipy.ndimage import binary_closing
    except ImportError:
        m = mask.astype(bool, copy=True)
        d = m.copy()
        for k in range(1, size):
            d[:-k] |= m[k:]
            d[k:] |= m[:-k]
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

    See module docstring of the previous version — semantics unchanged.
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
