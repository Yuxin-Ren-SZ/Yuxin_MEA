"""Second-stage clustering over *detected bursts* (assign a type to each burst).

The first stage (``ml_burst_cluster``) clusters bin-level features with HDBSCAN
and produces one row per burst in ``network_bursts``. This module runs a second,
**per-well** clustering over those bursts so every burst gets an integer
``burst_type`` label.

Per-burst feature vector
------------------------
Scalars carried on each event dict:
  duration_s, within_burst_fr (= total_spikes / duration_s), participation,
  llr_aggregate, posterior_peak, posterior_mean, ff_peak, n_distinct_clusters.
Composition (per-well, variable width): for each bin-level HDBSCAN cluster id
``c`` present in the well, ``cluster_ratio_<c>`` = fraction of the burst's bins
assigned to ``c`` (incl. ``-1`` noise). Each burst's ratios sum to 1.

Why KMeans/GMM (not HDBSCAN)?
-----------------------------
The request is to *label each* burst — every burst must get a type. HDBSCAN
would dump outliers to noise (-1, unlabeled). Per-well burst counts are small
(typ. 10-135), so we standardize the heterogeneous features and pick k by
silhouette (KMeans) or BIC (GMM), with guards for the small-N / degenerate
regimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Scalar feature columns, in fixed order. Composition columns are appended after
# these by ``build_burst_feature_matrix``.
SCALAR_FEATURES: tuple[str, ...] = (
    "duration_s",
    "within_burst_fr",
    "participation",
    "llr_aggregate",
    "posterior_peak",
    "posterior_mean",
    "ff_peak",
    "n_distinct_clusters",
)


@dataclass
class BurstTyping:
    """Output of :func:`cluster_bursts`.

    labels
        (n_bursts,) integer ``burst_type`` per burst, 0..k-1. All-zero when
        typing is skipped (too few bursts / degenerate).
    method
        "kmeans" or "gmm".
    k
        Number of burst types actually used (1 when skipped).
    score
        Selection metric of the chosen k: mean silhouette (kmeans, higher
        better) or BIC (gmm, lower better). NaN when skipped or k==1.
    per_type_counts
        {type_id -> count} burst membership per type.
    skipped_reason
        None on success; else one of "no_bursts", "too_few_bursts",
        "degenerate" (fewer than 2 distinguishable rows / k<2).
    feature_names
        Column order of the matrix that was clustered (echoed for diagnostics).
    """

    labels: np.ndarray
    method: str = "kmeans"
    k: int = 1
    score: float = float("nan")
    per_type_counts: dict = field(default_factory=dict)
    skipped_reason: Optional[str] = None
    feature_names: list = field(default_factory=list)


def build_burst_feature_matrix(
    events: list[dict],
    bin_labels: np.ndarray,
    t_centers: np.ndarray,
) -> tuple[np.ndarray, list[str], list[dict]]:
    """Assemble the per-burst feature matrix for second-stage clustering.

    Parameters
    ----------
    events
        Materialised burst dicts (post-gate ``network_bursts``). Each must carry
        ``start``/``end`` plus the scalar feature keys in ``SCALAR_FEATURES``.
    bin_labels
        (n_bins,) bin-level HDBSCAN cluster id per bin (``assignment.labels``).
    t_centers
        (n_bins,) bin centre times — used to slice each burst's bins with the
        same ``[start, end)`` convention as the detector.

    Returns
    -------
    X
        (n_bursts, n_feat) float matrix: scalar columns then ``cluster_ratio_*``.
    feature_names
        Column names matching ``X``.
    comps
        Per-burst composition dicts {cluster_id -> ratio} (for traceability).
    """
    n = len(events)
    bin_labels = np.asarray(bin_labels)
    t_centers = np.asarray(t_centers)

    # Per-burst composition over the bins each burst spans.
    comps: list[dict] = []
    for ev in events:
        in_ev = (t_centers >= ev["start"]) & (t_centers < ev["end"])
        labs = bin_labels[in_ev]
        if labs.size == 0:
            comps.append({})
            continue
        ids, counts = np.unique(labs, return_counts=True)
        total = float(labs.size)
        comps.append({int(c): float(cnt) / total for c, cnt in zip(ids, counts)})

    # Well-wide sorted union of cluster ids → fixed composition column order.
    union_ids = sorted({c for comp in comps for c in comp})
    ratio_names = [f"cluster_ratio_{c}" for c in union_ids]
    feature_names = list(SCALAR_FEATURES) + ratio_names

    X = np.zeros((n, len(feature_names)), dtype=float)
    n_scalar = len(SCALAR_FEATURES)
    for i, ev in enumerate(events):
        for j, name in enumerate(SCALAR_FEATURES):
            X[i, j] = float(ev.get(name, 0.0))
        comp = comps[i]
        for j, c in enumerate(union_ids):
            X[i, n_scalar + j] = comp.get(c, 0.0)
    return X, feature_names, comps


def _standardize(X: np.ndarray) -> np.ndarray:
    """Z-norm columns; zero-variance columns map to 0 (std forced to 1)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def cluster_bursts(
    X: np.ndarray,
    *,
    method: str = "kmeans",
    max_k: int = 6,
    k: int = 0,
    min_bursts: int = 8,
    random_state: int = 42,
    feature_names: Optional[list] = None,
) -> BurstTyping:
    """Cluster the per-burst feature matrix; assign a type to every burst.

    ``k == 0`` auto-selects k in ``2..min(max_k, n_bursts-1)`` by silhouette
    (kmeans) or BIC (gmm). ``k > 0`` fixes k (clamped to the valid range).
    Small-N / degenerate inputs fall back to a single type (all-zero labels).
    """
    feature_names = list(feature_names or [])
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    def _single(reason: Optional[str]) -> BurstTyping:
        labels = np.zeros(n, dtype=int)
        return BurstTyping(
            labels=labels,
            method=method,
            k=1 if n else 0,
            score=float("nan"),
            per_type_counts={0: int(n)} if n else {},
            skipped_reason=reason,
            feature_names=feature_names,
        )

    if n == 0:
        return _single("no_bursts")
    if n < int(min_bursts):
        return _single("too_few_bursts")

    Xn = _standardize(X)
    # Distinct-row guard: clustering is meaningless if every burst is identical.
    if np.unique(np.round(Xn, 9), axis=0).shape[0] < 2:
        return _single("degenerate")

    k_max = min(int(max_k), n - 1)
    if k_max < 2:
        return _single("degenerate")

    if int(k) > 0:
        k_candidates = [min(int(k), n - 1)]
        if k_candidates[0] < 2:
            return _single("degenerate")
    else:
        k_candidates = list(range(2, k_max + 1))

    best = _fit_select(Xn, k_candidates, method=method, random_state=random_state)
    if best is None:
        return _single("degenerate")
    labels, k_used, score = best
    counts = {int(t): int((labels == t).sum()) for t in sorted(set(labels.tolist()))}
    return BurstTyping(
        labels=labels.astype(int),
        method=method,
        k=int(k_used),
        score=float(score),
        per_type_counts=counts,
        skipped_reason=None,
        feature_names=feature_names,
    )


def _fit_select(
    Xn: np.ndarray,
    k_candidates: list[int],
    *,
    method: str,
    random_state: int,
) -> Optional[tuple[np.ndarray, int, float]]:
    """Fit each candidate k, return (labels, k, score) for the best.

    KMeans selects max mean silhouette; GMM selects min BIC. Returns None if no
    candidate produced a valid (>=2 distinct labels) partition.
    """
    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        best = None  # (bic, labels, k)
        for kk in k_candidates:
            gm = GaussianMixture(
                n_components=kk,
                covariance_type="diag",
                reg_covar=1e-6,
                random_state=int(random_state),
            )
            labels = gm.fit_predict(Xn)
            if len(set(labels.tolist())) < 2:
                continue
            bic = float(gm.bic(Xn))
            if best is None or bic < best[0]:
                best = (bic, labels, kk)
        if best is None:
            return None
        return best[1], best[2], best[0]

    # default: kmeans + silhouette
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best = None  # (sil, labels, k)
    for kk in k_candidates:
        km = KMeans(n_clusters=kk, n_init=10, random_state=int(random_state))
        labels = km.fit_predict(Xn)
        if len(set(labels.tolist())) < 2:
            continue
        sil = float(silhouette_score(Xn, labels))
        if best is None or sil > best[0]:
            best = (sil, labels, kk)
    if best is None:
        return None
    return best[1], best[2], best[0]
