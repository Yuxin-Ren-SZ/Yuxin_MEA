"""F6b — Developmental migration of resting- and bursting-state manifolds.

For one physical well, **all its recordings across DIV are pooled into a single
UMAP embedding** (one StandardScaler + one UMAP fit over the concatenated per-bin
feature matrices). This is the crucial design point: because the embedding is fit
once on every timepoint together, the axes/loadings are identical for DIV 7 and
DIV 30, so where a bin lands is comparable across development — separate per-day
UMAPs would have incomparable axes.

Rows = a Control well and IVH wells (same chip, CX118, to avoid a chip confound).
Columns are **treatment-relative timepoints** — Day 0, Day 14, Day 21 of
treatment (``tau`` = days post-treatment; nearest available recording is used and
its actual day annotated; if Day 21 was not recorded the latest day is shown and
labelled). On each timepoint's fixed axes the resting-state bins (blue) and
bursting-state bins (red) reveal how the network's dynamics move after treatment.

Reuses the pooled-embedding idea from ``scripts/well_recording_umap_overlay.py``.
Requires ``debug=True`` ML-burst runs (``debug_trace.pkl`` present).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load as L
from . import stats as S
from .report_style import save_fig

# UMAP params mirror the pipeline; n_components=2 for display.
_UMAP_KW = dict(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)
# The manifold is *fit* on a subsample for speed, then every bin is *transformed*
# onto those fixed axes — so displayed coverage is 100% while the (slow) fit stays
# bounded. Raise _FIT_SAMPLE for a finer manifold at the cost of fit time.
_FIT_SAMPLE = 20000          # bins used to FIT the UMAP per well
_GREY_SAMPLE = 12000         # bins drawn for the light context layer (display only)
ROW_ARMS = ["Control", "IVH_Early", "IVH_Late"]
TARGET_DAYS = [0, 14, 21]    # treatment-relative days (tau) to display
_REST_C = "#4477aa"
_BURST_C = "#d62728"


def _stride_keep(n: int, k: int, seed: int = 0) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def _pick_well(tidy, analysis_root, arm: str):
    """Well of this arm with the most DIV-spanning debug traces (chip CX118)."""
    wi = S.well_index(tidy)
    wells = wi[(wi.arm == arm) & (wi.chip == "CX118")].well_uid
    best = None
    for w in wells:
        sub = tidy[tidy.well_uid == w]
        divs = [int(r.DIV) for _, r in sub.iterrows()
                if L.ml_trace_path(analysis_root, r).exists()]
        if divs and min(divs) <= 10 and max(divs) >= 28:
            if best is None or len(divs) > best[1]:
                best = (w, len(divs))
    return None if best is None else best[0]


def _pool_well(tidy, analysis_root, well_uid: str):
    """Pool ALL bins of all recordings of one well → shared 2-D UMAP.

    UMAP is *fit* on a random ``_FIT_SAMPLE`` subsample (speed), then every bin is
    *transformed* onto the fitted axes, so the returned embedding is 100% of bins.
    Returns dict with emb (N,2), div/tau/is_burst (N,) or None if no data.
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    sub = tidy[tidy.well_uid == well_uid].sort_values("DIV")
    Xs, divs, taus, burst = [], [], [], []
    for _, r in sub.iterrows():
        p = L.ml_trace_path(analysis_root, r)
        if not p.exists():
            continue
        tr = L.load_ml_trace(analysis_root, r)
        X = np.asarray(getattr(tr, "feature_matrix", None), dtype=float)
        labels = np.asarray(getattr(tr, "hdbscan_labels", []))
        if X is None or X.ndim != 2 or len(labels) != X.shape[0]:
            continue
        bl = set(getattr(tr, "burst_labels", []) or [])
        Xs.append(X)                                    # keep every bin
        divs.append(np.full(X.shape[0], int(r.DIV)))
        taus.append(np.full(X.shape[0], int(r.tau)))
        burst.append(np.array([lab in bl for lab in labels]))
    if not Xs:
        return None
    Xall = np.nan_to_num(np.vstack(Xs))
    Z = StandardScaler().fit_transform(Xall)
    fit_idx = _stride_keep(Z.shape[0], _FIT_SAMPLE)     # subsample only for the fit
    reducer = umap.UMAP(**_UMAP_KW).fit(Z[fit_idx])
    emb = np.asarray(reducer.transform(Z))              # transform 100% of bins
    return dict(emb=emb, div=np.concatenate(divs),
                tau=np.concatenate(taus), is_burst=np.concatenate(burst))


def _nearest_day(avail: np.ndarray, target: int) -> int | None:
    """Nearest available post-treatment day (tau ≥ 0) to a target."""
    post = np.unique(avail[avail >= 0])
    if len(post) == 0:
        return None
    return int(post[np.argmin(np.abs(post - target))])


def build_umap_migration(tidy, analysis_root):
    import matplotlib.pyplot as plt

    pooled = {}
    for arm in ROW_ARMS:
        w = _pick_well(tidy, analysis_root, arm)
        pooled[arm] = (_pool_well(tidy, analysis_root, w), w) if w else (None, None)

    n_rows, n_cols = len(ROW_ARMS), len(TARGET_DAYS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows),
                             squeeze=False)
    for ri, arm in enumerate(ROW_ARMS):
        d, _ = pooled[arm]
        grey_idx = (_stride_keep(len(d["emb"]), _GREY_SAMPLE)
                    if d is not None else None)
        for ci, target in enumerate(TARGET_DAYS):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            if d is None:
                ax.text(0.5, 0.5, "no debug traces", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                continue
            day = _nearest_day(d["tau"], target)
            # faint context: subsample of the full embedding (display only)
            g = d["emb"][grey_idx]
            ax.scatter(g[:, 0], g[:, 1], s=1.0, c="0.88", lw=0, zorder=1,
                       rasterized=True)
            if day is not None:
                # colored day = 100% of that day's bins
                sel = d["tau"] == day
                rest = sel & ~d["is_burst"]
                brst = sel & d["is_burst"]
                ax.scatter(d["emb"][rest, 0], d["emb"][rest, 1], s=2,
                           c=_REST_C, lw=0, alpha=0.5, zorder=2, rasterized=True)
                ax.scatter(d["emb"][brst, 0], d["emb"][brst, 1], s=4,
                           c=_BURST_C, lw=0, alpha=0.75, zorder=3, rasterized=True)
            latest = (target == TARGET_DAYS[-1]
                      and day is not None and day < target)
            note = f"Day {day}" + (" (latest)" if latest else "") if day is not None else "n/a"
            ax.text(0.03, 0.96, note, transform=ax.transAxes, fontsize=6.5,
                    va="top", ha="left", color="0.25")
            if ri == 0:
                ax.set_title(f"Day {target} of treatment", fontsize=8,
                             fontweight="bold")
            if ci == 0:
                ax.set_ylabel(arm, fontsize=8, fontweight="bold")
    # single resting/burst legend
    axes[0, -1].scatter([], [], s=8, c=_REST_C, label="resting")
    axes[0, -1].scatter([], [], s=8, c=_BURST_C, label="bursting")
    axes[0, -1].legend(loc="lower right", fontsize=6, markerscale=1.5)
    fig.suptitle("Figure 6b — Post-treatment network-state dynamics "
                 "(per-well pooled UMAP; fixed axes across treatment days)",
                 fontsize=8.5, y=1.0)
    return fig, {arm: (None if d is None else
                       {int(t): int((d["tau"] == t).sum())
                        for t in [_nearest_day(d["tau"], x) for x in TARGET_DAYS]
                        if t is not None})
                 for arm, (d, _) in pooled.items()}


def render(tidy, analysis_root, figure_root):
    fig, meta = build_umap_migration(tidy, analysis_root)
    return save_fig(fig, "F6b_umap_migration", figure_root, subdir="main"), meta
