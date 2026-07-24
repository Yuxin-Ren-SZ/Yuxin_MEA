"""F6 — ML burst characterization (methods novelty).

Panels (single representative well for A-B, whole cohort for D):

* **A. HMM burst posteriors** — per-unit P(burst) heatmap over a time window
  with the population co-burst signal (fraction of units with P>0.5) beneath.
* **B. Feature-space embedding** — 2-D UMAP of the 26-D per-bin feature matrix,
  bins coloured by whether HDBSCAN assigned them to a burst cluster. Shows the
  low-density burst manifold the detector recovers.
* **D. Burst modulation index across arms** — cohort-level distribution of the
  HMM burst-modulation index (well_uid unit), tying the ML method to biology.

(Per-well burst-type clustering moved to the cohort-wide F6c, where bursts are
pooled across wells and clustered in one shared feature space — see
``fig6c_bursttypes.py``. Panel labels here read A / B / D to match F6d output.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .report_style import (
    METRIC_LABELS, STATE_BURST, UMAP_REST_GREY, group_color, ordered_groups,
)


# UMAP params mirror the pipeline (ml_burst_detection config), n_components=2 for viz.
_UMAP_KW = dict(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)
_BURST_RED = STATE_BURST         # network-state palette (shared across F6/F6b)
_NOISE_GREY = UMAP_REST_GREY


def _pick_example(tidy: pd.DataFrame) -> pd.Series:
    cand = tidy[(tidy.burst_type_k >= 2) & (tidy.nb_count >= 8)
                & (tidy.n_curated >= 25)]
    return cand.sort_values("nb_rate", ascending=False).iloc[0]


def _panel_posterior(ax, ax_trace, cax, tr, t_win=(0.0, 60.0)) -> None:
    bins = np.asarray(tr.bins)
    ncols = tr.posterior_matrix.shape[1]
    centers = (bins[:-1] + bins[1:]) / 2 if len(bins) == ncols + 1 else bins[:ncols]
    m = (centers >= t_win[0]) & (centers <= t_win[1])
    P = tr.posterior_matrix[:, m]
    x0, x1 = float(centers[m][0]), float(centers[m][-1])
    # order units by burst-time activity for a cleaner heatmap
    order = np.argsort(-P.mean(axis=1))
    im = ax.imshow(P[order], aspect="auto", cmap="magma", vmin=0, vmax=1,
                   extent=[x0, x1, 0, P.shape[0]], origin="lower",
                   interpolation="nearest")
    ax.set_ylabel("unit")
    ax.set_xlim(x0, x1)
    ax.tick_params(labelbottom=False)
    # colorbar in its own axis so it does NOT steal width from the heatmap
    # (that width theft is what mis-aligned the heatmap vs the trace below).
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("P(burst)", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    # population co-burst trace (post_frac_gt_0_5 = feature col 0), same x-axis
    fn = list(tr.feature_names)
    co = tr.feature_matrix[:, fn.index("post_frac_gt_0_5")]
    fc = centers[: len(co)]
    mm = (fc >= x0) & (fc <= x1)
    ax_trace.plot(fc[mm], co[: len(fc)][mm], color="#333333", lw=0.8)
    ax_trace.set_ylabel("co-burst\nfrac", fontsize=6)
    ax_trace.set_xlabel("time (s)")
    ax_trace.set_xlim(x0, x1)
    ax.set_title("A  HMM burst posteriors", loc="left", fontweight="bold")


def _panel_umap(ax, tr, seed: int = 42) -> None:
    import umap

    X = np.asarray(tr.feature_matrix, dtype=float)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[sd == 0] = 1.0
    Z = np.nan_to_num((X - mu) / sd)
    emb = umap.UMAP(**_UMAP_KW).fit_transform(Z)
    labels = np.asarray(tr.hdbscan_labels)
    burst_clusters = set(getattr(tr, "burst_labels", []) or [])
    is_burst = np.array([lab in burst_clusters for lab in labels])
    # fixed point size for both classes so burst vs rest is read by colour, not
    # size; single shared legend is added once by the caller.
    ax.scatter(emb[~is_burst, 0], emb[~is_burst, 1], s=3, c=_NOISE_GREY,
               alpha=0.6, lw=0)
    ax.scatter(emb[is_burst, 0], emb[is_burst, 1], s=3, c=_BURST_RED,
               alpha=0.85, lw=0)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_modulation(ax, tidy) -> None:
    metric = "burst_modulation_index"
    per_well = tidy.groupby(["well_uid"]).agg(
        arm=("canonical_group", lambda s: next((g for g in s if g != "Control"), "Control")),
        val=(metric, "median")).reset_index()
    arms = ordered_groups(per_well.arm.unique())
    data = [per_well.loc[per_well.arm == a, "val"].dropna().to_numpy() for a in arms]
    for i, (a, d) in enumerate(zip(arms, data)):
        if len(d) >= 8:
            bp = ax.boxplot(d, positions=[i], widths=0.6, patch_artist=True,
                            showfliers=False, zorder=1)
            for box in bp["boxes"]:
                box.set(facecolor=group_color(a), alpha=0.35, lw=0.8)
            for med in bp["medians"]:
                med.set(color=group_color(a), lw=1.2)
        jitter = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.3
        ax.scatter(np.full(len(d), i) + jitter, d, s=8, color=group_color(a),
                   alpha=0.8, lw=0, zorder=3)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=40, ha="right", fontsize=6)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title("D  Burst modulation index (cohort)", loc="left", fontweight="bold")
def _posterior_img(ax, tr, t_win=(0.0, 60.0)):
    """Compact per-unit P(burst) heatmap over a time window (no trace/colorbar)."""
    bins = np.asarray(tr.bins)
    ncols = tr.posterior_matrix.shape[1]
    centers = (bins[:-1] + bins[1:]) / 2 if len(bins) == ncols + 1 else bins[:ncols]
    m = (centers >= t_win[0]) & (centers <= t_win[1])
    P = tr.posterior_matrix[:, m]
    order = np.argsort(-P.mean(axis=1))
    ax.imshow(P[order], aspect="auto", cmap="magma", vmin=0, vmax=1,
              extent=[float(centers[m][0]), float(centers[m][-1]), 0, P.shape[0]],
              origin="lower", interpolation="nearest")
    ax.set_xlabel("time (s)", fontsize=7); ax.set_ylabel("unit", fontsize=7)
    ax.tick_params(labelsize=6)


_MOD_ARMS = ["Control", "IVH_Early", "IVH_Late"]
