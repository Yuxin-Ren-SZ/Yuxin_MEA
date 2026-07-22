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

from . import load as L
from .report_style import (
    METRIC_LABELS, group_color, ordered_groups, save_fig,
)

# UMAP params mirror the pipeline (ml_burst_detection config), n_components=2 for viz.
_UMAP_KW = dict(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)
_BURST_RED = "#d62728"
_NOISE_GREY = "#c8c8c8"


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
    ax.scatter(emb[~is_burst, 0], emb[~is_burst, 1], s=2, c=_NOISE_GREY,
               alpha=0.5, lw=0, label="baseline bins")
    ax.scatter(emb[is_burst, 0], emb[is_burst, 1], s=3, c=_BURST_RED,
               alpha=0.8, lw=0, label="burst bins")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=6, loc="best", markerscale=2)
    ax.set_title("B  Feature-space embedding", loc="left", fontweight="bold")


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


def build_f6(tidy, analysis_root):
    import matplotlib.pyplot as plt

    row = _pick_example(tidy)
    tr = L.load_ml_trace(analysis_root, row)

    fig = plt.figure(figsize=(7.6, 6.4))
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1.0, 0.28, 1.3],
        width_ratios=[1, 2, 1], hspace=0.5, wspace=0.05)
    # Panel A: heatmap + trace stacked, sharing x; thin colorbar column at right
    # so the colorbar never shrinks the heatmap out of alignment with the trace.
    gsA = gs[0, :].subgridspec(1, 2, width_ratios=[1, 0.02], wspace=0.02)
    gsT = gs[1, :].subgridspec(1, 2, width_ratios=[1, 0.02], wspace=0.02)
    ax_post = fig.add_subplot(gsA[0, 0])
    cax = fig.add_subplot(gsA[0, 1])
    ax_trace = fig.add_subplot(gsT[0, 0], sharex=ax_post)
    fig.add_subplot(gsT[0, 1]).set_visible(False)  # keep trace width == heatmap
    ax_umap = fig.add_subplot(gs[2, 1])            # centred square UMAP

    _panel_posterior(ax_post, ax_trace, cax, tr)
    _panel_umap(ax_umap, tr)
    ax_umap.set_box_aspect(1)

    tag = f"{row.well_uid.split('|')[0]} {row.well_name} DIV{int(row.DIV)} ({row.canonical_group})"
    fig.suptitle(f"Figure 6 — ML burst characterization   ·   example well: {tag}",
                 fontsize=9, y=1.0)
    return fig, {"example": row.well_uid}


def build_f6_modulation(tidy):
    """Panel D as a standalone cohort figure (kept separate: whole-cohort scope)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    _panel_modulation(ax, tidy)
    fig.tight_layout()
    return fig


def render(tidy, analysis_root, figure_root):
    fig, meta = build_f6(tidy, analysis_root)
    paths = save_fig(fig, "F6_ml_characterization", figure_root, subdir="main")
    figd = build_f6_modulation(tidy)
    paths += save_fig(figd, "F6d_burst_modulation", figure_root, subdir="main")
    return paths, meta
