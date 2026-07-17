"""S4 — Cross-group network-burst metrics at matched developmental stage.

The treatment-response axis (F5) is the confirmatory analysis; this cross-
sectional view is supplementary (demoted per review: the paired design is more
sensitive and the cross-sectional comparison is chip-confounded). Wells are
compared in a fixed DIV window so groups are matched for maturation. Unit of
analysis = physical well (well_uid). Mann-Whitney vs Control, BH-FDR; boxes only
for arms with ≥8 wells, strips for smaller/exploratory arms.
"""
from __future__ import annotations

import numpy as np

from . import stats as S
from .report_style import METRIC_LABELS, group_color, ordered_groups, save_fig

DIV_WINDOW = (18, 24)
METRICS = ["nb_rate", "nb_duration_mean", "nb_spikes_per_burst_mean",
           "nb_ibi_mean", "median_firing_rate"]


def _stars(q: float) -> str:
    if np.isnan(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def _panel(ax, matched, metric: str) -> None:
    mw = S.mwu_vs_control(matched, metric).set_index("arm")
    arms = ordered_groups(matched.arm.unique())
    for i, arm in enumerate(arms):
        d = matched.loc[matched.arm == arm, metric].dropna().to_numpy()
        if len(d) == 0:
            continue
        col = group_color(arm)
        if len(d) >= 8:
            bp = ax.boxplot(d, positions=[i], widths=0.6, patch_artist=True,
                            showfliers=False, zorder=1)
            bp["boxes"][0].set(facecolor=col, alpha=0.30, lw=0.8)
            bp["medians"][0].set(color=col, lw=1.2)
        jit = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.3
        ax.scatter(np.full(len(d), i) + jit, d, s=7, color=col, alpha=0.85,
                   lw=0, zorder=3)
        if arm != "Control" and arm in mw.index:
            s = _stars(mw.loc[arm, "q_bh"])
            if s:
                ax.text(i, ax.get_ylim()[1], s, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=40, ha="right", fontsize=6)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))


def build_s4(tidy):
    import matplotlib.pyplot as plt

    matched = S.matched_div_wells(tidy, *DIV_WINDOW, metrics=METRICS)
    n = len(METRICS)
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 3.4))
    for ax, m in zip(axes, METRICS):
        _panel(ax, matched, m)
    fig.suptitle(
        f"Figure S4 — Cross-group metrics at DIV {DIV_WINDOW[0]}-{DIV_WINDOW[1]} "
        f"(well_uid unit; * = q<0.05 vs Control)", fontsize=9, y=1.03)
    fig.tight_layout()
    return fig, matched


def render(tidy, figure_root):
    fig, matched = build_s4(tidy)
    paths = save_fig(fig, "S4_crossgroup_matched_div", figure_root, subdir="supp")
    return paths, matched
