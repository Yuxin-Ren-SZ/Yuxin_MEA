"""S4 — Cross-group network-burst metrics at matched developmental stage.

The treatment-response axis (F5) is the confirmatory analysis; this cross-
sectional view is supplementary (demoted per review: the paired design is more
sensitive and the cross-sectional comparison is chip-confounded). Wells are
compared in a fixed DIV window so groups are matched for maturation.

**Biological replicate = chip** (`sample_id`), well = technical replicate. Each
arm is drawn as its **per-chip means** (bold, ~3 points) with a mean bar and the
wells faint behind. No significance stars: the comparison is cross-sectional and
chip-confounded with only ~3 chips per arm, so it is **descriptive** — the paired
chip-level DiD in F5-F8 carries the inference.
"""
from __future__ import annotations

import numpy as np

from . import stats as S
from .report_style import (
    METRIC_LABELS, caption, group_color, ordered_groups, save_fig)

DIV_WINDOW = (18, 24)
METRICS = ["nb_rate", "nb_duration_mean", "nb_spikes_per_burst_mean",
           "nb_ibi_mean", "median_firing_rate"]


def _panel(ax, matched, metric: str, n1_arms=(), banner=False) -> None:
    from .forest import _short_group

    arms = ordered_groups(matched.arm.unique())
    cm = matched.groupby(["chip", "arm"], as_index=False)[metric].mean()
    rng = np.random.default_rng(0)
    for i, arm in enumerate(arms):
        w = matched.loc[matched.arm == arm, metric].dropna().to_numpy()   # wells
        cp = cm.loc[cm.arm == arm, metric].dropna().to_numpy()            # chips
        col = group_color(arm)
        if len(w):
            jit = (rng.random(len(w)) - 0.5) * 0.30
            ax.scatter(np.full(len(w), i) + jit, w, s=6, color=col, alpha=0.16,
                       lw=0, zorder=2)
        if len(cp):
            ax.hlines(float(np.mean(cp)), i - 0.28, i + 0.28, color=col, lw=1.8,
                      zorder=4)
            jit = (rng.random(len(cp)) - 0.5) * 0.16
            ax.scatter(np.full(len(cp), i) + jit, cp, s=26, facecolor=col,
                       edgecolor="white", linewidth=0.5, zorder=5)
        if arm in n1_arms:                 # flag single-chip arms explicitly
            ax.annotate("n=1", xy=(i, 0.01), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=5.5, color="0.45")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([_short_group(a) for a in arms], rotation=30, ha="right",
                       fontsize=6)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    if banner:                             # persistent caveat, in the panel
        ax.set_title("descriptive · chip-confounded · no significance test",
                     loc="left", fontsize=6.5, color="#A83B00", fontstyle="italic")


def build_s4(tidy):
    import matplotlib.pyplot as plt

    matched = S.matched_div_wells(tidy, *DIV_WINDOW, metrics=METRICS)
    # arms represented by a single chip (e.g. NPH / AraC) — flagged as n=1.
    chips_per_arm = matched.groupby("arm").chip.nunique()
    n1_arms = set(chips_per_arm[chips_per_arm <= 1].index)
    n = len(METRICS)
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 3.6))
    for j, (ax, m) in enumerate(zip(axes, METRICS)):
        _panel(ax, matched, m, n1_arms=n1_arms, banner=(j == 0))
    fig.suptitle(
        f"Figure S4 — Cross-group metrics at DIV {DIV_WINDOW[0]}-{DIV_WINDOW[1]} "
        f"(bold = per-chip means, faint = wells; descriptive, chip-confounded)",
        fontsize=9, y=1.03)
    fig.tight_layout()
    caption(fig,
        f"Cross-group network-burst metrics at a matched developmental stage "
        f"(DIV {DIV_WINDOW[0]}-{DIV_WINDOW[1]}), one panel per metric. Each arm's "
        f"wells are shown faint; the bold points are its per-chip means (the "
        f"biological replicates) with a horizontal mean bar. This cross-sectional "
        f"view is supplementary and chip-confounded (arms differ by chip as well "
        f"as treatment), so it carries no significance test — the paired "
        f"within-well difference-in-differences in F5-F8 is the inferential "
        f"analysis. Unit of replication: chip (n≈3 per arm; AraC/NPH are "
        f"single-chip).")
    return fig, matched


def render(tidy, figure_root):
    fig, matched = build_s4(tidy)
    paths = save_fig(fig, "S4_crossgroup_matched_div", figure_root, subdir="supp")
    return paths, matched
