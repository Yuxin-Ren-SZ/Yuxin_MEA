"""S3 — Representative network-burst rasters per condition.

One example well-recording per treatment arm, drawn from the matched DIV window
so the qualitative burst phenotype is compared at equal maturation. Reuses the
curated-spike raster panel from F4.
"""
from __future__ import annotations

import numpy as np

from .fig4_activity_dev import _panel_raster
from .report_style import caption, group_color, ordered_groups, save_fig

DIV_WINDOW = (18, 24)


def _pick_per_group(tidy):
    """For each arm, the in-window recording with the most curated units."""
    win = tidy[(tidy.DIV >= DIV_WINDOW[0]) & (tidy.DIV <= DIV_WINDOW[1])]
    win = win.dropna(subset=["n_curated"])
    picks = {}
    for arm in ordered_groups(win.canonical_group.unique()):
        sub = win[win.canonical_group == arm]
        if sub.empty:
            continue
        picks[arm] = sub.sort_values("n_curated", ascending=False).iloc[0]
    return picks


def build_s3(tidy, analysis_root):
    import matplotlib.pyplot as plt

    picks = _pick_per_group(tidy)
    arms = list(picks)
    ncols = 2
    nrows = (len(arms) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0, 1.7 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for ax, arm in zip(flat, arms):
        r = picks[arm]
        nb = "?" if np.isnan(r.nb_count) else int(r.nb_count)
        _panel_raster(ax, analysis_root, r,
                      f"{arm} — {r.well_uid.split('|')[0]} {r.well_name} "
                      f"DIV{int(r.DIV)} (nb={nb})")
        ax.title.set_color(group_color(arm))
    for ax in flat[len(arms):]:
        ax.set_visible(False)
    fig.suptitle(f"Figure S3 — Representative rasters at DIV "
                 f"{DIV_WINDOW[0]}-{DIV_WINDOW[1]}", fontsize=9, y=1.0)
    fig.tight_layout()
    caption(fig,
        f"Representative spike rasters, one per group, from wells recorded at a "
        f"matched developmental stage (DIV {DIV_WINDOW[0]}-{DIV_WINDOW[1]}) so "
        f"the groups are comparable for maturation. Each row = a unit, each tick "
        f"= a spike; synchronous vertical bands = network bursts. Illustrative "
        f"examples of the raw activity behind the quantitative figures.")
    return fig, picks


def render(tidy, analysis_root, figure_root):
    fig, picks = build_s3(tidy, analysis_root)
    return save_fig(fig, "S3_representative_rasters", figure_root, subdir="supp"), picks
