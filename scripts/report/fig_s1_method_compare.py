"""S1 — ML vs traditional burst detection agreement.

Reuses the pre-computed ``burst_method_comparison/per_well.csv`` (one row per
well-recording). Three panels: count agreement, Bland-Altman of burst duration,
and the per-well event-overlap (IoU) distribution.
"""
from __future__ import annotations

import numpy as np

from . import load as L
from .report_style import OKABE_ITO, caption, save_fig


def build_s1(figure_root):
    import matplotlib.pyplot as plt

    df = L.load_method_compare(figure_root)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))

    # A. count agreement (log-log, identity)
    ax = axes[0]
    x = df.trad_count.to_numpy(float)
    y = df.ml_count.to_numpy(float)
    ax.scatter(x + 1, y + 1, s=5, alpha=0.3, lw=0, color=OKABE_ITO["blue"])
    lim = [1, np.nanmax([x, y]) + 5]
    ax.plot(lim, lim, "--", color="0.5", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("traditional burst count (+1)")
    ax.set_ylabel("ML burst count (+1)")
    ax.set_title("A  Burst-count agreement", loc="left", fontweight="bold")

    # B. Bland-Altman of duration
    ax = axes[1]
    td, md = df.trad_duration_s.to_numpy(float), df.ml_duration_s.to_numpy(float)
    mean = (td + md) / 2
    diff = md - td
    ok = ~np.isnan(mean) & ~np.isnan(diff)
    ax.scatter(mean[ok], diff[ok], s=5, alpha=0.3, lw=0, color=OKABE_ITO["vermillion"])
    mu, sd = np.nanmean(diff[ok]), np.nanstd(diff[ok])
    for yv, ls in [(mu, "-"), (mu + 1.96 * sd, "--"), (mu - 1.96 * sd, "--")]:
        ax.axhline(yv, ls=ls, color="0.4", lw=0.8)
    ax.set_xlabel("mean burst duration (s)")
    ax.set_ylabel("ML − traditional (s)")
    ax.set_title("B  Duration Bland-Altman", loc="left", fontweight="bold")

    # C. IoU distribution
    ax = axes[2]
    iou = df.iou_mean.dropna().to_numpy(float)
    ax.hist(iou, bins=25, color=OKABE_ITO["green"], alpha=0.8)
    ax.axvline(np.median(iou), ls="--", color="0.3", lw=0.9,
               label=f"median {np.median(iou):.2f}")
    ax.set_xlabel("per-well mean event IoU")
    ax.set_ylabel("wells")
    ax.legend(fontsize=6)
    ax.set_title("C  Event overlap", loc="left", fontweight="bold")

    fig.suptitle("Figure S1 — ML vs traditional burst detection", fontsize=9, y=1.02)
    fig.tight_layout()
    caption(fig,
        "Agreement between the ML burst detector (used throughout) and the "
        "traditional threshold method, one point per well. (A) Per-well "
        "network-burst counts, ML vs traditional (dashed = identity). "
        "(B) Bland-Altman of burst duration (difference vs mean; solid = bias, "
        "dashed = 95% limits of agreement). (C) Event overlap (per-burst "
        "intersection-over-union). Method-validation figure, not a group "
        "comparison.")
    return fig


def render(figure_root):
    fig = build_s1(figure_root)
    return save_fig(fig, "S1_method_comparison", figure_root, subdir="supp")
