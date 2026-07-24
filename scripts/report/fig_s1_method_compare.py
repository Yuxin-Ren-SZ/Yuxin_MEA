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

    _ML = OKABE_ITO["blue"]           # one accent for ML, one for traditional
    _TRAD = OKABE_ITO["vermillion"]

    # A. count agreement (log-log, identity); flag method-only detections.
    ax = axes[0]
    x = df.trad_count.to_numpy(float)
    y = df.ml_count.to_numpy(float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    trad_only = ok & (x > 0) & (y == 0)
    ml_only = ok & (y > 0) & (x == 0)
    both = ok & ~trad_only & ~ml_only
    ax.scatter(x[both] + 1, y[both] + 1, s=5, alpha=0.3, lw=0, color=_ML,
               label="both detect")
    if trad_only.any():
        ax.scatter(x[trad_only] + 1, y[trad_only] + 1, s=12, lw=0, color=_TRAD,
                   label="traditional-only")
    if ml_only.any():
        ax.scatter(x[ml_only] + 1, y[ml_only] + 1, s=12, lw=0,
                   color=OKABE_ITO["orange"], label="ML-only")
    lim = [1, np.nanmax([x, y]) + 5]
    ax.plot(lim, lim, "--", color="0.5", lw=0.8)
    ax.annotate("identity", xy=(lim[1], lim[1]), ha="right", va="bottom",
                fontsize=5.5, color="0.5")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("traditional burst count (+1)")
    ax.set_ylabel("ML burst count (+1)")
    ax.legend(fontsize=5.5, loc="upper left")
    ax.set_title("A  Burst-count agreement", loc="left", fontweight="bold")

    # B. Bland-Altman of duration, with labelled bias & 95% LoA lines.
    ax = axes[1]
    td, md = df.trad_duration_s.to_numpy(float), df.ml_duration_s.to_numpy(float)
    mean = (td + md) / 2
    diff = md - td
    ok = ~np.isnan(mean) & ~np.isnan(diff)
    ax.scatter(mean[ok], diff[ok], s=5, alpha=0.3, lw=0, color=_ML)
    mu, sd = np.nanmean(diff[ok]), np.nanstd(diff[ok])
    for yv, ls, lab in [(mu, "-", "bias"), (mu + 1.96 * sd, "--", "+95% LoA"),
                        (mu - 1.96 * sd, "--", "−95% LoA")]:
        ax.axhline(yv, ls=ls, color="0.4", lw=0.8)
        ax.annotate(f"{lab} {yv:+.2f}s", xy=(0.99, yv),
                    xycoords=("axes fraction", "data"), ha="right",
                    va="bottom", fontsize=5.5, color="0.35")
    ax.annotate("spread widens with duration\n(proportional bias)", xy=(0.02, 0.03),
                xycoords="axes fraction", ha="left", va="bottom", fontsize=5.5,
                color="0.4", style="italic")
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
