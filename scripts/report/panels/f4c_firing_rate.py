"""F4c — Median firing rate vs treatment day, with the per-unit distribution."""
from __future__ import annotations

import numpy as np

from .. import panel_data as D
from ..panel_base import panel_main
from ..raster import FULL_WINDOW, load_spikes
from ..report_style import MUTED, group_color
from ..trajectory import plot_trajectory
from . import _f4_common as C

NAME = "F4c_firing_rate"
FIGSIZE = (3.8, 3.4)
CAPTION = (
    "Single-neuron firing rate across maturation, aligned on treatment day (τ), "
    "not culture age — chips differ in absolute DIV. Top: per-arm median of the "
    "well-median firing rate (each well contributes one value per τ bin, so "
    "recordings are not counted as replicates); shaded band = 95% bootstrap CI "
    "across wells. Dashed verticals and the tinted span mark the experiment "
    "window, treatment day τ0 to τ+21. Bottom: distribution of individual unit "
    "firing rates in one representative Control well immediately before treatment "
    "versus 14 days later, showing that the whole population shifts rather than a "
    "few units dominating the median."
)
METRIC = "median_firing_rate"


def _unit_rate_strip(ax, ctx):
    """Per-unit firing rates, pre vs post, for the F4 example Control well.

    Drawn as its own strip rather than an inset: the trajectory ribbons fill the
    axes, so an inset would sit on top of the data it is meant to support.
    """
    try:
        pre, post, wuid, _n = C.pair(ctx)
    except RuntimeError:
        ax.set_visible(False)
        return
    for row, colour, lab in ((pre, MUTED, f"τ{int(pre.tau):+d}"),
                             (post, group_color("Control"), f"τ{int(post.tau):+d}")):
        spikes = load_spikes(ctx.analysis_root, row)
        rates = np.array([len(np.asarray(s, float)) / FULL_WINDOW
                          for s in spikes.values()], float)
        rates = rates[rates > 0]
        if not len(rates):
            continue
        ax.hist(np.log10(rates), bins=28, histtype="step", lw=1.1, color=colour,
                label=f"{lab}  (n={len(rates)} units)", density=True)
    ax.set_xlabel("log₁₀ single-unit firing rate (Hz)", fontsize=6.5)
    ax.set_ylabel("density", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    ax.annotate(f"one Control well · {wuid.split('|')[0]}", xy=(0.99, 0.95),
                xycoords="axes fraction", ha="right", va="top",
                fontsize=5.5, color=MUTED)


def build_fig(fig, ctx):
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.55)
    plot_trajectory(fig.add_subplot(gs[0]), ctx.main, METRIC, D.MAIN_ARMS, legend=True)
    _unit_rate_strip(fig.add_subplot(gs[1]), ctx)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
