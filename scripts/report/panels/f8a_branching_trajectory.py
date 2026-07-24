"""F8a — Branching ratio vs treatment day."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ..report_style import MUTED
from ..trajectory import plot_trajectory

NAME = "F8a_branching_trajectory"
FIGSIZE = (3.6, 2.7)
CAPTION = (
    "Branching ratio of neuronal avalanches across treatment time, estimated with "
    "the multistep-regression (MR) estimator, which is robust to the subsampling "
    "inherent in recording a fraction of the network. A branching ratio of 1 "
    "(horizontal dashed line) is the critical value: each spike triggers on "
    "average one more, so activity neither dies out nor runs away. Line = per-arm "
    "median across wells (one value per well per treatment-day bin), shaded band "
    "= 95% bootstrap CI across wells. Vertical dashed lines and the tinted span "
    "mark the experiment window, treatment day τ0 to τ+21."
)


def build(ax, ctx):
    plot_trajectory(ax, ctx.main, "branching_ratio_mr", D.MAIN_ARMS, ref=1.0,
                    ylabel="branching ratio (MR)", legend=True)
    ax.annotate("m = 1: critical", xy=(0.99, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=5.5, color=MUTED)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
