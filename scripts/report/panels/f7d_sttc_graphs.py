"""F7d — Undirected STTC connectivity graphs, one representative well per group."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._f7_common import TAU_WINDOW, group_strip, plot_sttc_graph

NAME = "F7d_sttc_graphs"
FIGSIZE = (7.4, 1.9)
CAPTION = (
    "Functional connectivity graphs of one representative well per group, "
    "{lo}–{hi} days after treatment. Each dot is a curated unit at its position on "
    "the electrode plane; each line is a statistically significant spike-time "
    "tiling coefficient (STTC) edge, with line opacity proportional to coupling "
    "strength. All three panels are drawn on the same fixed chip rectangle with "
    "equal aspect, so node positions and edge lengths are directly comparable "
    "between groups. STTC is the pairwise correlation measure of Cutts & Eglen "
    "(2014), which is insensitive to firing rate — so a denser graph means more "
    "coordination, not simply more spikes. The representative well is the one "
    "nearest its group's median coupling strength; illustrative, not a group "
    "statistic (the quantitative comparison is the difference-in-differences "
    "panels)."
).format(lo=TAU_WINDOW[0], hi=TAU_WINDOW[1])


def build_fig(fig, ctx):
    return group_strip(fig, ctx, D.MAIN_ARMS,
                       lambda ax, c, row: plot_sttc_graph(ax, c, row))


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
