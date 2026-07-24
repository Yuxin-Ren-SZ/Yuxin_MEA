"""F7q — Directed transfer-entropy graphs, one representative well per group."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._f7_common import TAU_WINDOW, group_strip

NAME = "F7q_te_graph"
FIGSIZE = (7.4, 1.9)
CAPTION = (
    "Directed interactions on the electrode plane for one representative well per "
    "group, {lo}–{hi} days after treatment. Dots are units; arrows are the "
    "strongest transfer-entropy edges (top 15%), pointing from the unit whose "
    "past predicts the other's future. Caveat: transfer entropy computed on "
    "binary spike trains is inflated by network-burst co-activation, so these "
    "graphs are dense by construction and their edge count should not be read as "
    "a connectivity measure — the interpretable directed read-outs are degree "
    "asymmetry and flow hierarchy in F7s. Illustrative single wells, not a group "
    "statistic."
).format(lo=TAU_WINDOW[0], hi=TAU_WINDOW[1])


def _draw(ax, ctx, row):
    from ..fig7b_node_directed import _panel_directed_graph

    _panel_directed_graph(ax, ctx.analysis_root, row)
    ax.set_title("")                 # composed-figure title; group_strip sets its own


def build_fig(fig, ctx):
    return group_strip(fig, ctx, D.MAIN_ARMS, _draw)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
