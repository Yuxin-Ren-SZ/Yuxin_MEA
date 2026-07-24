"""F7p — Node cartography of one representative well per group."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._f7_common import TAU_WINDOW, group_strip, pick_recording

NAME = "F7p_node_cartography"
FIGSIZE = (7.0, 2.4)
CAPTION = (
    "Where each unit sits in its network, for one representative well per group, "
    "{lo}–{hi} days after treatment. Every point is a unit, placed by its "
    "participation coefficient P (how evenly its connections spread across the "
    "network's communities, x-axis) against its within-module degree z-score (how "
    "well connected it is inside its own community, y-axis) — the Guimerà–Amaral "
    "cartography also used by MEA-NAP. The dashed horizontal line at z = 2.5 is "
    "the hub threshold and the dotted vertical at P = 0.62 separates provincial "
    "hubs, which serve one community, from connector hubs, which bridge several. "
    "Colour marks the resulting role. Illustrative single wells, not a group "
    "statistic."
).format(lo=TAU_WINDOW[0], hi=TAU_WINDOW[1])


def _draw(ax, ctx, row):
    from ..fig7b_node_directed import _panel_cartography

    _panel_cartography(ax, ctx.analysis_root, row, "")


def build_fig(fig, ctx):
    return group_strip(fig, ctx, D.MAIN_ARMS, _draw)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
