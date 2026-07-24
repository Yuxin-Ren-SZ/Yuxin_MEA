"""F7e — Spatial activity fields, one representative well per group."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._f7_common import TAU_WINDOW, group_strip, plot_activity_field

NAME = "F7e_activity_fields"
FIGSIZE = (7.4, 1.9)
CAPTION = (
    "Spatial distribution of firing across the electrode plane for one "
    "representative well per group, {lo}–{hi} days after treatment. Brightness is "
    "the smoothed local firing rate; the colour scale is per panel, so these show "
    "*where* activity sits rather than how much of it there is. Every panel is "
    "drawn on the same fixed chip rectangle with equal aspect, so the panels cover "
    "identical physical area and a well whose units occupy only part of the chip "
    "shows a correspondingly small patch instead of being stretched to fill the "
    "frame. Illustrative single wells, not a group statistic."
).format(lo=TAU_WINDOW[0], hi=TAU_WINDOW[1])


def build_fig(fig, ctx):
    return group_strip(fig, ctx, D.MAIN_ARMS,
                       lambda ax, c, row: plot_activity_field(ax, c, row))


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
