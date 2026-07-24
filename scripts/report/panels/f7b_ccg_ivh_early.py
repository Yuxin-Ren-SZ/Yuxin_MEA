"""F7b — Cross-correlograms, IVH_Early example."""
from __future__ import annotations

from ..panel_base import panel_main
from ._f7_common import CCG_CAPTION, TAU_WINDOW, ccg_grid

NAME = "F7b_ccg_ivh_early"
FIGSIZE = (4.2, 2.9)
GROUP = "IVH_Early"
CAPTION = CCG_CAPTION.format(group=GROUP.replace("_", " "),
                             win="{}-{}".format(*TAU_WINDOW))


def build_fig(fig, ctx):
    return ccg_grid(fig, ctx, GROUP, n=4, ncols=2)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
