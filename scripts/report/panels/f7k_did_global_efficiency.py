"""F7k — Global efficiency: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import RATIO_YLABEL, MIXED_YLABEL, did_build, did_caption

NAME = "F7k_did_global_efficiency"
FIGSIZE = (2.5, 2.4)
CAPTION = did_caption(
    "Change in global efficiency (mean inverse shortest path — how easily activity can reach across the network)."
)

build = did_build("global_efficiency", D.F7_METRICS, ylabel=RATIO_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
