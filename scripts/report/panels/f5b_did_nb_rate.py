"""F5b — Network-burst rate: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import RATIO_YLABEL, MIXED_YLABEL, did_build, did_caption

NAME = "F5b_did_nb_rate"
FIGSIZE = (2.5, 2.4)
CAPTION = did_caption(
    "Change in network-burst rate (how often the whole network bursts)."
)

build = did_build("nb_rate", D.F5_METRICS, ylabel=RATIO_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
