"""F5c — Network-burst duration: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import RATIO_YLABEL, MIXED_YLABEL, did_build, did_caption

NAME = "F5c_did_nb_duration"
FIGSIZE = (2.5, 2.4)
CAPTION = did_caption(
    "Change in mean network-burst duration (how long each burst lasts)."
)

build = did_build("nb_duration_mean", D.F5_METRICS, ylabel=RATIO_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
