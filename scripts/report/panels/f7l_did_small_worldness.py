"""F7l — Small-worldness: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import RATIO_YLABEL, MIXED_YLABEL, did_build, did_caption

NAME = "F7l_did_small_worldness"
FIGSIZE = (2.5, 2.4)
CAPTION = did_caption(
    "Change in small-worldness (clustering relative to a random graph with the same path length)."
)

build = did_build("small_worldness", D.F7_METRICS, ylabel=RATIO_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
