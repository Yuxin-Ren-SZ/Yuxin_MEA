"""F8d — Distance to criticality (DCC): difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import RATIO_YLABEL, MIXED_YLABEL, did_build, did_caption

NAME = "F8d_did_dcc"
FIGSIZE = (2.5, 2.4)
CAPTION = did_caption(
    "Change in the distance-to-criticality coefficient; larger means further from the critical scaling relation."
)

build = did_build("dcc", D.F8_METRICS, ylabel=MIXED_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
