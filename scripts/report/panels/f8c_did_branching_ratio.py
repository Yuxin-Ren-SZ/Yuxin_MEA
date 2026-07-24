"""F8c — Branching ratio: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import RATIO_YLABEL, MIXED_YLABEL, did_build, did_caption

NAME = "F8c_did_branching_ratio"
FIGSIZE = (2.5, 2.4)
CAPTION = did_caption(
    "Change in the branching ratio (MR estimator); m = 1 is the critical value, so a negative change is a move toward sub-critical dynamics."
)

build = did_build("branching_ratio_mr", D.F8_METRICS, ylabel=MIXED_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
