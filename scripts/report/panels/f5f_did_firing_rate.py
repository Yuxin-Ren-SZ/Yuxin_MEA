"""F5f — Median firing rate: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import RATIO_YLABEL, MIXED_YLABEL, did_build, did_caption

NAME = "F5f_did_firing_rate"
FIGSIZE = (2.5, 2.4)
CAPTION = did_caption(
    "Change in median single-unit firing rate."
)

build = did_build("median_firing_rate", D.F5_METRICS, ylabel=RATIO_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
