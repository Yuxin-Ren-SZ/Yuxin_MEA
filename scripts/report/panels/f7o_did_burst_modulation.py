"""F7o — Burst modulation index: difference-in-differences vs Control."""
from __future__ import annotations

from ..panel_base import panel_main
from ._did import MIXED_YLABEL, did_build, did_caption

NAME = "F7o_did_burst_modulation"
FIGSIZE = (2.6, 2.5)
#: Its own family: the modulation index is a single pre-specified ML read-out,
#: not one of the six STTC graph metrics, so it is corrected on its own.
FAMILY = ["burst_modulation_index"]
CAPTION = did_caption(
    "Change in the burst modulation index (how sharply bursting departs from a "
    "Poisson process; higher means crisper bursts). This is a single "
    "pre-specified read-out and is corrected as its own family, so here q equals p."
)

build = did_build("burst_modulation_index", FAMILY, ylabel=MIXED_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
