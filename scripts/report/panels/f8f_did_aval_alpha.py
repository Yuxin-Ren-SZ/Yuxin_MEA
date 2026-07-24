"""F8f — Avalanche duration exponent alpha: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import RATIO_YLABEL, MIXED_YLABEL, did_build, did_caption

NAME = "F8f_did_aval_alpha"
FIGSIZE = (2.5, 2.4)
CAPTION = did_caption(
    "Change in the avalanche duration exponent α (slope of the duration distribution)."
)

build = did_build("aval_alpha", D.F8_METRICS, ylabel=RATIO_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
