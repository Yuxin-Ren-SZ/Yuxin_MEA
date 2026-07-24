"""F7g — Mean STTC: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import RATIO_YLABEL, MIXED_YLABEL, did_build, did_caption

NAME = "F7g_did_mean_sttc"
FIGSIZE = (2.5, 2.4)
CAPTION = did_caption(
    "Change in mean spike-time tiling coefficient (average pairwise functional coupling strength)."
)

build = did_build("mean_sttc", D.F7_METRICS, ylabel=MIXED_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
