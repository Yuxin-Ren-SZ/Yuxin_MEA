"""F5d — Spikes per burst: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import RATIO_YLABEL, MIXED_YLABEL, did_build, did_caption

NAME = "F5d_did_spikes_per_burst"
FIGSIZE = (2.5, 2.4)
CAPTION = did_caption(
    "Change in spikes per network burst (how much the network fires per burst)."
)

build = did_build("nb_spikes_per_burst_mean", D.F5_METRICS, ylabel=RATIO_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
