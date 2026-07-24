"""F7j — Modularity: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import RATIO_YLABEL, MIXED_YLABEL, did_build, did_caption

NAME = "F7j_did_modularity"
FIGSIZE = (2.5, 2.4)
CAPTION = did_caption(
    "Change in Newman weighted modularity Q of the STTC graph (how cleanly the network splits into communities)."
)

build = did_build("modularity", D.F7_METRICS, ylabel=MIXED_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
