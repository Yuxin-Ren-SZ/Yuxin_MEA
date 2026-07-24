"""F7i — Clustering coefficient: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import RATIO_YLABEL, MIXED_YLABEL, did_build, did_caption

NAME = "F7i_did_clustering"
FIGSIZE = (2.5, 2.4)
CAPTION = did_caption(
    "Change in the clustering coefficient (how often a unit's partners are themselves connected)."
)

build = did_build("clustering_coeff", D.F7_METRICS, ylabel=RATIO_YLABEL)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
