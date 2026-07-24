"""F7r — Node-level graph metrics: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import DID_METHOD, did_grid_build

NAME = "F7r_node_metrics_did"
FIGSIZE = (6.8, 4.2)
CAPTION = (
    "Node-level graph structure: how each unit sits inside the network, rather "
    "than the network's global properties. Hub and leaf fractions count units "
    "with unusually many or few connections; the participation coefficient "
    "measures how much a unit's edges spread across communities rather than "
    "staying inside one; betweenness counts how often a unit lies on the shortest "
    "path between others; the rich-club coefficient asks whether the "
    "best-connected units preferentially connect to each other; assortativity "
    "asks whether units connect to partners of similar degree. " + DID_METHOD
)

build_fig = did_grid_build(D.F7_NODE_METRICS, ncols=3)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
