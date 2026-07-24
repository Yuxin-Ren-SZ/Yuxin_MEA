"""F7s — Directed (transfer-entropy) metrics: difference-in-differences vs Control."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import DID_METHOD, did_grid_build

NAME = "F7s_directed_metrics_did"
FIGSIZE = (5.4, 4.2)
CAPTION = (
    "Directed connectivity from transfer entropy, which asks how much one unit's "
    "past reduces uncertainty about another's future. Mean TE is the average "
    "directed coupling; degree asymmetry contrasts out- and in-degree; "
    "reciprocity is the fraction of edges that run both ways; flow hierarchy is "
    "the fraction that are not part of a cycle, i.e. how feed-forward the network "
    "is. Caveat: transfer entropy between binary spike trains is inflated by "
    "network-burst co-activation, so edge counts are dense and the interpretable "
    "read-outs are the asymmetry and hierarchy measures, not the raw density. "
    + DID_METHOD
)

build_fig = did_grid_build(D.F7_DIR_METRICS, ncols=2)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
