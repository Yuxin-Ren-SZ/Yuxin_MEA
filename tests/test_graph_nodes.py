"""Tests for node-level graph metrics (analysis/graph_nodes.py)."""
import numpy as np
import pandas as pd

from yuxin_mea.analysis.graph_nodes import compute_node_metrics


def _edges(pairs, sttc=0.2):
    return pd.DataFrame([{"u": u, "v": v, "sttc": sttc} for u, v in pairs])


def test_leaf_and_isolated_handling():
    # star: hub 0 connected to 1,2,3,4; node 5 isolated; 6 a leaf off node 1
    edges = _edges([(0, 1), (0, 2), (0, 3), (0, 4), (1, 6)])
    res = compute_node_metrics(edges, node_ids=[0, 1, 2, 3, 4, 5, 6])
    nm = res.node_metrics
    # leaves = degree<=1 : nodes 2,3,4,5(isolated),6
    leaves = set(nm.index[nm.role == "leaf"])
    assert {2, 3, 4, 6}.issubset(leaves)
    assert res.scalars["n_nodes"] == 7
    assert res.scalars["leaf_fraction"] > 0
    # participation defined in [0,1]
    assert nm.participation.between(0, 1).all()


def test_degenerate_small_graph():
    res = compute_node_metrics(_edges([(0, 1)]), node_ids=[0, 1])
    assert res.scalars["n_nodes"] == 2
    assert np.isnan(res.scalars["hub_fraction"])  # <3 nodes -> empty


def test_no_edges():
    res = compute_node_metrics(pd.DataFrame(columns=["u", "v", "sttc"]),
                               node_ids=[0, 1, 2, 3])
    assert res.scalars["n_edges"] == 0
    assert res.diagnostics.get("reason") == "no edges"


def test_two_modules_participation():
    # two triangles (0,1,2) and (3,4,5) joined by one bridge 2-3
    edges = _edges([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)])
    res = compute_node_metrics(edges, node_ids=list(range(6)))
    # bridge nodes (2,3) should have higher participation than triangle-only nodes
    p = res.node_metrics.participation
    assert p.loc[2] > p.loc[0]
    assert res.diagnostics["n_modules"] >= 2
