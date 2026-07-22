"""Node-level graph metrics for the STTC functional-connectivity graph.

Computed from the **significant-edge graph already on disk**
(``connectivity_data/.../connectivity/edges.parquet`` = ``[u, v, sttc, dist_um]``
plus the full unit-id set from ``sttc_sweep.npz``) — so this needs **no STTC
recompute**, only a rollup pass over existing artifacts.

Methods follow the Brain Connectivity Toolbox (Rubinov & Sporns 2010) and the
node-cartography roles of Guimerà & Amaral (2005), as used by the HD-MEA network
pipeline MEA-NAP (Cell Reports Methods 2024):

* per-node: degree, weighted strength, betweenness centrality, clustering,
  within-module degree z-score ``z``, participation coefficient ``P``,
  local efficiency;
* node roles: **hub** if ``z > 2.5``; among hubs, **connector** (``P > 0.62``,
  Guimerà R6/R7) vs **provincial** (``P ≤ 0.62``, R5); **leaf** if degree ≤ 1;
* well-level scalars summarising the node distribution (hub/leaf fractions,
  betweenness, participation, degree dispersion, assortativity, rich-club).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

Z_HUB = 2.5            # within-module degree z-score above which a node is a hub
P_CONNECTOR = 0.62     # participation coefficient splitting connector/provincial hubs

_NODE_COLUMNS = [
    "degree", "strength", "betweenness", "clustering", "local_efficiency",
    "module", "z_within", "participation", "role",
]


@dataclass
class NodeMetricsResults:
    node_metrics: pd.DataFrame   # index = unit_id
    scalars: dict
    diagnostics: dict


def _empty(reason: str, n_nodes: int = 0) -> NodeMetricsResults:
    scalars = {k: float("nan") for k in (
        "hub_fraction", "leaf_fraction", "mean_betweenness", "max_betweenness",
        "participation_mean", "degree_mean", "degree_cv", "degree_skew",
        "assortativity", "rich_club",
    )}
    scalars.update(n_nodes=int(n_nodes), n_edges=0, n_hubs=0,
                   n_connector_hubs=0, n_provincial_hubs=0, n_leaves=0)
    return NodeMetricsResults(
        pd.DataFrame(columns=_NODE_COLUMNS), scalars, {"reason": reason})


def _participation(G, node, modules) -> float:
    """Guimerà participation coefficient P_i = 1 - Σ_m (k_i(m)/k_i)^2."""
    k = G.degree(node)
    if k == 0:
        return 0.0
    per_mod: dict = {}
    for nb in G.neighbors(node):
        m = modules[nb]
        per_mod[m] = per_mod.get(m, 0) + 1
    return float(1.0 - sum((c / k) ** 2 for c in per_mod.values()))


def _within_module_z(G, modules) -> dict:
    """Within-module degree z-score per node (Guimerà & Amaral 2005)."""
    # group nodes by module, compute within-module degree
    mod_nodes: dict = {}
    for n, m in modules.items():
        mod_nodes.setdefault(m, []).append(n)
    z = {}
    for m, nodes in mod_nodes.items():
        nodeset = set(nodes)
        kin = {n: sum(1 for nb in G.neighbors(n) if nb in nodeset) for n in nodes}
        vals = np.array(list(kin.values()), dtype=float)
        mu, sd = vals.mean(), vals.std()
        for n in nodes:
            z[n] = float((kin[n] - mu) / sd) if sd > 1e-12 else 0.0
    return z


def _role(z: float, p: float, degree: int) -> str:
    if degree <= 1:
        return "leaf"
    if z > Z_HUB:
        return "connector_hub" if p > P_CONNECTOR else "provincial_hub"
    return "peripheral"


def compute_node_metrics(edges: pd.DataFrame, node_ids) -> NodeMetricsResults:
    """Node-level metrics for one well.

    ``edges`` : DataFrame with columns ``u, v`` (unit ids) and ``sttc`` (weight).
    ``node_ids`` : the full unit-id set (isolated nodes included so fractions use
    the right denominator).
    """
    import networkx as nx

    node_ids = [int(x) for x in np.asarray(node_ids).ravel()]
    n_nodes = len(node_ids)
    if n_nodes < 3:
        return _empty("fewer than 3 nodes", n_nodes)

    G = nx.Graph()
    G.add_nodes_from(node_ids)
    if edges is not None and len(edges):
        for _, e in edges.iterrows():
            u, v = int(e["u"]), int(e["v"])
            if u in G and v in G and u != v:
                G.add_edge(u, v, weight=float(e.get("sttc", 1.0)))
    n_edges = G.number_of_edges()
    if n_edges == 0:
        return _empty("no edges", n_nodes)

    # communities (weighted Louvain)
    import community as community_louvain
    modules = community_louvain.best_partition(G, weight="weight", random_state=0)

    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    betw = nx.betweenness_centrality(G, weight=None, normalized=True)
    clustering = nx.clustering(G)
    z_within = _within_module_z(G, modules)

    rows = []
    for n in node_ids:
        p = _participation(G, n, modules)
        deg = degree.get(n, 0)
        z = z_within.get(n, 0.0)
        # local efficiency of the node's neighbourhood
        try:
            sub = G.subgraph(list(G.neighbors(n)))
            le = nx.global_efficiency(sub) if sub.number_of_nodes() > 1 else 0.0
        except Exception:  # noqa: BLE001
            le = 0.0
        rows.append(dict(
            unit_id=n, degree=deg, strength=float(strength.get(n, 0.0)),
            betweenness=float(betw.get(n, 0.0)), clustering=float(clustering.get(n, 0.0)),
            local_efficiency=float(le), module=int(modules.get(n, -1)),
            z_within=z, participation=p, role=_role(z, p, deg),
        ))
    node_df = pd.DataFrame(rows).set_index("unit_id")

    deg_arr = node_df.degree.to_numpy(float)
    roles = node_df.role.value_counts().to_dict()
    n_hubs = int(roles.get("connector_hub", 0) + roles.get("provincial_hub", 0))
    try:
        assort = float(nx.degree_assortativity_coefficient(G))
    except Exception:  # noqa: BLE001
        assort = float("nan")
    # rich-club at k = median degree (normalised coefficient); guard small graphs
    try:
        kmed = int(max(1, np.median(deg_arr)))
        rc = nx.rich_club_coefficient(G, normalized=False)
        rich_club = float(rc.get(kmed, np.nan))
    except Exception:  # noqa: BLE001
        rich_club = float("nan")

    scalars = dict(
        n_nodes=n_nodes, n_edges=n_edges,
        n_hubs=n_hubs,
        n_connector_hubs=int(roles.get("connector_hub", 0)),
        n_provincial_hubs=int(roles.get("provincial_hub", 0)),
        n_leaves=int(roles.get("leaf", 0)),
        hub_fraction=n_hubs / n_nodes,
        leaf_fraction=roles.get("leaf", 0) / n_nodes,
        mean_betweenness=float(node_df.betweenness.mean()),
        max_betweenness=float(node_df.betweenness.max()),
        participation_mean=float(node_df.participation.mean()),
        degree_mean=float(deg_arr.mean()),
        degree_cv=float(deg_arr.std() / deg_arr.mean()) if deg_arr.mean() > 0 else float("nan"),
        degree_skew=_skew(deg_arr),
        assortativity=assort,
        rich_club=rich_club,
    )
    diag = dict(n_modules=len(set(modules.values())), z_hub=Z_HUB,
                p_connector=P_CONNECTOR)
    return NodeMetricsResults(node_df, scalars, diag)


def _skew(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    mu, sd = x.mean(), x.std()
    return float(np.mean(((x - mu) / sd) ** 3)) if sd > 1e-12 else float("nan")
