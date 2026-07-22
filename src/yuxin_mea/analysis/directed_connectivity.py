"""Directed / effective connectivity via transfer entropy (TE) on spike trains.

STTC gives a symmetric (undirected) graph; TE adds **direction** — TE(X→Y)
measures how much X's past reduces uncertainty about Y's future beyond Y's own
past (Schreiber 2000). For binned binary spike trains with history length 1 this
is a fast plug-in estimate; a source-delay sweep takes the peak TE (delayed TE,
Ito et al. 2011). Significance is assessed against a circular-shift null.

Compute is O(n²) in units, so the unit set is capped to the ``max_units`` most
active units per well (directed effective networks are usually inferred on the
active core). Directed graph scalars: mean TE, edge density, degree asymmetry,
reciprocity, flow hierarchy, causal density.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class DirectedError(ValueError):
    pass


@dataclass
class DirectedConfig:
    bin_s: float = 0.005          # binarising bin width (5 ms)
    delays: tuple = (1, 2, 3)     # source-delay sweep (bins); TE = peak over delays
    max_units: int = 64           # cap the active-unit core (O(n^2) TE)
    min_units: int = 5
    min_spikes_per_unit: int = 20
    n_shuffle: int = 20           # circular-shift null
    thresh_percentile: float = 99.0   # strict: co-bursting inflates raw TE
    seed: int = 0

    @classmethod
    def from_task_params(cls, p: dict) -> "DirectedConfig":
        f = {k: p[k] for k in cls.__dataclass_fields__ if k in p}
        return cls(**f)


@dataclass
class DirectedResults:
    te_matrix: np.ndarray          # (n, n) directed TE (row=source, col=target)
    adjacency: np.ndarray          # bool significance mask
    unit_ids: np.ndarray
    metrics: dict
    diagnostics: dict


def _binarize(spike_times: dict, bin_s: float, cfg: DirectedConfig):
    """Return (binary matrix [n_units, n_bins], unit_ids) for the active core."""
    items = [(u, np.asarray(s, float)) for u, s in spike_times.items()
             if len(s) >= cfg.min_spikes_per_unit]
    if len(items) < cfg.min_units:
        raise DirectedError(f"only {len(items)} eligible units (< {cfg.min_units})")
    # keep the most active units (cap O(n^2))
    items.sort(key=lambda kv: -len(kv[1]))
    items = items[: cfg.max_units]
    t0 = min(s[0] for _, s in items)
    t1 = max(s[-1] for _, s in items)
    nbins = max(2, int(np.ceil((t1 - t0) / bin_s)))
    B = np.zeros((len(items), nbins), dtype=np.int8)
    uids = []
    for i, (u, s) in enumerate(items):
        idx = np.clip(((s - t0) / bin_s).astype(int), 0, nbins - 1)
        B[i, idx] = 1
        uids.append(int(u))
    return B, np.array(uids)


def _te_pair(x: np.ndarray, y: np.ndarray, delay: int) -> float:
    """Plug-in transfer entropy TE(x→y), binary, history 1, source delay."""
    if delay < 1 or delay >= len(y) - 1:
        return 0.0
    yn = y[delay + 1:]        # y_{t+1}
    yo = y[delay:-1]          # y_t
    xs = x[: -delay - 1]      # x_{t-delay}
    m = min(len(yn), len(yo), len(xs))
    yn, yo, xs = yn[:m], yo[:m], xs[:m]
    state = 4 * yn + 2 * yo + xs               # 0..7
    c = np.bincount(state, minlength=8).astype(float)
    N = c.sum()
    if N == 0:
        return 0.0
    p = c / N
    # marginals over the 3 binary vars indexed as bit2=yn, bit1=yo, bit0=xs
    te = 0.0
    for s in range(8):
        if p[s] <= 0:
            continue
        yn_b, yo_b, xs_b = (s >> 2) & 1, (s >> 1) & 1, s & 1
        p_yo_xs = sum(p[ss] for ss in range(8)
                      if ((ss >> 1) & 1) == yo_b and (ss & 1) == xs_b)
        p_yo = sum(p[ss] for ss in range(8) if ((ss >> 1) & 1) == yo_b)
        p_yn_yo = sum(p[ss] for ss in range(8)
                      if ((ss >> 2) & 1) == yn_b and ((ss >> 1) & 1) == yo_b)
        num = p[s] / p_yo_xs if p_yo_xs > 0 else 0.0          # p(yn|yo,xs)
        den = p_yn_yo / p_yo if p_yo > 0 else 0.0             # p(yn|yo)
        if num > 0 and den > 0:
            te += p[s] * np.log2(num / den)
    return float(max(te, 0.0))


def _te_matrix(B: np.ndarray, cfg: DirectedConfig) -> np.ndarray:
    n = B.shape[0]
    TE = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            TE[i, j] = max(_te_pair(B[i], B[j], d) for d in cfg.delays)
    return TE


def _significance(B, TE, cfg) -> tuple:
    """Circular-shift null → effective (bias-corrected) TE + high-percentile mask.

    Binned spike trains share network-wide co-bursting, which inflates raw TE for
    almost every pair. Subtracting the shuffle-null mean gives *effective* TE
    (the excess over what shared statistics predict), and a strict percentile
    threshold keeps only clearly directed pairs.
    """
    rng = np.random.default_rng(cfg.seed)
    n = B.shape[0]
    null = []
    for _ in range(cfg.n_shuffle):
        shifts = rng.integers(1, B.shape[1], size=n)
        Bs = np.array([np.roll(B[i], shifts[i]) for i in range(n)])
        for _ in range(min(200, n * (n - 1))):
            i, j = rng.integers(0, n), rng.integers(0, n)
            if i != j:
                null.append(max(_te_pair(Bs[i], B[j], d) for d in cfg.delays))
    null = np.asarray(null) if null else np.array([0.0])
    null_mean = float(null.mean())
    thr = float(np.percentile(null, cfg.thresh_percentile))
    te_eff = np.clip(TE - null_mean, 0.0, None)   # effective TE
    np.fill_diagonal(te_eff, 0.0)
    adj = (TE > thr)
    np.fill_diagonal(adj, False)
    return adj, thr, te_eff


def _directed_metrics(TE, adj) -> dict:
    import networkx as nx
    n = adj.shape[0]
    n_edges = int(adj.sum())
    if n_edges == 0:
        return {k: float("nan") for k in
                ("mean_te", "te_edge_density", "degree_asymmetry",
                 "reciprocity", "flow_hierarchy", "causal_density")}
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                G.add_edge(i, j, weight=float(TE[i, j]))
    indeg = np.array([G.in_degree(i) for i in range(n)])
    outdeg = np.array([G.out_degree(i) for i in range(n)])
    try:
        reciprocity = float(nx.reciprocity(G) or 0.0)
    except Exception:  # noqa: BLE001
        reciprocity = float("nan")
    try:
        flow = float(nx.flow_hierarchy(G))
    except Exception:  # noqa: BLE001
        flow = float("nan")
    return {
        "mean_te": float(TE[adj].mean()),
        "te_edge_density": n_edges / (n * (n - 1)),
        "degree_asymmetry": float(np.mean(np.abs(indeg - outdeg))),
        "reciprocity": reciprocity,
        "flow_hierarchy": flow,
        "causal_density": n_edges / (n * (n - 1)),
    }


def compute_directed(spike_times: dict,
                     config: DirectedConfig | None = None) -> DirectedResults:
    cfg = config or DirectedConfig()
    B, uids = _binarize(spike_times, cfg.bin_s, cfg)
    TE = _te_matrix(B, cfg)
    adj, thr, te_eff = _significance(B, TE, cfg)
    TE = te_eff                              # report effective (bias-corrected) TE
    metrics = _directed_metrics(TE, adj)
    diag = {"n_units": int(B.shape[0]), "n_bins": int(B.shape[1]),
            "bin_s": cfg.bin_s, "delays": list(cfg.delays),
            "thresh_value": thr, "n_shuffle": cfg.n_shuffle}
    return DirectedResults(TE, adj, uids, metrics, diag)


def empty_directed_results(reason: str = "") -> DirectedResults:
    nan = float("nan")
    metrics = {k: nan for k in ("mean_te", "te_edge_density", "degree_asymmetry",
                                "reciprocity", "flow_hierarchy", "causal_density")}
    return DirectedResults(np.zeros((0, 0)), np.zeros((0, 0), bool),
                           np.array([]), metrics, {"reason": reason})


def write_directed(results: DirectedResults, output_dir) -> None:
    import json
    from pathlib import Path

    import pandas as pd
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "te_matrix.npy", results.te_matrix)
    np.save(out / "unit_ids.npy", results.unit_ids)
    # significant directed edges
    rows = []
    if results.adjacency.size:
        ii, jj = np.where(results.adjacency)
        for i, j in zip(ii, jj):
            rows.append(dict(source=int(results.unit_ids[i]),
                             target=int(results.unit_ids[j]),
                             te=float(results.te_matrix[i, j])))
    pd.DataFrame(rows, columns=["source", "target", "te"]).to_parquet(
        out / "directed_edges.parquet")
    (out / "directed_metrics.json").write_text(json.dumps(results.metrics))
    (out / "diagnostics.json").write_text(json.dumps(results.diagnostics))
