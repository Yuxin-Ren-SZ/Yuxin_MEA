"""Functional connectivity via the Spike Time Tiling Coefficient (STTC).

New computation (see ``doc/spatial_connectivity_plan.md`` §B). STTC (Cutts &
Eglen, *J Neurosci* 2014) is the standard MEA pairwise-coupling measure; it is
insensitive to firing-rate differences, which makes it comparable across wells.

Pipeline per well:

  1. ``sttc_matrix`` — symmetric ``(n_units, n_units)`` STTC at the primary tiling
     window ``dt`` (plus a ``dt`` sweep for diagnostics).
  2. ``sttc_significance`` — a circular-shuffle null; edges above the null's
     percentile are kept, so edge density is comparable across wells.
  3. ``graph_metrics`` — thresholded adjacency → networkx topology scalars.
  4. ``edges`` — significant edges paired with inter-unit distance for the
     distance-vs-connectivity decay curve and the electrode-plane graph render.

House style mirrors :mod:`yuxin_mea.analysis.burst_detector`: a frozen
``ConnectivityConfig`` (+ ``from_task_params``), a ``ConnectivityResults``
container, a ``compute_connectivity`` entry over the
``dict[unit_id -> np.ndarray(seconds)]`` spike-times contract, and a
``ConnectivityError`` for insufficient input.

STTC definition used (identical trains → 1, disjoint → ≤0)::

    STTC = ½ · [ (P_A − T_B)/(1 − P_A·T_B) + (P_B − T_A)/(1 − P_B·T_A) ]

    T_X = fraction of the recording window covered by ±dt tiles around X's spikes
    P_A = fraction of A's spikes lying within ±dt of any B spike (and vice versa)
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, fields as _dc_fields
from pathlib import Path

import numpy as np
import pandas as pd


class ConnectivityError(ValueError):
    """Raised when spike data is insufficient to compute connectivity."""


@dataclass(frozen=True)
class ConnectivityConfig:
    """Tunable parameters for STTC connectivity."""

    dt: float = 0.02  # primary tiling window (s) = 20 ms
    dt_sweep: tuple = (0.005, 0.01, 0.02, 0.05)  # 5/10/20/50 ms
    n_shuffle: int = 100
    thresh_percentile: float = 95.0  # keep edges above this pct of the null
    min_units: int = 3
    min_spikes_per_unit: int = 5
    random_state: int = 42
    n_jobs: int = 1  # intra-well joblib; keep 1 (pipeline parallelism = CLI --jobs)

    @classmethod
    def from_task_params(cls, params: dict) -> "ConnectivityConfig":
        """Build a config from a task-params dict; cast by default type, ignore extras."""
        kwargs: dict = {}
        for f in _dc_fields(cls):
            if f.name not in params:
                continue
            v = params[f.name]
            if f.name == "dt_sweep":
                kwargs[f.name] = tuple(float(x) for x in v)
            elif isinstance(f.default, bool):
                kwargs[f.name] = bool(v)
            elif isinstance(f.default, int):
                kwargs[f.name] = int(v)
            elif isinstance(f.default, float):
                kwargs[f.name] = float(v)
            elif isinstance(f.default, str):
                kwargs[f.name] = str(v)
            else:
                kwargs[f.name] = v
        return cls(**kwargs)


@dataclass
class ConnectivityResults:
    """Structured output of :func:`compute_connectivity`.

    ``sttc_matrix`` : ``(n, n)`` STTC at the primary dt (unit order = ``unit_ids``).
    ``sttc_sweep``  : ``{dt_s: (n, n) matrix}`` for each swept window.
    ``adjacency``   : ``(n, n)`` boolean significance mask (thresholded null).
    ``graph_metrics`` : well-level topology scalars (schema = :func:`graph_metrics`).
    ``edges``       : significant edges ``[u, v, sttc, dist_um]`` (upper triangle).
    ``unit_ids``    : row/col order for the matrices.
    ``diagnostics`` : dt_primary, n_units, n_edges, thresh_method, n_shuffle, T.
    """

    sttc_matrix: np.ndarray
    sttc_sweep: dict
    adjacency: np.ndarray
    graph_metrics: dict
    edges: pd.DataFrame
    unit_ids: list
    diagnostics: dict


# ---------------------------------------------------------------------------
# STTC primitives
# ---------------------------------------------------------------------------
def _tiling_fraction(spikes: np.ndarray, dt: float, t_start: float, t_end: float) -> float:
    """Fraction of [t_start, t_end] covered by the union of ±dt tiles around spikes."""
    T = t_end - t_start
    if T <= 0 or spikes.size == 0:
        return 0.0
    lo = np.clip(spikes - dt, t_start, t_end)
    hi = np.clip(spikes + dt, t_start, t_end)
    order = np.argsort(lo, kind="mergesort")
    lo, hi = lo[order], hi[order]
    covered = 0.0
    cur_lo, cur_hi = lo[0], hi[0]
    for i in range(1, lo.size):
        if lo[i] <= cur_hi:
            if hi[i] > cur_hi:
                cur_hi = hi[i]
        else:
            covered += cur_hi - cur_lo
            cur_lo, cur_hi = lo[i], hi[i]
    covered += cur_hi - cur_lo
    return float(covered / T)


def _proportion_within(a: np.ndarray, b: np.ndarray, dt: float) -> float:
    """Fraction of spikes in ``a`` with at least one ``b`` spike within ±dt."""
    if a.size == 0 or b.size == 0:
        return 0.0
    idx = np.searchsorted(b, a)
    left = np.clip(idx - 1, 0, b.size - 1)
    right = np.clip(idx, 0, b.size - 1)
    dmin = np.minimum(np.abs(a - b[left]), np.abs(a - b[right]))
    return float(np.mean(dmin <= dt))


def _sttc_from(pa: float, pb: float, ta: float, tb: float) -> float:
    """Combine the two half-terms, guarding the 0/0 at P·T == 1."""
    denom1 = 1.0 - pa * tb
    denom2 = 1.0 - pb * ta
    term1 = (pa - tb) / denom1 if abs(denom1) > 1e-12 else 0.0
    term2 = (pb - ta) / denom2 if abs(denom2) > 1e-12 else 0.0
    return 0.5 * (term1 + term2)


def sttc_pair(
    a: np.ndarray,
    b: np.ndarray,
    dt: float,
    T: float,
    t_start: float = 0.0,
) -> float:
    """STTC for two spike trains over a recording window of length ``T``.

    ``a``/``b`` are spike times (s). The window is ``[t_start, t_start + T]``.
    Returns 0.0 when either train is empty (STTC undefined).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    t_end = t_start + T
    ta = _tiling_fraction(a, dt, t_start, t_end)
    tb = _tiling_fraction(b, dt, t_start, t_end)
    pa = _proportion_within(a, b, dt)
    pb = _proportion_within(b, a, dt)
    return _sttc_from(pa, pb, ta, tb)


def _spike_bounds(spikes: dict) -> tuple[float, float]:
    lo, hi = np.inf, -np.inf
    for st in spikes.values():
        st = np.asarray(st, dtype=float)
        if st.size:
            lo = min(lo, float(st[0]) if st[0] <= st[-1] else float(st.min()))
            hi = max(hi, float(st.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 0.0
    return lo, hi


def sttc_matrix(
    spikes: dict,
    dt: float = 0.02,
    T: float | None = None,
    t_start: float | None = None,
    unit_ids: list | None = None,
) -> np.ndarray:
    """Symmetric ``(n, n)`` STTC matrix. Diagonal = 1.

    ``T``/``t_start`` default to the global spike span across all trains
    (max − min), matching ``burst_detector``'s recording-length convention.
    """
    if unit_ids is None:
        unit_ids = list(spikes.keys())
    arrs = [np.sort(np.asarray(spikes[u], dtype=float)) for u in unit_ids]
    n = len(arrs)
    W = np.eye(n, dtype=float)
    if n < 2:
        return W
    if t_start is None or T is None:
        lo, hi = _spike_bounds(spikes)
        if t_start is None:
            t_start = lo
        if T is None:
            T = hi - lo
    t_end = t_start + T
    # Precompute per-unit tiling fractions once (O(n)).
    tf = np.array([_tiling_fraction(a, dt, t_start, t_end) for a in arrs])
    for i in range(n):
        ai = arrs[i]
        for j in range(i + 1, n):
            bj = arrs[j]
            pa = _proportion_within(ai, bj, dt)
            pb = _proportion_within(bj, ai, dt)
            w = _sttc_from(pa, pb, tf[i], tf[j])
            W[i, j] = W[j, i] = w
    return W


# ---------------------------------------------------------------------------
# Significance (circular-shuffle null)
# ---------------------------------------------------------------------------
def _circular_shift(arrs: list, T: float, t_start: float, rng) -> list:
    out = []
    for a in arrs:
        if a.size == 0:
            out.append(a)
            continue
        shift = rng.uniform(0.0, T)
        shifted = t_start + np.mod((a - t_start) + shift, T)
        out.append(np.sort(shifted))
    return out


def sttc_significance(
    spikes: dict,
    dt: float,
    T: float,
    t_start: float = 0.0,
    n_shuffle: int = 100,
    thresh_percentile: float = 95.0,
    random_state: int = 42,
    unit_ids: list | None = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, float]:
    """Circular-shuffle null → (adjacency mask, scalar threshold).

    Returns a boolean ``(n, n)`` adjacency where the observed STTC exceeds the
    ``thresh_percentile`` of a pooled circular-jitter null, plus that scalar
    threshold. A pooled (global) null keeps this O(n_shuffle · n² · spikes)
    tractable and yields comparable density across wells.
    """
    if unit_ids is None:
        unit_ids = list(spikes.keys())
    arrs = [np.sort(np.asarray(spikes[u], dtype=float)) for u in unit_ids]
    n = len(arrs)
    adjacency = np.zeros((n, n), dtype=bool)
    if n < 2 or T <= 0:
        return adjacency, float("nan")
    t_end = t_start + T

    def _one_shuffle(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sh = _circular_shift(arrs, T, t_start, rng)
        tf = np.array([_tiling_fraction(a, dt, t_start, t_end) for a in sh])
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                pa = _proportion_within(sh[i], sh[j], dt)
                pb = _proportion_within(sh[j], sh[i], dt)
                vals.append(_sttc_from(pa, pb, tf[i], tf[j]))
        return np.asarray(vals, dtype=float)

    seeds = [random_state + k for k in range(max(1, n_shuffle))]
    if n_jobs != 1:
        from joblib import Parallel, delayed
        null_chunks = Parallel(n_jobs=n_jobs)(
            delayed(_one_shuffle)(s) for s in seeds
        )
    else:
        null_chunks = [_one_shuffle(s) for s in seeds]
    null = np.concatenate(null_chunks) if null_chunks else np.array([0.0])
    thresh = float(np.percentile(null, thresh_percentile))

    W = sttc_matrix(spikes, dt=dt, T=T, t_start=t_start, unit_ids=unit_ids)
    iu = np.triu_indices(n, k=1)
    sig = W[iu] > thresh
    adjacency[iu[0][sig], iu[1][sig]] = True
    adjacency |= adjacency.T
    return adjacency, thresh


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------
def graph_metrics(W: np.ndarray, adjacency: np.ndarray | None = None,
                  thresh: float | None = None) -> dict:
    """Topology scalars from an STTC matrix + a boolean adjacency.

    ``adjacency`` (preferred) is the significance mask from
    :func:`sttc_significance`. If omitted, edges are ``W > thresh`` (or ``W > 0``).
    Small-worldness uses a fast analytic Erdős–Rényi reference
    (C_rand ≈ density, L_rand ≈ ln N / ln <k>); a degree-preserving null is a
    future refinement (the doc's aspiration) — noted, not silently claimed.
    """
    import networkx as nx

    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    empty = {
        "mean_sttc": float("nan"), "edge_density": 0.0, "mean_degree": 0.0,
        "clustering_coeff": 0.0, "modularity": float("nan"),
        "global_efficiency": 0.0, "small_worldness": float("nan"),
        "n_nodes": int(n), "n_edges": 0,
    }
    if n < 2:
        return empty
    if adjacency is None:
        thr = 0.0 if thresh is None else float(thresh)
        adjacency = (W > thr)
        np.fill_diagonal(adjacency, False)
    adjacency = np.triu(adjacency, k=1)
    iu = np.where(adjacency)
    n_edges = int(iu[0].size)
    if n_edges == 0:
        return {**empty, "mean_sttc": float("nan")}

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v in zip(iu[0].tolist(), iu[1].tolist()):
        G.add_edge(int(u), int(v), weight=float(W[u, v]))

    edge_w = np.array([W[u, v] for u, v in zip(iu[0], iu[1])], dtype=float)
    density = nx.density(G)
    clustering = nx.average_clustering(G)
    try:
        comms = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, comms)
    except Exception:
        modularity = float("nan")
    efficiency = nx.global_efficiency(G)

    # Analytic small-worldness (Humphries-Gurney σ with an ER reference).
    mean_deg = 2.0 * n_edges / n
    if mean_deg > 1.0 and density > 0 and clustering > 0:
        c_rand = density
        l_rand = np.log(n) / np.log(mean_deg)
        # Observed path length on the largest connected component.
        comps = list(nx.connected_components(G))
        gc = G.subgraph(max(comps, key=len)) if comps else G
        l_obs = nx.average_shortest_path_length(gc) if gc.number_of_nodes() > 1 else np.nan
        if np.isfinite(l_obs) and l_rand > 0 and c_rand > 0 and l_obs > 0:
            small_world = (clustering / c_rand) / (l_obs / l_rand)
        else:
            small_world = float("nan")
    else:
        small_world = float("nan")

    return {
        "mean_sttc": float(edge_w.mean()),
        "edge_density": float(density),
        "mean_degree": float(mean_deg),
        "clustering_coeff": float(clustering),
        "modularity": float(modularity),
        "global_efficiency": float(efficiency),
        "small_worldness": float(small_world),
        "n_nodes": int(n),
        "n_edges": n_edges,
    }


# ---------------------------------------------------------------------------
# Entry point + writer
# ---------------------------------------------------------------------------
def compute_connectivity(
    spike_times: dict,
    positions: dict | None = None,
    config: ConnectivityConfig | None = None,
) -> ConnectivityResults:
    """Compute STTC connectivity for one well.

    ``spike_times`` : ``{unit_id -> np.ndarray(seconds)}`` (curated).
    ``positions``   : optional ``{unit_id -> (x, y) µm}`` for the edge distance
    column and the electrode-plane render. Missing → ``dist_um`` is NaN.
    """
    cfg = config or ConnectivityConfig()
    # Keep units with enough spikes; STTC on 1-2 spike trains is meaningless.
    unit_ids = [
        u for u in spike_times
        if np.asarray(spike_times[u]).size >= cfg.min_spikes_per_unit
    ]
    if len(unit_ids) < cfg.min_units:
        raise ConnectivityError(
            f"only {len(unit_ids)} units with ≥{cfg.min_spikes_per_unit} spikes; "
            f"need ≥{cfg.min_units} for a connectivity graph"
        )
    lo, hi = _spike_bounds({u: spike_times[u] for u in unit_ids})
    t_start, T = lo, hi - lo
    if T <= 0:
        raise ConnectivityError("degenerate recording length (T <= 0)")

    # Primary matrix + dt sweep.
    W = sttc_matrix(spike_times, dt=cfg.dt, T=T, t_start=t_start, unit_ids=unit_ids)
    sweep = {}
    for d in cfg.dt_sweep:
        sweep[float(d)] = (
            W if abs(d - cfg.dt) < 1e-12
            else sttc_matrix(spike_times, dt=d, T=T, t_start=t_start, unit_ids=unit_ids)
        )

    adjacency, thresh = sttc_significance(
        spike_times, dt=cfg.dt, T=T, t_start=t_start,
        n_shuffle=cfg.n_shuffle, thresh_percentile=cfg.thresh_percentile,
        random_state=cfg.random_state, unit_ids=unit_ids, n_jobs=cfg.n_jobs,
    )
    gm = graph_metrics(W, adjacency=adjacency)

    # Significant-edge table with inter-unit distance.
    iu = np.where(np.triu(adjacency, k=1))
    rows = []
    for a, b in zip(iu[0].tolist(), iu[1].tolist()):
        ua, ub = unit_ids[a], unit_ids[b]
        if positions is not None and ua in positions and ub in positions:
            pa, pb = np.asarray(positions[ua], float), np.asarray(positions[ub], float)
            dist = float(np.hypot(*(pa - pb)))
        else:
            dist = float("nan")
        rows.append({"u": ua, "v": ub, "sttc": float(W[a, b]), "dist_um": dist})
    edges = pd.DataFrame(rows, columns=["u", "v", "sttc", "dist_um"])

    diagnostics = {
        "dt_primary": float(cfg.dt),
        "dt_sweep": [float(d) for d in cfg.dt_sweep],
        "n_units": int(len(unit_ids)),
        "n_edges": int(len(edges)),
        "thresh_method": "circular_shuffle_percentile",
        "thresh_percentile": float(cfg.thresh_percentile),
        "thresh_value": float(thresh),
        "n_shuffle": int(cfg.n_shuffle),
        "T": float(T),
    }
    return ConnectivityResults(
        sttc_matrix=W,
        sttc_sweep=sweep,
        adjacency=adjacency,
        graph_metrics=gm,
        edges=edges,
        unit_ids=list(unit_ids),
        diagnostics=diagnostics,
    )


def empty_connectivity_results(
    config: ConnectivityConfig | None = None, reason: str = ""
) -> ConnectivityResults:
    """A COMPLETE-but-empty bundle for wells too sparse for a connectivity graph.

    Mirrors the burst detectors returning zero events (COMPLETE, not FAILED) on
    sparse wells: empty matrices, graph scalars all NaN/0, no edges.
    """
    cfg = config or ConnectivityConfig()
    W = np.zeros((0, 0), dtype=float)
    return ConnectivityResults(
        sttc_matrix=W,
        sttc_sweep={float(d): W for d in cfg.dt_sweep},
        adjacency=np.zeros((0, 0), dtype=bool),
        graph_metrics=graph_metrics(W),
        edges=pd.DataFrame(columns=["u", "v", "sttc", "dist_um"]),
        unit_ids=[],
        diagnostics={
            "dt_primary": float(cfg.dt),
            "dt_sweep": [float(d) for d in cfg.dt_sweep],
            "n_units": 0, "n_edges": 0,
            "thresh_method": "circular_shuffle_percentile",
            "thresh_percentile": float(cfg.thresh_percentile),
            "thresh_value": float("nan"),
            "n_shuffle": int(cfg.n_shuffle), "T": 0.0,
            "empty_reason": reason,
        },
    )


def _atomic_json_write(data: dict, dest: Path) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, dest)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_connectivity(results: ConnectivityResults, output_dir: Path) -> None:
    """Persist a ``ConnectivityResults`` bundle.

    Layout::

        sttc_matrix.npy      (n, n) at primary dt
        sttc_sweep.npz       one array per dt (keys "dt_<ms>") + unit_ids
        graph_metrics.json   well-level scalars
        edges.parquet        significant edges [u, v, sttc, dist_um]
        diagnostics.json     dt_primary, n_units, n_edges, thresh_method, n_shuffle
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "sttc_matrix.npy", results.sttc_matrix)
    sweep_arrays = {
        f"dt_{int(round(d * 1000))}ms": m for d, m in results.sttc_sweep.items()
    }
    sweep_arrays["unit_ids"] = np.asarray(results.unit_ids)
    np.savez(output_dir / "sttc_sweep.npz", **sweep_arrays)
    results.edges.to_parquet(output_dir / "edges.parquet")
    _atomic_json_write(results.graph_metrics, output_dir / "graph_metrics.json")
    _atomic_json_write(results.diagnostics, output_dir / "diagnostics.json")
