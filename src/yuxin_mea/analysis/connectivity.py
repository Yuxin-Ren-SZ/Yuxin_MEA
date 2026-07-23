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
  5. ``compute_ccg`` — a lag-resolved cross-correlogram per significant edge, with
     a partially-hollow-Gaussian baseline (Stark & Abeles, *J Neurosci Methods*
     2009) so the broad network-burst co-activation bump is stripped before any
     peak is called.

STTC is symmetric and lag-free; the CCG is what answers *which* unit leads and at
what latency. Sign convention, fixed everywhere::

    reference = u, target = v; the histogram is of (t_v - t_u).
    A peak at POSITIVE lag means v fires AFTER u, i.e. u leads v.

``edges`` stores only the upper triangle, so this convention is the only thing
making a peak lag interpretable — do not change it without changing the readers.

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
    # ---- cross-correlograms (significant edges only) ----------------------
    ccg_enable: bool = True
    ccg_window: float = 0.1     # +/- half-width (s); dt sweep tops out at 50 ms
    ccg_bin: float = 0.001      # bin width (s) -> 201 bins at the default window
    ccg_hollow_sigma: float = 0.01   # baseline kernel sigma (s)
    ccg_hollow_frac: float = 0.6     # centre-bin hollowing (Stark & Abeles 2009)
    ccg_alpha: float = 0.001         # per-bin Poisson tail threshold
    ccg_syn_lo: float = 0.0008       # causal window low edge (s)
    ccg_syn_hi: float = 0.008        # causal window high edge (s)
    ccg_max_pairs: int = 5000        # cap; over it keep the highest-STTC edges

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
    ``edges``       : significant edges ``[u, v, sttc, dist_um]`` (upper triangle)
    plus the per-edge ``ccg_*`` columns when CCGs ran.
    ``unit_ids``    : row/col order for the matrices.
    ``diagnostics`` : dt_primary, n_units, n_edges, thresh_method, n_shuffle, T.
    ``ccg_counts``/``ccg_baseline`` : ``(n_pairs, n_bins)`` raw counts and the
    hollow-Gaussian baseline. Rows are keyed by ``ccg_pairs`` — **never** by
    position in ``edges``, which can be longer when ``ccg_max_pairs`` trips.
    ``ccg_pairs``   : ``(n_pairs, 2)`` ``(u, v)`` = (reference, target).
    """

    sttc_matrix: np.ndarray
    sttc_sweep: dict
    adjacency: np.ndarray
    graph_metrics: dict
    edges: pd.DataFrame
    unit_ids: list
    diagnostics: dict
    ccg_counts: np.ndarray | None = None
    ccg_baseline: np.ndarray | None = None
    ccg_lags_ms: np.ndarray | None = None
    ccg_pairs: np.ndarray | None = None
    ccg_n_ref_spikes: np.ndarray | None = None
    ccg_n_tgt_spikes: np.ndarray | None = None
    ccg_meta: dict | None = None


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
    """Combine the two half-terms, resolving the 0/0 at P·T == 1.

    Per Cutts & Eglen (2014) / Elephant: when a half-term's denominator vanishes
    (``P·T == 1``, i.e. one train's ±dt window tiles the whole recording) that
    half-term equals 1.0, so STTC = 0.5·(1 + other); both degenerate → 1.0.
    (An earlier version returned 0.0 here, deflating STTC for densely-tiling
    units and dropping their edges.)
    """
    denom1 = 1.0 - pa * tb
    denom2 = 1.0 - pb * ta
    term1 = (pa - tb) / denom1 if abs(denom1) > 1e-12 else 1.0
    term2 = (pb - ta) / denom2 if abs(denom2) > 1e-12 else 1.0
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
# Cross-correlograms
# ---------------------------------------------------------------------------
CCG_COLUMNS = ["ccg_peak_lag_ms", "ccg_peak_z", "ccg_sig", "ccg_syn_dir",
               "ccg_asymmetry"]

_LAM_FLOOR = 1e-6  # keeps z finite in near-empty bins (p-value needs no floor)


def ccg_lags(bin_s: float = 0.001, window_s: float = 0.1) -> np.ndarray:
    """Bin centres (s) for a ±``window_s`` correlogram: symmetric, odd count."""
    n_half = int(round(window_s / bin_s))
    return np.arange(-n_half, n_half + 1, dtype=float) * bin_s


def ccg_pair(
    ref: np.ndarray,
    tgt: np.ndarray,
    bin_s: float = 0.001,
    window_s: float = 0.1,
) -> np.ndarray:
    """Cross-correlogram counts of ``t_tgt − t_ref`` over ±``window_s``.

    A peak at positive lag means ``tgt`` fires *after* ``ref`` (``ref`` leads).
    Bin *centres* are ``ccg_lags(bin_s, window_s)``, so a delta at exactly +5 ms
    lands in the +5 ms bin.

    ``tgt`` **must be sorted** — ``np.searchsorted`` returns garbage (without
    raising) otherwise. :func:`compute_ccg` sorts once per unit before calling.
    """
    ref = np.asarray(ref, dtype=float)
    tgt = np.asarray(tgt, dtype=float)
    n_half = int(round(window_s / bin_s))
    n_bins = 2 * n_half + 1
    if ref.size == 0 or tgt.size == 0:
        return np.zeros(n_bins, dtype=np.int32)

    # Half a bin of slack so the outermost bins collect their full width.
    edge = window_s + 0.5 * bin_s
    lo = np.searchsorted(tgt, ref - edge, side="left")
    hi = np.searchsorted(tgt, ref + edge, side="right")
    deltas = [tgt[a:b] - t for t, a, b in zip(ref.tolist(), lo.tolist(), hi.tolist())
              if b > a]
    if not deltas:
        return np.zeros(n_bins, dtype=np.int32)
    idx = np.rint(np.concatenate(deltas) / bin_s).astype(np.int64) + n_half
    idx = idx[(idx >= 0) & (idx < n_bins)]
    return np.bincount(idx, minlength=n_bins).astype(np.int32)


def _hollow_gaussian(sigma_bins: float, hollow_frac: float) -> np.ndarray:
    """Gaussian kernel with a partially hollowed centre bin, summing to 1.

    Stark & Abeles (2009): hollowing the centre keeps a sharp peak from
    predicting its own baseline, so the slow (network-burst) component survives
    the convolution while a monosynaptic bump does not.
    """
    sigma_bins = max(float(sigma_bins), 1e-3)
    half = max(1, int(round(3.0 * sigma_bins)))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k[half] *= (1.0 - float(hollow_frac))
    s = k.sum()
    return k / s if s > 0 else k


def ccg_stats(
    counts: np.ndarray,
    lags_s: np.ndarray,
    config: "ConnectivityConfig | None" = None,
) -> dict:
    """Hollow-Gaussian baseline + per-bin Poisson test for one correlogram.

    Returns ``baseline`` (same shape as ``counts``) plus the scalars persisted
    per edge: ``peak_lag_ms``, ``peak_z``, ``sig``, ``syn_dir``, ``asymmetry``.

    ``sig``/``syn_dir`` are decided *inside the causal window*
    ``[ccg_syn_lo, ccg_syn_hi]`` on either side of zero — the broad central bump
    that every burst-dominated MEA pair carries is not evidence of anything.
    ``asymmetry`` is ``(Σ⁺ − Σ⁻)/(Σ⁺ + Σ⁻)`` over that window: +1 = u leads v
    exclusively, −1 = the reverse, 0 = symmetric.
    """
    from scipy.stats import poisson

    cfg = config or ConnectivityConfig()
    counts = np.asarray(counts, dtype=float)
    lags_s = np.asarray(lags_s, dtype=float)
    empty = {
        "baseline": np.zeros_like(counts),
        "peak_lag_ms": float("nan"), "peak_z": float("nan"),
        "sig": False, "syn_dir": "", "asymmetry": float("nan"),
    }
    if counts.size == 0 or counts.sum() <= 0:
        return empty

    k = _hollow_gaussian(cfg.ccg_hollow_sigma / cfg.ccg_bin, cfg.ccg_hollow_frac)
    # Normalise by the kernel mass actually available: without this the first and
    # last bins get a baseline biased low purely by truncation.
    denom = np.convolve(np.ones(counts.size), k, mode="same")
    lam = np.convolve(counts, k, mode="same") / np.where(denom > 0, denom, 1.0)
    lam_f = np.maximum(lam, _LAM_FLOOR)
    z = (counts - lam) / np.sqrt(lam_f)
    p = poisson.sf(counts - 1, lam_f)

    ipk = int(np.argmax(z))
    pos = (lags_s >= cfg.ccg_syn_lo) & (lags_s <= cfg.ccg_syn_hi)
    neg = (lags_s <= -cfg.ccg_syn_lo) & (lags_s >= -cfg.ccg_syn_hi)
    sig_pos = bool(np.any(p[pos] < cfg.ccg_alpha)) if pos.any() else False
    sig_neg = bool(np.any(p[neg] < cfg.ccg_alpha)) if neg.any() else False
    if sig_pos and sig_neg:
        zp = float(np.max(z[pos]))
        zn = float(np.max(z[neg]))
        syn_dir = "u->v" if zp >= zn else "v->u"
    elif sig_pos:
        syn_dir = "u->v"
    elif sig_neg:
        syn_dir = "v->u"
    else:
        syn_dir = ""

    s_pos = float(counts[pos].sum())
    s_neg = float(counts[neg].sum())
    total = s_pos + s_neg
    asym = (s_pos - s_neg) / total if total > 0 else float("nan")

    return {
        "baseline": lam,
        "peak_lag_ms": float(lags_s[ipk] * 1000.0),
        "peak_z": float(z[ipk]),
        "sig": sig_pos or sig_neg,
        "syn_dir": syn_dir,
        "asymmetry": asym,
    }


def compute_ccg(
    spike_times: dict,
    edges: pd.DataFrame,
    config: "ConnectivityConfig | None" = None,
) -> dict:
    """Correlograms for every edge in ``edges`` (reference = ``u``, target = ``v``).

    Returns a dict with ``counts``/``baseline`` ``(n_pairs, n_bins)``, ``lags_ms``,
    ``pairs`` ``(n_pairs, 2)``, the per-pair spike counts, a ``per_pair``
    DataFrame (``u``, ``v`` + :data:`CCG_COLUMNS`) and ``meta``.

    When ``len(edges) > ccg_max_pairs`` only the highest-STTC edges are computed,
    so ``n_pairs`` can be *smaller* than ``len(edges)``: join on ``(u, v)``,
    never on row position.
    """
    cfg = config or ConnectivityConfig()
    lags_s = ccg_lags(cfg.ccg_bin, cfg.ccg_window)
    n_bins = int(lags_s.size)
    out = {
        "counts": np.zeros((0, n_bins), dtype=np.int32),
        "baseline": np.zeros((0, n_bins), dtype=np.float32),
        "lags_ms": (lags_s * 1000.0).astype(np.float64),
        "pairs": np.zeros((0, 2)),
        "n_ref_spikes": np.zeros(0, dtype=np.int64),
        "n_tgt_spikes": np.zeros(0, dtype=np.int64),
        "per_pair": pd.DataFrame(columns=["u", "v", *CCG_COLUMNS]),
        "meta": {
            "ccg_enabled": bool(cfg.ccg_enable),
            "ccg_bin_ms": float(cfg.ccg_bin * 1000.0),
            "ccg_window_ms": float(cfg.ccg_window * 1000.0),
            "ccg_syn_lo_ms": float(cfg.ccg_syn_lo * 1000.0),
            "ccg_syn_hi_ms": float(cfg.ccg_syn_hi * 1000.0),
            "ccg_n_pairs": 0,
            "ccg_pairs_capped": 0,
        },
    }
    if not cfg.ccg_enable or edges is None or not len(edges):
        return out

    ed = edges.reset_index(drop=True)
    keep = np.arange(len(ed))
    capped = 0
    if len(ed) > cfg.ccg_max_pairs > 0:
        # Highest-STTC edges win the budget; keep them in table order afterwards.
        rank = np.argsort(-ed["sttc"].to_numpy(dtype=float), kind="mergesort")
        keep = np.sort(rank[: cfg.ccg_max_pairs])
        capped = int(len(ed) - keep.size)

    # Sort every unit's train exactly once. A manually merged unit is two sorted
    # trains concatenated — i.e. NOT sorted — and searchsorted would be silently
    # wrong on it.
    needed = set(ed["u"].tolist()) | set(ed["v"].tolist())
    trains = {
        u: np.sort(np.asarray(spike_times[u], dtype=float))
        for u in needed if u in spike_times
    }

    counts, baseline, pairs, n_ref, n_tgt, rows = [], [], [], [], [], []
    for i in keep.tolist():
        u = ed.at[i, "u"]
        v = ed.at[i, "v"]
        ref = trains.get(u)
        tgt = trains.get(v)
        if ref is None or tgt is None:
            continue
        c = ccg_pair(ref, tgt, bin_s=cfg.ccg_bin, window_s=cfg.ccg_window)
        st = ccg_stats(c, lags_s, cfg)
        counts.append(c)
        baseline.append(np.asarray(st["baseline"], dtype=np.float32))
        pairs.append((u, v))
        n_ref.append(ref.size)
        n_tgt.append(tgt.size)
        rows.append({
            "u": u, "v": v,
            "ccg_peak_lag_ms": st["peak_lag_ms"], "ccg_peak_z": st["peak_z"],
            "ccg_sig": st["sig"], "ccg_syn_dir": st["syn_dir"],
            "ccg_asymmetry": st["asymmetry"],
        })

    if counts:
        out["counts"] = np.vstack(counts).astype(np.int32)
        out["baseline"] = np.vstack(baseline).astype(np.float32)
        out["pairs"] = np.asarray(pairs)
        out["n_ref_spikes"] = np.asarray(n_ref, dtype=np.int64)
        out["n_tgt_spikes"] = np.asarray(n_tgt, dtype=np.int64)
        out["per_pair"] = pd.DataFrame(rows, columns=["u", "v", *CCG_COLUMNS])
    out["meta"]["ccg_n_pairs"] = int(len(counts))
    out["meta"]["ccg_pairs_capped"] = capped
    return out


def attach_ccg(edges: pd.DataFrame, per_pair: pd.DataFrame) -> pd.DataFrame:
    """Left-join the per-pair CCG columns onto ``edges`` on ``(u, v)``.

    Edges dropped by ``ccg_max_pairs`` keep NaN/``pd.NA`` in every ``ccg_*``
    column — they are "not computed", not "not significant".
    """
    if edges is None:
        return edges
    if per_pair is None or not len(per_pair):
        out = edges.copy()
        for c in CCG_COLUMNS:
            out[c] = pd.NA if c in ("ccg_sig", "ccg_syn_dir") else np.nan
    else:
        out = edges.merge(per_pair, on=["u", "v"], how="left")
    out["ccg_sig"] = out["ccg_sig"].astype("boolean")
    return out


def ccg_summary(per_pair: pd.DataFrame) -> dict:
    """Well-level CCG scalars merged into ``graph_metrics.json``.

    ``ccg_flow_asymmetry`` is the mean **absolute** asymmetry over significant
    pairs — signed asymmetry would average to ~0 because the ``(u, v)`` order is
    just the upper-triangle index order, not a direction. 0 = purely symmetric
    (co-activation), 1 = every significant pair is one-directional.

    **``frac_ccg_sig`` has a floor.** Measured on a real 116-unit well (4527
    edges): 43 significant pairs observed vs 8-17 on circularly-shifted trains,
    i.e. ~3-5x enrichment over the empirical null but a false-positive floor
    around 0.2-0.4%. Read differences between wells against that floor, not
    against zero. (The naive binomial expectation over the causal-window bins is
    much higher than the shuffled count because the hollow-Gaussian baseline is
    adaptive — neighbouring bins are not independent Poisson draws.)
    """
    empty = {
        "n_ccg_sig_pairs": 0, "frac_ccg_sig": float("nan"),
        "mean_abs_ccg_lag_ms": float("nan"), "ccg_flow_asymmetry": float("nan"),
    }
    if per_pair is None or not len(per_pair):
        return empty
    sig = per_pair["ccg_sig"].to_numpy(dtype=bool)
    lag = per_pair["ccg_peak_lag_ms"].to_numpy(dtype=float)
    asym = per_pair["ccg_asymmetry"].to_numpy(dtype=float)
    n_sig = int(sig.sum())
    return {
        "n_ccg_sig_pairs": n_sig,
        "frac_ccg_sig": float(sig.mean()),
        "mean_abs_ccg_lag_ms": (
            float(np.nanmean(np.abs(lag[sig]))) if n_sig else float("nan")
        ),
        "ccg_flow_asymmetry": (
            float(np.nanmean(np.abs(asym[sig]))) if n_sig else float("nan")
        ),
    }


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
        # Detect AND score on the same (STTC-weighted) objective.
        comms = nx.community.greedy_modularity_communities(G, weight="weight")
        modularity = nx.community.modularity(G, comms, weight="weight")
    except Exception:
        modularity = float("nan")
    efficiency = nx.global_efficiency(G)

    mean_deg = 2.0 * n_edges / n            # full-graph mean degree (reported)

    # Analytic small-worldness (Humphries-Gurney σ with an ER reference).
    # All four quantities are evaluated on the SAME node set — the largest
    # connected component — so the observed path length is comparable to its ER
    # reference. (Mixing a full-n numerator with a giant-component-only l_obs
    # inflated σ, and the bias differed by graph fragmentation.)
    comps = list(nx.connected_components(G))
    gc = G.subgraph(max(comps, key=len)) if comps else G
    n_gc = gc.number_of_nodes()
    m_gc = gc.number_of_edges()
    mean_deg_gc = (2.0 * m_gc / n_gc) if n_gc else 0.0
    if n_gc > 2 and mean_deg_gc > 1.0:
        clustering_gc = nx.average_clustering(gc)
        c_rand = nx.density(gc)
        l_rand = np.log(n_gc) / np.log(mean_deg_gc)
        l_obs = nx.average_shortest_path_length(gc)
        if (np.isfinite(l_obs) and l_obs > 0 and l_rand > 0
                and c_rand > 0 and clustering_gc > 0):
            small_world = (clustering_gc / c_rand) / (l_obs / l_rand)
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

    # Correlograms for the surviving edges (lag-resolved view of the same pairs).
    ccg = compute_ccg(spike_times, edges, config=cfg)
    edges = attach_ccg(edges, ccg["per_pair"])
    gm.update(ccg_summary(ccg["per_pair"]))

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
        **ccg["meta"],
    }
    return ConnectivityResults(
        sttc_matrix=W,
        sttc_sweep=sweep,
        adjacency=adjacency,
        graph_metrics=gm,
        edges=edges,
        unit_ids=list(unit_ids),
        diagnostics=diagnostics,
        ccg_counts=ccg["counts"],
        ccg_baseline=ccg["baseline"],
        ccg_lags_ms=ccg["lags_ms"],
        ccg_pairs=ccg["pairs"],
        ccg_n_ref_spikes=ccg["n_ref_spikes"],
        ccg_n_tgt_spikes=ccg["n_tgt_spikes"],
        ccg_meta=ccg["meta"],
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
    ccg = compute_ccg({}, None, config=cfg)  # zero-row arrays, full schema
    edges = attach_ccg(
        pd.DataFrame(columns=["u", "v", "sttc", "dist_um"]), ccg["per_pair"],
    )
    return ConnectivityResults(
        sttc_matrix=W,
        sttc_sweep={float(d): W for d in cfg.dt_sweep},
        adjacency=np.zeros((0, 0), dtype=bool),
        # Empty and full wells must share one graph_metrics schema.
        graph_metrics={**graph_metrics(W), **ccg_summary(ccg["per_pair"])},
        edges=edges,
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
            **ccg["meta"],
        },
        ccg_counts=ccg["counts"],
        ccg_baseline=ccg["baseline"],
        ccg_lags_ms=ccg["lags_ms"],
        ccg_pairs=ccg["pairs"],
        ccg_n_ref_spikes=ccg["n_ref_spikes"],
        ccg_n_tgt_spikes=ccg["n_tgt_spikes"],
        ccg_meta=ccg["meta"],
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
        graph_metrics.json   well-level scalars (+ the ccg_* summary)
        edges.parquet        significant edges [u, v, sttc, dist_um] + ccg_* columns
        ccg.npz              correlogram counts/baseline, keyed by `pairs` (u, v)
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
    if results.ccg_counts is not None:
        write_ccg_npz(results, output_dir / "ccg.npz")
    _atomic_json_write(results.graph_metrics, output_dir / "graph_metrics.json")
    _atomic_json_write(results.diagnostics, output_dir / "diagnostics.json")


def write_ccg_npz(results: ConnectivityResults, dest: Path) -> None:
    """Persist the correlogram block. ``pairs`` is the join key, not row order.

    Compressed: a dense well (~4.5k edges x 201 bins) is 7.4 MB raw and 3.6 MB
    compressed, for 0.16 s of write and 0.02 s of read — worth it at ~1900 wells.
    """
    meta = results.ccg_meta or {}
    np.savez_compressed(
        dest,
        counts=results.ccg_counts,
        baseline=results.ccg_baseline,
        lags_ms=results.ccg_lags_ms,
        pairs=results.ccg_pairs,
        n_ref_spikes=results.ccg_n_ref_spikes,
        n_tgt_spikes=results.ccg_n_tgt_spikes,
        bin_ms=np.asarray(meta.get("ccg_bin_ms", float("nan"))),
        window_ms=np.asarray(meta.get("ccg_window_ms", float("nan"))),
        syn_lo_ms=np.asarray(meta.get("ccg_syn_lo_ms", float("nan"))),
        syn_hi_ms=np.asarray(meta.get("ccg_syn_hi_ms", float("nan"))),
    )
