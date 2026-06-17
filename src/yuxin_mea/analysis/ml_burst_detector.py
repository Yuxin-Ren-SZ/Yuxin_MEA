"""End-to-end orchestrator for ML-based burst detection.

Wires together the four building blocks:

  1. Adaptive binning (mirrors iterative detector) ─────────────────────────────
  2. Per-unit 2-state Poisson HMM → (λ_bg_u, λ_burst_u, posterior) per unit
  3. Bin-level feature matrix (HMM aggregates + population + per-unit LLR/ISI
     + temporal derivatives)
  4. HDBSCAN cluster → burst-label bin mask → temporal merge → hierarchy
     (burstlets → strict-merge network_bursts → clustered-merge superbursts)
  5. ``BurstResults`` materialisation (same layout as iterative detector so
     dashboard tooling and ``PickleBurstOutputWriter`` work unchanged)

The output schema additions for the ML detector are the four per-event quality
columns: ``posterior_peak``, ``posterior_mean``, ``llr_aggregate``, ``ff_peak``.

This module owns ``MLBurstConfig`` (frozen dataclass with every tunable),
``MLBurstTrace`` (optional debug bundle), and ``MLBurstError``. It depends on
:mod:`yuxin_mea.analysis.ml_burst_hmm`, :mod:`ml_burst_features`,
:mod:`ml_burst_cluster`, and reuses helpers from
:mod:`burst_common` (multi-scale FF, hierarchy merges, candidate
materialisation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .burst_common import (
    _compute_spike_matrix,
    _level_metrics,
)
from .ml_burst_cluster import (
    ClusterAssignment,
    burst_bin_mask,
    cluster_bins,
    temporal_merge,
)
from .ml_burst_features import build_feature_matrix
from .ml_burst_hmm import UnitHMMFit, fit_all_units, fit_to_dict


class MLBurstError(ValueError):
    """Raised when input data is insufficient for ML burst detection."""


@dataclass(frozen=True)
class MLBurstConfig:
    """All tunable parameters for the ML burst detector.

    Grouped by stage for legibility — ParamSpec defaults in the task class
    mirror these in order.
    """

    # ---- Binning ----------------------------------------------------------
    bin_size_mode: str = "adaptive"
    fixed_bin_size_s: float = 0.02

    # ---- HMM --------------------------------------------------------------
    hmm_max_iter: int = 100
    hmm_tol: float = 1e-3
    hmm_min_spikes: int = 50
    hmm_init_strategy: str = "quantile"
    hmm_min_rate_ratio: float = 1.5
    hmm_random_state: int = 42
    hmm_n_jobs: int = 1

    # ---- Features ---------------------------------------------------------
    ff_scale_multipliers: tuple = (0.5, 1.0, 2.0, 5.0)
    posterior_quantile: float = 0.9
    isi_window_bins: int = 25
    deriv_sigma_short_bins: float = 1.5
    deriv_sigma_long_bins: float = 8.0
    background_quantile: float = 0.5
    unit_agg_quantile: float = 0.9

    # ---- Dim reduction ----------------------------------------------------
    pca_n_components: int = 0  # 0 = disabled

    # ---- Clustering embedding & multi-cluster burst selection -------------
    cluster_embedding_mode: str = "none"  # "none" | "umap"
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.0
    umap_n_components: int = 5
    burst_mad_scale: float = 3.0

    # ---- HDBSCAN ----------------------------------------------------------
    hdbscan_min_cluster_size: int = 30
    hdbscan_min_samples: int = 5
    hdbscan_cluster_selection_epsilon: float = 0.0
    hdbscan_cluster_selection_method: str = "eom"
    hdbscan_metric: str = "euclidean"
    cluster_ranking_feature: str = "post_frac_gt_0_5"
    fallback_posterior_threshold: float = 0.3

    # ---- Temporal merge / hierarchy --------------------------------------
    closing_bins: int = 3
    merge_mad_scale: float = 0.75
    merge_floor_frac: float = 0.70
    network_merge_gap_min_s: float = 0.75
    min_burst_modulation: float = 0.1


@dataclass
class MLBurstTrace:
    """Diagnostic bundle persisted when ``debug=True``.

    All fields are optional / default-empty so a future inspector page can read
    a partial trace without crashing on missing keys.
    """

    t_centers: Optional[np.ndarray] = None
    bin_size: Optional[float] = None
    bins: Optional[np.ndarray] = None
    unit_ids: Optional[list] = None
    feature_names: Optional[list] = None
    hmm_fits: list[dict] = field(default_factory=list)
    posterior_matrix: Optional[np.ndarray] = None
    feature_matrix: Optional[np.ndarray] = None
    scaler_mean: Optional[np.ndarray] = None
    scaler_std: Optional[np.ndarray] = None
    hdbscan_labels: Optional[np.ndarray] = None
    hdbscan_probabilities: Optional[np.ndarray] = None
    cluster_ranking: Optional[dict] = None
    burst_label: Optional[int] = None
    burst_labels: Optional[list] = None
    burst_mask_pre_merge: Optional[np.ndarray] = None
    burst_mask_post_closing: Optional[np.ndarray] = None
    merge_threshold: Optional[float] = None
    candidates_pre_hierarchy: Optional[list] = None
    burstlets_pre_gate: Optional[list] = None
    gate_decision: Optional[dict] = None
    cluster_decision: Optional[str] = None
    fallback_threshold: Optional[float] = None


# ---------------------------------------------------------------------------
# Internal helpers (event materialisation / finalisation)
# ---------------------------------------------------------------------------


def _ml_finalize_event(
    evs: list[dict],
    s: float,
    e: float,
    *,
    units: list,
    spike_times: dict,
    n_units: int,
    composite: np.ndarray,        # ranking_signal (post_frac_gt_0_5)
    t_centers: np.ndarray,
    bin_size: float,
    spike_counts_total: np.ndarray,
    pfr: np.ndarray,
    ws_sharp: np.ndarray,
    ws_smooth: np.ndarray,
    ff1: np.ndarray,
    llr_signal: np.ndarray,
) -> dict:
    """Aggregate sub-events into a single merged event spanning [s, e).

    Renames the composite-derived quality columns to reflect their ML meaning:

      composite_peak  →  posterior_peak
      composite_mean  →  posterior_mean

    ``llr_aggregate`` and ``ff_peak`` keep their original names to minimise
    downstream surprise.
    """
    in_ev = (t_centers >= s) & (t_centers < e)
    participating = sum(
        1 for u in units if np.any((spike_times[u] >= s) & (spike_times[u] < e))
    )
    total_spikes = int(spike_counts_total[in_ev].sum())
    best = max(evs, key=lambda x: x["peak_synchrony"])

    comp_vals = composite[in_ev]
    return {
        "start": float(s),
        "end": float(e),
        "duration_s": float(e - s),
        "peak_synchrony": float(ws_sharp[in_ev].max()) if in_ev.any() else 0.0,
        "peak_time": float(best["peak_time"]),
        "synchrony_energy": float(ws_smooth[in_ev].sum() * bin_size) if in_ev.any() else 0.0,
        "participation": participating / max(1, n_units),
        "total_spikes": total_spikes,
        "burst_peak": float(pfr[in_ev].max()) if in_ev.any() else 0.0,
        "fragment_count": sum(ev.get("fragment_count", 1) for ev in evs),
        "n_sub_events": len(evs),
        "llr_aggregate": float(llr_signal[in_ev].mean()) if in_ev.any() else 0.0,
        "posterior_peak": float(comp_vals.max()) if comp_vals.size > 0 else 0.0,
        "posterior_mean": float(comp_vals.mean()) if comp_vals.size > 0 else 0.0,
        "ff_peak": float(ff1[in_ev].max()) if in_ev.any() else 0.0,
    }


def _adaptive_bin_size_s(spike_times: dict, units: list) -> tuple[float, float]:
    """Median log-ISI based bin size, clamped to [10, 30] ms.

    Identical to the iterative detector's choice so outputs are comparable.
    """
    all_log_isis: list[float] = []
    for u in units:
        t = np.unique(np.sort(spike_times[u]))
        if len(t) >= 2:
            isi = np.diff(t)
            isi = isi[isi > 0]
            if isi.size > 0:
                all_log_isis.extend(np.log10(isi).tolist())
    biological_isi_s = 10 ** float(np.median(all_log_isis)) if all_log_isis else 0.1
    return float(np.clip(biological_isi_s * 1000, 10, 30) / 1000.0), biological_isi_s


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_ml_bursts(
    spike_times: dict[str, np.ndarray],
    config: MLBurstConfig | None = None,
    trace: "MLBurstTrace | None" = None,
):
    """Run the ML burst detector end-to-end.

    Returns a ``BurstResults`` instance (re-using the iterative/standard detector
    schema) with an added set of debug-friendly ``diagnostics`` keys.

    Raises
    ------
    MLBurstError
        When inputs are too sparse to detect anything meaningful (no units, no
        spikes, sub-microsecond span, or all HMM fits get skipped).
    """
    from .burst_detector import BurstResults

    if config is None:
        config = MLBurstConfig()

    units = list(spike_times.keys())
    if not units:
        raise MLBurstError("spike_times contains no units")

    non_empty = [spike_times[u] for u in units if len(spike_times[u]) > 0]
    if not non_empty:
        raise MLBurstError("spike_times contains no spikes")

    all_spikes = np.sort(np.concatenate(non_empty))
    rec_start, rec_end = float(all_spikes[0]), float(all_spikes[-1])
    total_dur = rec_end - rec_start
    if total_dur < 1e-6:
        raise MLBurstError("spike_times spans insufficient duration (< 1 µs)")

    # ---- 1. Binning -------------------------------------------------------
    if config.bin_size_mode == "fixed":
        bin_size = float(config.fixed_bin_size_s)
        biological_isi_s = bin_size
    else:
        bin_size, biological_isi_s = _adaptive_bin_size_s(spike_times, units)
    bins = np.arange(rec_start, rec_end + bin_size, bin_size)
    t_centers = (bins[:-1] + bins[1:]) / 2
    n_bins = len(t_centers)
    n_units = len(units)
    if n_bins < 8:
        raise MLBurstError("recording too short for the chosen bin size")

    isi_bins = biological_isi_s / bin_size
    sigma_fast = float(np.clip(isi_bins, 1, 2))
    sigma_slow = float(np.clip(5.0 * isi_bins, 3, 8))
    burstlet_merge_gap_s = 3.0 * biological_isi_s
    network_merge_gap_s = max(10.0 * biological_isi_s, config.network_merge_gap_min_s)

    # ---- 2. Per-unit HMM --------------------------------------------------
    spike_matrix = _compute_spike_matrix(spike_times, units, bins, n_bins)
    fits, posteriors = fit_all_units(
        spike_matrix,
        units,
        bin_size,
        max_iter=int(config.hmm_max_iter),
        tol=float(config.hmm_tol),
        min_spikes=int(config.hmm_min_spikes),
        init_strategy=str(config.hmm_init_strategy),
        min_rate_ratio=float(config.hmm_min_rate_ratio),
        random_state=int(config.hmm_random_state),
        n_jobs=int(config.hmm_n_jobs),
    )
    n_units_fit = sum(1 for f in fits if f.skipped_reason is None)
    if n_units_fit == 0:
        raise MLBurstError(
            "no HMM fits succeeded — try lowering hmm_min_spikes or "
            "hmm_min_rate_ratio for sparse recordings"
        )

    # Standard ws_sharp/ws_smooth (also used as peak_synchrony source for the
    # output events, so existing viewers keep working).
    spike_counts_total = spike_matrix.sum(axis=0)
    active_unit_counts = (spike_matrix > 0).sum(axis=0).astype(float)
    pfr = spike_counts_total / bin_size
    rate_per_unit = pfr / max(1, n_units)
    participation_raw = active_unit_counts / max(1, n_units)
    ws_sharp = gaussian_filter1d(participation_raw, sigma_fast)
    ws_smooth = gaussian_filter1d(rate_per_unit, sigma_slow)

    # ---- 3. Bin-level feature matrix --------------------------------------
    X, feature_names = build_feature_matrix(
        spike_times,
        units,
        bins,
        t_centers,
        bin_size,
        fits,
        posteriors,
        ff_scale_multipliers=config.ff_scale_multipliers,
        posterior_quantile=float(config.posterior_quantile),
        isi_window_bins=int(config.isi_window_bins),
        deriv_sigma_short_bins=float(config.deriv_sigma_short_bins),
        deriv_sigma_long_bins=float(config.deriv_sigma_long_bins),
        unit_agg_quantile=float(config.unit_agg_quantile),
        spike_matrix=spike_matrix,
    )

    # ---- 4. Cluster + temporal merge --------------------------------------
    assignment: ClusterAssignment = cluster_bins(
        X,
        feature_names,
        ranking_feature=str(config.cluster_ranking_feature),
        background_quantile=float(config.background_quantile),
        min_cluster_size=int(config.hdbscan_min_cluster_size),
        min_samples=int(config.hdbscan_min_samples),
        cluster_selection_epsilon=float(config.hdbscan_cluster_selection_epsilon),
        cluster_selection_method=str(config.hdbscan_cluster_selection_method),
        metric=str(config.hdbscan_metric),
        fallback_posterior_threshold=float(config.fallback_posterior_threshold),
        pca_n_components=int(config.pca_n_components),
        cluster_embedding_mode=str(config.cluster_embedding_mode),
        umap_n_neighbors=int(config.umap_n_neighbors),
        umap_min_dist=float(config.umap_min_dist),
        umap_n_components=int(config.umap_n_components),
        burst_mad_scale=float(config.burst_mad_scale),
    )
    mask_pre_merge = burst_bin_mask(assignment)

    # Composite/ranking signal used by valley logic and per-event quality cols
    rank_idx = feature_names.index(str(config.cluster_ranking_feature))
    ranking_signal = X[:, rank_idx]

    candidates, closed_mask, merge_threshold = temporal_merge(
        mask_pre_merge,
        t_centers,
        bins,
        bin_size,
        ranking_signal,
        closing_bins=int(config.closing_bins),
        merge_mad_scale=float(config.merge_mad_scale),
        merge_floor_frac=float(config.merge_floor_frac),
        merge_gap_s=float(burstlet_merge_gap_s),
    )

    # ---- 5. Materialise burstlet events -----------------------------------
    # Use 1×-scale FF column (FF1 by default) as the representative FF.
    ff_col_name = "FF1" if "FF1" in feature_names else "FF0"
    ff1 = X[:, feature_names.index(ff_col_name)]
    # Per-unit LLR mean column is the closest analog to the iterative
    # detector's llr_signal — used for per-event llr_aggregate and the soft
    # quality gate.
    llr_signal = X[:, feature_names.index("llr_hmm_mean")] if "llr_hmm_mean" in feature_names else np.zeros(n_bins)

    burstlets_raw: list[dict] = []
    pre_gate_events: list[dict] = []
    for c in candidates:
        s_t, e_t = c["start"], c["end"]
        duration_s = e_t - s_t
        if duration_s <= 0:
            continue
        in_ev = (t_centers >= s_t) & (t_centers < e_t)
        if not in_ev.any():
            continue
        participating = sum(
            1 for u in units if np.any((spike_times[u] >= s_t) & (spike_times[u] < e_t))
        )
        total_spikes = int(spike_counts_total[in_ev].sum())
        peak_abs_idx = int(np.where(in_ev)[0][np.argmax(ws_sharp[in_ev])])
        peak_synchrony = float(ws_sharp[peak_abs_idx])
        peak_time = float(t_centers[peak_abs_idx])
        comp_vals = ranking_signal[in_ev]
        event = {
            "start": float(s_t),
            "end": float(e_t),
            "duration_s": float(duration_s),
            "peak_synchrony": peak_synchrony,
            "peak_time": peak_time,
            "synchrony_energy": float(ws_smooth[in_ev].sum() * bin_size),
            "participation": participating / max(1, n_units),
            "total_spikes": total_spikes,
            "burst_peak": float(pfr[in_ev].max()),
            "fragment_count": 1,
            "llr_aggregate": float(llr_signal[in_ev].mean()),
            "posterior_peak": float(comp_vals.max()),
            "posterior_mean": float(comp_vals.mean()),
            "ff_peak": float(ff1[in_ev].max()),
        }
        pre_gate_events.append(dict(event))
        burstlets_raw.append(event)

    # Soft burst-modulation gate on llr_aggregate (mirrors iterative detector)
    burst_modulation_scores = [float(ev["llr_aggregate"]) for ev in burstlets_raw]
    burst_modulation_index = max(burst_modulation_scores) if burst_modulation_scores else 0.0
    gate_state: dict | None = None
    if config.min_burst_modulation > 0:
        kept = [
            ev for ev in burstlets_raw
            if float(ev["llr_aggregate"]) >= config.min_burst_modulation
        ]
        gate_state = {
            "threshold": float(config.min_burst_modulation),
            "n_pre": len(burstlets_raw),
            "n_post": len(kept),
        }
        burstlets_raw = kept

    # ---- 6. Hierarchy merge (reuse iterative-detector machinery) ----------
    # _merge_strict_hier / _merge_clustered_hier call _finalize_event from
    # the iterative detector module, which writes "composite_peak"/
    # "composite_mean" keys onto the merged events. For ML-flavored events we
    # want "posterior_peak"/"posterior_mean" instead, so we patch the context
    # dict to use the ML-flavored finaliser by monkey-routing through a
    # wrapper module attribute.
    #
    # Concretely: _merge_*_hier closes over the module-level _finalize_event
    # symbol, so we can't substitute it from here. Easiest path: re-implement
    # the two-level merge inline using the same gap / valley conditions but
    # with our _ml_finalize_event.
    hier_ctx = dict(
        units=units,
        spike_times=spike_times,
        n_units=n_units,
        composite=ranking_signal,
        t_centers=t_centers,
        bin_size=bin_size,
        spike_counts_total=spike_counts_total,
        pfr=pfr,
        ws_sharp=ws_sharp,
        ws_smooth=ws_smooth,
        ff1=ff1,
        llr_signal=llr_signal,
    )
    network_bursts = _merge_strict_local(
        burstlets_raw,
        gap=burstlet_merge_gap_s,
        threshold=merge_threshold,
        ctx=hier_ctx,
    )
    superbursts = _merge_clustered_local(
        network_bursts,
        gap=network_merge_gap_s,
        baseline=float(np.median(ranking_signal[~closed_mask])) if (~closed_mask).any() else 0.0,
        threshold=merge_threshold,
        ctx=hier_ctx,
    )

    # ---- 7. Build BurstResults --------------------------------------------
    def _to_df(evs: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(evs) if evs else pd.DataFrame()

    metrics = {
        "burstlets": _level_metrics(burstlets_raw, total_dur),
        "network_bursts": _level_metrics(network_bursts, total_dur),
        "superbursts": _level_metrics(superbursts, total_dur),
    }

    diagnostics = {
        "adaptive_bin_ms": float(bin_size * 1000.0),
        "biological_isi_s": float(biological_isi_s),
        "sigma_fast_bins": float(sigma_fast),
        "sigma_slow_bins": float(sigma_slow),
        "n_units": int(n_units),
        "n_units_fit": int(n_units_fit),
        "n_units_skipped": int(n_units - n_units_fit),
        "skipped_reasons": {
            f.unit_id: f.skipped_reason for f in fits if f.skipped_reason is not None
        },
        "lambda_bg_per_unit": {
            f.unit_id: float(f.lambda_bg) for f in fits if f.skipped_reason is None
        },
        "lambda_burst_per_unit": {
            f.unit_id: float(f.lambda_burst) for f in fits if f.skipped_reason is None
        },
        "cluster_decision": assignment.decision,
        "cluster_embedding_mode": str(config.cluster_embedding_mode),
        "cluster_n_clusters": int(assignment.n_clusters),
        "cluster_burst_label": int(assignment.burst_label),
        "cluster_burst_labels": [int(c) for c in (assignment.burst_labels or [])],
        "cluster_ranking": {int(k): float(v) for k, v in assignment.cluster_rank.items()},
        "merge_threshold": float(merge_threshold),
        "burst_modulation_index": float(burst_modulation_index),
        "burst_activity_detected": bool(burstlets_raw),
        "burstlet_merge_gap_s": float(burstlet_merge_gap_s),
        "network_merge_gap_s": float(network_merge_gap_s),
        "fallback_posterior_threshold": (
            float(assignment.fallback_threshold)
            if assignment.fallback_threshold is not None
            else None
        ),
        "ranking_feature": str(config.cluster_ranking_feature),
        "feature_names": list(feature_names),
    }
    if gate_state is not None:
        diagnostics["bmi_gate"] = gate_state

    plot_data = {
        "t": t_centers,
        "participation_signal": ws_sharp,
        "rate_signal": ws_smooth,
        "ranking_signal": ranking_signal,
        "ff_signal": ff1,
        "llr_signal": llr_signal,
        "posterior_matrix_mean": np.nan_to_num(np.nanmean(posteriors, axis=0), nan=0.0),
        "burst_peak_times": np.array([b["peak_time"] for b in network_bursts]),
        "burst_peak_values": np.array([b["peak_synchrony"] for b in network_bursts]),
        "merge_threshold": float(merge_threshold),
    }

    if trace is not None:
        trace.t_centers = t_centers.copy()
        trace.bin_size = float(bin_size)
        trace.bins = bins.copy()
        trace.unit_ids = [str(u) for u in units]
        trace.feature_names = list(feature_names)
        trace.hmm_fits = [fit_to_dict(f) for f in fits]
        trace.posterior_matrix = posteriors.copy()
        trace.feature_matrix = X.copy()
        trace.scaler_mean = assignment.scaler_mean.copy()
        trace.scaler_std = assignment.scaler_std.copy()
        trace.hdbscan_labels = assignment.labels.copy()
        trace.hdbscan_probabilities = assignment.probabilities.copy()
        trace.cluster_ranking = {int(k): float(v) for k, v in assignment.cluster_rank.items()}
        trace.burst_label = int(assignment.burst_label)
        trace.burst_labels = [int(c) for c in (assignment.burst_labels or [])]
        trace.burst_mask_pre_merge = mask_pre_merge.copy()
        trace.burst_mask_post_closing = closed_mask.copy()
        trace.merge_threshold = float(merge_threshold)
        trace.candidates_pre_hierarchy = [dict(c) for c in candidates]
        trace.burstlets_pre_gate = [dict(ev) for ev in pre_gate_events]
        trace.gate_decision = gate_state
        trace.cluster_decision = assignment.decision
        trace.fallback_threshold = assignment.fallback_threshold

    return BurstResults(
        burstlets=_to_df(burstlets_raw),
        network_bursts=_to_df(network_bursts),
        superbursts=_to_df(superbursts),
        metrics=metrics,
        diagnostics=diagnostics,
        plot_data=plot_data,
    )


# ---------------------------------------------------------------------------
# Local hierarchy merge — copies of the iterative-detector logic but using
# _ml_finalize_event so the merged events carry posterior_* keys instead of
# composite_* keys.
# ---------------------------------------------------------------------------


def _valley_min(prev: dict, nxt: dict, composite: np.ndarray, t_centers: np.ndarray) -> float | None:
    mask = (t_centers >= prev["end"]) & (t_centers <= nxt["start"])
    if not mask.any():
        return None
    v = composite[mask]
    return float(v.min()) if v.size > 0 else None


def _merge_strict_local(events: list[dict], gap: float, threshold: float, ctx: dict) -> list[dict]:
    if not events:
        return []
    composite, t_centers, bin_size = ctx["composite"], ctx["t_centers"], ctx["bin_size"]
    events = sorted(events, key=lambda x: x["start"])
    merged, curr_evs = [], [events[0]]
    s, e = events[0]["start"], events[0]["end"]
    for nxt in events[1:]:
        vm = _valley_min(curr_evs[-1], nxt, composite, t_centers)
        valley_ok = (nxt["start"] - e <= bin_size) if vm is None else (vm >= threshold)
        if (nxt["start"] - e) <= gap and valley_ok:
            curr_evs.append(nxt)
            e = max(e, nxt["end"])
        else:
            merged.append(_ml_finalize_event(curr_evs, s, e, **ctx))
            curr_evs, s, e = [nxt], nxt["start"], nxt["end"]
    merged.append(_ml_finalize_event(curr_evs, s, e, **ctx))
    return [m for m in merged if m["duration_s"] > 0]


def _merge_clustered_local(
    events: list[dict],
    gap: float,
    baseline: float,
    threshold: float,
    ctx: dict,
) -> list[dict]:
    if not events:
        return []
    composite, t_centers, bin_size = ctx["composite"], ctx["t_centers"], ctx["bin_size"]
    events = sorted(events, key=lambda x: x["start"])
    merged, curr_evs = [], [events[0]]
    s, e = events[0]["start"], events[0]["end"]
    for nxt in events[1:]:
        vm = _valley_min(curr_evs[-1], nxt, composite, t_centers)
        valley_ok = (nxt["start"] - e <= bin_size) if vm is None else (baseline < vm < threshold)
        if (nxt["start"] - e) <= gap and valley_ok:
            curr_evs.append(nxt)
            e = max(e, nxt["end"])
        else:
            merged.append(_ml_finalize_event(curr_evs, s, e, **ctx))
            curr_evs, s, e = [nxt], nxt["start"], nxt["end"]
    merged.append(_ml_finalize_event(curr_evs, s, e, **ctx))
    return [m for m in merged if m["duration_s"] > 0 and m["n_sub_events"] >= 2]
