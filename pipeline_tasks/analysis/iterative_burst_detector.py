from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


class IterativeBurstError(ValueError):
    """Raised when spike data is insufficient for iterative burst detection."""


@dataclass(frozen=True)
class IterativeBurstConfig:
    # Phase 1 — initial detection
    permissive_mad_scale: float = 0.30
    permissive_percentile: float = 70.0
    mad_fallback_threshold: float = 0.01

    # Phase 3 — iteration
    composite_mad_scale: float = 0.75
    extent_frac: float = 0.30
    merge_floor_frac: float = 0.70
    network_merge_gap_min_s: float = 0.75
    max_iterations: int = 20
    convergence_eps: float = 0.005

    # Fisher regularization
    fisher_alpha_frac: float = 1e-3

    # FF multi-scale bin-size multipliers
    ff_scale_multipliers: tuple = (0.5, 1.0, 2.0, 5.0)


# ---------------------------------------------------------------------------
# Helpers — statistics
# ---------------------------------------------------------------------------

def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0}
    mean_val = float(x.mean())
    std_val = float(x.std())
    cv = std_val / mean_val if abs(mean_val) > 1e-12 else float("nan")
    return {"mean": mean_val, "std": std_val, "cv": float(cv)}


def _level_metrics(events: list[dict], total_dur: float) -> dict:
    if not events:
        return {}
    starts = [ev["start"] for ev in events]
    return {
        "count": len(events),
        "rate": len(events) / total_dur,
        "duration": _stats([ev["duration_s"] for ev in events]),
        "inter_event_interval": _stats(np.diff(starts)) if len(starts) > 1 else _stats([]),
        "intensity": _stats([ev["synchrony_energy"] for ev in events]),
        "participation": _stats([ev["participation"] for ev in events]),
        "spikes_per_burst": _stats([ev["total_spikes"] for ev in events]),
        "burst_peak": _stats([ev["burst_peak"] for ev in events]),
        "peak_synchrony": _stats([ev["peak_synchrony"] for ev in events]),
    }


# ---------------------------------------------------------------------------
# Helpers — feature computation
# ---------------------------------------------------------------------------

def _compute_spike_matrix(
    spike_times: dict,
    units: list,
    bins: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Shape (n_units, n_bins): integer spike counts per unit per bin."""
    matrix = np.zeros((len(units), n_bins), dtype=np.float32)
    for i, u in enumerate(units):
        spk = np.asarray(spike_times[u])
        if spk.size > 0:
            counts, _ = np.histogram(spk, bins=bins)
            matrix[i] = counts
    return matrix


def _compute_multiscale_ff(
    spike_matrix: np.ndarray,
    bins: np.ndarray,
    t_centers: np.ndarray,
    bin_size: float,
    ff_scale_multipliers: tuple,
) -> np.ndarray:
    """Spatial Fano Factor at multiple scales. Returns (n_bins, n_scales)."""
    n_bins = len(t_centers)
    n_scales = len(ff_scale_multipliers)
    ff_signals = np.zeros((n_bins, n_scales))
    rec_start = float(bins[0])
    rec_end = float(bins[-1])

    for k, mult in enumerate(ff_scale_multipliers):
        scale_dt = float(np.clip(mult * bin_size, 0.005, 0.1))

        coarse_bins = np.arange(rec_start, rec_end + scale_dt, scale_dt)
        n_coarse = len(coarse_bins) - 1
        if n_coarse < 1:
            continue

        coarse_idx = np.clip(
            np.searchsorted(coarse_bins, t_centers, side="right") - 1,
            0, n_coarse - 1,
        )

        ff_coarse = np.zeros(n_coarse)
        for j in range(n_coarse):
            fine_mask = coarse_idx == j
            if not fine_mask.any():
                continue
            unit_counts = spike_matrix[:, fine_mask].sum(axis=1)
            mean_c = float(unit_counts.mean())
            if mean_c > 1e-9:
                ff_coarse[j] = float(unit_counts.var()) / mean_c

        ff_fine = ff_coarse[coarse_idx]
        ff_signals[:, k] = gaussian_filter1d(ff_fine, sigma=1.5)

    return ff_signals


def _compute_llr_signal(
    spike_matrix: np.ndarray,
    lambda_bg_per_unit: np.ndarray,
    bin_size: float,
    sigma: float,
) -> np.ndarray:
    """Signed per-unit Poisson LLR, averaged across units, then smoothed."""
    expected = (lambda_bg_per_unit * bin_size)[:, np.newaxis]  # (n_units, 1)
    n_u = spike_matrix

    safe_expected = np.where(expected > 1e-12, expected, 1.0)
    safe_n = np.where(n_u > 0, n_u, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        llr_u = np.where(
            (n_u > 0) & (expected > 1e-12),
            2.0 * (n_u * np.log(safe_n / safe_expected) - (n_u - expected)),
            np.where(expected > 1e-12, 2.0 * (0.0 - (0.0 - expected)), 0.0),
        )
        # Sign: + when elevated above background, - when suppressed
        signed = np.sign(n_u - expected) * np.abs(llr_u)

    llr = np.mean(signed, axis=0)
    return gaussian_filter1d(llr, sigma=sigma)


def _compute_burstiness(
    spike_times: dict,
    units: list,
    bins: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Per-bin mean instantaneous firing rate (1/ISI), smoothed."""
    raw = np.zeros(n_bins)
    cnt = np.zeros(n_bins)

    for u in units:
        spk = np.sort(np.asarray(spike_times[u]))
        if len(spk) < 2:
            continue
        isi = np.diff(spk)
        midpoints = (spk[:-1] + spk[1:]) / 2
        idx = np.clip(np.searchsorted(bins[1:], midpoints), 0, n_bins - 1)
        inst = 1.0 / np.maximum(isi, 1e-6)
        np.add.at(raw, idx, inst)
        np.add.at(cnt, idx, 1)

    result = np.divide(raw, cnt, out=np.zeros_like(raw), where=cnt > 0)
    return gaussian_filter1d(result, sigma=2.0)


# ---------------------------------------------------------------------------
# Helpers — Fisher's discriminant
# ---------------------------------------------------------------------------

def _znorm(
    X: np.ndarray,
    bg_mask: np.ndarray,
) -> np.ndarray:
    """Z-score X using background (bg_mask=True) bin statistics."""
    bg = X[bg_mask]
    mu = bg.mean(axis=0)
    std = bg.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mu) / std


def _fit_fisher(
    X_norm: np.ndarray,
    labels: np.ndarray,
    alpha_frac: float,
) -> np.ndarray | None:
    """Fisher LDA direction. Returns unit-norm weight vector or None if degenerate."""
    n_features = X_norm.shape[1]
    idx1 = labels == 1
    idx0 = labels == 0

    if idx1.sum() < 3 or idx0.sum() < 3:
        return None

    X1, X0 = X_norm[idx1], X_norm[idx0]
    mu1, mu0 = X1.mean(axis=0), X0.mean(axis=0)

    S_W = (X1 - mu1).T @ (X1 - mu1) + (X0 - mu0).T @ (X0 - mu0)
    alpha = alpha_frac * float(np.trace(S_W)) / n_features
    S_W_reg = S_W + alpha * np.eye(n_features)

    try:
        w = np.linalg.solve(S_W_reg, mu1 - mu0)
    except np.linalg.LinAlgError:
        return None

    norm = float(np.linalg.norm(w))
    return w / norm if norm > 1e-12 else None


# ---------------------------------------------------------------------------
# Helpers — candidate management
# ---------------------------------------------------------------------------

def _mask_to_candidates(mask: np.ndarray, bins: np.ndarray) -> list[dict]:
    """Boolean mask → list of {start, end, start_idx, end_idx}."""
    candidates = []
    n = len(mask)
    in_b = False
    s_idx = 0
    for t in range(n):
        if mask[t] and not in_b:
            s_idx = t
            in_b = True
        elif not mask[t] and in_b:
            candidates.append({"start": float(bins[s_idx]), "end": float(bins[t]),
                                "start_idx": s_idx, "end_idx": t - 1})
            in_b = False
    if in_b:
        candidates.append({"start": float(bins[s_idx]), "end": float(bins[n]),
                            "start_idx": s_idx, "end_idx": n - 1})
    return candidates


def _candidates_to_mask(candidates: list[dict], t_centers: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(t_centers), dtype=bool)
    for c in candidates:
        s, e = c.get("start_idx"), c.get("end_idx")
        if s is not None and e is not None:
            mask[s:e + 1] = True
        else:
            mask |= (t_centers >= c["start"]) & (t_centers < c["end"])
    return mask


def _trim_candidate(
    c: dict,
    composite: np.ndarray,
    threshold: float,
    extent_frac: float,
    bins: np.ndarray,
    n_bins: int,
) -> dict | None:
    s, e = c["start_idx"], c["end_idx"]
    if s > e:
        return None

    peak_rel = int(np.argmax(composite[s:e + 1]))
    peak_idx = s + peak_rel
    peak_val = composite[peak_idx]
    ext_thr = max(threshold, extent_frac * peak_val)

    while s < peak_idx and composite[s] < ext_thr:
        s += 1
    while e > peak_idx and composite[e] < ext_thr:
        e -= 1

    if s > e or float(bins[e + 1]) - float(bins[s]) <= 0:
        return None
    return {"start": float(bins[s]), "end": float(bins[e + 1]),
            "start_idx": s, "end_idx": e}


def _valley_min(
    prev: dict,
    nxt: dict,
    composite: np.ndarray,
    t_centers: np.ndarray,
) -> float | None:
    mask = (t_centers >= prev["end"]) & (t_centers <= nxt["start"])
    if not mask.any():
        return None
    v = composite[mask]
    return float(v.min()) if v.size > 0 else None


def _iter_merge(
    candidates: list[dict],
    composite: np.ndarray,
    t_centers: np.ndarray,
    bin_size: float,
    gap_s: float,
    threshold: float,
    floor_frac: float,
) -> list[dict]:
    """Merge candidates during iteration with relaxed valley condition."""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x["start"])
    merged = []
    cur = candidates[0].copy()
    for nxt in candidates[1:]:
        gap = nxt["start"] - cur["end"]
        vm = _valley_min(cur, nxt, composite, t_centers)
        valley_ok = (gap <= bin_size) if vm is None else (vm >= floor_frac * threshold)
        if gap <= gap_s and valley_ok:
            cur = {"start": cur["start"], "end": nxt["end"],
                   "start_idx": cur["start_idx"], "end_idx": nxt["end_idx"]}
        else:
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)
    return merged


# ---------------------------------------------------------------------------
# Helpers — hierarchy finalization
# ---------------------------------------------------------------------------

def _finalize_event(
    evs: list[dict],
    s: float,
    e: float,
    units: list,
    spike_times: dict,
    n_units: int,
    composite: np.ndarray,
    t_centers: np.ndarray,
    spike_counts_total: np.ndarray,
    pfr: np.ndarray,
    ws_sharp: np.ndarray,
    ws_smooth: np.ndarray,
    ff1: np.ndarray,
    llr_signal: np.ndarray,
    bin_size: float,
) -> dict:
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
        "participation": participating / n_units,
        "total_spikes": total_spikes,
        "burst_peak": float(pfr[in_ev].max()) if in_ev.any() else 0.0,
        "fragment_count": sum(ev.get("fragment_count", 1) for ev in evs),
        "n_sub_events": len(evs),
        "llr_aggregate": float(llr_signal[in_ev].mean()) if in_ev.any() else 0.0,
        "composite_peak": float(comp_vals.max()) if comp_vals.size > 0 else 0.0,
        "composite_mean": float(comp_vals.mean()) if comp_vals.size > 0 else 0.0,
        "ff_peak": float(ff1[in_ev].max()) if in_ev.any() else 0.0,
    }


def _merge_strict_hier(
    events: list[dict],
    gap: float,
    threshold: float,
    **ctx,
) -> list[dict]:
    """Merge burstlets → network bursts: valley must stay above threshold."""
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
            merged.append(_finalize_event(curr_evs, s, e, **ctx))
            curr_evs, s, e = [nxt], nxt["start"], nxt["end"]
    merged.append(_finalize_event(curr_evs, s, e, **ctx))
    return [m for m in merged if m["duration_s"] > 0]


def _merge_clustered_hier(
    events: list[dict],
    gap: float,
    baseline: float,
    threshold: float,
    **ctx,
) -> list[dict]:
    """Merge network bursts → superbursts: valley between baseline and threshold."""
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
            merged.append(_finalize_event(curr_evs, s, e, **ctx))
            curr_evs, s, e = [nxt], nxt["start"], nxt["end"]
    merged.append(_finalize_event(curr_evs, s, e, **ctx))
    return [m for m in merged if m["duration_s"] > 0 and m["n_sub_events"] >= 2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_iterative_bursts(
    spike_times: dict[str, np.ndarray],
    config: IterativeBurstConfig | None = None,
) -> "BurstResults":
    """Iterative contrast-maximizing network burst detector.

    Learns a Fisher LDA composite signal over 8 features (PFR, participation,
    multi-scale Fano Factor, per-unit Poisson LLR, burstiness) and iterates
    until the burst/non-burst partition converges.
    """
    from .burst_detector import BurstResults

    if config is None:
        config = IterativeBurstConfig()

    units = list(spike_times.keys())
    if not units:
        raise IterativeBurstError("spike_times contains no units")

    non_empty = [spike_times[u] for u in units if len(spike_times[u]) > 0]
    if not non_empty:
        raise IterativeBurstError("spike_times contains no spikes")

    all_spikes = np.sort(np.concatenate(non_empty))
    rec_start, rec_end = float(all_spikes[0]), float(all_spikes[-1])
    total_dur = rec_end - rec_start

    # -----------------------------------------------------------------------
    # Phase 1a: Adaptive binning
    # -----------------------------------------------------------------------
    all_log_isis: list[float] = []
    for u in units:
        t = np.unique(np.sort(spike_times[u]))
        if len(t) >= 2:
            isi = np.diff(t)
            isi = isi[isi > 0]
            if isi.size > 0:
                all_log_isis.extend(np.log10(isi).tolist())

    biological_isi_s = 10 ** float(np.median(all_log_isis)) if all_log_isis else 0.1
    adaptive_bin_ms = float(np.clip(biological_isi_s * 1000, 10, 30))
    bin_size = adaptive_bin_ms / 1000.0

    bins = np.arange(rec_start, rec_end + bin_size, bin_size)
    t_centers = (bins[:-1] + bins[1:]) / 2
    n_bins = len(t_centers)
    n_units = len(units)

    isi_bins = biological_isi_s / bin_size
    sigma_fast = float(np.clip(isi_bins, 1, 2))
    sigma_slow = float(np.clip(5.0 * isi_bins, 3, 8))

    burstlet_merge_gap_s = 3.0 * biological_isi_s
    network_merge_gap_s = max(10.0 * biological_isi_s, config.network_merge_gap_min_s)

    # -----------------------------------------------------------------------
    # Phase 1b: Population signals + initial participation-based candidates
    # -----------------------------------------------------------------------
    spike_matrix = _compute_spike_matrix(spike_times, units, bins, n_bins)
    spike_counts_total = spike_matrix.sum(axis=0)
    active_unit_counts = (spike_matrix > 0).sum(axis=0).astype(float)

    participation_raw = active_unit_counts / max(1, n_units)
    pfr = spike_counts_total / bin_size
    rate_per_unit = pfr / max(1, n_units)

    ws_sharp = gaussian_filter1d(participation_raw, sigma_fast)
    ws_smooth = gaussian_filter1d(rate_per_unit, sigma_slow)

    baseline_init = float(np.median(ws_sharp))
    spread_mad_init = float(np.median(np.abs(ws_sharp - baseline_init)))

    if spread_mad_init > config.mad_fallback_threshold:
        init_threshold = baseline_init + config.permissive_mad_scale * spread_mad_init
        init_method = "mad"
    else:
        init_threshold = float(np.percentile(ws_sharp, config.permissive_percentile))
        init_method = "percentile"

    candidates = _mask_to_candidates(ws_sharp >= init_threshold, bins)

    # -----------------------------------------------------------------------
    # Phase 2: Static feature signals (FF, burstiness — don't depend on λ_bg)
    # -----------------------------------------------------------------------
    ff_signals = _compute_multiscale_ff(
        spike_matrix, bins, t_centers, bin_size, config.ff_scale_multipliers
    )
    ff_scales_ms = [
        float(np.clip(m * adaptive_bin_ms, 5.0, 100.0))
        for m in config.ff_scale_multipliers
    ]

    burstiness = _compute_burstiness(spike_times, units, bins, n_bins)

    n_ff = len(config.ff_scale_multipliers)
    n_features = 2 + n_ff + 2  # PFR, P, FF×n_ff, LLR, burstiness
    llr_idx = 2 + n_ff          # index of LLR column in X

    # Placeholder X — LLR column updated each iteration
    X = np.column_stack([pfr, participation_raw, ff_signals, np.zeros(n_bins), burstiness])

    # Bootstrap Fisher weights (participation-heavy prior)
    w_prior = np.array([1.0, 1.5] + [0.5] * n_ff + [1.0, 0.5])
    w_prior = w_prior[:n_features]
    if len(w_prior) < n_features:
        w_prior = np.concatenate([w_prior, np.full(n_features - len(w_prior), 0.5)])
    w = w_prior / float(np.linalg.norm(w_prior))

    # Global fallback background rate (used when per-unit estimate is unreliable)
    global_lambda_bg = float(pfr.mean()) / max(1, n_units)

    # -----------------------------------------------------------------------
    # Phase 3: Iterative refinement
    # -----------------------------------------------------------------------
    prev_mask = _candidates_to_mask(candidates, t_centers)
    composite = np.zeros(n_bins)
    composite_threshold = 0.0
    composite_baseline = 0.0
    n_iterations = 0
    convergence_delta = 1.0
    lambda_bg_per_unit = np.full(n_units, global_lambda_bg)

    for iteration in range(config.max_iterations):
        n_iterations = iteration + 1

        # 3a. Background estimation
        candidate_mask = _candidates_to_mask(candidates, t_centers)
        bg_mask = ~candidate_mask
        if bg_mask.sum() < 2:
            bg_mask = np.ones(n_bins, dtype=bool)

        lambda_bg_per_unit = np.array([
            float(spike_matrix[i, bg_mask].sum() / (bg_mask.sum() * bin_size))
            for i in range(n_units)
        ])
        lambda_bg_per_unit = np.where(
            lambda_bg_per_unit > 0, lambda_bg_per_unit, global_lambda_bg
        )

        # 3b. Recompute LLR with updated background
        llr_signal = _compute_llr_signal(spike_matrix, lambda_bg_per_unit, bin_size, sigma_slow)
        X[:, llr_idx] = llr_signal

        # 3c. Z-normalize relative to background
        X_norm = _znorm(X, bg_mask)

        # 3d. Learn Fisher weights (skip iteration 0 — use prior bootstrap)
        if iteration > 0:
            w_new = _fit_fisher(X_norm, candidate_mask.astype(int), config.fisher_alpha_frac)
            if w_new is not None:
                w = w_new

        composite = X_norm @ w

        # 3e. Re-derive threshold from background distribution of composite
        comp_bg = composite[bg_mask]
        composite_baseline = float(np.median(comp_bg))
        composite_mad = float(np.median(np.abs(comp_bg - composite_baseline)))
        composite_threshold = composite_baseline + config.composite_mad_scale * max(
            composite_mad, 1e-6
        )

        new_candidates_raw = _mask_to_candidates(composite >= composite_threshold, bins)

        # 3f. Trim boundaries
        trimmed: list[dict] = []
        for c in new_candidates_raw:
            tc = _trim_candidate(c, composite, composite_threshold, config.extent_frac, bins, n_bins)
            if tc is not None:
                trimmed.append(tc)

        # 3g. Merge nearby candidates (relaxed valley during iteration)
        candidates = _iter_merge(
            trimmed, composite, t_centers, bin_size,
            burstlet_merge_gap_s, composite_threshold, config.merge_floor_frac,
        )

        new_mask = _candidates_to_mask(candidates, t_centers)
        convergence_delta = float(np.sum(new_mask != prev_mask)) / n_bins
        prev_mask = new_mask

        if convergence_delta < config.convergence_eps:
            break

    # -----------------------------------------------------------------------
    # Phase 4: Build burstlet events from converged candidates
    # -----------------------------------------------------------------------
    ff1 = ff_signals[:, min(1, n_ff - 1)]  # 1× bin-scale FF for quality output

    burstlets_raw: list[dict] = []
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
        peak_time = float(t_centers[peak_abs_idx])
        comp_vals = composite[in_ev]

        burstlets_raw.append({
            "start": float(s_t),
            "end": float(e_t),
            "duration_s": float(duration_s),
            "peak_synchrony": float(ws_sharp[in_ev].max()),
            "peak_time": peak_time,
            "synchrony_energy": float(ws_smooth[in_ev].sum() * bin_size),
            "participation": participating / n_units,
            "total_spikes": total_spikes,
            "burst_peak": float(pfr[in_ev].max()),
            "fragment_count": 1,
            "llr_aggregate": float(llr_signal[in_ev].mean()),
            "composite_peak": float(comp_vals.max()),
            "composite_mean": float(comp_vals.mean()),
            "ff_peak": float(ff1[in_ev].max()),
        })

    # -----------------------------------------------------------------------
    # Phase 4: Hierarchy merge (burstlets → network bursts → superbursts)
    # -----------------------------------------------------------------------
    hier_ctx = dict(
        units=units, spike_times=spike_times, n_units=n_units,
        composite=composite, t_centers=t_centers, bin_size=bin_size,
        spike_counts_total=spike_counts_total, pfr=pfr,
        ws_sharp=ws_sharp, ws_smooth=ws_smooth, ff1=ff1, llr_signal=llr_signal,
    )

    network_bursts = _merge_strict_hier(
        burstlets_raw, gap=burstlet_merge_gap_s, threshold=composite_threshold, **hier_ctx
    )
    superbursts = _merge_clustered_hier(
        network_bursts, gap=network_merge_gap_s,
        baseline=composite_baseline, threshold=composite_threshold, **hier_ctx
    )

    # -----------------------------------------------------------------------
    # Phase 5: Output
    # -----------------------------------------------------------------------
    def _to_df(evs: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(evs) if evs else pd.DataFrame()

    metrics = {
        "burstlets": _level_metrics(burstlets_raw, total_dur),
        "network_bursts": _level_metrics(network_bursts, total_dur),
        "superbursts": _level_metrics(superbursts, total_dur),
    }

    diagnostics = {
        "adaptive_bin_ms": adaptive_bin_ms,
        "biological_isi_s": biological_isi_s,
        "n_iterations": n_iterations,
        "convergence_delta": float(convergence_delta),
        "feature_weights_final": w.tolist(),
        "lambda_bg_per_unit": {u: float(lambda_bg_per_unit[i]) for i, u in enumerate(units)},
        "composite_threshold": float(composite_threshold),
        "composite_baseline": float(composite_baseline),
        "ff_scales_ms": ff_scales_ms,
        "init_method": init_method,
        "burstlet_merge_gap_s": float(burstlet_merge_gap_s),
        "network_merge_gap_s": float(network_merge_gap_s),
        "n_units": n_units,
        "sigma_fast_bins": sigma_fast,
        "sigma_slow_bins": sigma_slow,
    }

    plot_data = {
        "t": t_centers,
        "participation_signal": ws_sharp,
        "rate_signal": ws_smooth,
        "composite_signal": composite,
        "ff_signal": ff1,
        "llr_signal": llr_signal,
        "burst_peak_times": np.array([b["peak_time"] for b in network_bursts]),
        "burst_peak_values": np.array([b["peak_synchrony"] for b in network_bursts]),
        "participation_baseline": float(baseline_init),
        "participation_threshold": float(composite_threshold),
    }

    return BurstResults(
        burstlets=_to_df(burstlets_raw),
        network_bursts=_to_df(network_bursts),
        superbursts=_to_df(superbursts),
        metrics=metrics,
        diagnostics=diagnostics,
        plot_data=plot_data,
    )
