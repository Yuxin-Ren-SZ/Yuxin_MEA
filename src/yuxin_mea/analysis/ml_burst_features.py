"""Bin-level feature matrix for the ML burst detector.

Combines per-unit HMM posteriors with population statistics, per-unit Poisson
LLR (using HMM-fit two-rate λ), inverted ISI, ISI shape, ΔFR vs each unit's
own background, and short/long temporal derivatives. The output is a single
(n_bins, n_features) float matrix that HDBSCAN clusters on.

Why so many features?
---------------------
The plan rests on two ideas:
  1. **Per-unit calibration** — a tonic high-rate unit and a sparse unit that
     only fires during bursts look identical in pooled population signals; the
     per-unit HMM posterior and per-unit two-rate LLR fix that.
  2. **Aggregation across units, not over them** — HDBSCAN sees one point per
     bin, so we collapse the unit axis using four complementary statistics
     (fraction, mean, std+top-quantile, entropy) per per-unit signal. Each
     captures something different about coordination.

Population/dynamics features (PFR, participation, FF, derivatives) are kept
because they encode information the per-unit HMM cannot: total throughput,
multi-scale spatial coordination, and how rapidly the network is changing.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .iterative_burst_detector import _compute_multiscale_ff, _compute_spike_matrix
from .ml_burst_hmm import UnitHMMFit


def _nanaggregate(arr: np.ndarray, func, axis: int = 0) -> np.ndarray:
    """np.nan* aggregation that returns 0 (not NaN) when every value is NaN.

    Some bins may have zero fit units overall (e.g. all units skipped); we want
    a numeric feature value there rather than propagating NaN downstream.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        out = func(arr, axis=axis)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _posterior_aggregates(
    posteriors: np.ndarray,
    quantile: float = 0.9,
) -> dict[str, np.ndarray]:
    """Five summary statistics across units for the per-unit burst posterior.

    Each is a length-n_bins array. NaN rows (skipped units) are excluded by
    np.nan* aggregation. Result keys mirror the feature column names.
    """
    valid_mask = ~np.isnan(posteriors)
    n_valid_per_bin = valid_mask.sum(axis=0)
    safe_denom = np.where(n_valid_per_bin > 0, n_valid_per_bin, 1)

    # Fraction of fit units with p(burst) > 0.5
    frac = (np.where(valid_mask, posteriors, 0.0) > 0.5).sum(axis=0) / safe_denom
    frac = np.where(n_valid_per_bin > 0, frac, 0.0)

    mean = _nanaggregate(posteriors, np.nanmean, axis=0)
    std = _nanaggregate(posteriors, np.nanstd, axis=0)
    q = _nanaggregate(
        posteriors,
        lambda a, axis: np.nanquantile(a, float(quantile), axis=axis),
        axis=0,
    )
    # Per-unit Bernoulli entropy of p(burst), averaged across fit units.
    # entropy(p) = -p log(p) - (1-p) log(1-p); equals 0 at p∈{0,1}.
    p = np.clip(posteriors, 1e-9, 1.0 - 1e-9)
    ent = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    ent = _nanaggregate(ent, np.nanmean, axis=0)

    return {
        "post_frac_gt_0_5": frac.astype(float),
        "post_mean": mean.astype(float),
        "post_std": std.astype(float),
        "post_q90": q.astype(float),
        "post_entropy": ent.astype(float),
    }


def _per_unit_llr_from_hmm(
    spike_matrix: np.ndarray,
    fits: list[UnitHMMFit],
    bin_size: float,
) -> np.ndarray:
    """Per-unit two-rate Poisson LLR using HMM-fit (λ_bg, λ_burst).

    For unit u in bin t:
        LLR_u(t) = 2 [ n_u(t) · ln(λ_burst_u / λ_bg_u)
                       − (λ_burst_u − λ_bg_u) · Δt ]

    Returns
    -------
    llr_matrix : (n_units, n_bins) float array. Rows for skipped units are NaN.
    """
    n_units, n_bins = spike_matrix.shape
    out = np.full((n_units, n_bins), np.nan, dtype=float)
    for i, fit in enumerate(fits):
        if fit.skipped_reason is not None:
            continue
        lam_b = max(float(fit.lambda_bg), 1e-9)
        lam_burst = max(float(fit.lambda_burst), 1e-9)
        if lam_burst <= lam_b:
            continue
        ratio = np.log(lam_burst / lam_b)
        diff = (lam_burst - lam_b) * float(bin_size)
        out[i] = 2.0 * (spike_matrix[i] * ratio - diff)
    return out


def _aggregate_unit_matrix(
    matrix: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mean, std, top-quantile aggregation across units (axis 0). NaN-aware."""
    mean = _nanaggregate(matrix, np.nanmean, axis=0)
    std = _nanaggregate(matrix, np.nanstd, axis=0)
    q = _nanaggregate(
        matrix,
        lambda a, axis: np.nanquantile(a, float(quantile), axis=axis),
        axis=0,
    )
    return mean.astype(float), std.astype(float), q.astype(float)


def _inverted_isi_per_unit(
    spike_times: dict,
    units: list,
    bins: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Per-unit instantaneous firing rate (1/ISI) placed at the ISI midpoint bin.

    Returns (n_units, n_bins) with zeros where a unit had no ISI midpoint.
    """
    out = np.zeros((len(units), n_bins), dtype=float)
    cnt = np.zeros((len(units), n_bins), dtype=float)
    for i, u in enumerate(units):
        spk = np.sort(np.asarray(spike_times[u], dtype=float))
        if len(spk) < 2:
            continue
        isi = np.diff(spk)
        midpoints = (spk[:-1] + spk[1:]) / 2
        idx = np.clip(np.searchsorted(bins[1:], midpoints), 0, n_bins - 1)
        inst = 1.0 / np.maximum(isi, 1e-6)
        np.add.at(out[i], idx, inst)
        np.add.at(cnt[i], idx, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, out / np.where(cnt > 0, cnt, 1.0), 0.0)
    return out


def _cv_isi_and_lv_per_unit(
    spike_times: dict,
    units: list,
    bins: np.ndarray,
    n_bins: int,
    window_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit windowed CV(ISI) and Shinomoto LV, broadcast to each bin.

    For each unit, ISIs are computed and placed at the right-spike bin index.
    A sliding window of ``window_bins`` aggregates CV(ISI) = std/mean and
    LV = mean( 3*(ISI[i]-ISI[i+1])^2 / (ISI[i]+ISI[i+1])^2 ). Bins with no ISIs
    in their window get 0.

    Returns (cv_matrix, lv_matrix) each (n_units, n_bins).
    """
    cv_matrix = np.zeros((len(units), n_bins), dtype=float)
    lv_matrix = np.zeros((len(units), n_bins), dtype=float)

    half = max(1, window_bins // 2)
    for i, u in enumerate(units):
        spk = np.sort(np.asarray(spike_times[u], dtype=float))
        if len(spk) < 3:
            continue
        isi = np.diff(spk)
        # Bin index for each ISI (use right-spike position so the metric refers
        # to "what just happened up to this bin").
        idx = np.clip(np.searchsorted(bins[1:], spk[1:]), 0, n_bins - 1)
        # LV is defined on consecutive ISI pairs
        if len(isi) >= 2:
            paired_idx = idx[1:]  # bin index of the *second* ISI in the pair
            lv_pairs = 3.0 * (isi[:-1] - isi[1:]) ** 2 / np.maximum(
                (isi[:-1] + isi[1:]) ** 2, 1e-12
            )
        else:
            paired_idx = np.array([], dtype=int)
            lv_pairs = np.array([], dtype=float)

        # Sliding-window CV and LV using cumulative-sum tricks would be cleaner
        # but the bin count is O(10^4) and unit count is O(10^2), so a simple
        # per-bin sweep is fine for the metric's intended granularity (broad
        # smoothing follows in the aggregator).
        for t in range(n_bins):
            sel_lo, sel_hi = max(0, t - half), min(n_bins - 1, t + half)
            mask = (idx >= sel_lo) & (idx <= sel_hi)
            n = mask.sum()
            if n >= 2:
                isi_win = isi[mask]
                mean_w = isi_win.mean()
                if mean_w > 1e-9:
                    cv_matrix[i, t] = float(isi_win.std() / mean_w)
            if lv_pairs.size > 0:
                pmask = (paired_idx >= sel_lo) & (paired_idx <= sel_hi)
                if pmask.any():
                    lv_matrix[i, t] = float(lv_pairs[pmask].mean())

    return cv_matrix, lv_matrix


def _delta_fr_per_unit(
    spike_matrix: np.ndarray,
    fits: list[UnitHMMFit],
    bin_size: float,
) -> np.ndarray:
    """Per-unit (count/Δt − λ_bg_u) / λ_bg_u. NaN row for skipped units.

    Captures "how unusual is this bin given the unit's own background rate".
    Saturates around 0 when n=λ_bg·Δt; large positive for elevated firing.
    """
    n_units, n_bins = spike_matrix.shape
    out = np.full((n_units, n_bins), np.nan, dtype=float)
    for i, fit in enumerate(fits):
        if fit.skipped_reason is not None:
            continue
        rate = spike_matrix[i] / max(bin_size, 1e-12)
        lam_b = max(float(fit.lambda_bg), 1e-6)
        out[i] = (rate - lam_b) / lam_b
    return out


def _temporal_derivatives(
    signal: np.ndarray,
    sigma_short: float,
    sigma_long: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First-difference of a 1D signal, smoothed at two scales."""
    d = np.diff(signal, prepend=signal[0])
    return (
        gaussian_filter1d(d, sigma=max(sigma_short, 1e-6)),
        gaussian_filter1d(d, sigma=max(sigma_long, 1e-6)),
    )


def build_feature_matrix(
    spike_times: dict,
    units: list,
    bins: np.ndarray,
    t_centers: np.ndarray,
    bin_size: float,
    fits: list[UnitHMMFit],
    posteriors: np.ndarray,
    *,
    ff_scale_multipliers: Iterable[float] = (0.5, 1.0, 2.0, 5.0),
    posterior_quantile: float = 0.9,
    isi_window_bins: int = 25,
    deriv_sigma_short_bins: float = 1.5,
    deriv_sigma_long_bins: float = 8.0,
    unit_agg_quantile: float = 0.9,
    spike_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the bin-level feature matrix.

    Returns
    -------
    X : (n_bins, n_features) float64
    column_names : list[str]
    """
    n_bins = len(t_centers)
    if spike_matrix is None:
        spike_matrix = _compute_spike_matrix(spike_times, units, bins, n_bins)

    n_units = spike_matrix.shape[0]
    n_units_fit = int(sum(1 for f in fits if f.skipped_reason is None))

    # ---- 1. HMM posterior aggregates (5 columns) -----------------------------
    post_agg = _posterior_aggregates(posteriors, quantile=float(posterior_quantile))

    # ---- 2. Population: PFR + participation (2 columns) ----------------------
    spike_counts_total = spike_matrix.sum(axis=0)
    pfr = spike_counts_total / max(bin_size, 1e-12)
    active_unit_counts = (spike_matrix > 0).sum(axis=0).astype(float)
    # Use number of *fit* units for participation when any units are fit; else fall back to all units.
    denom = float(n_units_fit) if n_units_fit > 0 else float(max(1, n_units))
    participation = active_unit_counts / denom

    # ---- 3. Multi-scale Fano Factor (len(ff_scale_multipliers) columns) ------
    ff_mults = tuple(float(x) for x in ff_scale_multipliers)
    ff_signals = _compute_multiscale_ff(spike_matrix, bins, t_centers, bin_size, ff_mults)

    # ---- 4. Per-unit LLR aggregates (3 columns) ------------------------------
    llr_matrix = _per_unit_llr_from_hmm(spike_matrix, fits, bin_size)
    llr_mean, llr_std, llr_q = _aggregate_unit_matrix(llr_matrix, unit_agg_quantile)

    # ---- 5. Inverted ISI aggregates (2 columns) ------------------------------
    inv_isi_matrix = _inverted_isi_per_unit(spike_times, units, bins, n_bins)
    inv_isi_mean = inv_isi_matrix.mean(axis=0)
    inv_isi_std = inv_isi_matrix.std(axis=0)

    # ---- 6. CV(ISI) and LV aggregates (2 columns) ----------------------------
    cv_matrix, lv_matrix = _cv_isi_and_lv_per_unit(
        spike_times, units, bins, n_bins, int(isi_window_bins)
    )
    cv_isi_mean = cv_matrix.mean(axis=0)
    lv_mean = lv_matrix.mean(axis=0)

    # ---- 7. ΔFR vs unit's own λ_bg aggregates (2 columns) --------------------
    dfr_matrix = _delta_fr_per_unit(spike_matrix, fits, bin_size)
    dfr_mean, dfr_std, _ = _aggregate_unit_matrix(dfr_matrix, unit_agg_quantile)

    # ---- 8. Temporal derivatives at short + long scale (6 columns) -----------
    dPFR_short, dPFR_long = _temporal_derivatives(pfr, deriv_sigma_short_bins, deriv_sigma_long_bins)
    dP_short, dP_long = _temporal_derivatives(
        participation, deriv_sigma_short_bins, deriv_sigma_long_bins
    )
    dLLR_short, dLLR_long = _temporal_derivatives(
        llr_mean, deriv_sigma_short_bins, deriv_sigma_long_bins
    )

    columns: list[tuple[str, np.ndarray]] = [
        ("post_frac_gt_0_5", post_agg["post_frac_gt_0_5"]),
        ("post_mean", post_agg["post_mean"]),
        ("post_std", post_agg["post_std"]),
        ("post_q90", post_agg["post_q90"]),
        ("post_entropy", post_agg["post_entropy"]),
        ("PFR", pfr),
        ("participation", participation),
    ]
    for k in range(ff_signals.shape[1]):
        columns.append((f"FF{k}", ff_signals[:, k]))
    columns += [
        ("llr_hmm_mean", llr_mean),
        ("llr_hmm_std", llr_std),
        ("llr_hmm_q90", llr_q),
        ("inv_isi_mean", inv_isi_mean),
        ("inv_isi_std", inv_isi_std),
        ("cv_isi_mean", cv_isi_mean),
        ("lv_mean", lv_mean),
        ("dfr_unit_mean", dfr_mean),
        ("dfr_unit_std", dfr_std),
        ("dPFR_short", dPFR_short),
        ("dPFR_long", dPFR_long),
        ("dParticipation_short", dP_short),
        ("dParticipation_long", dP_long),
        ("dLLR_short", dLLR_short),
        ("dLLR_long", dLLR_long),
    ]

    column_names = [name for name, _ in columns]
    X = np.column_stack([col for _, col in columns]).astype(float, copy=False)
    # Guard against NaN/Inf leaking from numerical edge cases
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, column_names


# Canonical column order for callers that want the names without invoking build.
def feature_names_for(ff_scale_multipliers: Iterable[float]) -> list[str]:
    n_ff = len(tuple(ff_scale_multipliers))
    return (
        [
            "post_frac_gt_0_5",
            "post_mean",
            "post_std",
            "post_q90",
            "post_entropy",
            "PFR",
            "participation",
        ]
        + [f"FF{k}" for k in range(n_ff)]
        + [
            "llr_hmm_mean",
            "llr_hmm_std",
            "llr_hmm_q90",
            "inv_isi_mean",
            "inv_isi_std",
            "cv_isi_mean",
            "lv_mean",
            "dfr_unit_mean",
            "dfr_unit_std",
            "dPFR_short",
            "dPFR_long",
            "dParticipation_short",
            "dParticipation_long",
            "dLLR_short",
            "dLLR_long",
        ]
    )
