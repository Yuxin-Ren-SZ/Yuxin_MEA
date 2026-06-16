"""Shared low-level helpers for burst detectors.

These utilities are detector-agnostic building blocks — adaptive spike-count
matrices, multi-scale Fano Factor, candidate↔mask conversion, relaxed-valley
iteration merges, and per-level summary statistics. They are used by the ML
burst detector (:mod:`ml_burst_detector`, :mod:`ml_burst_features`,
:mod:`ml_burst_cluster`) and contain no detector-specific configuration.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


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


def _compute_spike_matrix(
    spike_times: dict,
    units: list,
    bins: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Build the (n_units × n_bins) spike count matrix.

    Each row is a unit; each column is an adaptive time bin. This matrix is
    the shared substrate for all per-bin features: summing columns gives PFR
    and participation; computing Var/Mean across rows at each column gives the
    spatial Fano Factor; comparing against per-unit background expectations
    gives the LLR signal.
    """
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
    """Spatial Fano Factor at multiple temporal scales. Returns (n_bins, n_scales).

    Fano Factor (FF) at a given temporal scale Δt:
        FF(t) = Var( spike_counts_per_unit in [t, t+Δt) ) /
                Mean( spike_counts_per_unit in [t, t+Δt) )

    For independent Poisson firing:  FF ≈ 1  (variance equals mean).
    During synchronised bursting:    FF >> 1  (units fire together, so the
        variance across units is much larger than the mean count).
    During silence:                  FF → 0  (all units have 0 counts).

    Why multiple scales?
    A short burstlet (< 100 ms) shows elevated FF at fine scales (5–20 ms)
    but is too brief to leave a strong signal at coarse scales. A long network
    burst shows elevated FF at all scales. By providing 4 scales, Fisher's
    discriminant can learn which temporal resolution is most informative for
    each recording, rather than committing to a single scale up front.

    Implementation:
    Rather than re-histogramming from raw spike times at each scale, we reuse
    the fine-resolution spike_matrix by summing columns that fall within each
    coarse bin. This is equivalent to re-binning but avoids re-reading spike
    times. The coarse FF value is then broadcast back to all fine bins within
    that coarse window, and lightly smoothed (sigma=1.5 bins) to reduce
    single-bin artefacts.
    """
    n_bins = len(t_centers)
    n_scales = len(ff_scale_multipliers)
    ff_signals = np.zeros((n_bins, n_scales))
    rec_start = float(bins[0])
    rec_end = float(bins[-1])

    # TODO is there a faster way to do this without a Python loop over scales? The loop is only over 4 scales so it's not a bottleneck, but it would be cleaner if we could vectorize it.
    for k, mult in enumerate(ff_scale_multipliers):
        # Clamp scale to [5 ms, 100 ms] regardless of the adaptive bin size
        # TODO parameterize the min and max.
        scale_dt = float(np.clip(mult * bin_size, 0.005, 0.1))

        coarse_bins = np.arange(rec_start, rec_end + scale_dt, scale_dt)
        n_coarse = len(coarse_bins) - 1
        if n_coarse < 1:
            continue

        # Map each fine-resolution bin center to its coarse bin index
        coarse_idx = np.clip(
            np.searchsorted(coarse_bins, t_centers, side="right") - 1,
            0, n_coarse - 1,
        )

        # Aggregate per-unit spike counts into coarse bins via bincount (vectorized)
        n_units_ff = spike_matrix.shape[0]
        unit_coarse = np.zeros((n_units_ff, n_coarse), dtype=np.float64)
        for i in range(n_units_ff):
            unit_coarse[i] = np.bincount(coarse_idx, weights=spike_matrix[i], minlength=n_coarse)
        mean_c = unit_coarse.mean(axis=0)   # (n_coarse,)
        var_c = unit_coarse.var(axis=0)     # (n_coarse,)
        valid = mean_c > 1e-9
        ff_coarse = np.where(valid, var_c / np.where(valid, mean_c, 1.0), 0.0)

        # Broadcast coarse FF back to fine resolution, then lightly smooth
        # TODO is the smoothing necessary? Lookslike yes.
        ff_fine = ff_coarse[coarse_idx]
        ff_signals[:, k] = gaussian_filter1d(ff_fine, sigma=1.5)

    return ff_signals


def _mask_to_candidates(mask: np.ndarray, bins: np.ndarray) -> list[dict]:
    """Convert a boolean bin mask to a list of contiguous above-threshold regions.

    Each candidate dict stores both the continuous time boundaries (start/end
    in seconds, using bin edges) and the array indices (start_idx/end_idx) so
    downstream steps can slice the feature matrix without recomputing indices.
    """
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


def _valley_min(
    prev: dict,
    nxt: dict,
    composite: np.ndarray,
    t_centers: np.ndarray,
) -> float | None:
    """Minimum composite value in the valley between two adjacent candidates.

    Returns None when there are no bins between the two candidates (they are
    immediately adjacent), in which case callers use the gap duration as a
    proxy.
    """
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
    gap_tolerance_bins: int = 0,
) -> list[dict]:
    """Merge adjacent candidates during iteration using a relaxed valley condition.

    Two candidates are merged when their temporal gap is ≤ ``gap_s`` AND
    one of three valley conditions holds:

      1. No bins lie in the valley (truly adjacent) — gap-width acts as a
         proxy and the merge fires if gap ≤ bin_size.
      2. The gap spans ≤ ``gap_tolerance_bins`` × bin_size — the merge fires
         regardless of valley depth.  This handles brief composite dips
         (1–3 bins) inside one true network burst.
      3. The valley minimum is ≥ ``floor_frac × threshold`` — the
         classical relaxed-floor criterion ("still ``floor_frac`` of burst
         regime").

    The relaxed valley criteria allow the algorithm to merge candidates that
    are clearly part of the same burst even if the composite signal dips
    briefly in the valley, which is common for fast oscillations within a
    burst.  The final hierarchy merge (Phase 4) applies a stricter criterion
    on the converged candidates.
    """
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x["start"])
    merged = []
    cur = candidates[0].copy()
    tol_s = float(gap_tolerance_bins) * bin_size
    for nxt in candidates[1:]:
        gap = nxt["start"] - cur["end"]
        vm = _valley_min(cur, nxt, composite, t_centers)
        if vm is None:
            valley_ok = (gap <= bin_size)
        elif gap_tolerance_bins > 0 and gap <= tol_s:
            valley_ok = True
        else:
            valley_ok = (vm >= floor_frac * threshold)
        if gap <= gap_s and valley_ok:
            # Extend current candidate to absorb nxt
            cur = {"start": cur["start"], "end": nxt["end"],
                   "start_idx": cur["start_idx"], "end_idx": nxt["end_idx"]}
        else:
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)
    return merged
