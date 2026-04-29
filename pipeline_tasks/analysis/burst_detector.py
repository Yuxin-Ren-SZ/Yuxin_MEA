from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class BurstDetectorError(ValueError):
    """Raised when spike data is insufficient to run burst detection."""


@dataclass(frozen=True)
class BurstDetectorConfig:
    """All tunable parameters for the parameter-free network burst detector.

    Parameters not yet wired to active filtering (gamma, min_burstlet_participation,
    min_absolute_rate_hz, min_burst_density_hz, min_relative_height) are kept for
    API stability; their corresponding filter conditions remain disabled.
    """

    gamma: float = 1.0
    min_burstlet_participation: float = 0.20
    min_absolute_rate_hz: float = 0.5
    min_burst_density_hz: float = 1.0
    min_relative_height: float = 0.1
    extent_frac: float = 0.30
    network_merge_gap_min_s: float = 0.75


@dataclass
class BurstResults:
    """Structured output of compute_network_bursts.

    Each event-level DataFrame has rows = events, columns = per-event scalars.
    Burstlet columns: start, end, duration_s, peak_synchrony, peak_time,
        synchrony_energy, participation, total_spikes, burst_peak.
    Network-burst / superburst columns add: fragment_count, n_sub_events.
    """

    burstlets: pd.DataFrame
    network_bursts: pd.DataFrame
    superbursts: pd.DataFrame
    metrics: dict
    diagnostics: dict
    plot_data: dict


# ---------------------------------------------------------------------------
# Internal helpers
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


def _get_valley_min(
    prev: dict,
    nxt: dict,
    ws_sharp: np.ndarray,
    t_centers: np.ndarray,
) -> float | None:
    valley_mask = (t_centers >= prev["end"]) & (t_centers <= nxt["start"])
    if not np.any(valley_mask):
        return None
    valley_vals = ws_sharp[valley_mask]
    return float(np.min(valley_vals)) if valley_vals.size > 0 else None


def _finalize(
    evs: list[dict],
    s: float,
    e: float,
    units: list,
    spike_times: dict,
    n_units: int,
) -> dict:
    best = max(evs, key=lambda x: x["peak_synchrony"])
    participating_units = sum(
        1 for u in units
        if np.any((spike_times[u] >= s) & (spike_times[u] < e))
    )
    return {
        "start": s,
        "end": e,
        "duration_s": e - s,
        "peak_synchrony": best["peak_synchrony"],
        "peak_time": best["peak_time"],
        "synchrony_energy": sum(ev["synchrony_energy"] for ev in evs),
        "fragment_count": sum(ev.get("fragment_count", 1) for ev in evs),
        "total_spikes": sum(ev["total_spikes"] for ev in evs),
        "participation": participating_units / n_units,
        "burst_peak": max(ev["burst_peak"] for ev in evs),
        "n_sub_events": len(evs),
    }


def _merge_strict(
    events: list[dict],
    gap: float,
    floor_val: float,
    ws_sharp: np.ndarray,
    t_centers: np.ndarray,
    bin_size: float,
    units: list,
    spike_times: dict,
    n_units: int,
    min_dur: float = 0.0,
) -> list[dict]:
    if not events:
        return []

    events = sorted(events, key=lambda x: x["start"])
    merged: list[dict] = []
    curr = [events[0]]
    s = events[0]["start"]
    e = events[0]["end"]

    for nxt in events[1:]:
        valley_duration = nxt["start"] - e
        valley_min = _get_valley_min(curr[-1], nxt, ws_sharp, t_centers)

        if valley_min is None:
            valley_ok = valley_duration <= bin_size
        else:
            valley_ok = valley_min >= floor_val

        if valley_duration <= gap and valley_ok:
            curr.append(nxt)
            e = max(e, nxt["end"])
        else:
            merged.append(_finalize(curr, s, e, units, spike_times, n_units))
            curr = [nxt]
            s = nxt["start"]
            e = nxt["end"]

    merged.append(_finalize(curr, s, e, units, spike_times, n_units))
    return [m for m in merged if m["duration_s"] >= min_dur]


def _merge_clustered(
    events: list[dict],
    gap: float,
    baseline_val: float,
    threshold_val: float,
    ws_sharp: np.ndarray,
    t_centers: np.ndarray,
    bin_size: float,
    units: list,
    spike_times: dict,
    n_units: int,
    min_dur: float = 0.0,
) -> list[dict]:
    if not events:
        return []

    events = sorted(events, key=lambda x: x["start"])
    merged: list[dict] = []
    curr = [events[0]]
    s = events[0]["start"]
    e = events[0]["end"]

    for nxt in events[1:]:
        valley_duration = nxt["start"] - e
        valley_min = _get_valley_min(curr[-1], nxt, ws_sharp, t_centers)

        if valley_min is None:
            valley_ok = valley_duration <= bin_size
        else:
            # Relaxed: allow dip below burst threshold but not to silence
            valley_ok = baseline_val < valley_min < threshold_val

        if valley_duration <= gap and valley_ok:
            curr.append(nxt)
            e = max(e, nxt["end"])
        else:
            merged.append(_finalize(curr, s, e, units, spike_times, n_units))
            curr = [nxt]
            s = nxt["start"]
            e = nxt["end"]

    merged.append(_finalize(curr, s, e, units, spike_times, n_units))
    return [
        m for m in merged
        if m["duration_s"] >= min_dur and m["n_sub_events"] >= 2
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_network_bursts(
    spike_times: dict[str, np.ndarray],
    config: BurstDetectorConfig | None = None,
) -> BurstResults:
    """Detect burstlets, network bursts, and superbursts from population spike trains.

    Args:
        spike_times: Mapping of unit_id → spike time array (seconds).
        config: Detection parameters. Uses BurstDetectorConfig defaults when None.

    Returns:
        BurstResults with per-level DataFrames, aggregate metrics, diagnostics,
        and raw time-domain signals for visualization.

    Raises:
        BurstDetectorError: When spike_times is empty or contains no spikes.
    """
    if config is None:
        config = BurstDetectorConfig()

    units = list(spike_times.keys())
    if not units:
        raise BurstDetectorError("spike_times contains no units")

    non_empty = [spike_times[u] for u in units if len(spike_times[u]) > 0]
    if not non_empty:
        raise BurstDetectorError("spike_times contains no spikes")
    all_spikes = np.sort(np.concatenate(non_empty))

    rec_start = float(all_spikes[0])
    rec_end = float(all_spikes[-1])
    total_dur = rec_end - rec_start

    # ------------------------------------------------------------------
    # 1. Biological calibration — adaptive bin size from median log-ISI
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Population signals
    # ------------------------------------------------------------------
    n_bins = len(t_centers)
    n_units = len(units)

    active_unit_counts = np.zeros(n_bins)
    spike_counts_total = np.zeros(n_bins)

    for u in units:
        spk = np.asarray(spike_times[u])
        if spk.size == 0:
            continue
        counts, _ = np.histogram(spk, bins=bins)
        active_unit_counts += (counts > 0)
        spike_counts_total += counts

    participation_signal_raw = active_unit_counts / max(1, n_units)
    rate_signal_raw = spike_counts_total / bin_size / max(1, n_units)
    pfr = spike_counts_total / bin_size

    # ------------------------------------------------------------------
    # 3. Smoothing — fast (participation) and slow (rate)
    # ------------------------------------------------------------------
    isi_bins = biological_isi_s / bin_size
    sigma_fast = float(np.clip(isi_bins, 1, 2))
    sigma_slow = float(np.clip(5.0 * isi_bins, 3, 8))

    ws_sharp = gaussian_filter1d(participation_signal_raw, sigma_fast)
    ws_smooth = gaussian_filter1d(rate_signal_raw, sigma_slow)

    burstlet_merge_gap_s = 3 * biological_isi_s
    network_merge_gap_s = max(10 * biological_isi_s, config.network_merge_gap_min_s)

    # ------------------------------------------------------------------
    # 4. Detection thresholds
    # ------------------------------------------------------------------
    participation_floor_count = (
        max(5, 0.15 * n_units) if n_units < 50 else max(10, 0.05 * n_units)
    )
    participation_floor = participation_floor_count / max(1, n_units)

    baseline_val = float(np.median(ws_sharp))
    spread_mad = float(np.median(np.abs(ws_sharp - baseline_val)))
    relative_threshold_val = max(participation_floor, baseline_val + 0.75 * spread_mad)

    # ------------------------------------------------------------------
    # 5. Peak detection
    # ------------------------------------------------------------------
    min_prominence = max(0.5 * spread_mad, 0.02)
    peaks, _ = find_peaks(
        ws_sharp,
        height=relative_threshold_val,
        prominence=min_prominence,
    )

    # ------------------------------------------------------------------
    # 6. Burstlet extraction
    # ------------------------------------------------------------------
    burstlets: list[dict] = []

    for p_idx in peaks:
        peak_val = ws_sharp[p_idx]
        extent_threshold = max(relative_threshold_val, config.extent_frac * peak_val)

        s = p_idx
        while s > 0 and ws_sharp[s - 1] >= extent_threshold:
            s -= 1
        e = p_idx
        while e < n_bins - 1 and ws_sharp[e + 1] >= extent_threshold:
            e += 1

        start_t = float(bins[s])
        end_t = float(bins[e + 1])
        duration_s = end_t - start_t
        if duration_s <= 0:
            continue

        participating = sum(
            1 for u in units
            if np.any((spike_times[u] >= start_t) & (spike_times[u] < end_t))
        )
        participation_frac = participating / n_units

        # Filters below remain disabled — see BurstDetectorConfig docstring
        # if participation_frac < config.min_burstlet_participation:
        #     continue

        total_spikes = int(np.sum(spike_counts_total[s:e + 1]))
        denom = duration_s * max(1, participating)
        burst_density = total_spikes / denom if denom > 0 else 0.0
        peak_drive_rate = float(np.max(rate_signal_raw[s:e + 1]))

        # if burst_density < config.min_burst_density_hz:
        #     continue
        # if peak_drive_rate < config.min_absolute_rate_hz:
        #     continue

        burstlets.append({
            "start": start_t,
            "end": end_t,
            "duration_s": duration_s,
            "peak_synchrony": float(peak_val),
            "peak_time": float(t_centers[p_idx]),
            "synchrony_energy": float(np.sum(ws_smooth[s:e + 1]) * bin_size),
            "participation": participation_frac,
            "total_spikes": total_spikes,
            "burst_peak": float(np.max(pfr[s:e + 1])),
        })

    # ------------------------------------------------------------------
    # 7. Merge: burstlets → network bursts → superbursts
    # ------------------------------------------------------------------
    _merge_ctx = dict(
        ws_sharp=ws_sharp,
        t_centers=t_centers,
        bin_size=bin_size,
        units=units,
        spike_times=spike_times,
        n_units=n_units,
    )

    network_bursts = _merge_strict(
        burstlets,
        gap=burstlet_merge_gap_s,
        floor_val=relative_threshold_val,
        **_merge_ctx,
    )

    superbursts = _merge_clustered(
        network_bursts,
        gap=network_merge_gap_s,
        baseline_val=baseline_val,
        threshold_val=relative_threshold_val,
        **_merge_ctx,
    )

    # ------------------------------------------------------------------
    # 8. Build DataFrames (empty DataFrame on no events)
    # ------------------------------------------------------------------
    def _to_df(events: list[dict]) -> pd.DataFrame:
        if not events:
            return pd.DataFrame()
        return pd.DataFrame(events)

    burstlets_df = _to_df(burstlets)
    network_bursts_df = _to_df(network_bursts)
    superbursts_df = _to_df(superbursts)

    # ------------------------------------------------------------------
    # 9. Aggregate metrics
    # ------------------------------------------------------------------
    metrics = {
        "burstlets": _level_metrics(burstlets, total_dur),
        "network_bursts": _level_metrics(network_bursts, total_dur),
        "superbursts": _level_metrics(superbursts, total_dur),
    }

    diagnostics = {
        "adaptive_bin_ms": adaptive_bin_ms,
        "biological_isi_s": biological_isi_s,
        "baseline_value": baseline_val,
        "spread_mad": spread_mad,
        "merge_floor": relative_threshold_val,
        "burstlet_merge_gap_s": burstlet_merge_gap_s,
        "network_merge_gap_s": network_merge_gap_s,
        "n_units": n_units,
        "sigma_fast_bins": sigma_fast,
        "sigma_slow_bins": sigma_slow,
    }

    plot_data = {
        "t": t_centers,
        "participation_signal": ws_sharp,
        "rate_signal": ws_smooth,
        "burst_peak_times": np.array([b["peak_time"] for b in network_bursts]),
        "burst_peak_values": np.array([b["peak_synchrony"] for b in network_bursts]),
        "participation_baseline": baseline_val,
        "participation_threshold": relative_threshold_val,
    }

    return BurstResults(
        burstlets=burstlets_df,
        network_bursts=network_bursts_df,
        superbursts=superbursts_df,
        metrics=metrics,
        diagnostics=diagnostics,
        plot_data=plot_data,
    )
