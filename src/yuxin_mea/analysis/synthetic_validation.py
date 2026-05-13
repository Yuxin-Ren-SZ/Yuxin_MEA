"""Synthetic spike-train generators + ground-truth evaluation.

Lets the burst-detector test suite exercise the core algorithms against
known inputs. The notebook ``notebooks/06_iterative_burst_detector_synthetic_validation.ipynb``
used to inline all of this; Phase 4 promotes the pieces worth having
unit tests for into this module.

Public surface:
- ``SyntheticDataset`` dataclass: the data contract.
- Interval utilities: ``merge_intervals``, ``complement_intervals``,
  ``poisson_spikes_in_intervals``, ``make_unit_ids``.
- Generators: ``generate_poisson_baseline``, ``generate_cascade_culture``.
- Evaluation: ``score_detection``.

Notes for the curious:
- All generators take a NumPy ``rng`` and an explicit ``seed`` so results
  are reproducible.
- All ground-truth ``burst_intervals`` are sorted + non-overlapping by
  construction (the generators merge before returning). ``score_detection``
  relies on this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@dataclass
class SyntheticDataset:
    """One synthetic recording with ground-truth burst/silence intervals.

    The detector under test receives ``spike_times`` and produces detected
    burst intervals; tests compare those against ``burst_intervals`` via
    :func:`score_detection`.
    """

    name: str
    culture_type: str
    description: str
    spike_times: dict[str, np.ndarray]
    duration_s: float
    burst_intervals: list[tuple[float, float]]
    silence_intervals: list[tuple[float, float]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Interval utilities
# ---------------------------------------------------------------------------


def merge_intervals(
    intervals: list[tuple[float, float]],
    duration_s: float,
) -> list[tuple[float, float]]:
    """Clip to ``[0, duration_s]``, sort, and merge overlapping spans."""
    clipped = sorted(
        (max(0.0, s), min(duration_s, e))
        for s, e in intervals
        if e > 0 and s < duration_s
    )
    if not clipped:
        return []
    merged: list[tuple[float, float]] = [clipped[0]]
    for s, e in clipped[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def complement_intervals(
    duration_s: float,
    blocked: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the gaps between merged ``blocked`` intervals within ``[0, duration_s]``."""
    blocked = merge_intervals(blocked, duration_s)
    allowed: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in blocked:
        if s > cursor:
            allowed.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration_s:
        allowed.append((cursor, duration_s))
    return allowed


def poisson_spikes_in_intervals(
    rate_hz: float,
    intervals: list[tuple[float, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a homogeneous Poisson process at ``rate_hz`` restricted to ``intervals``."""
    if rate_hz <= 0 or not intervals:
        return np.array([], dtype=float)
    pieces: list[np.ndarray] = []
    for s, e in intervals:
        n = rng.poisson(rate_hz * max(0.0, e - s))
        if n:
            pieces.append(rng.uniform(s, e, n))
    if not pieces:
        return np.array([], dtype=float)
    return np.sort(np.concatenate(pieces).astype(float))


def make_unit_ids(n_units: int) -> list[str]:
    """Return ``["unit_000", "unit_001", ...]``."""
    return [f"unit_{i:03d}" for i in range(n_units)]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_poisson_baseline(
    n_units: int,
    duration_s: float,
    rate_hz: float = 1.0,
    seed: int = 0,
) -> SyntheticDataset:
    """No bursts: every unit fires Poisson at ``rate_hz`` for the whole recording.

    The detector should produce zero or near-zero network bursts on this
    dataset — it's the negative control.
    """
    rng = np.random.default_rng(seed)
    spike_times = {
        uid: poisson_spikes_in_intervals(rate_hz, [(0.0, duration_s)], rng)
        for uid in make_unit_ids(n_units)
    }
    return SyntheticDataset(
        name=f"poisson-baseline-{n_units}u-{rate_hz:.2g}hz-seed{seed}",
        culture_type="independent",
        description=(
            "Independent Poisson units at uniform rate — negative control: "
            "no network bursts expected."
        ),
        spike_times=spike_times,
        duration_s=duration_s,
        burst_intervals=[],
        silence_intervals=[],
        params={"n_units": n_units, "rate_hz": rate_hz, "seed": seed},
    )


def generate_cascade_culture(
    n_units: int,
    duration_s: float,
    burst_centers_s: list[float],
    burst_duration_s: float = 0.5,
    burst_rate_hz: float = 50.0,
    baseline_rate_hz: float = 0.5,
    recruitment: float = 0.8,
    seed: int = 0,
) -> SyntheticDataset:
    """A network with discrete burst epochs at given centers.

    During each burst, a fraction ``recruitment`` of units fires at
    ``burst_rate_hz``. Between bursts every unit fires at
    ``baseline_rate_hz``. This is the simplest positive control.

    ``burst_intervals`` are the (center - duration/2, center + duration/2)
    intervals, merged if they overlap.
    """
    rng = np.random.default_rng(seed)
    half = burst_duration_s / 2
    burst_intervals = merge_intervals(
        [(c - half, c + half) for c in burst_centers_s],
        duration_s,
    )
    baseline_intervals = complement_intervals(duration_s, burst_intervals)

    unit_ids = make_unit_ids(n_units)
    n_recruited = max(1, int(round(recruitment * n_units)))
    recruited = set(rng.choice(unit_ids, size=n_recruited, replace=False).tolist())

    spike_times: dict[str, np.ndarray] = {}
    for uid in unit_ids:
        baseline = poisson_spikes_in_intervals(baseline_rate_hz, baseline_intervals, rng)
        burst = (
            poisson_spikes_in_intervals(burst_rate_hz, burst_intervals, rng)
            if uid in recruited
            else np.array([], dtype=float)
        )
        spike_times[uid] = np.sort(np.concatenate([baseline, burst]))

    return SyntheticDataset(
        name=f"cascade-{n_units}u-{len(burst_centers_s)}b-seed{seed}",
        culture_type="connected",
        description=(
            "Discrete network bursts at fixed centers. "
            f"{n_recruited}/{n_units} units recruited per burst."
        ),
        spike_times=spike_times,
        duration_s=duration_s,
        burst_intervals=burst_intervals,
        silence_intervals=[],
        params={
            "n_units": n_units,
            "burst_centers_s": list(burst_centers_s),
            "burst_duration_s": burst_duration_s,
            "burst_rate_hz": burst_rate_hz,
            "baseline_rate_hz": baseline_rate_hz,
            "recruitment": recruitment,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def score_detection(
    detected: list[tuple[float, float]],
    ground_truth: list[tuple[float, float]],
    min_overlap_s: float = 0.03,
) -> dict[str, float | int]:
    """Score detected burst intervals against ground truth.

    A detected interval ``d`` is a true positive iff it overlaps some
    ground-truth interval ``g`` by ≥ ``min_overlap_s`` AND that ``g`` is
    not already claimed by a previous detection. Each ground-truth burst
    can match at most one detection (the earliest-overlapping one) so
    fragmented detections inflate FP, not TP.

    Returns ``{"tp", "fp", "fn", "precision", "recall", "f1"}``.
    ``ground_truth=[]`` short-circuits to ``recall=nan`` so a Poisson
    baseline can still be evaluated for FP rate.
    """
    detected_sorted = sorted(detected)
    gt_sorted = sorted(ground_truth)
    claimed = [False] * len(gt_sorted)

    tp = 0
    for d_start, d_end in detected_sorted:
        for gi, (g_start, g_end) in enumerate(gt_sorted):
            if claimed[gi]:
                continue
            overlap = max(0.0, min(d_end, g_end) - max(d_start, g_start))
            if overlap >= min_overlap_s:
                claimed[gi] = True
                tp += 1
                break
        # Un-matched detections are accounted for via `fp = len(detected) - tp`.

    fp = len(detected_sorted) - tp
    fn = len(gt_sorted) - tp

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if (precision != precision) or (recall != recall) or (precision + recall == 0):  # NaN-safe
        f1: float = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }
