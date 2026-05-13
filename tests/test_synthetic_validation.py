"""Tests for ``yuxin_mea.analysis.synthetic_validation``.

Focus areas:
- Interval algebra (merge/complement) edge cases.
- Generator output shape + sortedness + reproducibility under seed.
- ``score_detection`` precision/recall/F1 on known inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from yuxin_mea.analysis.synthetic_validation import (
    SyntheticDataset,
    complement_intervals,
    generate_cascade_culture,
    generate_poisson_baseline,
    make_unit_ids,
    merge_intervals,
    poisson_spikes_in_intervals,
    score_detection,
)


# ---------------------------------------------------------------------------
# Interval utilities
# ---------------------------------------------------------------------------


def test_merge_intervals_clips_and_merges():
    assert merge_intervals([(-1.0, 2.0), (1.5, 3.0), (5.0, 7.0)], duration_s=6.0) == [
        (0.0, 3.0),
        (5.0, 6.0),
    ]


def test_merge_intervals_drops_out_of_range():
    assert merge_intervals([(-2.0, -1.0), (8.0, 9.0)], duration_s=5.0) == []


def test_merge_intervals_empty_input():
    assert merge_intervals([], duration_s=10.0) == []


def test_complement_intervals_basic():
    assert complement_intervals(10.0, [(2.0, 4.0), (6.0, 8.0)]) == [
        (0.0, 2.0),
        (4.0, 6.0),
        (8.0, 10.0),
    ]


def test_complement_of_empty_is_full_span():
    assert complement_intervals(5.0, []) == [(0.0, 5.0)]


def test_complement_of_full_span_is_empty():
    assert complement_intervals(5.0, [(0.0, 5.0)]) == []


# ---------------------------------------------------------------------------
# Poisson generator + helpers
# ---------------------------------------------------------------------------


def test_poisson_spikes_obey_intervals():
    rng = np.random.default_rng(0)
    spikes = poisson_spikes_in_intervals(50.0, [(1.0, 2.0), (4.0, 4.5)], rng)
    assert (spikes >= 1.0).all() and ((spikes <= 2.0) | ((spikes >= 4.0) & (spikes <= 4.5))).all()
    assert np.all(np.diff(spikes) >= 0)  # sorted


def test_poisson_zero_rate_returns_empty():
    rng = np.random.default_rng(0)
    assert poisson_spikes_in_intervals(0.0, [(0.0, 10.0)], rng).size == 0


def test_make_unit_ids_zero_padding():
    ids = make_unit_ids(5)
    assert ids == ["unit_000", "unit_001", "unit_002", "unit_003", "unit_004"]


# ---------------------------------------------------------------------------
# Poisson baseline generator
# ---------------------------------------------------------------------------


def test_generate_poisson_baseline_shape():
    ds = generate_poisson_baseline(n_units=10, duration_s=30.0, rate_hz=2.0, seed=42)
    assert isinstance(ds, SyntheticDataset)
    assert ds.duration_s == 30.0
    assert ds.burst_intervals == []
    assert len(ds.spike_times) == 10
    for spikes in ds.spike_times.values():
        assert spikes.dtype == np.float64
        assert np.all(np.diff(spikes) >= 0)  # sorted
        assert (spikes >= 0).all() and (spikes <= 30.0).all()


def test_generate_poisson_baseline_reproducible_under_seed():
    a = generate_poisson_baseline(n_units=5, duration_s=10.0, seed=123)
    b = generate_poisson_baseline(n_units=5, duration_s=10.0, seed=123)
    for unit_id in a.spike_times:
        np.testing.assert_array_equal(a.spike_times[unit_id], b.spike_times[unit_id])


# ---------------------------------------------------------------------------
# Cascade generator
# ---------------------------------------------------------------------------


def test_generate_cascade_records_burst_intervals():
    ds = generate_cascade_culture(
        n_units=20, duration_s=60.0,
        burst_centers_s=[10.0, 30.0, 50.0],
        burst_duration_s=1.0,
        seed=7,
    )
    assert ds.burst_intervals == [(9.5, 10.5), (29.5, 30.5), (49.5, 50.5)]
    assert len(ds.spike_times) == 20


def test_cascade_recruited_units_fire_more_during_bursts():
    """Recruited units should produce more spikes in burst intervals than baseline-only."""
    ds = generate_cascade_culture(
        n_units=30, duration_s=60.0,
        burst_centers_s=[5.0, 15.0, 25.0, 35.0, 45.0, 55.0],
        burst_duration_s=2.0, burst_rate_hz=100.0, baseline_rate_hz=0.1,
        recruitment=0.5, seed=1,
    )
    # Total spike counts vary widely between recruited and non-recruited units.
    counts = sorted(len(s) for s in ds.spike_times.values())
    # Lowest 25% should be small (baseline-only); highest 25% should be much larger.
    q1 = counts[len(counts) // 4]
    q3 = counts[3 * len(counts) // 4]
    assert q3 > q1 * 3  # crude but stable bimodality check


# ---------------------------------------------------------------------------
# score_detection
# ---------------------------------------------------------------------------


def test_score_detection_perfect_match():
    gt = [(1.0, 2.0), (5.0, 6.0)]
    detected = [(1.05, 1.95), (5.1, 5.9)]
    s = score_detection(detected, gt, min_overlap_s=0.5)
    assert s["tp"] == 2
    assert s["fp"] == 0
    assert s["fn"] == 0
    assert s["precision"] == 1.0
    assert s["recall"] == 1.0
    assert s["f1"] == 1.0


def test_score_detection_extra_detection_is_fp():
    gt = [(1.0, 2.0)]
    detected = [(1.0, 2.0), (5.0, 6.0)]
    s = score_detection(detected, gt, min_overlap_s=0.5)
    assert s["tp"] == 1
    assert s["fp"] == 1
    assert s["fn"] == 0
    assert s["precision"] == 0.5
    assert s["recall"] == 1.0


def test_score_detection_missed_ground_truth_is_fn():
    gt = [(1.0, 2.0), (5.0, 6.0)]
    detected = [(1.0, 2.0)]
    s = score_detection(detected, gt, min_overlap_s=0.5)
    assert s["tp"] == 1
    assert s["fp"] == 0
    assert s["fn"] == 1
    assert s["recall"] == 0.5


def test_score_detection_overlap_threshold_enforced():
    gt = [(1.0, 2.0)]
    # 0.01s overlap — below the default 0.03s threshold
    detected = [(1.99, 3.0)]
    s = score_detection(detected, gt, min_overlap_s=0.03)
    assert s["tp"] == 0
    assert s["fp"] == 1
    assert s["fn"] == 1


def test_score_detection_fragmented_detection_inflates_fp():
    """Two short detections within one GT burst: only one is a TP, the other is FP."""
    gt = [(0.0, 1.0)]
    detected = [(0.0, 0.4), (0.5, 0.9)]
    s = score_detection(detected, gt, min_overlap_s=0.1)
    assert s["tp"] == 1
    assert s["fp"] == 1


def test_score_detection_empty_ground_truth():
    """For a Poisson baseline (no GT bursts), recall is NaN; precision still computed."""
    s = score_detection([(1.0, 2.0)], [], min_overlap_s=0.5)
    assert s["tp"] == 0
    assert s["fp"] == 1
    assert s["fn"] == 0
    assert math.isnan(s["recall"])
    assert s["precision"] == 0.0


def test_score_detection_empty_inputs():
    s = score_detection([], [], min_overlap_s=0.5)
    assert s["tp"] == 0 and s["fp"] == 0 and s["fn"] == 0
    assert math.isnan(s["precision"])
    assert math.isnan(s["recall"])
    assert math.isnan(s["f1"])
