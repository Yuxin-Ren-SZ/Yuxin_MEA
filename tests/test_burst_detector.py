from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from pipeline_tasks.analysis.burst_detector import (
    BurstDetectorConfig,
    BurstDetectorError,
    BurstResults,
    compute_network_bursts,
)
from pipeline_tasks.analysis.burst_output import ParquetBurstOutputWriter


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_bursty_spike_trains(
    n_units: int = 10,
    duration_s: float = 120.0,
    burst_starts: tuple[float, ...] = (15.0, 35.0, 55.0, 75.0, 95.0),
    burst_duration_s: float = 0.5,
    burst_rate_hz: float = 20.0,
    background_rate_hz: float = 0.3,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Synthetic spike trains — used for schema tests (not accuracy)."""
    rng = np.random.default_rng(seed)
    spike_times: dict[str, np.ndarray] = {}

    for i in range(n_units):
        spikes: list[float] = []
        t = 0.0
        while t < duration_s:
            t += rng.exponential(1.0 / background_rate_hz)
            spikes.append(t)
        for bs in burst_starts:
            n = rng.poisson(burst_rate_hz * burst_duration_s)
            burst_spikes = bs + rng.uniform(0, burst_duration_s, max(1, n))
            spikes.extend(burst_spikes.tolist())
        spike_times[f"unit_{i}"] = np.sort([s for s in spikes if 0 < s < duration_s])

    return spike_times


def _make_strong_spike_trains(
    n_units: int = 20,
    duration_s: float = 120.0,
    burst_starts: tuple[float, ...] = (15.0, 35.0, 55.0, 75.0, 95.0),
    burst_duration_s: float = 0.5,
    seed: int = 99,
) -> dict[str, np.ndarray]:
    """Prominent bursts guaranteed to exceed the adaptive participation floor.

    Uses 100 Hz burst rate and near-silent background so that every injected
    burst produces participation well above the floor threshold (0.25 for 20
    units).  At 10ms bins and 100 Hz: P(active per bin) ≈ 0.63, giving 12.6
    expected active units vs. a floor of max(5, 3) / 20 = 0.25.
    """
    burst_rate_hz = 100.0
    background_rate_hz = 0.02

    rng = np.random.default_rng(seed)
    spike_times: dict[str, np.ndarray] = {}

    for i in range(n_units):
        spikes: list[float] = []
        t = 0.0
        while t < duration_s:
            t += rng.exponential(1.0 / background_rate_hz)
            spikes.append(t)
        for bs in burst_starts:
            n = rng.poisson(burst_rate_hz * burst_duration_s)
            burst_spikes = bs + rng.uniform(0, burst_duration_s, max(5, n))
            spikes.extend(burst_spikes.tolist())
        spike_times[f"unit_{i}"] = np.sort([s for s in spikes if 0 < s < duration_s])

    return spike_times


def _make_minimal_spike_trains() -> dict[str, np.ndarray]:
    """Two-unit recording with exactly one synchronised burst."""
    return {
        "u0": np.array([10.0, 10.05, 10.10, 10.15, 10.20, 50.0, 100.0]),
        "u1": np.array([10.02, 10.07, 10.12, 10.17, 10.22, 60.0, 110.0]),
    }


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class BurstDetectorConfigTests(unittest.TestCase):
    def test_defaults_are_frozen(self):
        cfg = BurstDetectorConfig()
        with self.assertRaises(Exception):
            cfg.extent_frac = 0.99  # type: ignore[misc]

    def test_custom_values_stored(self):
        cfg = BurstDetectorConfig(extent_frac=0.5, network_merge_gap_min_s=2.0)
        self.assertAlmostEqual(cfg.extent_frac, 0.5)
        self.assertAlmostEqual(cfg.network_merge_gap_min_s, 2.0)


# ---------------------------------------------------------------------------
# Error case tests
# ---------------------------------------------------------------------------

class BurstDetectorErrorCasesTests(unittest.TestCase):
    def test_empty_spike_times_raises(self):
        with self.assertRaises(BurstDetectorError):
            compute_network_bursts({})

    def test_all_empty_arrays_raises(self):
        with self.assertRaises(BurstDetectorError):
            compute_network_bursts({"u0": np.array([]), "u1": np.array([])})

    def test_none_config_uses_defaults(self):
        result = compute_network_bursts(_make_minimal_spike_trains(), config=None)
        self.assertIsInstance(result, BurstResults)


# ---------------------------------------------------------------------------
# Schema tests (structure of BurstResults)
# ---------------------------------------------------------------------------

class BurstResultsSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = compute_network_bursts(_make_bursty_spike_trains())

    def test_returns_burst_results(self):
        self.assertIsInstance(self.result, BurstResults)

    def test_all_event_fields_are_dataframes(self):
        for attr in ("burstlets", "network_bursts", "superbursts"):
            self.assertIsInstance(getattr(self.result, attr), pd.DataFrame)

    def test_burstlet_required_columns(self):
        required = {
            "start", "end", "duration_s", "peak_synchrony", "peak_time",
            "synchrony_energy", "participation", "total_spikes", "burst_peak",
        }
        if not self.result.burstlets.empty:
            self.assertTrue(required.issubset(set(self.result.burstlets.columns)))

    def test_network_burst_has_merge_columns(self):
        if not self.result.network_bursts.empty:
            self.assertTrue({"fragment_count", "n_sub_events"}.issubset(
                set(self.result.network_bursts.columns)
            ))

    def test_superburst_has_n_sub_events(self):
        if not self.result.superbursts.empty:
            self.assertIn("n_sub_events", self.result.superbursts.columns)

    def test_metrics_has_all_levels(self):
        for level in ("burstlets", "network_bursts", "superbursts"):
            self.assertIn(level, self.result.metrics)

    def test_diagnostics_required_keys(self):
        required = {
            "adaptive_bin_ms", "biological_isi_s", "baseline_value",
            "spread_mad", "merge_floor", "n_units",
        }
        self.assertTrue(required.issubset(self.result.diagnostics.keys()))

    def test_plot_data_required_keys(self):
        required = {
            "t", "participation_signal", "rate_signal",
            "burst_peak_times", "burst_peak_values",
            "participation_baseline", "participation_threshold",
        }
        self.assertTrue(required.issubset(self.result.plot_data.keys()))

    def test_plot_signals_are_ndarrays(self):
        for key in ("t", "participation_signal", "rate_signal"):
            self.assertIsInstance(self.result.plot_data[key], np.ndarray)


# ---------------------------------------------------------------------------
# Accuracy tests (uses strong bursts so detection is reliable)
# ---------------------------------------------------------------------------

class BurstDetectionAccuracyTests(unittest.TestCase):
    """Detector should reliably find all injected bursts in high-SNR data."""

    BURST_STARTS = (15.0, 35.0, 55.0, 75.0, 95.0)

    @classmethod
    def setUpClass(cls):
        # Strong bursts: 100 Hz, 20 units, near-silent background.
        # participation_floor = max(5,3)/20 = 0.25; peak participation ≈ 0.63.
        cls.spike_times = _make_strong_spike_trains(burst_starts=cls.BURST_STARTS)
        cls.result = compute_network_bursts(cls.spike_times)

    def test_detects_network_bursts(self):
        self.assertFalse(
            self.result.network_bursts.empty,
            "Expected network bursts to be detected in high-SNR data",
        )

    def test_network_burst_count_near_injected(self):
        n = len(self.result.network_bursts)
        # 5 injected bursts; allow ±1 for edge merges/splits
        self.assertGreaterEqual(n, 4, f"Too few network bursts detected: {n}")
        self.assertLessEqual(n, 7, f"Too many network bursts detected: {n}")

    def test_event_start_before_end(self):
        for attr in ("burstlets", "network_bursts", "superbursts"):
            df = getattr(self.result, attr)
            if not df.empty:
                self.assertTrue((df["end"] > df["start"]).all(), attr)

    def test_participation_in_valid_range(self):
        for attr in ("burstlets", "network_bursts", "superbursts"):
            df = getattr(self.result, attr)
            if not df.empty:
                self.assertTrue((df["participation"] >= 0).all())
                self.assertTrue((df["participation"] <= 1).all())

    def test_diagnostics_n_units_correct(self):
        self.assertEqual(self.result.diagnostics["n_units"], 20)

    def test_lower_extent_frac_produces_longer_events(self):
        """Lower extent_frac → lower boundary threshold → events extend further from peak.

        extent_threshold = max(relative_threshold_val, extent_frac * peak_val)
        A lower extent_frac means extent_threshold stays closer to
        relative_threshold_val, so the while-loop boundary expands further
        out from the peak, yielding longer events.
        """
        low_frac = compute_network_bursts(
            self.spike_times, BurstDetectorConfig(extent_frac=0.10)
        )
        high_frac = compute_network_bursts(
            self.spike_times, BurstDetectorConfig(extent_frac=0.60)
        )

        if low_frac.burstlets.empty or high_frac.burstlets.empty:
            self.skipTest("No burstlets detected — cannot compare extent_frac effect")

        low_mean = low_frac.burstlets["duration_s"].mean()
        high_mean = high_frac.burstlets["duration_s"].mean()
        self.assertGreaterEqual(
            low_mean, high_mean,
            f"extent_frac=0.10 produced shorter events ({low_mean:.4f}s) "
            f"than extent_frac=0.60 ({high_mean:.4f}s)",
        )


# ---------------------------------------------------------------------------
# Parquet round-trip tests
# ---------------------------------------------------------------------------

class ParquetRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = compute_network_bursts(_make_bursty_spike_trains())

    def test_roundtrip_event_dataframes(self):
        writer = ParquetBurstOutputWriter()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "burst_out"
            writer.write(self.original, out)
            loaded = writer.read(out)

        for attr in ("burstlets", "network_bursts", "superbursts"):
            orig_df: pd.DataFrame = getattr(self.original, attr)
            load_df: pd.DataFrame = getattr(loaded, attr)
            if orig_df.empty:
                self.assertTrue(load_df.empty)
            else:
                pd.testing.assert_frame_equal(
                    orig_df.reset_index(drop=True),
                    load_df.reset_index(drop=True),
                    check_like=True,
                )

    def test_roundtrip_diagnostics(self):
        writer = ParquetBurstOutputWriter()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "burst_out"
            writer.write(self.original, out)
            loaded = writer.read(out)

        self.assertEqual(self.original.diagnostics.keys(), loaded.diagnostics.keys())
        for k, v in self.original.diagnostics.items():
            self.assertAlmostEqual(v, loaded.diagnostics[k], places=6)

    def test_roundtrip_plot_signals(self):
        writer = ParquetBurstOutputWriter()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "burst_out"
            writer.write(self.original, out)
            loaded = writer.read(out)

        for key in ("t", "participation_signal", "rate_signal"):
            np.testing.assert_array_almost_equal(
                self.original.plot_data[key],
                loaded.plot_data[key],
            )

    def test_expected_output_files_created(self):
        writer = ParquetBurstOutputWriter()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "burst_out"
            writer.write(self.original, out)

            self.assertEqual(
                {f.name for f in out.iterdir()},
                {
                    "burstlets.parquet",
                    "network_bursts.parquet",
                    "superbursts.parquet",
                    "metrics.json",
                    "diagnostics.json",
                    "plot_signals.npy",
                },
            )


# ---------------------------------------------------------------------------
# Reference equivalence tests
# ---------------------------------------------------------------------------

_REFERENCE_PATH = Path("/Users/yuxinren/Code/SadeghLab/RBS_Readonly/IPNAnalysis")


def _import_reference_detector():
    sys.path.insert(0, str(_REFERENCE_PATH))
    try:
        import importlib
        mod = importlib.import_module("parameter_free_burst_detector")
        return mod.compute_network_bursts
    finally:
        if str(_REFERENCE_PATH) in sys.path:
            sys.path.remove(str(_REFERENCE_PATH))


@unittest.skipUnless(_REFERENCE_PATH.exists(), "RBS_Readonly reference not on disk")
class ReferenceEquivalenceTests(unittest.TestCase):
    """Verify the migrated algorithm is numerically identical to the original.

    The reference function returns a nested dict; we convert both outputs to
    comparable lists of event dicts for comparison.
    """

    @classmethod
    def setUpClass(cls):
        cls.ref_fn = _import_reference_detector()
        cls.spike_times = _make_strong_spike_trains()
        cls.ref_result = cls.ref_fn(SpikeTimes=cls.spike_times)
        cls.our_result = compute_network_bursts(cls.spike_times)

    def _our_events(self, attr: str) -> list[dict]:
        df: pd.DataFrame = getattr(self.our_result, attr)
        return df.to_dict("records") if not df.empty else []

    def _ref_events(self, level: str) -> list[dict]:
        return self.ref_result[level]["events"]

    def _compare_event_lists(self, ref_events: list[dict], our_events: list[dict], label: str):
        self.assertEqual(
            len(ref_events), len(our_events),
            f"{label}: event count mismatch — ref={len(ref_events)}, ours={len(our_events)}",
        )
        for i, (ref_ev, our_ev) in enumerate(zip(ref_events, our_events)):
            for key in ("start", "end", "duration_s", "peak_synchrony", "peak_time",
                        "total_spikes", "participation", "burst_peak"):
                if key not in ref_ev:
                    continue
                self.assertAlmostEqual(
                    ref_ev[key], our_ev[key], places=10,
                    msg=f"{label}[{i}].{key}: ref={ref_ev[key]}, ours={our_ev[key]}",
                )

    def test_burstlet_events_match_reference(self):
        self._compare_event_lists(
            self._ref_events("burstlets"),
            self._our_events("burstlets"),
            "burstlets",
        )

    def test_network_burst_events_match_reference(self):
        self._compare_event_lists(
            self._ref_events("network_bursts"),
            self._our_events("network_bursts"),
            "network_bursts",
        )

    def test_superburst_events_match_reference(self):
        self._compare_event_lists(
            self._ref_events("superbursts"),
            self._our_events("superbursts"),
            "superbursts",
        )

    def test_diagnostics_match_reference(self):
        ref_diag = self.ref_result["diagnostics"]
        our_diag = self.our_result.diagnostics
        for key in (
            "adaptive_bin_ms", "biological_isi_s", "baseline_value",
            "spread_mad", "merge_floor", "n_units",
            "sigma_fast_bins", "sigma_slow_bins",
        ):
            self.assertAlmostEqual(
                ref_diag[key], our_diag[key], places=10,
                msg=f"diagnostics[{key}]: ref={ref_diag[key]}, ours={our_diag[key]}",
            )

    def test_plot_signals_match_reference(self):
        ref_plot = self.ref_result["plot_data"]
        our_plot = self.our_result.plot_data
        for key in ("t", "participation_signal", "rate_signal",
                    "participation_baseline", "participation_threshold"):
            np.testing.assert_array_almost_equal(
                ref_plot[key], our_plot[key],
                decimal=10,
                err_msg=f"plot_data[{key}] differs from reference",
            )


if __name__ == "__main__":
    unittest.main()
