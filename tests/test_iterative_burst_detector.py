from __future__ import annotations

import unittest

import numpy as np

from pipeline_tasks.analysis.iterative_burst_detector import (
    IterativeBurstConfig,
    IterativeBurstTrace,
    compute_iterative_bursts,
)


def _poisson_spike_trains(
    rates_hz: np.ndarray,
    duration_s: float = 150.0,
    seed: int = 20260510,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    spike_times: dict[str, np.ndarray] = {}
    for i, rate in enumerate(rates_hz):
        n_spikes = rng.poisson(float(rate) * duration_s)
        spike_times[f"unit_{i:02d}"] = np.sort(rng.uniform(0.0, duration_s, n_spikes))
    return spike_times


def _cascade_spike_trains(
    n_units: int = 24,
    duration_s: float = 150.0,
    burst_starts: tuple[float, ...] = (18.0, 42.0, 66.0, 90.0, 114.0, 138.0),
    seed: int = 20260511,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    units = [f"unit_{i:02d}" for i in range(n_units)]
    groups = np.array_split(np.arange(n_units), 4)
    spike_times: dict[str, list[float]] = {unit: [] for unit in units}

    for unit in units:
        n_background = rng.poisson(0.6 * duration_s)
        spike_times[unit].extend(rng.uniform(0.0, duration_s, n_background).tolist())

    for burst_start in burst_starts:
        for group_idx, group in enumerate(groups):
            group_start = burst_start + group_idx * 0.08
            group_end = group_start + 0.22
            for unit_idx in group:
                if rng.random() > 0.85:
                    continue
                n_spikes = max(1, int(rng.poisson(60.0 * 0.22)))
                spikes = rng.normal(
                    loc=(group_start + group_end) / 2,
                    scale=0.22 / 7,
                    size=n_spikes,
                )
                spikes = spikes[(spikes >= group_start) & (spikes < group_end)]
                if spikes.size == 0:
                    spikes = np.array([rng.uniform(group_start, group_end)])
                spike_times[units[int(unit_idx)]].extend(spikes.tolist())

    return {
        unit: np.sort(np.asarray([s for s in spikes if 0.0 <= s < duration_s]))
        for unit, spikes in spike_times.items()
    }


def _heterogeneous_spike_trains(
    n_units: int = 24,
    duration_s: float = 150.0,
    burst_starts: tuple[float, ...] = (90.0, 108.0, 126.0, 138.0),
    seed: int = 20260512,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    units = [f"unit_{i:02d}" for i in range(n_units)]
    groups = np.array_split(np.arange(n_units), 4)
    spike_times: dict[str, list[float]] = {unit: [] for unit in units}

    for unit in units:
        n_background = rng.poisson(0.2 * duration_s)
        spike_times[unit].extend(rng.uniform(0.0, duration_s, n_background).tolist())

    for burst_start in burst_starts:
        for group_idx, group in enumerate(groups):
            group_start = burst_start + group_idx * 0.06
            group_end = group_start + 0.18
            for unit_idx in group:
                if rng.random() > 0.9:
                    continue
                n_spikes = max(1, int(rng.poisson(80.0 * 0.18)))
                spikes = rng.normal(
                    loc=(group_start + group_end) / 2,
                    scale=0.18 / 8,
                    size=n_spikes,
                )
                spikes = spikes[(spikes >= group_start) & (spikes < group_end)]
                if spikes.size == 0:
                    spikes = np.array([rng.uniform(group_start, group_end)])
                spike_times[units[int(unit_idx)]].extend(spikes.tolist())

    return {
        unit: np.sort(np.asarray([s for s in spikes if 0.0 <= s < duration_s]))
        for unit, spikes in spike_times.items()
    }


class IterativeBurstDetectorSyntheticControlsTests(unittest.TestCase):
    def test_independent_poisson_controls_do_not_produce_network_bursts(self):
        cases = {
            "low": np.full(24, 0.15),
            "medium": np.full(24, 0.8),
            "high": np.full(24, 3.0),
            "mixed": np.exp(np.linspace(np.log(0.05), np.log(5.0), 24)),
        }
        for label, rates in cases.items():
            with self.subTest(label=label):
                result = compute_iterative_bursts(_poisson_spike_trains(rates))
                self.assertTrue(
                    result.network_bursts.empty,
                    f"{label} independent Poisson control produced network bursts",
                )

    def test_cascade_bursts_are_detected_without_long_interburst_events(self):
        spike_times = _cascade_spike_trains()
        result = compute_iterative_bursts(spike_times, IterativeBurstConfig())

        n_network_bursts = len(result.network_bursts)
        self.assertGreaterEqual(n_network_bursts, 4)
        self.assertLessEqual(n_network_bursts, 8)
        self.assertTrue(
            (result.network_bursts["duration_s"] <= 2.0).all(),
            "Detected network bursts should not span long inter-burst background intervals",
        )

    def test_heterogeneous_recording_keeps_bursty_section(self):
        spike_times = _heterogeneous_spike_trains()
        result = compute_iterative_bursts(spike_times, IterativeBurstConfig())

        self.assertTrue(result.burstlets.shape[0] > 0)
        self.assertTrue(result.network_bursts.shape[0] > 0)
        self.assertTrue(
            (result.network_bursts["start"] >= 75.0).any(),
            "A recording with a silent first half and bursty second half should keep burst events in the bursty section",
        )
        self.assertTrue(result.diagnostics["burst_activity_detected"])

    def test_cluster_merges_similar_burstlets(self):
        spike_times = _cascade_spike_trains()
        unclustered = compute_iterative_bursts(
            spike_times,
            IterativeBurstConfig(min_burst_modulation=0.0, cluster_events=False),
        )
        clustered = compute_iterative_bursts(
            spike_times,
            IterativeBurstConfig(
                min_burst_modulation=0.1,
                cluster_events=True,
                cluster_initial_components=6,
            ),
        )

        self.assertGreater(len(unclustered.network_bursts), len(clustered.network_bursts))
        self.assertLessEqual(len(clustered.network_bursts), 8)
        self.assertTrue((clustered.network_bursts["duration_s"] <= 2.0).all())


class IterativeBurstDetectorTraceTests(unittest.TestCase):
    def test_trace_populates_and_preserves_behavior(self):
        spike_times = _cascade_spike_trains()
        config = IterativeBurstConfig()

        trace = IterativeBurstTrace()
        with_trace = compute_iterative_bursts(spike_times, config, trace=trace)
        without_trace = compute_iterative_bursts(spike_times, config)

        # Per-iteration snapshots populated, last one converged or hit max_iterations
        self.assertGreater(len(trace.iterations), 0)
        last_iter = trace.iterations[-1]
        for key in (
            "iter", "n_candidates", "candidates", "composite",
            "composite_threshold", "w", "candidate_mask",
            "convergence_delta", "converged",
        ):
            self.assertIn(key, last_iter)
        self.assertEqual(last_iter["composite"].shape, trace.t_centers.shape)

        # Pre-gate burstlets and gate snapshots populated
        self.assertGreater(len(trace.burstlets_pre_gates), 0)
        self.assertIsNotNone(trace.participation_gate)
        self.assertIn("floor", trace.participation_gate)
        self.assertIsNotNone(trace.bmi_gate)
        self.assertIn("threshold", trace.bmi_gate)
        self.assertIn("llr_aggregate", trace.bmi_gate)
        self.assertIn("n_pre", trace.bmi_gate)
        self.assertIn("n_post", trace.bmi_gate)

        # GMM stage either ran fully or recorded a skip reason
        self.assertIsNotNone(trace.gmm)
        if "decision" in trace.gmm:
            self.assertIn("X", trace.gmm)
            self.assertIn("X_scaled", trace.gmm)
            self.assertIn("labels", trace.gmm)
            self.assertIn("component_means_scaled", trace.gmm)
            self.assertIn("merged_groups", trace.gmm)
            self.assertIn("cluster_scores", trace.gmm)
            self.assertIn("kept_event_mask", trace.gmm)
            self.assertEqual(
                trace.gmm["X"].shape[0],
                trace.gmm["kept_event_mask"].shape[0],
            )
            self.assertEqual(trace.gmm["X"].shape[1], 6)
        else:
            self.assertIn("skipped", trace.gmm)

        # Behavior preservation: detector output unchanged whether trace is on
        self.assertTrue(with_trace.burstlets.equals(without_trace.burstlets))
        self.assertTrue(with_trace.network_bursts.equals(without_trace.network_bursts))
        self.assertTrue(with_trace.superbursts.equals(without_trace.superbursts))


if __name__ == "__main__":
    unittest.main()
