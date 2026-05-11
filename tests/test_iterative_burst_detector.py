from __future__ import annotations

import unittest

import numpy as np

from pipeline_tasks.analysis.iterative_burst_detector import (
    IterativeBurstConfig,
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


if __name__ == "__main__":
    unittest.main()
