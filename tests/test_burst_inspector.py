"""Smoke tests for burst_inspector pure-library figure builders.

Exercises the generic (traditional / ML) inspector path: a BurstResults
bundle loaded from disk or built in-process, with raster + composite figures
and the summary card. The iterative-specific machinery was removed, so these
tests no longer depend on an iterative trace.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import plotly.graph_objects as go

from yuxin_mea.analysis.burst_detector import compute_network_bursts
from yuxin_mea.analysis.burst_inspector import (
    InspectorBundle,
    fig_composite_basic,
    fig_raster_basic,
    load_generic_bundle,
    summary_card,
)
from yuxin_mea.analysis.burst_output import PickleBurstOutputWriter


def _synthetic_burst_spike_trains(
    n_units: int = 16, duration_s: float = 60.0, seed: int = 0,
) -> dict[str, np.ndarray]:
    """Background Poisson firing plus periodic synchronous network bursts."""
    rng = np.random.default_rng(seed)
    spike_times: dict[str, np.ndarray] = {}
    burst_onsets = np.arange(5.0, duration_s, 8.0)
    for u in range(n_units):
        # Sparse background.
        n_bg = rng.poisson(0.5 * duration_s)
        bg = rng.uniform(0.0, duration_s, size=n_bg)
        # Dense, tightly synchronised firing at each burst onset.
        burst = []
        for onset in burst_onsets:
            burst.append(onset + rng.uniform(0.0, 0.08, size=rng.integers(8, 16)))
        spikes = np.concatenate([bg, *burst]) if burst else bg
        spike_times[f"cluster_{u:03d}"] = np.sort(spikes[(spikes >= 0) & (spikes < duration_s)])
    return spike_times


def _make_bundle(method: str = "traditional") -> InspectorBundle:
    """Build an InspectorBundle from a fresh in-process traditional-detector run."""
    spike_times = _synthetic_burst_spike_trains()
    results = compute_network_bursts(spike_times)
    return InspectorBundle(
        spike_times=spike_times,
        burstlets=results.burstlets,
        recording_key="testkey",
        rec_name="rec0000",
        well_id="well000",
        source="on_demand",
        method=method,
        results=results,
    )


class FigureBuildersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = _make_bundle()

    def _assert_nonempty_figure(self, fig: go.Figure) -> None:
        self.assertIsInstance(fig, go.Figure)
        # Either there's data or there's at least one annotation
        # (empty-state figures are still legitimate).
        self.assertTrue(len(fig.data) > 0 or len(fig.layout.annotations) > 0)

    def test_fig_raster_basic(self):
        self._assert_nonempty_figure(fig_raster_basic(self.bundle))

    def test_fig_composite_basic(self):
        self._assert_nonempty_figure(fig_composite_basic(self.bundle))

    def test_summary_card_has_required_keys(self):
        card = summary_card(self.bundle)
        for key in ("well", "source", "method", "n_units", "n_burstlets"):
            self.assertIn(key, card)
        # results is populated, so the network/superburst counts appear.
        self.assertIn("n_network_bursts", card)
        self.assertIn("n_superbursts", card)


class LoaderTests(unittest.TestCase):
    def test_load_generic_bundle_round_trip(self):
        spike_times = _synthetic_burst_spike_trains(n_units=12, duration_s=40.0, seed=1)
        results = compute_network_bursts(spike_times)
        with TemporaryDirectory() as tmp:
            output_dir = (
                Path(tmp) / "any_key" / "rec0000" / "well000" / "burst_detection"
            )
            output_dir.mkdir(parents=True)
            PickleBurstOutputWriter().write(results, output_dir)

            bundle = load_generic_bundle(
                Path(tmp), "any_key", "rec0000", "well000", method="traditional",
            )
            self.assertEqual(bundle.source, "disk")
            self.assertEqual(bundle.method, "traditional")
            self.assertEqual(len(bundle.burstlets), len(results.burstlets))

    def test_load_generic_bundle_missing_dir_raises(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_generic_bundle(
                    Path(tmp), "any_key", "rec0000", "well000", method="traditional",
                )


if __name__ == "__main__":
    unittest.main()
