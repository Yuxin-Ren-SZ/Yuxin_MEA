"""Smoke tests for burst_inspector pure-library figure builders."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import plotly.graph_objects as go

from yuxin_mea.analysis.burst_inspector import (
    InspectorBundle,
    _classify_events,
    fig_composite_with_threshold,
    fig_event_gmm_clusters,
    fig_iteration_trajectory,
    fig_label_comparison_table,
    fig_pca_feature_space,
    fig_raster,
    load_inspector_bundle,
    summary_card,
)
from yuxin_mea.analysis.iterative_burst_detector import (
    IterativeBurstConfig,
    IterativeBurstTrace,
    compute_iterative_bursts,
)
from tests.test_iterative_burst_detector import _cascade_spike_trains


def _make_bundle() -> InspectorBundle:
    """Build an InspectorBundle from a fresh in-process detector run."""
    spike_times = _cascade_spike_trains(n_units=18, duration_s=80.0)
    config = IterativeBurstConfig(max_iterations=6)
    trace = IterativeBurstTrace()
    results = compute_iterative_bursts(spike_times, config=config, trace=trace)
    return InspectorBundle(
        trace=trace,
        spike_times=spike_times,
        burstlets=results.burstlets,
        config=config,
        recording_key="testkey",
        rec_name="rec0000",
        well_id="well000",
        source="on_demand",
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

    def test_fig_raster(self):
        self._assert_nonempty_figure(fig_raster(self.bundle))
        self._assert_nonempty_figure(fig_raster(self.bundle, iteration=0))

    def test_fig_composite_with_threshold(self):
        self._assert_nonempty_figure(fig_composite_with_threshold(self.bundle))

    def test_fig_iteration_trajectory(self):
        fig = fig_iteration_trajectory(self.bundle)
        self.assertEqual(len(fig.data), 3)  # n_cands, deltas, thrs

    def test_fig_pca_feature_space(self):
        fig = fig_pca_feature_space(self.bundle)
        # 2 scatter traces (bg + burst) + 2 loading bars (PC1 + PC2)
        self.assertGreaterEqual(len(fig.data), 3)

    def test_fig_event_gmm_clusters(self):
        # Either a populated figure or a legitimate "skipped" annotation.
        self._assert_nonempty_figure(fig_event_gmm_clusters(self.bundle))

    def test_fig_label_comparison_table_row_count(self):
        fig = fig_label_comparison_table(self.bundle)
        if not self.bundle.trace.burstlets_pre_gates:
            # No events to show — degenerate fixture; still legitimate.
            self.assertGreater(len(fig.layout.annotations), 0)
            return
        self.assertEqual(len(fig.data), 1)
        # First column's cell list length == number of pre-gate events
        self.assertEqual(
            len(fig.data[0].cells.values[0]),
            len(self.bundle.trace.burstlets_pre_gates),
        )

    def test_classify_events_kill_reason_invariants(self):
        df = _classify_events(self.bundle.trace)
        if df.empty:
            self.skipTest("Cascade fixture produced no pre-gate events.")

        allowed_reasons = {"—", "participation", "BMI", "GMM"}
        self.assertTrue(set(df["kill_reason"].unique()).issubset(allowed_reasons))

        # Survivors (kept == "Y") must report no kill reason.
        kept = df[df["kept"] == "Y"]
        self.assertTrue((kept["kill_reason"] == "—").all())

        # And kill_reason != "—" implies kept == "N".
        killed = df[df["kill_reason"] != "—"]
        self.assertTrue((killed["kept"] == "N").all())

        # Final burstlets should equal the kept rows.
        self.assertEqual(int((df["kept"] == "Y").sum()),
                         len(self.bundle.burstlets))

    def test_summary_card_has_required_keys(self):
        card = summary_card(self.bundle)
        for key in (
            "well", "source", "n_units", "n_iterations_run",
            "converged", "n_final_burstlets", "kill_breakdown",
        ):
            self.assertIn(key, card)


class LoaderTests(unittest.TestCase):
    def test_load_missing_disk_falls_back_to_on_demand(self):
        spike_times = _cascade_spike_trains(n_units=12, duration_s=60.0)
        with TemporaryDirectory() as tmp:
            bundle = load_inspector_bundle(
                Path(tmp),
                "any_key",
                "rec0000",
                "well000",
                on_demand_spike_times=spike_times,
                on_demand_config=IterativeBurstConfig(max_iterations=4),
            )
            self.assertEqual(bundle.source, "on_demand")
            self.assertGreater(len(bundle.trace.iterations), 0)

    def test_load_missing_disk_without_fallback_raises(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_inspector_bundle(
                    Path(tmp), "any_key", "rec0000", "well000",
                )


if __name__ == "__main__":
    unittest.main()
