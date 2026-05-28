"""End-to-end test for MLBurstDetectionTask on a small synthetic well."""
from __future__ import annotations

import importlib
import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from yuxin_mea.analysis.ml_burst_detector import MLBurstConfig, MLBurstTrace
from yuxin_mea.analysis.synthetic_validation import generate_cascade_culture
from yuxin_mea.tasks.ml_burst_detection import MLBurstDetectionTask


HDBSCAN_AVAILABLE = importlib.util.find_spec("hdbscan") is not None

_RECORDING_KEY = "SampleA/240415/PlateX/Network/001"
_WELL_ID = "rec0000/well000"
_REC_NAME = "rec0000"
_ACTUAL_WELL = "well000"


def _stage_curation_dir(tmp_path: Path) -> dict:
    """Plant a curated_spike_times.npy under the expected layout."""
    ds = generate_cascade_culture(
        n_units=18,
        duration_s=80.0,
        burst_centers_s=[10.0, 25.0, 40.0, 55.0, 70.0],
        burst_duration_s=0.5,
        burst_rate_hz=60.0,
        baseline_rate_hz=2.0,
        recruitment=0.9,
        seed=20260522,
    )
    curation_dir = (
        tmp_path
        / "curation"
        / _RECORDING_KEY
        / _REC_NAME
        / _ACTUAL_WELL
        / "auto_curation"
    )
    curation_dir.mkdir(parents=True)
    np.save(
        curation_dir / "curated_spike_times.npy",
        np.array(ds.spike_times, dtype=object),
        allow_pickle=True,
    )
    return ds.spike_times


def _params(tmp_path: Path, *, debug: bool, override: dict | None = None) -> dict:
    base = {
        "curation_output_root": str(tmp_path / "curation"),
        "analyzer_output_root": str(tmp_path / "analyzer"),
        "output_root": str(tmp_path / "ml_burst"),
        # Lower HMM cost on tiny synthetic data
        "hmm_max_iter": 50,
        "hmm_min_spikes": 30,
        # Lower HDBSCAN min sizes so the small synthetic burst is detectable
        "hdbscan_min_cluster_size": 10,
        "hdbscan_min_samples": 3,
        "fallback_posterior_threshold": 0.4,
        "debug": debug,
    }
    if override:
        base.update(override)
    return base


class MLBurstDetectionTaskOutputTests(unittest.TestCase):
    def test_writes_standard_burst_results_layout(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _stage_curation_dir(tmp_path)
            output_path = MLBurstDetectionTask().run(
                _RECORDING_KEY,
                _WELL_ID,
                tmp_path / "data.h5",
                _params(tmp_path, debug=False),
            )
            for name in (
                "burstlets.pkl",
                "network_bursts.pkl",
                "superbursts.pkl",
                "metrics.json",
                "diagnostics.json",
                "plot_signals.npy",
            ):
                self.assertTrue(
                    (output_path / name).exists(),
                    f"missing output file: {name}",
                )

    def test_debug_true_persists_trace_and_config(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _stage_curation_dir(tmp_path)
            output_path = MLBurstDetectionTask().run(
                _RECORDING_KEY,
                _WELL_ID,
                tmp_path / "data.h5",
                _params(tmp_path, debug=True),
            )
            trace_path = output_path / "debug_trace.pkl"
            spikes_path = output_path / "debug_spike_times.npy"
            cfg_path = output_path / "debug_config.json"
            self.assertTrue(trace_path.exists())
            self.assertTrue(spikes_path.exists())
            self.assertTrue(cfg_path.exists())

            with open(trace_path, "rb") as fh:
                trace = pickle.load(fh)
            self.assertIsInstance(trace, MLBurstTrace)
            self.assertIsNotNone(trace.t_centers)
            self.assertIsNotNone(trace.feature_names)
            self.assertIsNotNone(trace.unit_ids)
            self.assertIsNotNone(trace.posterior_matrix)
            self.assertIsNotNone(trace.feature_matrix)
            self.assertIsNotNone(trace.hdbscan_labels)

            # Config round-trip into the dataclass
            with open(cfg_path) as fh:
                raw = json.load(fh)
            raw["ff_scale_multipliers"] = tuple(raw["ff_scale_multipliers"])
            raw["gmm_k_range"] = tuple(raw["gmm_k_range"])
            cfg = MLBurstConfig(**raw)
            self.assertEqual(cfg.hmm_max_iter, 50)

    def test_debug_false_does_not_write_trace(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _stage_curation_dir(tmp_path)
            output_path = MLBurstDetectionTask().run(
                _RECORDING_KEY,
                _WELL_ID,
                tmp_path / "data.h5",
                _params(tmp_path, debug=False),
            )
            self.assertFalse((output_path / "debug_trace.pkl").exists())
            self.assertFalse((output_path / "debug_spike_times.npy").exists())
            self.assertFalse((output_path / "debug_config.json").exists())

    @unittest.skipUnless(HDBSCAN_AVAILABLE, "hdbscan not installed")
    def test_detects_at_least_one_burst_on_synthetic_data(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _stage_curation_dir(tmp_path)
            output_path = MLBurstDetectionTask().run(
                _RECORDING_KEY,
                _WELL_ID,
                tmp_path / "data.h5",
                _params(tmp_path, debug=True),
            )
            # Read back the diagnostics — at least one burstlet expected
            with open(output_path / "diagnostics.json") as fh:
                diag = json.load(fh)
            self.assertTrue(diag.get("burst_activity_detected", False))

    def test_debug_param_in_schema_and_defaults_false(self):
        defaults = MLBurstDetectionTask.default_params()
        schema = MLBurstDetectionTask.params_schema()
        self.assertIn("debug", defaults)
        self.assertIn("debug", schema)
        self.assertEqual(defaults["debug"], False)
        self.assertEqual(schema["debug"].type, "bool")

    def test_diffmap_gmm_is_the_default_algorithm(self):
        defaults = MLBurstDetectionTask.default_params()
        self.assertEqual(defaults["cluster_algorithm"], "diffmap_gmm")

    def test_hdbscan_algorithm_still_reachable_via_param(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _stage_curation_dir(tmp_path)
            output_path = MLBurstDetectionTask().run(
                _RECORDING_KEY,
                _WELL_ID,
                tmp_path / "data.h5",
                _params(tmp_path, debug=False,
                        override={"cluster_algorithm": "hdbscan"}),
            )
            with open(output_path / "diagnostics.json") as fh:
                diag = json.load(fh)
            self.assertEqual(diag["cluster_algorithm"], "hdbscan")
            # Decision tag should reflect the HDBSCAN code path, not diffmap_gmm
            self.assertIn(
                diag["cluster_decision"],
                {"hdbscan", "hdbscan_single", "hdbscan_all_noise", "fallback_threshold"},
            )

    def test_diffmap_gmm_diagnostics_carry_new_fields(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _stage_curation_dir(tmp_path)
            output_path = MLBurstDetectionTask().run(
                _RECORDING_KEY,
                _WELL_ID,
                tmp_path / "data.h5",
                _params(tmp_path, debug=True),
            )
            with open(output_path / "diagnostics.json") as fh:
                diag = json.load(fh)
            self.assertEqual(diag["cluster_algorithm"], "diffmap_gmm")
            self.assertIn(diag["cluster_decision"],
                          {"diffmap_gmm", "diffmap_singleton", "fallback_threshold"})
            # Always present (may be None on fallback paths)
            self.assertIn("burst_posterior_threshold", diag)
            self.assertIn("diffmap_n_components", diag)
            self.assertIn("diffmap_k_neighbors", diag)
            # New trace fields
            with open(output_path / "debug_trace.pkl", "rb") as fh:
                trace = pickle.load(fh)
            self.assertIsNotNone(trace.cluster_labels)
            # Legacy alias still populated for back-compat viewers
            self.assertIsNotNone(trace.hdbscan_labels)
            if diag["cluster_decision"] == "diffmap_gmm":
                self.assertIsNotNone(trace.embedding)
                self.assertIsNotNone(trace.burst_posterior)
                # Burst-posterior signal also persisted in plot_signals
                plot_signals = np.load(
                    output_path / "plot_signals.npy", allow_pickle=True,
                ).item()
                self.assertIn("burst_posterior_signal", plot_signals)
                self.assertEqual(
                    plot_signals["burst_posterior_signal"].shape,
                    plot_signals["t"].shape,
                )


class DiffmapGmmTrajectoryRecoveryTests(unittest.TestCase):
    """Focused unit test: with two attractors connected by a sparse trajectory
    in feature space, diffmap_gmm should put trajectory bins on the manifold
    (intermediate burst_posterior, not noise) — HDBSCAN labels them -1."""

    def _make_two_attractor_data(self, rng):
        d = 8
        bg = rng.normal(0, 1, size=(2000, d))
        burst = rng.normal(5, 1, size=(600, d))
        t = rng.uniform(0.2, 0.8, size=200)[:, None]
        traj = (np.zeros(d) + t * 5) + rng.normal(0, 0.5, size=(200, d))
        X = np.vstack([bg, burst, traj]).astype(float)
        truth = np.concatenate([
            np.zeros(2000), np.ones(600), np.full(200, 0.5),
        ])
        # Synthesize a "ranking feature" column so the clusterer's burst-label
        # selection logic has something to rank on. Use the mean along the
        # data axes — monotonic in distance from the bg attractor.
        ranking = X.mean(axis=1, keepdims=True)
        X_aug = np.hstack([ranking, X])
        feature_names = ["post_frac_gt_0_5"] + [f"f{i}" for i in range(d)]
        perm = rng.permutation(X_aug.shape[0])
        return X_aug[perm], feature_names, truth[perm]

    def test_diffmap_gmm_trajectory_bins_are_not_noise(self):
        from yuxin_mea.analysis.ml_burst_cluster import cluster_bins

        rng = np.random.default_rng(0)
        X, fnames, truth = self._make_two_attractor_data(rng)
        res = cluster_bins(X, fnames, algorithm="diffmap_gmm")

        self.assertEqual(res.decision, "diffmap_gmm")
        # GMM is exhaustive — no -1 labels under diffmap_gmm.
        self.assertFalse((res.labels == -1).any())
        # Burst attractor → high burst_posterior; bg → low; trajectory → intermediate.
        bp = res.burst_posterior
        self.assertIsNotNone(bp)
        self.assertGreater(float(bp[truth == 1].mean()), 0.9)
        self.assertLess(float(bp[truth == 0].mean()), 0.1)
        self.assertGreater(float(bp[truth == 0.5].mean()), 0.2)
        self.assertLess(float(bp[truth == 0.5].mean()), 0.9)

    @unittest.skipUnless(HDBSCAN_AVAILABLE, "hdbscan not installed")
    def test_hdbscan_labels_trajectory_as_noise_on_same_data(self):
        from yuxin_mea.analysis.ml_burst_cluster import cluster_bins

        rng = np.random.default_rng(0)
        X, fnames, truth = self._make_two_attractor_data(rng)
        res = cluster_bins(X, fnames, algorithm="hdbscan",
                           min_cluster_size=30, min_samples=5)
        # Sanity check the regression the new algo is meant to fix:
        # a meaningful fraction of trajectory bins falls into HDBSCAN noise.
        noise_frac = float((res.labels[truth == 0.5] == -1).mean())
        burst_noise_frac = float((res.labels[truth == 1] == -1).mean())
        self.assertGreater(noise_frac + burst_noise_frac, 0.05)


if __name__ == "__main__":
    unittest.main()
