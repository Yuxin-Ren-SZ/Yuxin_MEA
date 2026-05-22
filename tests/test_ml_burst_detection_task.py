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


if __name__ == "__main__":
    unittest.main()
