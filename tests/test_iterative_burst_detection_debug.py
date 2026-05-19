"""End-to-end test for IterativeBurstDetectionTask debug-mode persistence."""
from __future__ import annotations

import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from yuxin_mea.analysis.iterative_burst_detector import (
    IterativeBurstConfig,
    IterativeBurstTrace,
)
from yuxin_mea.tasks.iterative_burst_detection import IterativeBurstDetectionTask
from tests.test_iterative_burst_detector import _cascade_spike_trains


_RECORDING_KEY = "SampleA/240415/PlateX/Network/001"
_WELL_ID = "rec0000/well000"
_REC_NAME = "rec0000"
_ACTUAL_WELL = "well000"


def _stage_curation_dir(tmp_path: Path) -> tuple[Path, dict]:
    """Plant a curated_spike_times.npy under the expected layout."""
    spike_times = _cascade_spike_trains(n_units=18, duration_s=80.0)
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
        np.array(spike_times, dtype=object),
        allow_pickle=True,
    )
    return curation_dir, spike_times


def _params(tmp_path: Path, *, debug: bool) -> dict:
    return {
        "curation_output_root": str(tmp_path / "curation"),
        "output_root": str(tmp_path / "iterative_burst"),
        # Keep iterations cheap on synthetic data
        "max_iterations": 6,
        "debug": debug,
    }


class IterativeBurstDetectionDebugModeTests(unittest.TestCase):
    def test_debug_true_writes_trace_pickle_and_spike_times(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _stage_curation_dir(tmp_path)
            task = IterativeBurstDetectionTask()
            output_path = task.run(
                _RECORDING_KEY,
                _WELL_ID,
                tmp_path / "data.h5",
                _params(tmp_path, debug=True),
            )

            trace_path = output_path / "debug_trace.pkl"
            spikes_path = output_path / "debug_spike_times.npy"
            self.assertTrue(trace_path.exists(), "debug_trace.pkl should be written")
            self.assertTrue(
                spikes_path.exists(), "debug_spike_times.npy should be written"
            )

            with open(trace_path, "rb") as fh:
                trace = pickle.load(fh)
            self.assertIsInstance(trace, IterativeBurstTrace)
            self.assertGreater(
                len(trace.iterations),
                0,
                "trace should record at least one iteration",
            )
            self.assertIsNotNone(trace.t_centers)
            self.assertIsNotNone(trace.feature_names)
            self.assertIsNotNone(trace.unit_ids)
            # gmm may be skipped if too few events; either way it must be set
            # to a dict so the inspector page can introspect it.
            self.assertIsNotNone(trace.gmm)

            # spike_times round-trip
            loaded = np.load(spikes_path, allow_pickle=True).item()
            self.assertEqual(set(loaded.keys()), set(trace.unit_ids))

            # debug_config.json round-trips into a usable IterativeBurstConfig.
            config_path = output_path / "debug_config.json"
            self.assertTrue(config_path.exists())
            with open(config_path) as fh:
                raw = json.load(fh)
            raw["ff_scale_multipliers"] = tuple(raw["ff_scale_multipliers"])
            cfg = IterativeBurstConfig(**raw)
            self.assertEqual(cfg.max_iterations, 6)

    def test_debug_false_does_not_write_trace_pickle(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _stage_curation_dir(tmp_path)
            task = IterativeBurstDetectionTask()
            output_path = task.run(
                _RECORDING_KEY,
                _WELL_ID,
                tmp_path / "data.h5",
                _params(tmp_path, debug=False),
            )

            self.assertFalse((output_path / "debug_trace.pkl").exists())
            self.assertFalse((output_path / "debug_spike_times.npy").exists())
            self.assertFalse((output_path / "debug_config.json").exists())

    def test_debug_param_is_in_schema_and_defaults_false(self):
        defaults = IterativeBurstDetectionTask.default_params()
        schema = IterativeBurstDetectionTask.params_schema()
        self.assertIn("debug", defaults)
        self.assertIn("debug", schema)
        self.assertEqual(defaults["debug"], False)
        self.assertEqual(schema["debug"].type, "bool")


if __name__ == "__main__":
    unittest.main()
