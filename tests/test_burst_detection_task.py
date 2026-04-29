from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from pipeline_tasks.burst_detection import BurstDetectionTask
from pipeline_tasks.analysis.burst_detector import BurstDetectorConfig, BurstResults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RECORDING_KEY = "SampleA/240415/PlateX/Network/001"
_WELL_ID = "rec0000/well000"
_REC_NAME = "rec0000"
_ACTUAL_WELL = "well000"


def _make_fake_burst_results() -> BurstResults:
    return BurstResults(
        burstlets=pd.DataFrame([
            {"start": 10.0, "end": 10.5, "duration_s": 0.5, "peak_synchrony": 0.8,
             "peak_time": 10.25, "synchrony_energy": 0.4, "participation": 0.9,
             "total_spikes": 50, "burst_peak": 20.0},
        ]),
        network_bursts=pd.DataFrame([
            {"start": 10.0, "end": 10.5, "duration_s": 0.5, "peak_synchrony": 0.8,
             "peak_time": 10.25, "synchrony_energy": 0.4, "participation": 0.9,
             "total_spikes": 50, "burst_peak": 20.0, "fragment_count": 1,
             "n_sub_events": 1},
        ]),
        superbursts=pd.DataFrame(),
        metrics={
            "burstlets": {"count": 1},
            "network_bursts": {"count": 1},
            "superbursts": {"count": 0},
        },
        diagnostics={"n_units": 5, "adaptive_bin_ms": 15.0},
        plot_data={
            "t": np.array([0.0, 1.0]),
            "participation_signal": np.array([0.1, 0.9]),
            "rate_signal": np.array([0.05, 0.5]),
            "burst_peak_times": np.array([10.25]),
            "burst_peak_values": np.array([0.8]),
            "participation_baseline": 0.1,
            "participation_threshold": 0.3,
        },
    )


def _make_fake_spike_times() -> dict[str, np.ndarray]:
    return {f"unit_{i}": np.array([10.0 + i * 0.01, 50.0 + i * 0.01]) for i in range(5)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class BurstDetectionTaskStaticMethodTests(unittest.TestCase):
    def test_split_compound_well_id(self):
        self.assertEqual(
            BurstDetectionTask.split_compound_well_id("rec0000/well000"),
            ("rec0000", "well000"),
        )

    def test_split_invalid_well_id_raises(self):
        with self.assertRaises(ValueError):
            BurstDetectionTask.split_compound_well_id("well000")

    def test_build_output_path(self):
        path = BurstDetectionTask.build_output_path(
            output_root="/analysis",
            recording_key=_RECORDING_KEY,
            rec_name=_REC_NAME,
            well_id=_ACTUAL_WELL,
        )
        expected = (
            Path("/analysis")
            / _RECORDING_KEY
            / _REC_NAME
            / _ACTUAL_WELL
            / "burst_detection"
        )
        self.assertEqual(path, expected)

    def test_build_curation_output_path(self):
        path = BurstDetectionTask.build_curation_output_path(
            curation_output_root="/curation",
            recording_key=_RECORDING_KEY,
            rec_name=_REC_NAME,
            well_id=_ACTUAL_WELL,
        )
        expected = (
            Path("/curation")
            / _RECORDING_KEY
            / _REC_NAME
            / _ACTUAL_WELL
            / "auto_curation"
        )
        self.assertEqual(path, expected)


class BurstDetectionTaskDefaultParamsTests(unittest.TestCase):
    def test_default_params_has_config_keys(self):
        defaults = BurstDetectionTask.default_params()
        config_keys = {f.name for f in BurstDetectorConfig.__dataclass_fields__.values()}
        self.assertTrue(config_keys.issubset(defaults.keys()))

    def test_default_params_has_path_roots(self):
        defaults = BurstDetectionTask.default_params()
        self.assertIn("output_root", defaults)
        self.assertIn("curation_output_root", defaults)


class BurstDetectionTaskRunTests(unittest.TestCase):
    def _run_task(self, tmp_path: Path, extra_params: dict | None = None) -> Path:
        """Helper: set up curation dir, run task with mocked detector."""
        fake_spike_times = _make_fake_spike_times()
        fake_results = _make_fake_burst_results()

        curation_dir = (
            tmp_path
            / "curation"
            / _RECORDING_KEY
            / _REC_NAME
            / _ACTUAL_WELL
            / "auto_curation"
        )
        curation_dir.mkdir(parents=True)
        np.save(curation_dir / "curated_spike_times.npy", fake_spike_times)

        params = {
            "curation_output_root": str(tmp_path / "curation"),
            "output_root": str(tmp_path / "burst"),
        }
        if extra_params:
            params.update(extra_params)

        task = BurstDetectionTask()
        with patch(
            "pipeline_tasks.analysis.burst_detector.compute_network_bursts",
            return_value=fake_results,
        ) as mock_detector:
            output_path = task.run(_RECORDING_KEY, _WELL_ID, tmp_path / "data.h5", params)
            self._mock_detector = mock_detector

        return output_path

    def test_run_returns_output_path(self):
        with TemporaryDirectory() as tmp:
            output_path = self._run_task(Path(tmp))
        self.assertIsInstance(output_path, Path)
        self.assertEqual(output_path.name, "burst_detection")

    def test_run_writes_expected_files(self):
        with TemporaryDirectory() as tmp:
            output_path = self._run_task(Path(tmp))
            expected = {
                "burstlets.parquet",
                "network_bursts.parquet",
                "superbursts.parquet",
                "metrics.json",
                "diagnostics.json",
                "plot_signals.npy",
            }
            actual = {f.name for f in output_path.iterdir()}
        self.assertEqual(expected, actual)

    def test_run_passes_config_to_detector(self):
        with TemporaryDirectory() as tmp:
            self._run_task(Path(tmp), extra_params={"extent_frac": 0.55})
            _, call_kwargs = self._mock_detector.call_args
            cfg: BurstDetectorConfig = call_kwargs["config"]
        self.assertAlmostEqual(cfg.extent_frac, 0.55)

    def test_run_raises_when_curation_output_missing(self):
        with TemporaryDirectory() as tmp:
            task = BurstDetectionTask()
            with self.assertRaises(FileNotFoundError):
                task.run(
                    _RECORDING_KEY,
                    _WELL_ID,
                    Path(tmp) / "data.h5",
                    {
                        "curation_output_root": str(Path(tmp) / "does_not_exist"),
                        "output_root": str(Path(tmp) / "burst"),
                    },
                )

    def test_resolve_params_merges_defaults(self):
        task = BurstDetectionTask()
        merged = task.resolve_params({"extent_frac": 0.77})
        self.assertAlmostEqual(merged["extent_frac"], 0.77)
        # Default keys not in the override must still be present
        self.assertIn("gamma", merged)
        self.assertIn("network_merge_gap_min_s", merged)


class BurstDetectionTaskMetadataTests(unittest.TestCase):
    def test_task_name(self):
        self.assertEqual(BurstDetectionTask.task_name, "burst_detection")

    def test_dependencies(self):
        self.assertEqual(BurstDetectionTask.dependencies, ["auto_curation"])


if __name__ == "__main__":
    unittest.main()
