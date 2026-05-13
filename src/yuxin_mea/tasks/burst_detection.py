from __future__ import annotations

from pathlib import Path
from typing import Any

from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask
from yuxin_mea.tasks.preprocessing import PreprocessingTask


class BurstDetectionTask(BaseAnalysisTask):
    """Network burst detection for one curated well.

    Depends on 'auto_curation', which is expected to write a
    ``curated_spike_times.npy`` file containing a dict of
    {unit_id: np.ndarray} of spike times in seconds.
    """

    task_name = "burst_detection"
    dependencies = ["auto_curation"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "curation_output_root": "./curation_data",
            "output_root": "./burst_detection_data",
            # BurstDetectorConfig defaults — mirrors BurstDetectorConfig fields
            "gamma": 1.0,
            "min_burstlet_participation": 0.20,
            "min_absolute_rate_hz": 0.5,
            "min_burst_density_hz": 1.0,
            "min_relative_height": 0.1,
            "extent_frac": 0.30,
            "network_merge_gap_min_s": 0.75,
        }

    @classmethod
    def params_schema(cls) -> dict[str, ParamSpec]:
        defaults = cls.default_params()
        return {
            "curation_output_root": ParamSpec(
                "path", defaults["curation_output_root"],
                "Root directory where AutoCurationTask writes "
                "curated_spike_times.npy (per recording/well).",
            ),
            "output_root": ParamSpec(
                "path", defaults["output_root"],
                "Directory where burst-detection outputs (pickle bundle) "
                "are written (per recording/well).",
            ),
            "gamma": ParamSpec(
                "float", defaults["gamma"],
                "Reserved tuning exponent for the parameter-free detector. "
                "Currently inactive (see BurstDetectorConfig docstring); "
                "kept for API stability.",
                min=0,
            ),
            "min_burstlet_participation": ParamSpec(
                "float", defaults["min_burstlet_participation"],
                "Minimum fraction of units that must participate in a "
                "burstlet for it to be kept. Currently inactive filter "
                "(see BurstDetectorConfig docstring).",
                min=0, max=1,
            ),
            "min_absolute_rate_hz": ParamSpec(
                "float", defaults["min_absolute_rate_hz"],
                "Minimum peak population firing rate (Hz) for a burstlet. "
                "Currently inactive filter.",
                min=0,
            ),
            "min_burst_density_hz": ParamSpec(
                "float", defaults["min_burst_density_hz"],
                "Minimum spike density (spikes per participating-unit-second) "
                "within a burstlet. Currently inactive filter.",
                min=0,
            ),
            "min_relative_height": ParamSpec(
                "float", defaults["min_relative_height"],
                "Minimum relative peak height (fraction of dynamic range) "
                "required for a burstlet. Currently inactive filter.",
                min=0, max=1,
            ),
            "extent_frac": ParamSpec(
                "float", defaults["extent_frac"],
                "Burstlet extent threshold as a fraction of its peak "
                "synchrony; bin walks outward while the smoothed "
                "participation signal stays above extent_frac * peak.",
                min=0, max=1,
            ),
            "network_merge_gap_min_s": ParamSpec(
                "float", defaults["network_merge_gap_min_s"],
                "Lower bound (seconds) on the network-burst merge gap. "
                "Actual gap is max(10 * biological_ISI, this value).",
                min=0,
            ),
        }

    @staticmethod
    def split_compound_well_id(well_id: str) -> tuple[str, str]:
        return PreprocessingTask.split_compound_well_id(well_id)

    @staticmethod
    def build_curation_output_path(
        curation_output_root: str | Path,
        recording_key: str,
        rec_name: str,
        well_id: str,
    ) -> Path:
        """Path where AutoCurationTask writes curated_spike_times.npy."""
        return (
            Path(curation_output_root)
            / Path(recording_key)
            / rec_name
            / well_id
            / "auto_curation"
        )

    @staticmethod
    def build_output_path(
        output_root: str | Path,
        recording_key: str,
        rec_name: str,
        well_id: str,
    ) -> Path:
        return (
            Path(output_root)
            / Path(recording_key)
            / rec_name
            / well_id
            / "burst_detection"
        )

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        import numpy as np

        from yuxin_mea.analysis.burst_detector import (
            BurstDetectorConfig,
            compute_network_bursts,
        )
        from yuxin_mea.analysis.burst_output import PickleBurstOutputWriter

        p = self.resolve_params(params)
        rec_name, actual_well_id = self.split_compound_well_id(well_id)

        curation_dir = self.build_curation_output_path(
            p["curation_output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        spike_times_path = curation_dir / "curated_spike_times.npy"
        if not spike_times_path.exists():
            raise FileNotFoundError(
                f"curated_spike_times.npy not found at {spike_times_path}. "
                "Ensure auto_curation has completed successfully."
            )

        spike_times: dict = np.load(  # type: ignore[call-overload]
            spike_times_path, allow_pickle=True
        ).item()

        config = BurstDetectorConfig(
            gamma=float(p["gamma"]),
            min_burstlet_participation=float(p["min_burstlet_participation"]),
            min_absolute_rate_hz=float(p["min_absolute_rate_hz"]),
            min_burst_density_hz=float(p["min_burst_density_hz"]),
            min_relative_height=float(p["min_relative_height"]),
            extent_frac=float(p["extent_frac"]),
            network_merge_gap_min_s=float(p["network_merge_gap_min_s"]),
        )

        results = compute_network_bursts(spike_times, config=config)

        output_dir = self.build_output_path(
            p["output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        PickleBurstOutputWriter().write(results, output_dir)
        return output_dir
