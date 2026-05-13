from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline_manager import BaseAnalysisTask
from pipeline_tasks.preprocessing import PreprocessingTask


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

        from pipeline_tasks.analysis.burst_detector import (
            BurstDetectorConfig,
            compute_network_bursts,
        )
        from pipeline_tasks.analysis.burst_output import PickleBurstOutputWriter

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
