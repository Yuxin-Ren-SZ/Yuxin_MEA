from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask
from yuxin_mea.tasks.preprocessing import PreprocessingTask

logger = logging.getLogger(__name__)


class BurstDetectionTask(BaseAnalysisTask):
    """Network burst detection for one well.

    Prefers curated spike times from ``auto_curation``; falls back to all
    units from the ``SortingAnalyzer`` when auto_curation was skipped.
    """

    task_name = "burst_detection"
    dependencies = ["analyzer"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "curation_output_root": "./curation_data",
            "analyzer_output_root": "./analyzer_data",
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
            "analyzer_output_root": ParamSpec(
                "path", defaults["analyzer_output_root"],
                "Root directory of SortingAnalyzer outputs from AnalyzerTask. "
                "Used as fallback when auto_curation has not run.",
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
        from yuxin_mea.analysis.burst_detector import (
            BurstDetectorConfig,
            compute_network_bursts,
        )
        from yuxin_mea.analysis.burst_output import PickleBurstOutputWriter
        from yuxin_mea.tasks._spike_times import load_spike_times
        from yuxin_mea.tasks.analyzer import AnalyzerTask

        p = self.resolve_params(params)
        rec_name, actual_well_id = self.split_compound_well_id(well_id)

        curation_dir = self.build_curation_output_path(
            p["curation_output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        analyzer_path = AnalyzerTask.build_output_path(
            p["analyzer_output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        spike_times = load_spike_times(
            curation_dir, analyzer_path, logger,
            well_label=f"{recording_key}/{rec_name}/{actual_well_id}",
        )

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
