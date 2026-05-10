from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline_manager import BaseAnalysisTask
from pipeline_tasks.preprocessing import PreprocessingTask


class IterativeBurstDetectionTask(BaseAnalysisTask):
    """Iterative contrast-maximizing network burst detection for one curated well.

    Reads the same curated_spike_times.npy produced by auto_curation as
    BurstDetectionTask, but uses a Fisher LDA iterative refinement loop over
    multi-feature signals (PFR, participation, multi-scale Fano Factor,
    per-unit Poisson LLR, burstiness) instead of a fixed participation threshold.

    Output schema is identical to BurstDetectionTask (BurstResults pickle layout)
    with four additional per-event quality columns:
        llr_aggregate, composite_peak, composite_mean, ff_peak.
    """

    task_name = "iterative_burst_detection"
    dependencies = ["auto_curation"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "curation_output_root": "./curation_data",
            "output_root": "./iterative_burst_data",
            # IterativeBurstConfig fields
            "permissive_mad_scale": 0.30,
            "permissive_percentile": 70.0,
            "mad_fallback_threshold": 0.01,
            "composite_mad_scale": 0.75,
            "extent_frac": 0.30,
            "merge_floor_frac": 0.70,
            "network_merge_gap_min_s": 0.75,
            "max_iterations": 20,
            "convergence_eps": 0.005,
            "fisher_alpha_frac": 1e-3,
            "ff_scale_multipliers": [0.5, 1.0, 2.0, 5.0],
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
            / "iterative_burst_detection"
        )

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        import numpy as np

        from pipeline_tasks.analysis import (
            IterativeBurstConfig,
            compute_iterative_bursts,
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

        config = IterativeBurstConfig(
            permissive_mad_scale=float(p["permissive_mad_scale"]),
            permissive_percentile=float(p["permissive_percentile"]),
            mad_fallback_threshold=float(p["mad_fallback_threshold"]),
            composite_mad_scale=float(p["composite_mad_scale"]),
            extent_frac=float(p["extent_frac"]),
            merge_floor_frac=float(p["merge_floor_frac"]),
            network_merge_gap_min_s=float(p["network_merge_gap_min_s"]),
            max_iterations=int(p["max_iterations"]),
            convergence_eps=float(p["convergence_eps"]),
            fisher_alpha_frac=float(p["fisher_alpha_frac"]),
            ff_scale_multipliers=tuple(float(x) for x in p["ff_scale_multipliers"]),
        )

        results = compute_iterative_bursts(spike_times, config=config)

        output_dir = self.build_output_path(
            p["output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        PickleBurstOutputWriter().write(results, output_dir)
        return output_dir
