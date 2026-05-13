from __future__ import annotations

from pathlib import Path
from typing import Any

from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask
from yuxin_mea.tasks.preprocessing import PreprocessingTask


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
            "min_burst_modulation": 0.1,
            "cluster_events": True,
            "cluster_initial_components": 6,
            "cluster_min_events": 5,
            "cluster_min_separation": 1.5,
        }

    @classmethod
    def params_schema(cls) -> dict[str, ParamSpec]:
        defaults = cls.default_params()
        return {
            "curation_output_root": ParamSpec(
                "path", defaults["curation_output_root"],
                "Root directory containing auto_curation outputs "
                "(curated_spike_times.npy per recording/well).",
            ),
            "output_root": ParamSpec(
                "path", defaults["output_root"],
                "Directory where iterative burst detection results "
                "(BurstResults pickles) are written per recording/well.",
            ),
            "permissive_mad_scale": ParamSpec(
                "float", defaults["permissive_mad_scale"],
                "MAD multiplier for the initial (permissive) participation "
                "threshold. Lower values seed more candidates that later "
                "iterations must eliminate.",
                min=0,
            ),
            "permissive_percentile": ParamSpec(
                "float", defaults["permissive_percentile"],
                "Fallback initial percentile threshold when spread_mad is near "
                "zero (e.g. 70.0 -> top 30% of bins).",
                min=0, max=100,
            ),
            "mad_fallback_threshold": ParamSpec(
                "float", defaults["mad_fallback_threshold"],
                "If spread_mad falls below this value, use the percentile "
                "fallback instead of the MAD-based seed threshold.",
                min=0,
            ),
            "composite_mad_scale": ParamSpec(
                "float", defaults["composite_mad_scale"],
                "MAD multiplier applied to the composite signal background "
                "to set the burst/non-burst threshold each iteration.",
                min=0,
            ),
            "extent_frac": ParamSpec(
                "float", defaults["extent_frac"],
                "Edge-trimming fraction: candidate edges are trimmed inward "
                "until composite exceeds max(threshold, extent_frac * peak).",
                min=0,
            ),
            "merge_floor_frac": ParamSpec(
                "float", defaults["merge_floor_frac"],
                "Adjacent candidates are merged if their separating valley "
                "is above merge_floor_frac * threshold.",
                min=0,
            ),
            "network_merge_gap_min_s": ParamSpec(
                "float", defaults["network_merge_gap_min_s"],
                "Minimum gap (seconds) enforced for the network-burst merge "
                "stage in the hierarchy.",
                min=0,
            ),
            "max_iterations": ParamSpec(
                "int", defaults["max_iterations"],
                "Hard cap on refinement iterations (safety valve; typically "
                "converges in 5-10 iterations).",
                min=1,
            ),
            "convergence_eps": ParamSpec(
                "float", defaults["convergence_eps"],
                "Stop iterating when fewer than this fraction of bins flip "
                "burst/non-burst label between iterations.",
                min=0,
            ),
            "fisher_alpha_frac": ParamSpec(
                "float", defaults["fisher_alpha_frac"],
                "Ridge regularization for Fisher LDA within-class scatter: "
                "alpha = fisher_alpha_frac * trace(S_W) / n_features.",
                min=0,
            ),
            "ff_scale_multipliers": ParamSpec(
                "list_float", defaults["ff_scale_multipliers"],
                "Bin-size multipliers for multi-scale Fano Factor features. "
                "Each multiplier produces one FF feature; clamped to [5,100] ms.",
                min=1,
            ),
            "min_burst_modulation": ParamSpec(
                "float", defaults["min_burst_modulation"],
                "Minimum burstlet-level llr_aggregate required for an event "
                "to survive. <= 0 disables the gate.",
                min=0,
            ),
            "cluster_events": ParamSpec(
                "bool", defaults["cluster_events"],
                "After convergence, fit a GMM on per-event quality features "
                "and discard noise-like clusters.",
            ),
            "cluster_initial_components": ParamSpec(
                "int", defaults["cluster_initial_components"],
                "Initial number of GMM components used before similarity-based "
                "merging.",
                min=1,
            ),
            "cluster_min_events": ParamSpec(
                "int", defaults["cluster_min_events"],
                "Minimum detected events required to attempt GMM clustering.",
                min=1,
            ),
            "cluster_min_separation": ParamSpec(
                "float", defaults["cluster_min_separation"],
                "Maximum normalised Euclidean distance between component means "
                "(standardized feature space) for them to be merged.",
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

        from yuxin_mea.analysis import (
            IterativeBurstConfig,
            compute_iterative_bursts,
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
            min_burst_modulation=float(p["min_burst_modulation"]),
            cluster_events=bool(p["cluster_events"]),
            cluster_initial_components=int(p["cluster_initial_components"]),
            cluster_min_events=int(p["cluster_min_events"]),
            cluster_min_separation=float(p["cluster_min_separation"]),
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
