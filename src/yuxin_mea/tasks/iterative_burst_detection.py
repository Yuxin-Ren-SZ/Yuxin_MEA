from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask
from yuxin_mea.tasks.preprocessing import PreprocessingTask

logger = logging.getLogger(__name__)


class IterativeBurstDetectionTask(BaseAnalysisTask):
    """Iterative contrast-maximizing network burst detection for one well.

    Uses a Fisher LDA iterative refinement loop over multi-feature signals
    (PFR, participation, multi-scale Fano Factor, per-unit Poisson LLR,
    burstiness) instead of a fixed participation threshold.

    Prefers curated spike times from ``auto_curation``; falls back to all
    units from the ``SortingAnalyzer`` when auto_curation was skipped.

    Output schema is identical to BurstDetectionTask (BurstResults pickle layout)
    with four additional per-event quality columns:
        llr_aggregate, composite_peak, composite_mean, ff_peak.
    """

    task_name = "iterative_burst_detection"
    dependencies = ["analyzer"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "curation_output_root": "./curation_data",
            "analyzer_output_root": "./analyzer_data",
            "output_root": "./iterative_burst_data",
            # IterativeBurstConfig fields
            "permissive_mad_scale": 0.30,
            "permissive_percentile": 70.0,
            "mad_fallback_threshold": 0.01,
            "composite_mad_scale": 0.75,
            "extent_frac": 0.30,
            "merge_floor_frac": 0.50,
            "merge_gap_tolerance_bins": 3,
            "strict_merge_gap_tolerance_bins": 2,
            "merge_strict_floor_frac": 1.0,
            "network_merge_gap_min_s": 0.75,
            "max_iterations": 20,
            "convergence_eps": 0.005,
            "fisher_alpha_frac": 1e-3,
            "ff_scale_multipliers": [0.5, 1.0, 2.0, 5.0],
            "min_burst_modulation": 0.1,
            "peak_synchrony_floor_frac": 0.5,
            "participation_gate_mode": "and",
            "cluster_events": False,
            "cluster_initial_components": 6,
            "cluster_min_events": 5,
            "cluster_min_separation": 1.5,
            "debug": False,
            # Inner partitioner choice
            "inner_partitioner": "gmm_em",
            # Fisher LDA safeguards
            "lda_exclude_silence": True,
            "lda_sign_pinned_feature_names": ["PFR", "P", "LLR"],
            # GMM-EM partitioner (only used when inner_partitioner == "gmm_em")
            "gmm_k_range": [2, 5],
            "gmm_bic_margin": 5.0,
            "gmm_em_n_init": 5,
            "gmm_em_reg_covar": 1e-4,
            "gmm_burst_score_weights": [0.20, 0.25, 0.05, 0.10, 0.10, 0.05, 0.20, 0.05],
            "gmm_posterior_threshold": 0.5,
            "gmm_component_merge_distance": 0.5,
            "gmm_burst_top_fraction": 0.7,
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
            "analyzer_output_root": ParamSpec(
                "path", defaults["analyzer_output_root"],
                "Root directory of SortingAnalyzer outputs from AnalyzerTask. "
                "Used as fallback when auto_curation has not run.",
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
                "Iteration-time merge: adjacent candidates are merged if "
                "their separating valley stays above merge_floor_frac × "
                "threshold. Default 0.50 (lowered from 0.70) admits real "
                "network bursts whose composite signal briefly dips between "
                "sub-events without admitting genuine inter-burst silence.",
                min=0,
            ),
            "merge_gap_tolerance_bins": ParamSpec(
                "int", defaults["merge_gap_tolerance_bins"],
                "Iteration-time merge: two candidates separated by ≤ this "
                "many composite bins are merged regardless of valley depth. "
                "Handles 1–3-bin composite dips inside one true network "
                "burst that would otherwise fragment it.",
                min=0,
            ),
            "strict_merge_gap_tolerance_bins": ParamSpec(
                "int", defaults["strict_merge_gap_tolerance_bins"],
                "Post-iteration strict merge: same idea as "
                "merge_gap_tolerance_bins but for the burstlets → network "
                "bursts hierarchy step. Defaults one bin tighter (2) "
                "because this merge is intentionally more conservative.",
                min=0,
            ),
            "merge_strict_floor_frac": ParamSpec(
                "float", defaults["merge_strict_floor_frac"],
                "Post-iteration strict merge: valley floor as a fraction "
                "of the detection threshold. Default 1.0 reproduces the "
                "original rule 'valley must stay above the full detection "
                "threshold'. Lower (e.g. 0.7) when the gap-tolerance "
                "override is still not enough to reunite fragments.",
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
            "peak_synchrony_floor_frac": ParamSpec(
                "float", defaults["peak_synchrony_floor_frac"],
                "Per-bin peak-synchrony floor as a fraction of the "
                "window-wide participation_floor. Only consulted when "
                "participation_gate_mode is 'and'. Default 0.5 keeps a "
                "tight per-bin secondary floor; lower toward 0 to admit "
                "even more asynchronous events.",
                min=0, max=1,
            ),
            "participation_gate_mode": ParamSpec(
                "str", defaults["participation_gate_mode"],
                "Combines the two halves of the participation gate. 'and' "
                "(default) drops a burstlet only when BOTH window-wide "
                "participation AND single-bin peak_synchrony fall below "
                "their floors — admits asynchronous bursts. "
                "'peak_synchrony' uses the legacy single-axis rule on "
                "peak_synchrony only; stricter on noise but misses "
                "asynchronous bursts.",
                choices=["and", "peak_synchrony"],
            ),
            "cluster_events": ParamSpec(
                "bool", defaults["cluster_events"],
                "Optional post-iteration GMM filter on per-event quality "
                "features that discards noise-like clusters. Disabled by "
                "default because its scoring weights are hardcoded and "
                "may discard real events on multi-regime recordings; "
                "enable only with care.",
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
            "debug": ParamSpec(
                "bool", defaults["debug"],
                "When True, also persist `debug_trace.pkl` and "
                "`debug_spike_times.npy` with per-iteration state "
                "(composite, LDA weights, X_norm, candidate masks, GMM "
                "cluster labels) for use by the Burst inspector page. "
                "Adds a few MB per well.",
            ),
            "inner_partitioner": ParamSpec(
                "str", defaults["inner_partitioner"],
                "Which method discriminates burst vs background bins each "
                "iteration. 'gmm_em' (default) fits a BIC-adaptive Gaussian "
                "Mixture and uses the burst-component posterior — best for "
                "multi-regime recordings (silence + tonic + burst, or "
                "multiple burst types) where a single linear discriminant "
                "collapses real structure. 'fisher_lda' fits a Fisher linear "
                "discriminant on the z-normed feature matrix — use for clean "
                "2-regime data with one dominant burst direction.",
                choices=["fisher_lda", "gmm_em"],
            ),
            "lda_exclude_silence": ParamSpec(
                "bool", defaults["lda_exclude_silence"],
                "LDA only. Drop bins with zero active units from both the "
                "burst and background classes (and from background z-norm "
                "statistics). Prevents long silent stretches from pulling "
                "the background centroid toward zero and flipping the "
                "discriminant sign on heterogeneous recordings.",
            ),
            "lda_sign_pinned_feature_names": ParamSpec(
                "list_str", defaults["lda_sign_pinned_feature_names"],
                "LDA only. Feature names whose Fisher weight must be "
                "non-negative for the new direction to be accepted. These "
                "features are biologically constrained to be elevated "
                "during bursts; a negative weight signals the LDA found "
                "the wrong contrast and would otherwise destabilise the "
                "next iteration. Unknown names are silently ignored.",
                choices=["PFR", "P", "FF0", "FF1", "FF2", "FF3", "LLR", "burstiness"],
                multiselect=True,
            ),
            "gmm_k_range": ParamSpec(
                "list_int", defaults["gmm_k_range"],
                "GMM-EM only. Two integers [k_min, k_max] (inclusive) for "
                "the BIC sweep over component counts each iteration. "
                "Default (2, 5) lets BIC capture multi-regime structure "
                "(silence, tonic, burst sub-phases). Burst-component "
                "identity drift across iterations is bounded by "
                "gmm_bic_margin (k-flap hysteresis) and by burst-centroid "
                "anchoring. Multi-component burst regimes are unioned at "
                "the posterior stage via gmm_burst_top_fraction.",
                min=2, max=2,
            ),
            "gmm_bic_margin": ParamSpec(
                "float", defaults["gmm_bic_margin"],
                "GMM-EM only. New k* must beat the previous iteration's k* "
                "BIC by this margin before the chosen component count "
                "changes. Suppresses k-flapping when multiple k fit nearly "
                "equally well.",
                min=0,
            ),
            "gmm_em_n_init": ParamSpec(
                "int", defaults["gmm_em_n_init"],
                "GMM-EM only. Number of random restarts per GMM fit "
                "(sklearn GaussianMixture.n_init).",
                min=1,
            ),
            "gmm_em_reg_covar": ParamSpec(
                "float", defaults["gmm_em_reg_covar"],
                "GMM-EM only. Covariance regularisation "
                "(sklearn GaussianMixture.reg_covar). Slightly above the "
                "sklearn default because the z-normed feature matrix "
                "contains near-collinear FF columns.",
                min=0,
            ),
            "gmm_burst_score_weights": ParamSpec(
                "list_float", defaults["gmm_burst_score_weights"],
                "GMM-EM only. Burst-component scoring prior, aligned to "
                "bin feature order [PFR, P, FF0, FF1, FF2, FF3, LLR, "
                "burstiness]. After merging near-duplicate components, "
                "each merged group's centroid is scored by weights @ "
                "centroid; the highest-scoring group is the burst cluster. "
                "Note: 'burstiness' is the mean instantaneous ISI "
                "reciprocal (within-unit temporal tightness), computed "
                "once before iteration and never updated — not the burst "
                "label.",
            ),
            "gmm_posterior_threshold": ParamSpec(
                "float", defaults["gmm_posterior_threshold"],
                "GMM-EM only. Candidate threshold for the burst-component "
                "posterior. Clear the field to fall back to the MAD-based "
                "threshold used by the Fisher path "
                "(median(comp_bg) + composite_mad_scale * MAD).",
                min=0, max=1, nullable=True,
            ),
            "gmm_component_merge_distance": ParamSpec(
                "float", defaults["gmm_component_merge_distance"],
                "GMM-EM only. Standardised-Euclidean distance below which "
                "sibling GMM components are collapsed before centroid "
                "scoring. Prevents degenerate EM solutions (two "
                "near-identical components for the same regime) from "
                "splitting the burst posterior.",
                min=0,
            ),
            "gmm_burst_top_fraction": ParamSpec(
                "float", defaults["gmm_burst_top_fraction"],
                "GMM-EM only. Keep every merged group whose centroid "
                "score is at least gmm_burst_top_fraction × top_score. "
                "Default 0.7 pairs with the wider gmm_k_range default: "
                "when BIC picks a higher k and the burst regime splits "
                "into ramp / peak / plateau components, the posterior "
                "unions them so they survive downstream merging as one "
                "event. Set to 1.0 to keep only the single top-scoring "
                "component (stricter); lower values union an even wider "
                "multi-component burst regime.",
                min=0, max=1,
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
        import dataclasses
        import json
        import pickle

        import numpy as np

        from yuxin_mea.analysis import (
            IterativeBurstConfig,
            IterativeBurstTrace,
            compute_iterative_bursts,
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

        config = IterativeBurstConfig(
            permissive_mad_scale=float(p["permissive_mad_scale"]),
            permissive_percentile=float(p["permissive_percentile"]),
            mad_fallback_threshold=float(p["mad_fallback_threshold"]),
            composite_mad_scale=float(p["composite_mad_scale"]),
            extent_frac=float(p["extent_frac"]),
            merge_floor_frac=float(p["merge_floor_frac"]),
            merge_gap_tolerance_bins=int(p["merge_gap_tolerance_bins"]),
            strict_merge_gap_tolerance_bins=int(p["strict_merge_gap_tolerance_bins"]),
            merge_strict_floor_frac=float(p["merge_strict_floor_frac"]),
            network_merge_gap_min_s=float(p["network_merge_gap_min_s"]),
            max_iterations=int(p["max_iterations"]),
            convergence_eps=float(p["convergence_eps"]),
            fisher_alpha_frac=float(p["fisher_alpha_frac"]),
            ff_scale_multipliers=tuple(float(x) for x in p["ff_scale_multipliers"]),
            min_burst_modulation=float(p["min_burst_modulation"]),
            peak_synchrony_floor_frac=float(p["peak_synchrony_floor_frac"]),
            participation_gate_mode=str(p["participation_gate_mode"]),
            cluster_events=bool(p["cluster_events"]),
            cluster_initial_components=int(p["cluster_initial_components"]),
            cluster_min_events=int(p["cluster_min_events"]),
            cluster_min_separation=float(p["cluster_min_separation"]),
            inner_partitioner=str(p["inner_partitioner"]),
            lda_exclude_silence=bool(p["lda_exclude_silence"]),
            lda_sign_pinned_feature_names=tuple(
                str(x) for x in p["lda_sign_pinned_feature_names"]
            ),
            gmm_k_range=tuple(int(x) for x in p["gmm_k_range"]),  # type: ignore[arg-type]
            gmm_bic_margin=float(p["gmm_bic_margin"]),
            gmm_em_n_init=int(p["gmm_em_n_init"]),
            gmm_em_reg_covar=float(p["gmm_em_reg_covar"]),
            gmm_burst_score_weights=tuple(
                float(x) for x in p["gmm_burst_score_weights"]
            ),
            gmm_posterior_threshold=(
                None if p["gmm_posterior_threshold"] is None
                else float(p["gmm_posterior_threshold"])
            ),
            gmm_component_merge_distance=float(p["gmm_component_merge_distance"]),
            gmm_burst_top_fraction=float(p["gmm_burst_top_fraction"]),
        )

        debug_enabled = bool(p.get("debug", False))
        trace = IterativeBurstTrace() if debug_enabled else None
        results = compute_iterative_bursts(
            spike_times, config=config, trace=trace
        )

        output_dir = self.build_output_path(
            p["output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        PickleBurstOutputWriter().write(results, output_dir)

        if trace is not None:
            with open(output_dir / "debug_trace.pkl", "wb") as fh:
                pickle.dump(trace, fh)
            np.save(
                output_dir / "debug_spike_times.npy",
                np.array(spike_times, dtype=object),
                allow_pickle=True,
            )
            # Persist the *run-time* config so the inspector reads back the
            # parameters used here, not whatever the user has in their
            # pipeline_config.json at viewing time.
            config_dict = dataclasses.asdict(config)
            # JSON can't encode tuples; round-trip them as lists.
            for k, v in list(config_dict.items()):
                if isinstance(v, tuple):
                    config_dict[k] = list(v)
            with open(output_dir / "debug_config.json", "w") as fh:
                json.dump(config_dict, fh, indent=2, sort_keys=True)

        return output_dir
