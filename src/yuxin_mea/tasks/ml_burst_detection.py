"""MLBurstDetectionTask — pipeline wrapper for the ML burst detector.

Loads ``curated_spike_times.npy`` from ``auto_curation`` output, constructs an
``MLBurstConfig`` from the resolved params, runs ``compute_ml_bursts``, and
writes a single-level BurstResults bundle (``network_bursts.pkl`` — the bursts,
each row carrying ``cluster_ids`` — plus ``metrics.json``, ``diagnostics.json``,
``plot_signals.npy``) to its own output directory. The burstlet and superburst
tiers are intentionally not written.

When ``debug=True`` it also persists ``debug_trace.pkl``,
``debug_spike_times.npy``, and ``debug_config.json`` so a future ML-aware
burst-inspector page can share most of the loading code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask
from yuxin_mea.tasks.preprocessing import PreprocessingTask


class MLBurstDetectionTask(BaseAnalysisTask):
    """Unsupervised ML-based network burst detection.

    Reads the same ``curated_spike_times.npy`` as ``BurstDetectionTask`` but
    detects bursts via a per-unit 2-state Poisson HMM (calibrated λ_bg/λ_burst
    per unit) plus HDBSCAN clustering on a 26-column bin-level feature matrix.

    Output schema mirrors the other detectors (``BurstResults`` pickle layout)
    with four per-event quality columns: ``posterior_peak``, ``posterior_mean``,
    ``llr_aggregate``, ``ff_peak``.
    """

    task_name = "ml_burst_detection"
    dependencies = ["auto_curation"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            # ---- I/O ------------------------------------------------------
            "curation_output_root": "./curation_data",
            "output_root": "./ml_burst_data",
            # ---- Binning --------------------------------------------------
            "bin_size_mode": "adaptive",
            "fixed_bin_size_s": 0.02,
            # ---- HMM ------------------------------------------------------
            "hmm_max_iter": 100,
            "hmm_tol": 1e-3,
            "hmm_min_spikes": 50,
            "hmm_init_strategy": "quantile",
            "hmm_min_rate_ratio": 1.5,
            "hmm_random_state": 42,
            "hmm_n_jobs": 1,
            # ---- Features -------------------------------------------------
            "ff_scale_multipliers": [0.5, 1.0, 2.0, 5.0],
            "posterior_quantile": 0.9,
            "isi_window_bins": 25,
            "deriv_sigma_short_bins": 1.5,
            "deriv_sigma_long_bins": 8.0,
            "background_quantile": 0.5,
            "unit_agg_quantile": 0.9,
            # ---- Dim reduction -------------------------------------------
            "pca_n_components": 0,
            # ---- Clustering embedding & multi-cluster burst selection -----
            "cluster_embedding_mode": "none",
            "umap_n_neighbors": 30,
            "umap_min_dist": 0.0,
            "umap_n_components": 5,
            "burst_mad_scale": 3.0,
            # ---- HDBSCAN --------------------------------------------------
            "hdbscan_min_cluster_size": 30,
            "hdbscan_min_samples": 5,
            "hdbscan_cluster_selection_epsilon": 0.0,
            "hdbscan_cluster_selection_method": "eom",
            "hdbscan_metric": "euclidean",
            "cluster_ranking_feature": "post_frac_gt_0_5",
            "fallback_posterior_threshold": 0.3,
            # ---- Temporal merge / hierarchy ------------------------------
            "closing_bins": 3,
            "merge_mad_scale": 0.75,
            "merge_floor_frac": 0.70,
            "network_merge_gap_min_s": 0.75,
            "min_burst_modulation": 0.1,
            # ---- Burst typing (second-stage) ------------------------------
            "burst_typing_enabled": True,
            "burst_typing_method": "kmeans",
            "burst_typing_max_k": 6,
            "burst_typing_k": 0,
            "burst_typing_min_bursts": 8,
            "burst_typing_random_state": 42,
            # ---- Debug ----------------------------------------------------
            "debug": False,
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
                "Directory where ML burst detection results are written "
                "per recording/well.",
            ),
            "bin_size_mode": ParamSpec(
                "str", defaults["bin_size_mode"],
                "How to choose bin size. 'adaptive' uses the population median "
                "log-ISI (clamped to [10, 30] ms) — matches the iterative "
                "detector so outputs are comparable. 'fixed' uses "
                "fixed_bin_size_s instead.",
                choices=["adaptive", "fixed"],
            ),
            "fixed_bin_size_s": ParamSpec(
                "float", defaults["fixed_bin_size_s"],
                "Bin size in seconds when bin_size_mode='fixed'.",
                min=0.001,
            ),
            "hmm_max_iter": ParamSpec(
                "int", defaults["hmm_max_iter"],
                "Hard cap on Baum-Welch EM iterations per unit.",
                min=1,
            ),
            "hmm_tol": ParamSpec(
                "float", defaults["hmm_tol"],
                "Stop Baum-Welch when |Δ log-likelihood| < tol between iterations.",
                min=0,
            ),
            "hmm_min_spikes": ParamSpec(
                "int", defaults["hmm_min_spikes"],
                "Units with fewer total spikes are skipped (their per-bin "
                "posterior is NaN and dropped from aggregations).",
                min=0,
            ),
            "hmm_init_strategy": ParamSpec(
                "str", defaults["hmm_init_strategy"],
                "Initial λ_bg / λ_burst estimate for Baum-Welch. 'quantile' "
                "uses 25th / 95th percentile of bin counts (robust default). "
                "'kmeans' fits 1D k-means with k=2.",
                choices=["quantile", "kmeans"],
            ),
            "hmm_min_rate_ratio": ParamSpec(
                "float", defaults["hmm_min_rate_ratio"],
                "Reject the fit if λ_burst / λ_bg falls below this ratio "
                "(unit lacks a clear two-rate regime).",
                min=1.0,
            ),
            "hmm_random_state": ParamSpec(
                "int", defaults["hmm_random_state"],
                "Random seed used by k-means init (Baum-Welch is deterministic).",
                min=0,
            ),
            "hmm_n_jobs": ParamSpec(
                "int", defaults["hmm_n_jobs"],
                "joblib parallelism across units. 1 = serial; -1 = all cores. "
                "Use 1 inside the worker CLI (pipeline parallelism handles wells).",
            ),
            "ff_scale_multipliers": ParamSpec(
                "list_float", defaults["ff_scale_multipliers"],
                "Bin-size multipliers for the multi-scale Fano Factor feature. "
                "Same convention as the iterative detector; clamped to [5, 100] ms.",
                min=1,
            ),
            "posterior_quantile": ParamSpec(
                "float", defaults["posterior_quantile"],
                "Top quantile of per-unit HMM posteriors aggregated as the "
                "`post_q90` feature column.",
                min=0, max=1,
            ),
            "isi_window_bins": ParamSpec(
                "int", defaults["isi_window_bins"],
                "Sliding window (in bins) for CV(ISI) and Shinomoto LV features.",
                min=1,
            ),
            "deriv_sigma_short_bins": ParamSpec(
                "float", defaults["deriv_sigma_short_bins"],
                "Gaussian smoothing σ (bins) applied to short-scale ΔPFR, "
                "ΔParticipation, and ΔLLR derivative features.",
                min=0,
            ),
            "deriv_sigma_long_bins": ParamSpec(
                "float", defaults["deriv_sigma_long_bins"],
                "Gaussian smoothing σ (bins) applied to long-scale derivative "
                "features.",
                min=0,
            ),
            "background_quantile": ParamSpec(
                "float", defaults["background_quantile"],
                "Bottom fraction of the ranking feature treated as background "
                "for z-norm scaler statistics.",
                min=0, max=1,
            ),
            "unit_agg_quantile": ParamSpec(
                "float", defaults["unit_agg_quantile"],
                "Top quantile used when aggregating per-unit signals (e.g. "
                "per-unit LLR's top-quantile column).",
                min=0, max=1,
            ),
            "pca_n_components": ParamSpec(
                "int", defaults["pca_n_components"],
                "If > 0, apply PCA to the z-normed feature matrix before "
                "HDBSCAN. 0 disables PCA (default). Ignored when "
                "cluster_embedding_mode='umap'.",
                min=0,
            ),
            "cluster_embedding_mode": ParamSpec(
                "str", defaults["cluster_embedding_mode"],
                "Space HDBSCAN clusters in. 'none' = z-normed features (or PCA "
                "if pca_n_components>0); 'umap' = UMAP embedding of the z-normed "
                "features, which recovers the low-density burst trajectory that "
                "HDBSCAN discards as noise in the raw space.",
                choices=["none", "umap"],
            ),
            "umap_n_neighbors": ParamSpec(
                "int", defaults["umap_n_neighbors"],
                "UMAP n_neighbors when cluster_embedding_mode='umap'.",
                min=2,
            ),
            "umap_min_dist": ParamSpec(
                "float", defaults["umap_min_dist"],
                "UMAP min_dist when cluster_embedding_mode='umap' (0 packs "
                "points tightly, best for clustering).",
                min=0, max=1,
            ),
            "umap_n_components": ParamSpec(
                "int", defaults["umap_n_components"],
                "UMAP output dimensionality clustered by HDBSCAN when "
                "cluster_embedding_mode='umap'.",
                min=2,
            ),
            "burst_mad_scale": ParamSpec(
                "float", defaults["burst_mad_scale"],
                "Multi-cluster burst selection: a cluster is burst if its mean "
                "ranking feature exceeds median+mad_scale*MAD of the background "
                "(lower half). The top-ranked cluster is always included.",
                min=0,
            ),
            "hdbscan_min_cluster_size": ParamSpec(
                "int", defaults["hdbscan_min_cluster_size"],
                "HDBSCAN min_cluster_size: smallest cluster size considered "
                "real (smaller groups become noise).",
                min=2,
            ),
            "hdbscan_min_samples": ParamSpec(
                "int", defaults["hdbscan_min_samples"],
                "HDBSCAN min_samples: density threshold for cluster cores.",
                min=1,
            ),
            "hdbscan_cluster_selection_epsilon": ParamSpec(
                "float", defaults["hdbscan_cluster_selection_epsilon"],
                "HDBSCAN cluster_selection_epsilon: distance threshold for "
                "merging clusters. 0 disables.",
                min=0,
            ),
            "hdbscan_cluster_selection_method": ParamSpec(
                "str", defaults["hdbscan_cluster_selection_method"],
                "HDBSCAN cluster_selection_method: 'eom' (excess of mass, "
                "default) or 'leaf' (more fine-grained).",
                choices=["eom", "leaf"],
            ),
            "hdbscan_metric": ParamSpec(
                "str", defaults["hdbscan_metric"],
                "HDBSCAN distance metric.",
                choices=["euclidean", "manhattan"],
            ),
            "cluster_ranking_feature": ParamSpec(
                "str", defaults["cluster_ranking_feature"],
                "Which feature column is used to rank clusters and identify "
                "the burst cluster (highest mean wins).",
                choices=["post_frac_gt_0_5", "post_mean", "llr_hmm_mean"],
            ),
            "fallback_posterior_threshold": ParamSpec(
                "float", defaults["fallback_posterior_threshold"],
                "When HDBSCAN returns all-noise, fall back to thresholding "
                "the ranking feature at this value.",
                min=0, max=1,
            ),
            "closing_bins": ParamSpec(
                "int", defaults["closing_bins"],
                "1D morphological-closing structuring element size (bins). "
                "Fills gaps shorter than this in the burst-bin mask.",
                min=1,
            ),
            "merge_mad_scale": ParamSpec(
                "float", defaults["merge_mad_scale"],
                "MAD multiplier on the ranking signal used to derive the "
                "iter-merge threshold (mirrors the iterative detector).",
                min=0,
            ),
            "merge_floor_frac": ParamSpec(
                "float", defaults["merge_floor_frac"],
                "Adjacent candidates merge when valley ≥ floor_frac × threshold.",
                min=0,
            ),
            "network_merge_gap_min_s": ParamSpec(
                "float", defaults["network_merge_gap_min_s"],
                "Minimum gap (seconds) enforced for the network-burst merge "
                "stage in the hierarchy.",
                min=0,
            ),
            "min_burst_modulation": ParamSpec(
                "float", defaults["min_burst_modulation"],
                "Minimum llr_aggregate required for a burstlet to survive. "
                "≤ 0 disables the gate.",
                min=0,
            ),
            "burst_typing_enabled": ParamSpec(
                "bool", defaults["burst_typing_enabled"],
                "Run the second-stage clustering over detected bursts, adding a "
                "per-burst `burst_type` label. False omits the column.",
            ),
            "burst_typing_method": ParamSpec(
                "str", defaults["burst_typing_method"],
                "Clustering algorithm for burst typing. 'kmeans' picks k by max "
                "silhouette; 'gmm' (diag covariance) picks k by min BIC.",
                choices=["kmeans", "gmm"],
            ),
            "burst_typing_max_k": ParamSpec(
                "int", defaults["burst_typing_max_k"],
                "Upper bound on the number of burst types (k swept 2..max_k, "
                "clamped to n_bursts-1).",
                min=2,
            ),
            "burst_typing_k": ParamSpec(
                "int", defaults["burst_typing_k"],
                "Fixed number of burst types. 0 = auto-select k by the method's "
                "criterion.",
                min=0,
            ),
            "burst_typing_min_bursts": ParamSpec(
                "int", defaults["burst_typing_min_bursts"],
                "Skip typing (assign all bursts type 0) when a well has fewer "
                "bursts than this — too few to cluster meaningfully.",
                min=1,
            ),
            "burst_typing_random_state": ParamSpec(
                "int", defaults["burst_typing_random_state"],
                "Random seed for the burst-typing clusterer (KMeans/GMM init).",
                min=0,
            ),
            "debug": ParamSpec(
                "bool", defaults["debug"],
                "When True, also persist `debug_trace.pkl`, "
                "`debug_spike_times.npy`, and `debug_config.json` with the "
                "per-unit HMM fits, posterior matrix, feature matrix, HDBSCAN "
                "labels, and pre/post merge masks. Adds a few MB per well.",
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
            / "ml_burst_detection"
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

        from yuxin_mea.analysis.burst_output import PickleBurstOutputWriter
        from yuxin_mea.analysis.ml_burst_detector import (
            MLBurstConfig,
            MLBurstTrace,
            compute_ml_bursts,
        )

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

        config = MLBurstConfig.from_task_params(p)

        debug_enabled = bool(p.get("debug", False))
        trace = MLBurstTrace() if debug_enabled else None
        results = compute_ml_bursts(spike_times, config=config, trace=trace)

        output_dir = self.build_output_path(
            p["output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        # ML detector emits a single burst level — write only network_bursts.pkl
        # (burstlets/superbursts are dropped, not written as empty files).
        PickleBurstOutputWriter().write(results, output_dir, levels={"network_bursts"})

        if trace is not None:
            with open(output_dir / "debug_trace.pkl", "wb") as fh:
                pickle.dump(trace, fh)
            np.save(
                output_dir / "debug_spike_times.npy",
                np.array(spike_times, dtype=object),
                allow_pickle=True,
            )
            # Persist the run-time config so the inspector reads back the
            # parameters used here, not whatever the user has in their
            # pipeline_config.json at viewing time.
            config_dict = dataclasses.asdict(config)
            for k, v in list(config_dict.items()):
                if isinstance(v, tuple):
                    config_dict[k] = list(v)
            with open(output_dir / "debug_config.json", "w") as fh:
                json.dump(config_dict, fh, indent=2, sort_keys=True)

        return output_dir
