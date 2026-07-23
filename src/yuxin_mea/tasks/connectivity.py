"""ConnectivityTask — pipeline wrapper for STTC functional connectivity.

Reads ``curated_spike_times.npy`` and ``quality_metrics.pkl`` (for the edge
distance column) from the ``auto_curation`` output, computes the STTC matrix +
dt sweep + circular-shuffle significance + graph topology metrics, and writes
``sttc_matrix.npy`` + ``sttc_sweep.npz`` + ``graph_metrics.json`` +
``edges.parquet`` + ``ccg.npz`` + ``diagnostics.json`` per well.

Correlograms run on the edges that survive the significance threshold (reference
= ``u``, target = ``v``, positive lag = ``u`` leads); they are cheap next to the
shuffle null, so they are on by default.

STTC is computed over the full recording (``dependencies=["auto_curation"]``);
restricting to burst epochs is a future config toggle (would add a
``burst_detection`` dependency). Keep the task-config ``n_jobs`` at 1: pipeline
parallelism runs one well per worker (CLI ``--jobs``); intra-well joblib on top
would oversubscribe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask
from yuxin_mea.tasks.preprocessing import PreprocessingTask


class ConnectivityTask(BaseAnalysisTask):
    """Pairwise Spike Time Tiling Coefficient graph + network topology metrics."""

    task_name = "connectivity"
    dependencies = ["auto_curation"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            # ---- I/O ------------------------------------------------------
            "curation_output_root": "./curation_data",
            "output_root": "./connectivity_data",
            # ---- STTC -----------------------------------------------------
            "dt": 0.02,
            "dt_sweep": [0.005, 0.01, 0.02, 0.05],
            "min_units": 3,
            "min_spikes_per_unit": 5,
            # ---- Significance null ---------------------------------------
            "n_shuffle": 100,
            "thresh_percentile": 95.0,
            "random_state": 42,
            # ---- Cross-correlograms --------------------------------------
            "ccg_enable": True,
            "ccg_window": 0.1,
            "ccg_bin": 0.001,
            "ccg_hollow_sigma": 0.01,
            "ccg_hollow_frac": 0.6,
            "ccg_alpha": 0.001,
            "ccg_syn_lo": 0.0008,
            "ccg_syn_hi": 0.008,
            "ccg_max_pairs": 5000,
            # ---- Parallelism ---------------------------------------------
            "n_jobs": 1,
        }

    @classmethod
    def params_schema(cls) -> dict[str, ParamSpec]:
        defaults = cls.default_params()
        return {
            "curation_output_root": ParamSpec(
                "path", defaults["curation_output_root"],
                "Root of auto_curation outputs (curated_spike_times.npy + "
                "quality_metrics.pkl per recording/well).",
            ),
            "output_root": ParamSpec(
                "path", defaults["output_root"],
                "Directory where connectivity results are written per recording/well.",
            ),
            "dt": ParamSpec(
                "float", defaults["dt"],
                "Primary STTC tiling window (seconds). 0.02 = 20 ms.",
                min=0.0,
            ),
            "dt_sweep": ParamSpec(
                "list_float", defaults["dt_sweep"],
                "Tiling windows (seconds) reported in the diagnostics sweep. "
                "The primary dt should be one of these.",
                min=0.0,
            ),
            "min_units": ParamSpec(
                "int", defaults["min_units"],
                "Minimum eligible units for a connectivity graph; below this the "
                "well returns no graph (task fails on this well, like a sparse well).",
                min=2,
            ),
            "min_spikes_per_unit": ParamSpec(
                "int", defaults["min_spikes_per_unit"],
                "Units with fewer spikes are dropped before STTC (a 1-2 spike "
                "train's tiling fraction is meaningless).",
                min=1,
            ),
            "n_shuffle": ParamSpec(
                "int", defaults["n_shuffle"],
                "Circular-jitter shuffles building the significance null. The "
                "expensive step — cap at ~100.",
                min=1,
            ),
            "thresh_percentile": ParamSpec(
                "float", defaults["thresh_percentile"],
                "Keep edges whose STTC exceeds this percentile of the pooled "
                "shuffled null (95 → comparable density across wells).",
                min=0.0, max=100.0,
            ),
            "random_state": ParamSpec(
                "int", defaults["random_state"],
                "Seed base for the circular-shift null (reproducible edges).",
                min=0,
            ),
            "ccg_enable": ParamSpec(
                "bool", defaults["ccg_enable"],
                "Compute a cross-correlogram for every significant STTC edge "
                "(ccg.npz + the ccg_* columns on edges.parquet).",
            ),
            "ccg_window": ParamSpec(
                "float", defaults["ccg_window"],
                "Correlogram half-width (seconds). 0.1 = +/-100 ms.",
                min=0.0,
            ),
            "ccg_bin": ParamSpec(
                "float", defaults["ccg_bin"],
                "Correlogram bin width (seconds). 0.001 = 1 ms -> 201 bins at "
                "the default window.",
                min=0.0,
            ),
            "ccg_hollow_sigma": ParamSpec(
                "float", defaults["ccg_hollow_sigma"],
                "Sigma (seconds) of the partially-hollow Gaussian that estimates "
                "the slow baseline — the network-burst co-activation bump every "
                "MEA pair carries.",
                min=0.0,
            ),
            "ccg_hollow_frac": ParamSpec(
                "float", defaults["ccg_hollow_frac"],
                "Centre-bin hollowing fraction (Stark & Abeles 2009). 0.6 keeps a "
                "sharp peak from predicting its own baseline.",
                min=0.0, max=1.0,
            ),
            "ccg_alpha": ParamSpec(
                "float", defaults["ccg_alpha"],
                "Per-bin Poisson tail threshold against the baseline; a bin below "
                "it inside the causal window flags the edge.",
                min=0.0, max=1.0,
            ),
            "ccg_syn_lo": ParamSpec(
                "float", defaults["ccg_syn_lo"],
                "Low edge (seconds) of the causal window used for the "
                "significance flag, direction and asymmetry.",
                min=0.0,
            ),
            "ccg_syn_hi": ParamSpec(
                "float", defaults["ccg_syn_hi"],
                "High edge (seconds) of the causal window.",
                min=0.0,
            ),
            "ccg_max_pairs": ParamSpec(
                "int", defaults["ccg_max_pairs"],
                "Cap on correlograms per well; above it only the highest-STTC "
                "edges are computed and the rest keep NaN ccg_* columns.",
                min=1,
            ),
            "n_jobs": ParamSpec(
                "int", defaults["n_jobs"],
                "joblib parallelism across shuffles. 1 = serial; use 1 inside "
                "the worker CLI (pipeline parallelism handles wells).",
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
            / "connectivity"
        )

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        import numpy as np
        import pandas as pd

        from yuxin_mea.analysis.connectivity import (
            ConnectivityConfig,
            ConnectivityError,
            compute_connectivity,
            empty_connectivity_results,
            write_connectivity,
        )

        p = self.resolve_params(params)
        rec_name, actual_well_id = self.split_compound_well_id(well_id)

        curation_dir = self.build_curation_output_path(
            p["curation_output_root"], recording_key, rec_name, actual_well_id,
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

        # Positions for the edge distance column (optional).
        positions = None
        qm_path = curation_dir / "quality_metrics.pkl"
        if qm_path.exists():
            qm = pd.read_pickle(qm_path)
            if "loc_x" in qm.columns and "loc_y" in qm.columns:
                positions = {
                    uid: (float(row["loc_x"]), float(row["loc_y"]))
                    for uid, row in qm.iterrows()
                }

        config = ConnectivityConfig.from_task_params(p)
        try:
            results = compute_connectivity(
                spike_times, positions=positions, config=config
            )
        except ConnectivityError as exc:
            # Too few eligible units: write an empty graph, COMPLETE (not FAILED)
            # — mirrors the burst detectors on sparse wells.
            results = empty_connectivity_results(config=config, reason=str(exc))

        output_dir = self.build_output_path(
            p["output_root"], recording_key, rec_name, actual_well_id,
        )
        write_connectivity(results, output_dir)
        return output_dir
