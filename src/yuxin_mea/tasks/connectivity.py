"""ConnectivityTask — pipeline wrapper for STTC functional connectivity.

Reads ``curated_spike_times.npy`` and ``quality_metrics.pkl`` (for the edge
distance column) from the ``auto_curation`` output, computes the STTC matrix +
dt sweep + circular-shuffle significance + graph topology metrics, and writes
``sttc_matrix.npy`` + ``sttc_sweep.npz`` + ``graph_metrics.json`` +
``edges.parquet`` + ``diagnostics.json`` per well.

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
            compute_connectivity,
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
        results = compute_connectivity(spike_times, positions=positions, config=config)

        output_dir = self.build_output_path(
            p["output_root"], recording_key, rec_name, actual_well_id,
        )
        write_connectivity(results, output_dir)
        return output_dir
