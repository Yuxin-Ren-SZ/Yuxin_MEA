"""SpatialMapTask — pipeline wrapper for the spatial activity map + propagation.

Reads ``curated_spike_times.npy`` and ``quality_metrics.pkl`` (for ``loc_x``/
``loc_y``/``firing_rate``) from the ``auto_curation`` output, plus
``network_bursts.pkl`` from the traditional ``burst_detection`` output (for the
propagation plane fit), then writes an ``activity_field.npy`` +
``burst_propagation.parquet`` + ``spatial_metrics.json`` + ``diagnostics.json``
bundle to its own per-well directory.

Depends on both ``auto_curation`` (spikes + positions) and ``burst_detection``
(burst windows). A well with no network bursts still produces the activity field
(``n_bursts_used=0``, COMPLETE — not FAILED), mirroring the burst detectors on
sparse wells.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask
from yuxin_mea.tasks.burst_detection import BurstDetectionTask
from yuxin_mea.tasks.preprocessing import PreprocessingTask


class SpatialMapTask(BaseAnalysisTask):
    """Electrode-plane firing-rate field + network-burst propagation gradient."""

    task_name = "spatial_map"
    dependencies = ["auto_curation", "burst_detection"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            # ---- I/O ------------------------------------------------------
            "curation_output_root": "./curation_data",
            "burst_output_root": "./burst_detection_data",
            "output_root": "./spatial_map_data",
            # ---- Field ----------------------------------------------------
            "bins": 64,
            "sigma_um": 50.0,
            "active_thresh_frac": 0.05,
            # ---- Propagation ---------------------------------------------
            "min_participation": 5,
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
            "burst_output_root": ParamSpec(
                "path", defaults["burst_output_root"],
                "Root of traditional burst_detection outputs "
                "(network_bursts.pkl per recording/well) used for propagation.",
            ),
            "output_root": ParamSpec(
                "path", defaults["output_root"],
                "Directory where spatial-map results are written per recording/well.",
            ),
            "bins": ParamSpec(
                "int", defaults["bins"],
                "Grid resolution of the 2-D activity field (bins per axis).",
                min=8,
            ),
            "sigma_um": ParamSpec(
                "float", defaults["sigma_um"],
                "Gaussian-KDE smoothing sigma (µm) for the activity field; "
                "≈50 matches the analyzer sparsity radius.",
                min=0.0,
            ),
            "active_thresh_frac": ParamSpec(
                "float", defaults["active_thresh_frac"],
                "Fraction of the field's max that defines the 'active' support "
                "for active_area_frac.",
                min=0.0, max=1.0,
            ),
            "min_participation": ParamSpec(
                "int", defaults["min_participation"],
                "Minimum positioned, in-window units required to fit a burst's "
                "propagation plane (also floored at 3).",
                min=3,
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
            / "spatial_map"
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

        from yuxin_mea.analysis.spatial_map import (
            SpatialMapConfig,
            SpatialMapError,
            compute_spatial_map,
            empty_spatial_results,
            write_spatial_map,
        )

        p = self.resolve_params(params)
        rec_name, actual_well_id = self.split_compound_well_id(well_id)

        curation_dir = self.build_curation_output_path(
            p["curation_output_root"], recording_key, rec_name, actual_well_id,
        )
        spike_times_path = curation_dir / "curated_spike_times.npy"
        qm_path = curation_dir / "quality_metrics.pkl"
        if not spike_times_path.exists():
            raise FileNotFoundError(
                f"curated_spike_times.npy not found at {spike_times_path}. "
                "Ensure auto_curation has completed successfully."
            )
        if not qm_path.exists():
            raise FileNotFoundError(
                f"quality_metrics.pkl not found at {qm_path}. "
                "Ensure auto_curation has completed successfully."
            )

        spike_times: dict = np.load(  # type: ignore[call-overload]
            spike_times_path, allow_pickle=True
        ).item()
        quality_metrics = pd.read_pickle(qm_path)

        # Network bursts (optional — field-only if absent/empty).
        burst_dir = BurstDetectionTask.build_output_path(
            p["burst_output_root"], recording_key, rec_name, actual_well_id,
        )
        bursts_path = burst_dir / "network_bursts.pkl"
        bursts = pd.read_pickle(bursts_path) if bursts_path.exists() else None

        config = SpatialMapConfig.from_task_params(p)
        try:
            results = compute_spatial_map(
                spike_times, quality_metrics, bursts, config=config
            )
        except SpatialMapError as exc:
            # Sparse well (no positioned units): write an empty field, COMPLETE
            # (not FAILED) — mirrors the burst detectors on sparse wells.
            results = empty_spatial_results(bins=int(p["bins"]), reason=str(exc))

        output_dir = self.build_output_path(
            p["output_root"], recording_key, rec_name, actual_well_id,
        )
        write_spatial_map(results, output_dir)
        return output_dir
