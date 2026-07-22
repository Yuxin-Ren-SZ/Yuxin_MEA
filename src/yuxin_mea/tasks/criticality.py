"""CriticalityTask — pipeline wrapper for neuronal-avalanche criticality.

Reads ``curated_spike_times.npy`` from the ``auto_curation`` output, pools the
curated spikes into a population raster, detects avalanches (Beggs-Plenz), fits
size/duration power laws, and computes the branching ratio (MR estimator + naive)
and the Deviation-from-Criticality Coefficient. Writes ``avalanches.parquet`` +
``criticality_metrics.json`` + ``powerlaw_fits.json`` + ``diagnostics.json`` per
well. ``dependencies=["auto_curation"]``; keep ``n_jobs`` at 1 (pipeline
parallelism runs one well per worker).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask
from yuxin_mea.tasks.preprocessing import PreprocessingTask


class CriticalityTask(BaseAnalysisTask):
    """Neuronal-avalanche criticality: branching ratio, DCC, power-law exponents."""

    task_name = "criticality"
    dependencies = ["auto_curation"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "curation_output_root": "./curation_data",
            "output_root": "./criticality_data",
            "min_spikes": 200,
            "min_avalanches": 30,
            "bin_mode": "mean_iei",
            "fixed_bin_s": 0.004,
            "mr_max_bin_s": 0.05,
            "mr_kmax": 150,
            "seed": 0,
        }

    @classmethod
    def params_schema(cls) -> dict[str, ParamSpec]:
        d = cls.default_params()
        return {
            "curation_output_root": ParamSpec(
                "path", d["curation_output_root"],
                "Root of auto_curation outputs (curated_spike_times.npy per well)."),
            "output_root": ParamSpec(
                "path", d["output_root"],
                "Directory where criticality results are written per well."),
            "min_spikes": ParamSpec(
                "int", d["min_spikes"],
                "Pooled curated spikes required for any avalanche statistics.", min=1),
            "min_avalanches": ParamSpec(
                "int", d["min_avalanches"],
                "Avalanches required before a power-law exponent is fit.", min=1),
            "bin_mode": ParamSpec(
                "str", d["bin_mode"],
                "Avalanche bin width: 'mean_iei' (Beggs-Plenz) or 'fixed'.",
                choices=["mean_iei", "fixed"]),
            "fixed_bin_s": ParamSpec(
                "float", d["fixed_bin_s"],
                "Fixed avalanche bin width (s) when bin_mode='fixed'.", min=0.0),
            "mr_max_bin_s": ParamSpec(
                "float", d["mr_max_bin_s"],
                "Cap on the bin used for the MR branching-ratio estimator (s).", min=0.0),
            "mr_kmax": ParamSpec(
                "int", d["mr_kmax"],
                "Max autocorrelation steps for the MR estimator.", min=2),
            "seed": ParamSpec("int", d["seed"], "Random seed.", min=0),
        }

    @staticmethod
    def split_compound_well_id(well_id: str) -> tuple[str, str]:
        return PreprocessingTask.split_compound_well_id(well_id)

    @staticmethod
    def build_curation_output_path(curation_output_root, recording_key, rec_name, well_id) -> Path:
        return (Path(curation_output_root) / Path(recording_key) / rec_name
                / well_id / "auto_curation")

    @staticmethod
    def build_output_path(output_root, recording_key, rec_name, well_id) -> Path:
        return (Path(output_root) / Path(recording_key) / rec_name
                / well_id / "criticality")

    def run(self, recording_key: str, well_id: str, data_path: Path,
            params: dict[str, Any]) -> Path:
        import numpy as np

        from yuxin_mea.analysis.criticality import (
            CriticalityConfig, CriticalityError, compute_criticality,
            empty_criticality_results, write_criticality,
        )

        p = self.resolve_params(params)
        rec_name, actual_well_id = self.split_compound_well_id(well_id)
        curation_dir = self.build_curation_output_path(
            p["curation_output_root"], recording_key, rec_name, actual_well_id)
        spikes_path = curation_dir / "curated_spike_times.npy"
        if not spikes_path.exists():
            raise FileNotFoundError(
                f"curated_spike_times.npy not found at {spikes_path}. "
                "Ensure auto_curation has completed successfully.")
        spike_times = np.load(spikes_path, allow_pickle=True).item()

        config = CriticalityConfig.from_task_params(p)
        try:
            results = compute_criticality(spike_times, config=config)
        except CriticalityError as exc:
            results = empty_criticality_results(reason=str(exc))

        out = self.build_output_path(
            p["output_root"], recording_key, rec_name, actual_well_id)
        write_criticality(results, out)
        return out
