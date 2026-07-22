"""DirectedConnectivityTask — pipeline wrapper for transfer-entropy directed graphs.

Reads ``curated_spike_times.npy``, bins/binarises the most-active units, computes
pairwise transfer entropy (delayed, effective/bias-corrected), and writes
``te_matrix.npy`` + ``directed_edges.parquet`` + ``directed_metrics.json`` +
``diagnostics.json``. ``dependencies=["auto_curation"]``.

Caveat: binary spike-train TE is inflated by network-burst co-activation, so the
edge density saturates; the interpretable readouts are flow_hierarchy,
degree_asymmetry and effective mean_te. This is the most preliminary of the
connectivity tasks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask
from yuxin_mea.tasks.preprocessing import PreprocessingTask


class DirectedConnectivityTask(BaseAnalysisTask):
    """Directed / effective connectivity via transfer entropy."""

    task_name = "directed_connectivity"
    dependencies = ["auto_curation"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "curation_output_root": "./curation_data",
            "output_root": "./directed_connectivity_data",
            "bin_s": 0.005,
            "max_units": 64,
            "min_units": 5,
            "min_spikes_per_unit": 20,
            "n_shuffle": 20,
            "thresh_percentile": 99.0,
            "seed": 0,
        }

    @classmethod
    def params_schema(cls) -> dict[str, ParamSpec]:
        d = cls.default_params()
        return {
            "curation_output_root": ParamSpec("path", d["curation_output_root"],
                "Root of auto_curation outputs."),
            "output_root": ParamSpec("path", d["output_root"],
                "Directory where directed-connectivity results are written."),
            "bin_s": ParamSpec("float", d["bin_s"],
                "Binarising bin width (s) for transfer entropy.", min=0.0),
            "max_units": ParamSpec("int", d["max_units"],
                "Cap on the most-active-unit core (TE is O(n^2)).", min=2),
            "min_units": ParamSpec("int", d["min_units"],
                "Minimum eligible units for a directed graph.", min=2),
            "min_spikes_per_unit": ParamSpec("int", d["min_spikes_per_unit"],
                "Drop units below this spike count before TE.", min=1),
            "n_shuffle": ParamSpec("int", d["n_shuffle"],
                "Circular-shift shuffles for the TE significance null.", min=1),
            "thresh_percentile": ParamSpec("float", d["thresh_percentile"],
                "Percentile of the shuffled-TE null kept as significant edges.",
                min=0.0, max=100.0),
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
                / well_id / "directed")

    def run(self, recording_key: str, well_id: str, data_path: Path,
            params: dict[str, Any]) -> Path:
        import numpy as np

        from yuxin_mea.analysis.directed_connectivity import (
            DirectedConfig, DirectedError, compute_directed,
            empty_directed_results, write_directed)

        p = self.resolve_params(params)
        rec_name, actual_well_id = self.split_compound_well_id(well_id)
        curation_dir = self.build_curation_output_path(
            p["curation_output_root"], recording_key, rec_name, actual_well_id)
        spikes_path = curation_dir / "curated_spike_times.npy"
        if not spikes_path.exists():
            raise FileNotFoundError(f"curated_spike_times.npy not found at {spikes_path}.")
        spike_times = np.load(spikes_path, allow_pickle=True).item()

        config = DirectedConfig.from_task_params(p)
        try:
            results = compute_directed(spike_times, config=config)
        except DirectedError as exc:
            results = empty_directed_results(reason=str(exc))

        out = self.build_output_path(
            p["output_root"], recording_key, rec_name, actual_well_id)
        write_directed(results, out)
        return out
