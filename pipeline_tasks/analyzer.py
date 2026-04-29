from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pipeline_manager import BaseAnalysisTask
from pipeline_tasks.auto_merge import AutoMergeTask
from pipeline_tasks.preprocessing import PreprocessingTask


class AnalyzerTask(BaseAnalysisTask):
    """Creates a SpikeInterface SortingAnalyzer and computes standard extensions.

    Reads the preprocessed recording (zarr) from PreprocessingTask and the
    (optionally merged) sorting from AutoMergeTask.  Computes the full set
    of extensions needed for curation and downstream analysis.
    """

    task_name = "analyzer"
    dependencies = ["auto_merge"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "output_root": "./analyzer_data",
            "preprocessing_output_root": "./preprocessed_data",
            "auto_merge_output_root": "./auto_merge_data",
            "radius_um": 50,
            "ms_before": 1.0,
            "ms_after": 2.0,
            "unit_locations_method": "monopolar_triangulation",
            "n_jobs": 1,
        }

    @staticmethod
    def split_compound_well_id(well_id: str) -> tuple[str, str]:
        return PreprocessingTask.split_compound_well_id(well_id)

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
            / "analyzer"
        )

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        import spikeinterface.full as si

        p = self.resolve_params(params)
        rec_name, actual_well_id = self.split_compound_well_id(well_id)

        preprocessed_path = PreprocessingTask.build_output_path(
            p["preprocessing_output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        sorting_path = AutoMergeTask.build_output_path(
            p["auto_merge_output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        output_path = self.build_output_path(
            p["output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )

        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        recording = si.load(preprocessed_path)
        sorting = si.load(sorting_path)

        sparsity = si.estimate_sparsity(
            sorting,
            recording,
            method="radius",
            radius_um=int(p["radius_um"]),
            peak_sign="neg",
        )

        analyzer = si.create_sorting_analyzer(
            sorting,
            recording,
            format="binary_folder",
            folder=output_path,
            sparsity=sparsity,
            return_in_uV=True,
        )

        ext_list = [
            "random_spikes",
            "spike_amplitudes",
            "waveforms",
            "templates",
            "noise_levels",
            "quality_metrics",
            "template_metrics",
            "unit_locations",
        ]
        ext_params = {
            "waveforms": {
                "ms_before": float(p["ms_before"]),
                "ms_after": float(p["ms_after"]),
            },
            "unit_locations": {"method": str(p["unit_locations_method"])},
        }

        analyzer.compute(ext_list, extension_params=ext_params, n_jobs=int(p["n_jobs"]))
        return output_path
