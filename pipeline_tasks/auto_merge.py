from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pipeline_manager import BaseAnalysisTask
from pipeline_tasks.preprocessing import PreprocessingTask
from pipeline_tasks.sorting import SortingTask


class AutoMergeTask(BaseAnalysisTask):
    """Optional SI auto-merge step between sorting and analyzer.

    When enabled=False (default), loads the sorting output and saves it
    to the canonical output path so downstream tasks have a single stable
    path to read from regardless of whether merging was performed.

    When enabled=True, constructs a temporary SortingAnalyzer over the
    four lightweight extensions (random_spikes, templates,
    template_similarity, correlograms), runs sic.auto_merge_units(), then
    saves the merged sorting and tears down the temporary analyzer.
    """

    task_name = "auto_merge"
    dependencies = ["sorting"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "output_root": "./auto_merge_data",
            "sorting_output_root": "./spikesorted_data",
            "preprocessing_output_root": "./preprocessed_data",
            "enabled": False,
            "presets": ["firing_rate_similarity", "template_similarity"],
            "radius_um": 50,
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
            / "auto_merge"
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

        sorting_path, _ = SortingTask.build_output_paths(
            p["sorting_output_root"],
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
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sorting = si.load(sorting_path)

        if not p["enabled"]:
            sorting.save(folder=output_path, overwrite=True)
            return output_path

        import spikeinterface.curation as sic

        preprocessed_path = PreprocessingTask.build_output_path(
            p["preprocessing_output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        recording = si.load(preprocessed_path)
        n_jobs = int(p["n_jobs"])

        tmp_analyzer_folder = output_path.parent / "_tmp_merge_analyzer"
        try:
            if tmp_analyzer_folder.exists():
                shutil.rmtree(tmp_analyzer_folder)

            sparsity = si.estimate_sparsity(
                sorting,
                recording,
                method="radius",
                radius_um=int(p["radius_um"]),
                peak_sign="neg",
            )
            merge_analyzer = si.create_sorting_analyzer(
                sorting,
                recording,
                format="binary_folder",
                folder=tmp_analyzer_folder,
                sparsity=sparsity,
                return_in_uV=True,
            )
            merge_analyzer.compute(
                ["random_spikes", "templates", "template_similarity", "correlograms"],
                n_jobs=n_jobs,
            )

            try:
                merged = sic.auto_merge_units(
                    merge_analyzer,
                    presets=list(p["presets"]),
                    recursive=True,
                    n_jobs=n_jobs,
                )
            except TypeError:
                merged = sic.auto_merge_units(
                    merge_analyzer,
                    presets=list(p["presets"]),
                    recursive=True,
                )

            if isinstance(merged, tuple):
                merged = merged[0]
            merged_sorting = merged.sorting if hasattr(merged, "sorting") else merged
            merged_sorting.save(folder=output_path, overwrite=True)
        finally:
            if tmp_analyzer_folder.exists():
                shutil.rmtree(tmp_analyzer_folder)

        return output_path
