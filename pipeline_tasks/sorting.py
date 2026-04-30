from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline_manager import BaseAnalysisTask
from pipeline_tasks.preprocessing import PreprocessingTask


class SortingTask(BaseAnalysisTask):
    """SpikeInterface Kilosort4 sorting for one preprocessed Maxwell stream."""

    task_name = "sorting"
    dependencies = ["preprocessing"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "preprocessing_output_root": "./preprocessed_data",
            "output_root": "./spikesorted_data",
            "sorter": "kilosort4",
            "docker_image": None,
            "verbose": True,
            "remove_existing_folder": True,
            "delete_output_folder": False,
            "overwrite": True,
            "clean_excess_spikes": True,
            "remove_empty_units": True,
            "min_high_vram_gb": 14,
            "high_vram_sorter_kwargs": {
                "batch_size_seconds": 2.0,
                "clear_cache": True,
                "invert_sign": True,
                "cluster_downsampling": 20,
                "max_cluster_subset": None,
                "nblocks": 0,
                "dmin": 17,
                "do_correction": False,
            },
            "low_vram_sorter_kwargs": {
                "batch_size_seconds": 0.5,
                "clear_cache": True,
                "invert_sign": True,
                "cluster_downsampling": 30,
                "max_cluster_subset": 50000,
                "nblocks": 0,
                "do_correction": False,
            },
            "sorter_kwargs": {},
        }

    @staticmethod
    def split_compound_well_id(well_id: str) -> tuple[str, str]:
        return PreprocessingTask.split_compound_well_id(well_id)

    @staticmethod
    def build_output_paths(
        output_root: str | Path,
        recording_key: str,
        rec_name: str,
        well_id: str,
    ) -> tuple[Path, Path]:
        base = Path(output_root) / Path(recording_key) / rec_name / well_id
        output = base / "sorter_output"
        return output, output

    @staticmethod
    def _suppress_kilosort_console() -> None:
        import logging
        ks_log = logging.getLogger("kilosort")
        if not ks_log.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.WARNING)
            ks_log.addHandler(handler)

    @staticmethod
    def _detect_total_vram_gb(torch_module: Any) -> float:
        try:
            if not torch_module.cuda.is_available():
                return 0.0
            props = torch_module.cuda.get_device_properties(0)
            return float(props.total_memory) / (1024**3)
        except Exception:
            return 0.0

    @staticmethod
    def _build_kilosort_params(
        recording: Any,
        total_vram_gb: float,
        min_high_vram_gb: float,
        high_vram_sorter_kwargs: dict[str, Any],
        low_vram_sorter_kwargs: dict[str, Any],
        sorter_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        params = {
            **(
                high_vram_sorter_kwargs
                if total_vram_gb >= min_high_vram_gb
                else low_vram_sorter_kwargs
            ),
            **sorter_kwargs,
        }

        batch_size_seconds = params.pop("batch_size_seconds", None)
        if "batch_size" not in params and batch_size_seconds is not None:
            sampling_frequency = float(recording.get_sampling_frequency())
            params["batch_size"] = int(sampling_frequency * float(batch_size_seconds))

        return params

    def _resolve_sorting_params(self, params: dict[str, Any]) -> dict[str, Any]:
        defaults = self.default_params()
        resolved = self.resolve_params(params)

        # BaseAnalysisTask.resolve_params() is intentionally shallow. Preserve
        # nested preset defaults when config files override only selected knobs.
        for key in ("high_vram_sorter_kwargs", "low_vram_sorter_kwargs"):
            resolved[key] = {
                **dict(defaults[key]),
                **dict(params.get(key, {})),
            }
        return resolved

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        import spikeinterface.full as si
        import torch

        p = self._resolve_sorting_params(params)
        rec_name, actual_well_id = self.split_compound_well_id(well_id)

        # Get dependencies output path
        preprocessed_path = PreprocessingTask.build_output_path(
            p["preprocessing_output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        sorter_output, cleaned_output = self.build_output_paths(
            p["output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        sorter_output.parent.mkdir(parents=True, exist_ok=True)

        recording = si.load(preprocessed_path)
        # if not isinstance(recording, si.BaseRecording):
        #     raise ValueError(f"Expected a BaseRecording, got {type(recording)}")
        
        total_vram_gb = self._detect_total_vram_gb(torch)
        sorter_params = self._build_kilosort_params(
            recording,
            total_vram_gb,
            float(p["min_high_vram_gb"]),
            dict(p["high_vram_sorter_kwargs"]),
            dict(p["low_vram_sorter_kwargs"]),
            dict(p["sorter_kwargs"]),
        )

        if total_vram_gb > 0:
            torch.cuda.empty_cache()

        self._suppress_kilosort_console()
        sorting = si.run_sorter(
            sorter_name=p["sorter"],
            recording=recording,
            folder=str(sorter_output),
            delete_output_folder=bool(p["delete_output_folder"]),
            remove_existing_folder=bool(p["remove_existing_folder"]),
            verbose=bool(p["verbose"]),
            docker_image=p["docker_image"],
            **sorter_params,
        )

        # if not isinstance(sorting, si.BaseSorting):
        #     raise ValueError(f"Expected a BaseSorting, got {type(sorting)}")

        if bool(p["clean_excess_spikes"]):
            sorting = si.remove_excess_spikes(sorting, recording)
        if bool(p["remove_empty_units"]):
            sorting = sorting.remove_empty_units()

        sorting.save(folder=cleaned_output, overwrite=bool(p["overwrite"]))
        return cleaned_output
