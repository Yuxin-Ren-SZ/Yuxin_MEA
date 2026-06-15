from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask
from yuxin_mea.tasks.preprocessing import PreprocessingTask


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

    @classmethod
    def params_schema(cls) -> dict[str, ParamSpec]:
        defaults = cls.default_params()
        high_vram_defaults = defaults["high_vram_sorter_kwargs"]
        low_vram_defaults = defaults["low_vram_sorter_kwargs"]
        return {
            "preprocessing_output_root": ParamSpec(
                "path", defaults["preprocessing_output_root"],
                "Root directory where the upstream PreprocessingTask wrote its "
                "Zarr stores; used to locate the input recording.",
            ),
            "output_root": ParamSpec(
                "path", defaults["output_root"],
                "Directory where sorter output folders are written "
                "(per recording/well).",
            ),
            "sorter": ParamSpec(
                "str", defaults["sorter"],
                "SpikeInterface sorter name passed to si.run_sorter().",
                choices=[
                    "kilosort4",
                    "kilosort2",
                    "kilosort3",
                    "tridesclous2",
                    "mountainsort5",
                    "spykingcircus2",
                ],
            ),
            "docker_image": ParamSpec(
                "str", defaults["docker_image"],
                "Docker image to run the sorter in. Leave blank to run "
                "natively (None is passed to si.run_sorter()).",
                nullable=True,
            ),
            "verbose": ParamSpec(
                "bool", defaults["verbose"],
                "Verbose logging from the sorter.",
            ),
            "remove_existing_folder": ParamSpec(
                "bool", defaults["remove_existing_folder"],
                "If True, delete a pre-existing sorter output folder before "
                "running.",
            ),
            "delete_output_folder": ParamSpec(
                "bool", defaults["delete_output_folder"],
                "If True, delete the sorter output folder after sorting "
                "finishes (forwarded to si.run_sorter()).",
            ),
            "overwrite": ParamSpec(
                "bool", defaults["overwrite"],
                "Overwrite the cleaned sorting output folder when saving.",
            ),
            "clean_excess_spikes": ParamSpec(
                "bool", defaults["clean_excess_spikes"],
                "Run si.remove_excess_spikes() to drop spikes past the "
                "recording end.",
            ),
            "remove_empty_units": ParamSpec(
                "bool", defaults["remove_empty_units"],
                "Drop units with zero spikes after sorting.",
            ),
            "min_high_vram_gb": ParamSpec(
                "float", defaults["min_high_vram_gb"],
                "VRAM threshold (GB). At or above this, high_vram_sorter_kwargs "
                "are used; otherwise low_vram_sorter_kwargs.",
                min=0,
            ),
            "high_vram_sorter_kwargs": ParamSpec(
                "dict", defaults["high_vram_sorter_kwargs"],
                "Sorter kwargs preset for GPUs with >= min_high_vram_gb.",
                nested_schema={
                    "batch_size_seconds": ParamSpec(
                        "float", high_vram_defaults["batch_size_seconds"],
                        "Batch length in seconds; converted to samples via "
                        "the recording sampling rate.",
                        min=0,
                    ),
                    "clear_cache": ParamSpec(
                        "bool", high_vram_defaults["clear_cache"],
                        "Clear sorter internal caches between batches.",
                    ),
                    "invert_sign": ParamSpec(
                        "bool", high_vram_defaults["invert_sign"],
                        "Invert recording polarity before sorting.",
                    ),
                    "cluster_downsampling": ParamSpec(
                        "int", high_vram_defaults["cluster_downsampling"],
                        "Downsampling factor used during clustering.",
                        min=1,
                    ),
                    "max_cluster_subset": ParamSpec(
                        "int", high_vram_defaults["max_cluster_subset"],
                        "Max spikes per cluster used in clustering. Leave "
                        "blank for no limit (None).",
                        min=0, nullable=True,
                    ),
                    "nblocks": ParamSpec(
                        "int", high_vram_defaults["nblocks"],
                        "Number of drift-correction blocks; 0 disables block "
                        "drift correction.",
                        min=0,
                    ),
                    "dmin": ParamSpec(
                        "int", high_vram_defaults["dmin"],
                        "Minimum vertical spacing (µm) between detected "
                        "templates.",
                        min=0,
                    ),
                    "do_correction": ParamSpec(
                        "bool", high_vram_defaults["do_correction"],
                        "Enable Kilosort drift correction.",
                    ),
                },
            ),
            "low_vram_sorter_kwargs": ParamSpec(
                "dict", defaults["low_vram_sorter_kwargs"],
                "Sorter kwargs preset for GPUs with < min_high_vram_gb.",
                nested_schema={
                    "batch_size_seconds": ParamSpec(
                        "float", low_vram_defaults["batch_size_seconds"],
                        "Batch length in seconds; converted to samples via "
                        "the recording sampling rate.",
                        min=0,
                    ),
                    "clear_cache": ParamSpec(
                        "bool", low_vram_defaults["clear_cache"],
                        "Clear sorter internal caches between batches.",
                    ),
                    "invert_sign": ParamSpec(
                        "bool", low_vram_defaults["invert_sign"],
                        "Invert recording polarity before sorting.",
                    ),
                    "cluster_downsampling": ParamSpec(
                        "int", low_vram_defaults["cluster_downsampling"],
                        "Downsampling factor used during clustering.",
                        min=1,
                    ),
                    "max_cluster_subset": ParamSpec(
                        "int", low_vram_defaults["max_cluster_subset"],
                        "Max spikes per cluster used in clustering. Leave "
                        "blank for no limit (None).",
                        min=0, nullable=True,
                    ),
                    "nblocks": ParamSpec(
                        "int", low_vram_defaults["nblocks"],
                        "Number of drift-correction blocks; 0 disables block "
                        "drift correction.",
                        min=0,
                    ),
                    "do_correction": ParamSpec(
                        "bool", low_vram_defaults["do_correction"],
                        "Enable Kilosort drift correction.",
                    ),
                },
            ),
            "sorter_kwargs": ParamSpec(
                "dict", defaults["sorter_kwargs"],
                "Free-form raw overrides merged on top of the chosen VRAM "
                "preset before being passed to si.run_sorter().",
                nested_schema=None,
            ),
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
        return base / "sorter_output", base / "sorting"

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
        if not isinstance(recording, si.BaseRecording):
            raise ValueError(
                f"Expected a BaseRecording subclass at {preprocessed_path}, "
                f"got {type(recording).__name__}"
            )
        
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

        ks_filter = None
        if not bool(p["verbose"]):
            ks_filter = logging.Filter()
            ks_filter.filter = lambda _: False
            logging.getLogger("kilosort").addFilter(ks_filter)

        try:
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
        finally:
            if ks_filter is not None:
                logging.getLogger("kilosort").removeFilter(ks_filter)
                for h in logging.getLogger("kilosort").handlers[:]:
                    logging.getLogger("kilosort").removeHandler(h)

        if not isinstance(sorting, si.BaseSorting):
            raise ValueError(
                f"Expected a BaseSorting subclass from run_sorter, "
                f"got {type(sorting).__name__}"
            )

        if bool(p["clean_excess_spikes"]):
            sorting = si.remove_excess_spikes(sorting, recording)
        if bool(p["remove_empty_units"]):
            sorting = sorting.remove_empty_units()

        sorting.save(folder=cleaned_output, overwrite=bool(p["overwrite"]))
        return cleaned_output
