from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseAnalysisTask(ABC):
    """ABC for user-defined analysis tasks in the MEA pipeline.

    Subclasses must declare two class-level attributes:

        task_name:    str        — unique name matching PipelineManager registration
        dependencies: list[str]  — immediate upstream task names

    Example (with class-level defaults)::

        class PreprocessingTask(BaseAnalysisTask):
            task_name    = "SI-Preprocessing"
            dependencies = []

            @classmethod
            def default_params(cls):
                return {"bandpass_freq_min": 300, "bandpass_freq_max": 3000}

            def run(self, recording_key, well_id, data_path, params):
                p = self.resolve_params(params)   # JSON values override class defaults
                ...

    Example (no hard-coded defaults — all values come from the config file)::

        class SortingTask(BaseAnalysisTask):
            task_name    = "sorting"
            dependencies = ["SI-Preprocessing"]

            def run(self, recording_key, well_id, data_path, params):
                sorter = params["sorter"]   # must be present in pipeline_config.json
                ...
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only validate when the subclass provides a concrete run() implementation.
        # Abstract intermediaries (no run() in their own dict) are left unchecked
        # because __abstractmethods__ isn't populated yet at __init_subclass__ time.
        if "run" not in cls.__dict__:
            return
        if not isinstance(getattr(cls, "task_name", None), str):
            raise TypeError(
                f"{cls.__name__} must define 'task_name' as a class-level str."
            )
        if not isinstance(getattr(cls, "dependencies", None), list):
            raise TypeError(
                f"{cls.__name__} must define 'dependencies' as a class-level list."
            )

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """Optional class-level defaults. Override to provide fallback parameter values.

        These become the base layer in resolve_params(): any key present in the
        config-file dict will overwrite the corresponding default here.

        For SpikeInterface sorters::

            @classmethod
            def default_params(cls):
                return si.get_default_sorter_params("kilosort4")
        """
        return {}

    def resolve_params(self, config_params: dict[str, Any]) -> dict[str, Any]:
        """Merge class-level defaults with config_params. Config-file values win.

        Call at the top of run() to obtain the effective parameter dict::

            def run(self, recording_key, well_id, data_path, params):
                p = self.resolve_params(params)
                freq_min = p["bandpass_freq_min"]
        """
        return {**self.default_params(), **config_params}

    @abstractmethod
    def run(
        self,
        recording_key: str,
        well_id:       str,
        data_path:     Path,
        params:        dict[str, Any],
    ) -> Path:
        """Execute the analysis. Returns the output directory or file path.

        params is the dict provided by ConfigManager (values from the config file).
        Call self.resolve_params(params) to merge with any class-level defaults.
        """
