from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pipeline_manager import BaseAnalysisTask


class PreprocessingTask(BaseAnalysisTask):
    """SpikeInterface preprocessing for one Maxwell recording/well stream."""

    task_name = "preprocessing"
    dependencies: list[str] = []

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "output_root": "./preprocessed_data",
            "bandpass_freq_min": 300,
            "bandpass_freq_max": 3000,
            "reference": "local",
            "operator": "median",
            "local_radius": [0, 250],
            "dtype": "float32",
            "n_jobs": max(1, (os.cpu_count() or 3) - 2),
            "chunk_duration": "1s",
            "progress_bar": True,
            "overwrite": True,
        }

    @staticmethod
    def split_compound_well_id(well_id: str) -> tuple[str, str]:
        """Return (rec_name, well_id) from a PipelineManager compound well key."""
        parts = str(well_id).split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "PreprocessingTask expects compound well_id formatted as "
                "'rec0000/well000'."
            )
        return parts[0], parts[1]

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
            / "preprocessed.zarr"
        )

    @staticmethod
    def _apply_common_reference(rec: Any, spre: Any, params: dict[str, Any]) -> Any:
        reference = params["reference"]
        operator = params["operator"]

        if reference == "global":
            try:
                return spre.common_reference(
                    rec,
                    reference="global",
                    operator=operator,
                )
            except Exception as exc:
                raise RuntimeError(f"Global common reference failed: {exc}") from exc

        if reference == "local":
            try:
                return spre.common_reference(
                    rec,
                    reference="local",
                    operator=operator,
                    local_radius=tuple(params["local_radius"]),
                )
            except Exception as local_exc:
                try:
                    referenced = spre.common_reference(
                        rec,
                        reference="global",
                        operator=operator,
                    )
                except Exception as global_exc:
                    raise RuntimeError(
                        "Common reference failed: "
                        f"local reference error: {local_exc}; "
                        f"global fallback error: {global_exc}"
                    ) from global_exc

                # TODO(logger): warn that local CMR failed and global CMR fallback succeeded.
                return referenced

        raise ValueError(
            f"Invalid reference type: {reference}. Expected 'local' or 'global'."
        )

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        import spikeinterface.full as si
        import spikeinterface.preprocessing as spre

        p = self.resolve_params(params)
        rec_name, actual_well_id = self.split_compound_well_id(well_id)
        zarr_path = self.build_output_path(
            p["output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )

        rec = si.read_maxwell(
            str(data_path),
            stream_id=actual_well_id,
            rec_name=rec_name,
        )

        if rec.get_dtype().kind == "u":
            rec = spre.unsigned_to_signed(rec)
        # rec.annotate(is_signed=True)

        rec = spre.bandpass_filter(
            rec,
            freq_min=p["bandpass_freq_min"],
            freq_max=p["bandpass_freq_max"],
        )
        # rec.annotate(is_filtered=True)

        rec = self._apply_common_reference(rec, spre, p)
        # rec.annotate(is_common_referenced=True)


        # F. Force Float32 (Prevents Signal Crushing)
        # Int16 scaling caused Kilosort crash on artifacts. Float32 is safer.
        # -- Mandar M.P. Jan, 2026 --
        dtype = p.get("dtype")
        if dtype and rec.get_dtype() != dtype:
            rec = spre.astype(rec, dtype)

        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        rec.save(
            folder=zarr_path,
            format="zarr",
            overwrite=bool(p["overwrite"]),
            n_jobs=p["n_jobs"],
            chunk_duration=p["chunk_duration"],
            progress_bar=bool(p["progress_bar"]),
        )
        return zarr_path
