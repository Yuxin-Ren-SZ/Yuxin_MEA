"""Abstract base class for tasks that aggregate plate-level data per recording.

A `BasePlateLevelTask` runs once per recording (sentinel `well_id="__plate__"`)
and processes all 24 wells together. It depends on at least the
``burst_detection`` and ``auto_curation`` outputs being present per well
(those are what :func:`yuxin_mea.analysis.plate_raster_synchrony.load_plate_data`
reads).

Subclasses implement two hooks:

* ``aggregate_records(well_records, params) -> Any`` — produces the
  per-recording artifact (a figure, a CSV row, a JSON summary…).
* ``write_output(result, recording_key, params) -> Path`` — persists the
  artifact and returns the on-disk location.

The plate viewer used to be the only consumer; Phase 5 moved that
visualization to the dashboard, so this class is currently scaffolding
for future plate-level tasks (e.g., per-recording QC export). Keeping it
in ``pipeline/`` rather than ``tasks/`` signals "generic infrastructure,
not a specific task."
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

from yuxin_mea.pipeline.base_task import BaseAnalysisTask


class BasePlateLevelTask(BaseAnalysisTask):
    """ABC for tasks that aggregate all 24 wells of a recording into one artifact.

    Subclasses must declare ``task_name``, ``dependencies``, and
    ``default_params()`` (and ``params_schema()`` if they want to be
    editable from the dashboard). Then implement the two abstract hooks
    below and call ``self._run_template(...)`` from ``run()``::

        def run(self, recording_key, well_id, data_path, params):
            return self._run_template(recording_key, well_id, data_path, params)

    NOTE: ``_run_template`` is not ``run()`` itself because
    ``BaseAnalysisTask.__init_subclass__`` validates ``task_name`` /
    ``dependencies`` on any class that defines ``run`` in its own
    ``__dict__``. ``BasePlateLevelTask`` is an intermediate abstract
    class and must not trigger that check.
    """

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def aggregate_records(
        self,
        well_records: list,
        params: dict[str, Any],
    ) -> Any:
        """Produce a per-recording artifact from the 24 ``WellRecord`` objects.

        Args:
            well_records: 24 ``WellRecord`` items (status "ok" or "missing"),
                ordered by well number (``well000`` … ``well023``).
            params: fully resolved parameter dict (defaults merged with
                config).

        Returns:
            Any artifact (figure, dict, DataFrame, …) that ``write_output``
            then persists.
        """

    @abstractmethod
    def write_output(
        self,
        result: Any,
        recording_key: str,
        params: dict[str, Any],
    ) -> Path:
        """Persist the aggregated artifact and return its on-disk path.

        Args:
            result: return value of ``aggregate_records``.
            recording_key: e.g. ``"CX138/260329/T003346/Network/000029"``.
            params: fully resolved parameter dict.

        Returns:
            Path to the written output file.
        """

    # ------------------------------------------------------------------
    # Template orchestrator
    # ------------------------------------------------------------------

    def _run_template(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        """Orchestrate: resolve params → load 24 wells → aggregate → write."""
        # Lazy import: ``load_plate_data`` pulls in plotly transitively
        # via the analysis module. Subclasses that don't need it
        # (e.g. a future CSV-only summary task) can skip the helper
        # entirely by overriding ``_run_template``.
        from yuxin_mea.analysis.plate_raster_synchrony import load_plate_data

        p = self.resolve_params(params)
        well_records = load_plate_data(
            burst_detection_root=Path(p["burst_detection_root"]),
            curation_output_root=Path(p["curation_output_root"]),
            recording_key=recording_key,
            rec_name=str(p.get("rec_name", "auto")),
            experiment_cache_path=(
                Path(p["experiment_cache_path"])
                if p.get("experiment_cache_path")
                else None
            ),
        )
        result = self.aggregate_records(well_records, p)
        return self.write_output(result, recording_key, p)
