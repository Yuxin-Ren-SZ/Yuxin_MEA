"""Abstract base class for plate-level viewer tasks.

Subclasses implement two hooks:
  - build_figure(well_records, params) -> figure object
  - write_output(fig, recording_key, params) -> Path

All data-loading logic lives here as concrete helpers.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pipeline_manager.base_task import BaseAnalysisTask

if TYPE_CHECKING:
    from pipeline_tasks.analysis.plate_raster_synchrony import WellRecord


def _load_viewer_components():
    """Lazily import plotly-backed visualization components."""
    try:
        from pipeline_tasks.analysis.plate_raster_synchrony import (
            PlateViewerConfig,
            WellRecord,
            build_plate_figure,
            write_plate_viewer_html,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "plotly":
            raise RuntimeError(
                "BasePlateViewer requires plotly. Install the repo environment "
                "from environment.yml before running a plate viewer."
            ) from exc
        raise
    return PlateViewerConfig, WellRecord, build_plate_figure, write_plate_viewer_html


class BasePlateViewer(BaseAnalysisTask):
    """ABC for plate-level viewer tasks.

    Subclasses must declare task_name, dependencies, and default_params(),
    then implement build_figure() and write_output().

    Call self._run_template(...) from run() to execute the pipeline:
        def run(self, recording_key, well_id, data_path, params):
            return self._run_template(recording_key, well_id, data_path, params)

    NOTE: _run_template is not run() itself because BaseAnalysisTask.__init_subclass__
    validates task_name/dependencies on any class that defines run() in its own __dict__.
    BasePlateViewer is an intermediate abstract class and must not trigger that check.
    """

    # ------------------------------------------------------------------ #
    # Abstract hooks                                                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def build_figure(
        self,
        well_records: list[WellRecord],
        params: dict[str, Any],
    ) -> Any:
        """Build a figure from assembled well records.

        Args:
            well_records: 24 WellRecord objects (status "ok" or "missing").
            params: Fully resolved parameter dict (defaults merged with config).

        Returns:
            Any figure object (e.g., plotly go.Figure, matplotlib Figure).
        """

    @abstractmethod
    def write_output(
        self,
        fig: Any,
        recording_key: str,
        params: dict[str, Any],
    ) -> Path:
        """Persist the figure and return the output path.

        Args:
            fig: Return value of build_figure().
            recording_key: e.g. "CX138/260329/T003346/Network/000029"
            params: Fully resolved parameter dict.

        Returns:
            Path to the written output file.
        """

    # ------------------------------------------------------------------ #
    # Template orchestrator                                                #
    # ------------------------------------------------------------------ #

    def _run_template(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        """Orchestrate: resolve params → load wells → build → write."""
        p = self.resolve_params(params)
        well_records = self._assemble_well_records(recording_key, p)
        fig = self.build_figure(well_records, p)
        return self.write_output(fig, recording_key, p)

    # ------------------------------------------------------------------ #
    # Concrete data-loading helpers (shared by all subclasses)            #
    # ------------------------------------------------------------------ #

    def _assemble_well_records(
        self,
        recording_key: str,
        p: dict[str, Any],
    ) -> list:
        """Load WellRecord objects for all 24 wells."""
        burst_root = Path(p["burst_detection_root"])
        curation_root = Path(p["curation_output_root"])
        figures_root = Path(p["figures_root"])
        rec_name = str(p["rec_name"])
        _, WellRecord, _, _ = _load_viewer_components()

        cache_path = self._resolve_cache_path(
            Path(p["experiment_cache_path"]), figures_root
        )
        well_metadata, well_rec_names = self._load_recording_cache(
            cache_path, recording_key
        )
        discovered = self._discover_well_rec_names(
            recording_key, burst_root, curation_root
        )
        for well_id_str, discovered_rec_name in discovered.items():
            well_rec_names.setdefault(well_id_str, discovered_rec_name)

        well_records = []
        for well_num in range(24):
            well_id_str = f"well{well_num:03d}"
            well_records.append(
                self._load_well_record(
                    well_id_str,
                    recording_key,
                    rec_name,
                    burst_root,
                    curation_root,
                    well_metadata,
                    WellRecord,
                    well_rec_names,
                )
            )
        return well_records

    def _resolve_cache_path(self, cache_path: Path, figures_root: Path) -> Path:
        """Return the first existing experiment cache path from configured fallbacks."""
        candidates = [
            cache_path,
            figures_root.parent / "experiment_cache.json",
            figures_root.parent / "analysis" / "experiment_cache.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return cache_path

    def _load_well_metadata(
        self, cache_path: Path, recording_key: str
    ) -> dict[str, dict[str, Any]]:
        """Load well metadata from experiment_cache.json."""
        metadata, _ = self._load_recording_cache(cache_path, recording_key)
        return metadata

    def _load_recording_cache(
        self, cache_path: Path, recording_key: str
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        """Load well metadata and per-well rec name map from experiment cache."""
        metadata: dict[str, dict[str, Any]] = {}
        well_rec_names: dict[str, str] = {}
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            recording_data = cache.get(recording_key, {})
            wells_data = recording_data.get("wells", {})

            for well_id_str, well_info in wells_data.items():
                well_meta = well_info.get("metadata", {})
                metadata[well_id_str] = {
                    "well_name": well_meta.get("well_name", "?"),
                    "groupname": well_meta.get("groupname", "?"),
                }

            for rec_name, well_ids in recording_data.get("h5_recordings", {}).items():
                for wid in well_ids:
                    well_rec_names[str(wid)] = str(rec_name)
        except Exception as e:
            print(f"Warning: Failed to load experiment cache: {e}")

        return metadata, well_rec_names

    def _discover_well_rec_names(
        self,
        recording_key: str,
        burst_root: Path,
        curation_root: Path,
    ) -> dict[str, str]:
        """Discover well → rec name mapping from existing task output directories."""
        discovered: dict[str, str] = {}
        for root, terminal_dir in (
            (burst_root, "burst_detection"),
            (curation_root, "auto_curation"),
        ):
            recording_dir = root / recording_key
            if not recording_dir.exists():
                continue
            for rec_dir in sorted(recording_dir.glob("rec*")):
                if not rec_dir.is_dir():
                    continue
                for well_dir in sorted(rec_dir.glob("well*")):
                    if (well_dir / terminal_dir).exists():
                        discovered.setdefault(well_dir.name, rec_dir.name)
        return discovered

    def _rec_name_candidates(
        self,
        well_id_str: str,
        rec_name: str,
        well_rec_names: dict[str, str] | None,
        burst_root: Path,
        curation_root: Path,
        recording_key: str,
    ) -> list[str]:
        """Order rec-name candidates, using legacy hints only when their files exist."""
        candidates: list[str] = []
        rec_hint = rec_name if rec_name and rec_name.lower() != "auto" else None
        mapped_rec_name = (well_rec_names or {}).get(well_id_str)

        if rec_hint:
            hinted_burst = (
                burst_root / recording_key / rec_hint / well_id_str / "burst_detection"
            )
            hinted_curation = (
                curation_root / recording_key / rec_hint / well_id_str / "auto_curation"
            )
            if hinted_burst.exists() or hinted_curation.exists():
                candidates.append(rec_hint)

        if mapped_rec_name and mapped_rec_name not in candidates:
            candidates.append(mapped_rec_name)

        return candidates

    def _load_well_record(
        self,
        well_id_str: str,
        recording_key: str,
        rec_name: str,
        burst_root: Path,
        curation_root: Path,
        well_metadata: dict[str, dict[str, Any]],
        well_record_cls: Any,
        well_rec_names: dict[str, str] | None = None,
    ) -> Any:
        """Load spike times and plot signals for one well.

        Returns a WellRecord with status = "ok" or an error message.
        """
        meta = well_metadata.get(well_id_str, {})
        well_name = meta.get("well_name", "?")
        groupname = meta.get("groupname", "?")
        rec_names = self._rec_name_candidates(
            well_id_str,
            rec_name,
            well_rec_names,
            burst_root,
            curation_root,
            recording_key,
        )
        event_intervals = self._load_event_intervals(
            well_id_str,
            recording_key,
            rec_names,
            burst_root,
        )

        # Try to load plot_signals
        plot_signals = None
        for candidate_rec_name in rec_names:
            plot_signals_path = (
                burst_root
                / recording_key
                / candidate_rec_name
                / well_id_str
                / "burst_detection"
                / "plot_signals.npy"
            )
            if plot_signals_path.exists():
                try:
                    plot_signals = np.load(plot_signals_path, allow_pickle=True).item()
                except Exception:
                    return well_record_cls(
                        well_id=well_id_str,
                        well_name=well_name,
                        groupname=groupname,
                        status="plot_signals error",
                    )
                break

        # Try to load spike times
        spike_times = None
        for candidate_rec_name in rec_names:
            spike_times_path = (
                curation_root
                / recording_key
                / candidate_rec_name
                / well_id_str
                / "auto_curation"
                / "curated_spike_times.npy"
            )
            if spike_times_path.exists():
                try:
                    spike_times = np.load(spike_times_path, allow_pickle=True).item()
                except Exception:
                    return well_record_cls(
                        well_id=well_id_str,
                        well_name=well_name,
                        groupname=groupname,
                        status="spike_times error",
                    )
                break

        if plot_signals is None and spike_times is None:
            return self._make_well_record(
                well_record_cls,
                well_id=well_id_str,
                well_name=well_name,
                groupname=groupname,
                event_intervals=event_intervals,
                status="missing",
            )

        return self._make_well_record(
            well_record_cls,
            well_id=well_id_str,
            well_name=well_name,
            groupname=groupname,
            plot_signals=plot_signals,
            spike_times=spike_times,
            event_intervals=event_intervals,
            status="ok",
        )

    def _make_well_record(self, well_record_cls: Any, **kwargs: Any) -> Any:
        """Instantiate WellRecord, tolerating older test stubs without new fields."""
        try:
            return well_record_cls(**kwargs)
        except TypeError as exc:
            if "event_intervals" not in str(exc):
                raise
            kwargs.pop("event_intervals", None)
            return well_record_cls(**kwargs)

    def _load_event_intervals(
        self,
        well_id_str: str,
        recording_key: str,
        rec_names: list[str],
        burst_root: Path,
    ) -> dict[str, list[dict[str, Any]]]:
        """Load persisted event interval tables for one well."""
        event_intervals = {event_key: [] for event_key in self._event_type_keys()}
        for candidate_rec_name in rec_names:
            burst_dir = (
                burst_root
                / recording_key
                / candidate_rec_name
                / well_id_str
                / "burst_detection"
            )
            if not burst_dir.exists():
                continue
            for event_key in event_intervals:
                event_path = burst_dir / f"{event_key}.pkl"
                if event_path.exists():
                    event_intervals[event_key] = self._read_event_table(event_path)
            break
        return event_intervals

    def _event_type_keys(self) -> list[str]:
        """Return event table keys from the viewer registry."""
        try:
            from pipeline_tasks.analysis.plate_raster_synchrony import BURST_EVENT_TYPES
        except ModuleNotFoundError as exc:
            if exc.name == "plotly":
                return ["burstlets", "network_bursts", "superbursts"]
            raise
        return list(BURST_EVENT_TYPES.keys())

    def _read_event_table(self, event_path: Path) -> list[dict[str, Any]]:
        """Read one event table and keep valid numeric intervals."""
        try:
            table = pd.read_pickle(event_path)
        except Exception as exc:
            print(f"Warning: Failed to load event intervals from {event_path}: {exc}")
            return []
        if table is None or table.empty or "start" not in table or "end" not in table:
            return []

        intervals = []
        for row in table.to_dict(orient="records"):
            try:
                start = float(row["start"])
                end = float(row["end"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
                continue
            interval = {}
            for key, value in row.items():
                safe_value = self._json_safe_scalar(value)
                if safe_value is not None:
                    interval[key] = safe_value
            interval["start"] = start
            interval["end"] = end
            intervals.append(interval)
        return intervals

    def _json_safe_scalar(self, value: Any) -> Any:
        """Convert common numpy/pandas scalars to JSON-safe values."""
        if value is None:
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        return None
