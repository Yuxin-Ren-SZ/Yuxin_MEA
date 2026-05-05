"""Plate-level visualization task for burst detection results.

Reads burst detection + spike data for all 24 wells and assembles a plate viewer HTML.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_manager.base_task import BaseAnalysisTask
from pipeline_tasks.analysis.plate_raster_synchrony import (
    PlateViewerConfig,
    WellRecord,
    build_plate_figure,
)


class PlateViewerTask(BaseAnalysisTask):
    """Plate-level task: assembles a 24-well raster + synchrony viewer.

    Runs once per recording. Reads burst_detection_data for all 24 wells and
    generates a single HTML figure.

    This task is registered with well_id = "__plate__" (sentinel) and ignores
    the well_id parameter in run().
    """

    task_name = "plate_viewer"
    dependencies = ["burst_detection"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "burst_detection_root": "./burst_detection_data",
            "curation_output_root": "./curation_data",
            "figures_root": "./figures",
            "experiment_cache_path": "./data/analysis/experiment_cache.json",
            "display_mode": "both",
            "marker_size": 5.0,
            "line_width": 1.25,
            "width_px": 2400,
            "max_raster_points_per_well": 12000,
            "max_synchrony_points": 3000,
        }

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        """Generate plate viewer HTML for a recording.

        Args:
            recording_key: e.g. "CX138/260329/T003346/Network/000029"
            well_id: ignored (plate-level task, always "__plate__")
            data_path: ignored
            params: config dict with burst_detection_root, figures_root, etc.

        Returns:
            Path to the generated HTML file.
        """
        p = self.resolve_params(params)

        burst_root = Path(p["burst_detection_root"])
        curation_root = Path(p["curation_output_root"])
        figures_root = Path(p["figures_root"])
        cache_path = Path(p["experiment_cache_path"])

        # Create output directory
        output_dir = figures_root / recording_key
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load experiment cache for well metadata
        well_metadata = self._load_well_metadata(cache_path, recording_key)

        # Assemble well records for all 24 wells
        well_records = []
        for well_num in range(24):
            well_id_str = f"well{well_num:03d}"
            well_record = self._load_well_record(
                well_id_str,
                recording_key,
                burst_root,
                curation_root,
                well_metadata,
            )
            well_records.append(well_record)

        # Build figure
        config = PlateViewerConfig(
            display_mode=str(p["display_mode"]),
            marker_size=float(p["marker_size"]),
            line_width=float(p["line_width"]),
            width_px=int(p["width_px"]),
            max_raster_points_per_well=int(p["max_raster_points_per_well"]),
            max_synchrony_points=int(p["max_synchrony_points"]),
        )

        fig = build_plate_figure(well_records, config)

        # Write HTML
        output_path = output_dir / "plate_viewer.html"
        fig.write_html(str(output_path))

        return output_path

    def _load_well_metadata(
        self, cache_path: Path, recording_key: str
    ) -> dict[str, dict[str, Any]]:
        """Load well metadata from experiment_cache.json.

        Returns dict: well_id_str -> {well_name, groupname, ...}
        """
        metadata = {}
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
        except Exception as e:
            print(f"Warning: Failed to load experiment cache: {e}")

        return metadata

    def _load_well_record(
        self,
        well_id_str: str,
        recording_key: str,
        burst_root: Path,
        curation_root: Path,
        well_metadata: dict[str, dict[str, Any]],
    ) -> WellRecord:
        """Load spike times and plot signals for one well.

        Returns a WellRecord with status = "ok" or an error message.
        """
        meta = well_metadata.get(well_id_str, {})
        well_name = meta.get("well_name", "?")
        groupname = meta.get("groupname", "?")

        # Try to load plot_signals
        plot_signals_path = burst_root / recording_key / well_id_str / "burst_detection" / "plot_signals.npy"
        plot_signals = None
        if plot_signals_path.exists():
            try:
                plot_signals = np.load(plot_signals_path, allow_pickle=True).item()
            except Exception as e:
                return WellRecord(
                    well_id=well_id_str,
                    well_name=well_name,
                    groupname=groupname,
                    status=f"plot_signals error",
                )

        # Try to load spike times
        spike_times_path = curation_root / recording_key / well_id_str / "auto_curation" / "curated_spike_times.npy"
        spike_times = None
        if spike_times_path.exists():
            try:
                spike_times = np.load(spike_times_path, allow_pickle=True).item()
            except Exception as e:
                return WellRecord(
                    well_id=well_id_str,
                    well_name=well_name,
                    groupname=groupname,
                    status=f"spike_times error",
                )

        if plot_signals is None and spike_times is None:
            return WellRecord(
                well_id=well_id_str,
                well_name=well_name,
                groupname=groupname,
                status="missing",
            )

        return WellRecord(
            well_id=well_id_str,
            well_name=well_name,
            groupname=groupname,
            plot_signals=plot_signals,
            spike_times=spike_times,
            status="ok",
        )
