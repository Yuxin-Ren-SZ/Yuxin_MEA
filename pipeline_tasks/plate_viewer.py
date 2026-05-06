"""Plate-level visualization task for burst detection results.

Reads burst detection + spike data for all 24 wells and assembles a plate viewer HTML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline_tasks.base_plate_viewer import BasePlateViewer, _load_viewer_components


class PlateViewerTask(BasePlateViewer):
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
            "rec_name": "auto",
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
        return self._run_template(recording_key, well_id, data_path, params)

    def build_figure(self, well_records: list, params: dict[str, Any]) -> Any:
        """Build a Plotly 4×6 raster + synchrony figure with burst event zones."""
        PlateViewerConfig, _, build_plate_figure, _ = _load_viewer_components()
        config = PlateViewerConfig(
            display_mode=str(params["display_mode"]),
            marker_size=float(params["marker_size"]),
            line_width=float(params["line_width"]),
            width_px=int(params["width_px"]),
            max_raster_points_per_well=int(params["max_raster_points_per_well"]),
            max_synchrony_points=int(params["max_synchrony_points"]),
        )
        return build_plate_figure(well_records, config)

    def write_output(self, fig: Any, recording_key: str, params: dict[str, Any]) -> Path:
        """Write figure as HTML with burst-zone controls to figures_root/recording_key/plate_viewer.html."""
        _, _, _, write_plate_viewer_html = _load_viewer_components()
        output_path = Path(params["figures_root"]) / recording_key / "plate_viewer.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_plate_viewer_html(fig, output_path)
        return output_path
