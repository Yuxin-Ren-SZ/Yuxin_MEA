"""Plate-level visualization task for burst detection results.

Reads burst detection + spike data for all 24 wells and assembles a plate viewer HTML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yuxin_mea.config import ParamSpec
from yuxin_mea.tasks.base_plate_viewer import BasePlateViewer, _load_viewer_components


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

    @classmethod
    def params_schema(cls) -> dict[str, ParamSpec]:
        defaults = cls.default_params()
        return {
            "burst_detection_root": ParamSpec(
                "path", defaults["burst_detection_root"],
                "Directory containing burst_detection outputs read per well.",
            ),
            "curation_output_root": ParamSpec(
                "path", defaults["curation_output_root"],
                "Directory containing curation outputs (spike times per well).",
            ),
            "figures_root": ParamSpec(
                "path", defaults["figures_root"],
                "Directory where the plate_viewer.html figure is written.",
            ),
            "experiment_cache_path": ParamSpec(
                "path", defaults["experiment_cache_path"],
                "JSON cache providing well group/name metadata for the plate.",
            ),
            "rec_name": ParamSpec(
                "str", defaults["rec_name"],
                "Maxwell rec_name to display. Empty string or 'auto' triggers "
                "auto-detection from the recording.",
            ),
            "display_mode": ParamSpec(
                "str", defaults["display_mode"],
                "What to render per well: raster only, synchrony only, or both.",
                choices=["raster", "synchrony", "both"],
            ),
            "marker_size": ParamSpec(
                "float", defaults["marker_size"],
                "Spike marker size (pixels) in the raster subplots.",
                min=0,
            ),
            "line_width": ParamSpec(
                "float", defaults["line_width"],
                "Line width (pixels) for synchrony traces.",
                min=0,
            ),
            "width_px": ParamSpec(
                "int", defaults["width_px"],
                "Total figure width in pixels (height is derived from this).",
                min=1,
            ),
            "max_raster_points_per_well": ParamSpec(
                "int", defaults["max_raster_points_per_well"],
                "Cap on rendered spike markers per well (uniform downsample).",
                min=1,
            ),
            "max_synchrony_points": ParamSpec(
                "int", defaults["max_synchrony_points"],
                "Cap on rendered synchrony samples per well (uniform downsample).",
                min=1,
            ),
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
