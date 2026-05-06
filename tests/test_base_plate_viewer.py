from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# These imports will fail until we create base_plate_viewer.py
from pipeline_tasks.base_plate_viewer import BasePlateViewer


class _FakeViewer(BasePlateViewer):
    """Minimal concrete subclass for testing the ABC template."""

    task_name = "fake_viewer"
    dependencies = []

    def run(self, recording_key, well_id, data_path, params):
        return self._run_template(recording_key, well_id, data_path, params)

    def build_figure(self, well_records, params):
        self.captured_well_records = well_records
        self.captured_params = params
        return "FAKE_FIGURE"

    def write_output(self, fig, recording_key, params):
        self.captured_fig = fig
        self.captured_recording_key = recording_key
        return Path("/tmp/fake_output.html")


def test_run_template_assembles_24_well_records(tmp_path):
    """_run_template must produce exactly 24 WellRecords (all missing when no data)."""
    viewer = _FakeViewer()
    viewer._run_template(
        recording_key="test/recording",
        well_id="__plate__",
        data_path=tmp_path,
        params={
            "burst_detection_root": str(tmp_path),
            "curation_output_root": str(tmp_path),
            "figures_root": str(tmp_path),
            "experiment_cache_path": str(tmp_path / "missing_cache.json"),
            "rec_name": "auto",
        },
    )
    assert len(viewer.captured_well_records) == 24
    assert all(wr.status == "missing" for wr in viewer.captured_well_records)


def test_run_template_passes_resolved_params_to_hooks(tmp_path):
    """_run_template must call build_figure and write_output with resolved params."""
    viewer = _FakeViewer()
    viewer._run_template(
        recording_key="test/recording",
        well_id="__plate__",
        data_path=tmp_path,
        params={
            "burst_detection_root": str(tmp_path),
            "curation_output_root": str(tmp_path),
            "figures_root": str(tmp_path),
            "experiment_cache_path": str(tmp_path / "missing_cache.json"),
            "rec_name": "auto",
        },
    )
    assert "burst_detection_root" in viewer.captured_params


def test_run_template_threads_fig_from_build_to_write(tmp_path):
    """_run_template must pass build_figure return value to write_output."""
    viewer = _FakeViewer()
    viewer._run_template(
        recording_key="test/recording",
        well_id="__plate__",
        data_path=tmp_path,
        params={
            "burst_detection_root": str(tmp_path),
            "curation_output_root": str(tmp_path),
            "figures_root": str(tmp_path),
            "experiment_cache_path": str(tmp_path / "missing_cache.json"),
            "rec_name": "auto",
        },
    )
    assert viewer.captured_fig == "FAKE_FIGURE"


def test_run_template_returns_write_output_path(tmp_path):
    """_run_template must return exactly what write_output returns."""
    viewer = _FakeViewer()
    result = viewer._run_template(
        recording_key="test/recording",
        well_id="__plate__",
        data_path=tmp_path,
        params={
            "burst_detection_root": str(tmp_path),
            "curation_output_root": str(tmp_path),
            "figures_root": str(tmp_path),
            "experiment_cache_path": str(tmp_path / "missing_cache.json"),
            "rec_name": "auto",
        },
    )
    assert result == Path("/tmp/fake_output.html")


def test_base_plate_viewer_cannot_be_instantiated_without_hooks():
    """BasePlateViewer must be abstract — direct instantiation is forbidden."""
    with pytest.raises(TypeError):
        BasePlateViewer()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# PlateViewerTask concrete hook tests
# ---------------------------------------------------------------------------

import plotly.graph_objects as go
from pipeline_tasks.plate_viewer import PlateViewerTask
from pipeline_tasks.analysis.plate_raster_synchrony import WellRecord


def _missing_well_records() -> list[WellRecord]:
    return [
        WellRecord(
            well_id=f"well{i:03d}",
            well_name=f"W{i + 1}",
            groupname="test",
            status="missing",
        )
        for i in range(24)
    ]


def _default_vis_params() -> dict:
    return {
        "display_mode": "both",
        "marker_size": 5.0,
        "line_width": 1.25,
        "width_px": 2400,
        "max_raster_points_per_well": 12000,
        "max_synchrony_points": 3000,
    }


def test_plate_viewer_task_build_figure_returns_plotly_figure():
    task = PlateViewerTask()
    fig = task.build_figure(_missing_well_records(), _default_vis_params())
    assert isinstance(fig, go.Figure)


def test_plate_viewer_task_write_output_writes_html(tmp_path):
    """write_output calls write_plate_viewer_html and returns the correct path."""
    task = PlateViewerTask()
    mock_fig = MagicMock()
    mock_fig.to_html.return_value = "<html><body></body></html>"
    params = {"figures_root": str(tmp_path)}
    result = task.write_output(mock_fig, "myexp/recording", params)

    expected = tmp_path / "myexp" / "recording" / "plate_viewer.html"
    assert result == expected
    assert expected.exists()
    mock_fig.to_html.assert_called_once_with(full_html=True)


def test_plate_viewer_task_write_output_creates_parent_dirs(tmp_path):
    task = PlateViewerTask()
    mock_fig = MagicMock()
    mock_fig.to_html.return_value = "<html><body></body></html>"
    result = task.write_output(mock_fig, "a/b/c", {"figures_root": str(tmp_path)})
    assert result.parent.exists()


def test_plate_viewer_task_is_instance_of_base_plate_viewer():
    assert isinstance(PlateViewerTask(), BasePlateViewer)
