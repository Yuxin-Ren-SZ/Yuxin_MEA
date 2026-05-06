from __future__ import annotations

import numpy as np

from pipeline_tasks.analysis.plate_raster_synchrony import (
    PlateViewerConfig,
    WellRecord,
    build_plate_figure,
    plate_figure_to_html,
)


def _synthetic_well_records() -> list[WellRecord]:
    records = []
    for i in range(24):
        records.append(
            WellRecord(
                well_id=f"well{i:03d}",
                well_name=f"W{i + 1}",
                groupname="control" if i < 12 else "treated",
                plot_signals={
                    "t": np.array([0.0, 1.0, 2.0]),
                    "participation_signal": np.array([0.1, 0.3, 0.2]),
                    "participation_signal_smooth": np.array([0.12, 0.22, 0.18]),
                    "rate_signal": np.array([1.0, 1.5, 1.1]),
                    "burst_peak_times": np.array([1.0]),
                    "burst_peak_values": np.array([0.3]),
                    "participation_baseline": 0.1,
                    "participation_threshold": 0.25,
                },
                spike_times={
                    "unit_1": np.array([0.2, 1.4]),
                    "unit_2": np.array([0.7]),
                },
            )
        )
    return records


def _synthetic_well_records_with_events() -> list[WellRecord]:
    records = _synthetic_well_records()
    records[0].event_intervals = {
        "burstlets": [{"start": 0.2, "end": 0.4}],
        "network_bursts": [{"start": 0.3, "end": 0.8}],
        "superbursts": [{"start": 0.1, "end": 1.5}],
    }
    return records


def test_build_plate_figure_supports_secondary_y_subplots():
    fig = build_plate_figure(_synthetic_well_records(), PlateViewerConfig())

    assert len(fig.data) > 24
    assert "yaxis2" in fig.layout
    assert "yaxis48" in fig.layout
    assert len(fig.layout.annotations) == 24


def test_build_plate_figure_handles_missing_wells():
    records = [
        WellRecord(
            well_id=f"well{i:03d}",
            well_name=f"W{i + 1}",
            groupname="?",
            status="missing",
        )
        for i in range(24)
    ]

    fig = build_plate_figure(records, PlateViewerConfig())

    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 48


def test_build_plate_figure_orders_shuffled_well_titles_by_plate_position():
    records = list(reversed(_synthetic_well_records()))

    fig = build_plate_figure(records, PlateViewerConfig())

    assert fig.layout.annotations[0].text == "W1 / control"
    assert fig.layout.annotations[23].text == "W24 / treated"


def test_threshold_and_baseline_shapes_use_secondary_y_axes():
    fig = build_plate_figure(_synthetic_well_records(), PlateViewerConfig())

    threshold_shapes = [
        shape
        for shape in fig.layout.shapes
        if not str(getattr(shape, "name", "")).startswith("event-")
    ]
    assert len(threshold_shapes) == 48
    yrefs = [shape.yref for shape in threshold_shapes]

    assert yrefs[:2] == ["y2", "y2"]
    assert all(yref != "y" for yref in yrefs)
    assert all(int(yref.removeprefix("y")) % 2 == 0 for yref in yrefs)


def test_event_interval_shapes_are_toggleable_by_event_type():
    fig = build_plate_figure(_synthetic_well_records_with_events(), PlateViewerConfig())

    event_shapes = [
        shape
        for shape in fig.layout.shapes
        if str(getattr(shape, "name", "")).startswith("event-zone:")
    ]

    assert len(event_shapes) == 3
    visibility_by_name = {shape.name: shape.visible for shape in event_shapes}
    assert visibility_by_name["event-zone:burstlets"] is False
    assert visibility_by_name["event-zone:network_bursts"] is True
    assert visibility_by_name["event-zone:superbursts"] is False
    assert all(shape.type == "rect" for shape in event_shapes)
    assert all(shape.yref == "paper" for shape in event_shapes)
    assert all(shape.layer == "above" for shape in event_shapes)
    assert all(shape.line.width == 1 for shape in event_shapes)


def test_event_marginal_reserves_separate_plot_band():
    fig = build_plate_figure(_synthetic_well_records_with_events(), PlateViewerConfig())

    event_backgrounds = [
        shape
        for shape in fig.layout.shapes
        if str(getattr(shape, "name", "")) == "event-marginal-background"
    ]
    event_shapes = [
        shape
        for shape in fig.layout.shapes
        if str(getattr(shape, "name", "")).startswith("event-zone:")
    ]

    assert len(event_backgrounds) == 24
    assert len(event_shapes) == 3
    assert event_backgrounds[0].yref == "paper"
    assert event_shapes[0].y0 >= event_backgrounds[0].y0
    assert event_shapes[0].y1 <= event_backgrounds[0].y1
    assert fig.layout.yaxis.domain[1] < event_backgrounds[0].y0


def test_event_shapes_do_not_replace_secondary_threshold_shapes():
    fig = build_plate_figure(_synthetic_well_records_with_events(), PlateViewerConfig())

    threshold_shapes = [
        shape
        for shape in fig.layout.shapes
        if not str(getattr(shape, "name", "")).startswith("event-")
    ]

    assert len(threshold_shapes) == 48
    assert threshold_shapes[0].yref == "y2"


def test_plate_figure_html_contains_burst_zone_controls():
    fig = build_plate_figure(_synthetic_well_records_with_events(), PlateViewerConfig())
    html = plate_figure_to_html(fig)

    assert "data-burst-zone-control" in html
    assert "Burstlet" in html
    assert "Burst" in html
    assert "Superburst" in html
    assert '"burstlets"' in html
    assert '"network_bursts"' in html
    assert '"superbursts"' in html
    assert "event-zone:" in html


def test_default_viewer_plots_smooth_participation_not_rate_signal():
    fig = build_plate_figure(_synthetic_well_records(), PlateViewerConfig())

    smooth_traces = [
        trace
        for trace in fig.data
        if str(getattr(trace, "hovertemplate", "")).startswith("Smooth participation")
    ]

    assert len(smooth_traces) == 24
    assert list(smooth_traces[0].y) == [0.12, 0.22, 0.18]
    assert smooth_traces[0].yaxis == "y2"
    assert all(list(trace.y) != [1.0, 1.5, 1.1] for trace in fig.data)


def test_default_viewer_smooths_participation_when_smooth_field_missing():
    records = _synthetic_well_records()
    for record in records:
        record.plot_signals.pop("participation_signal_smooth")

    fig = build_plate_figure(records, PlateViewerConfig())

    smooth_traces = [
        trace
        for trace in fig.data
        if str(getattr(trace, "hovertemplate", "")).startswith("Smooth participation")
    ]

    assert len(smooth_traces) == 24
    assert smooth_traces[0].yaxis == "y2"
    assert all(0.0 <= y <= 1.0 for y in smooth_traces[0].y)
    assert list(smooth_traces[0].y) != [1.0, 1.5, 1.1]


def test_peak_markers_use_participation_values_on_secondary_y_axis():
    fig = build_plate_figure(_synthetic_well_records(), PlateViewerConfig())

    peak_traces = [
        trace
        for trace in fig.data
        if str(getattr(trace, "hovertemplate", "")).startswith("Peak:")
    ]

    assert len(peak_traces) == 24
    assert list(peak_traces[0].x) == [1.0]
    assert list(peak_traces[0].y) == [0.3]
    assert peak_traces[0].yaxis == "y2"
