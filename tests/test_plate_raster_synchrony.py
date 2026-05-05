from __future__ import annotations

import numpy as np

from pipeline_tasks.analysis.plate_raster_synchrony import (
    PlateViewerConfig,
    WellRecord,
    build_plate_figure,
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

    assert len(fig.layout.shapes) == 48
    yrefs = [shape.yref for shape in fig.layout.shapes]

    assert yrefs[:2] == ["y2", "y2"]
    assert all(yref != "y" for yref in yrefs)
    assert all(int(yref.removeprefix("y")) % 2 == 0 for yref in yrefs)


def test_default_viewer_does_not_plot_rate_signal():
    fig = build_plate_figure(_synthetic_well_records(), PlateViewerConfig())

    assert all(
        not str(getattr(trace, "hovertemplate", "")).startswith("Smooth sync")
        for trace in fig.data
    )
    assert all(list(trace.y) != [1.0, 1.5, 1.1] for trace in fig.data)


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
