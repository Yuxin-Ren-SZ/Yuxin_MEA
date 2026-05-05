"""Core visualization logic for plate raster + synchrony viewer.

No pipeline coupling — can be used standalone or by PlateViewerTask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class PlateViewerConfig:
    """Display parameters for plate visualization."""

    display_mode: str = "both"  # "raster", "synchrony", or "both"
    marker_size: float = 5.0
    line_width: float = 1.25
    width_px: int = 2400
    max_raster_points_per_well: int = 12000
    max_synchrony_points: int = 3000


@dataclass
class WellRecord:
    """Per-well input data and metadata."""

    well_id: str  # "well000", "well001", etc.
    well_name: str  # "A1", "A2", etc.
    groupname: str  # "NPH", "IVH Early", etc.
    plot_signals: Optional[dict] = None  # Keys: t, participation_signal, rate_signal, etc.
    spike_times: Optional[dict[str, np.ndarray]] = None  # unit_id -> spike times in seconds
    status: str = "ok"  # "ok", "missing", or error message string


def _well_id_to_position(well_id: str) -> tuple[int, int]:
    """Convert well000 -> (row=0, col=0), well001 -> (row=0, col=1), etc.

    24 wells in 4 rows x 6 cols, row-major order.
    """
    well_num = int(well_id.replace("well", ""))
    row = well_num // 6
    col = well_num % 6
    return (row, col)


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform stride downsampling of paired arrays to stay under max_points."""
    if len(x) <= max_points:
        return x, y
    stride = int(np.ceil(len(x) / max_points))
    return x[::stride], y[::stride]


def _downsample_array(arr: np.ndarray, max_points: int) -> np.ndarray:
    """Uniform stride downsampling of a single array."""
    if len(arr) <= max_points:
        return arr
    stride = int(np.ceil(len(arr) / max_points))
    return arr[::stride]


def _raster_payload_for_well(
    spike_times: dict[str, np.ndarray],
    max_points_per_well: int,
) -> tuple[list[dict], float]:
    """Build raster traces for all units in a well.

    Returns: (list of trace dicts, xmax)
    Each trace: {"x": [times...], "rank": int, "unit_id": str}
    """
    if not spike_times:
        return [], 0.0

    traces = []
    xmax = 0.0

    # Sort units by firing rate (descending)
    unit_ids = sorted(spike_times.keys())
    rates = [(uid, len(spike_times[uid])) for uid in unit_ids]
    rates.sort(key=lambda x: x[1], reverse=True)

    total_points = sum(len(spike_times[uid]) for uid in unit_ids)
    if total_points > 0:
        stride = max(1, int(np.ceil(total_points / max_points_per_well)))
    else:
        stride = 1

    for rank, (unit_id, _) in enumerate(rates):
        times = spike_times[unit_id]
        if times.size > 0:
            downsampled = times[::stride]
            traces.append(
                {
                    "x": downsampled.tolist(),
                    "y": [rank] * len(downsampled),
                    "unit_id": unit_id,
                    "hover_label": f"{unit_id}: {len(times)} spikes",
                }
            )
            xmax = max(xmax, float(np.max(times)))

    return traces, xmax


def _synchrony_payload_for_well(
    plot_signals: dict,
    max_points: int,
) -> tuple[dict, float]:
    """Build synchrony traces from plot_signals dict.

    Returns: (payload dict with signal/smooth/peaks/baseline/threshold, xmax)
    """
    payload = {
        "signal": None,
        "smooth": None,
        "peaks": None,
        "baseline": None,
        "threshold": None,
        "xmax": 0.0,
    }

    if not plot_signals:
        return payload, 0.0

    t = plot_signals.get("t")
    participation_signal = plot_signals.get("participation_signal")
    rate_signal = plot_signals.get("rate_signal")
    burst_peak_times = plot_signals.get("burst_peak_times")
    burst_peak_values = plot_signals.get("burst_peak_values")
    participation_baseline = plot_signals.get("participation_baseline")
    participation_threshold = plot_signals.get("participation_threshold")

    xmax = 0.0
    if t is not None and len(t) > 0:
        xmax = float(np.max(t))

    # Sharp synchrony (participation_signal)
    if t is not None and participation_signal is not None:
        t_down, sig_down = _downsample_xy(t, participation_signal, max_points)
        payload["signal"] = {"x": t_down.tolist(), "y": sig_down.tolist()}

    # Smooth synchrony (rate_signal)
    if t is not None and rate_signal is not None:
        t_down, sig_down = _downsample_xy(t, rate_signal, max_points)
        payload["smooth"] = {"x": t_down.tolist(), "y": sig_down.tolist()}

    # Burst peaks
    if burst_peak_times is not None and burst_peak_values is not None:
        if len(burst_peak_times) > 0:
            payload["peaks"] = {
                "x": burst_peak_times.tolist(),
                "y": burst_peak_values.tolist(),
            }

    # Baseline and threshold as scalar values
    if participation_baseline is not None:
        baseline_val = float(participation_baseline) if isinstance(
            participation_baseline, (np.ndarray, float, int)
        ) else None
        payload["baseline"] = baseline_val

    if participation_threshold is not None:
        threshold_val = float(participation_threshold) if isinstance(
            participation_threshold, (np.ndarray, float, int)
        ) else None
        payload["threshold"] = threshold_val

    payload["xmax"] = xmax
    return payload, xmax


def build_plate_figure(
    well_records: list[WellRecord],
    config: PlateViewerConfig,
) -> go.Figure:
    """Build a Plotly figure with 4x6 well subplots.

    Each subplot: raster (primary Y) + synchrony signals (secondary Y).
    """
    fig = make_subplots(
        rows=4,
        cols=6,
        shared_xaxes=True,
        secondary_y=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.03,
        subplot_titles=[f"{wr.well_name}\n{wr.groupname}" for wr in well_records],
    )

    # Compute global x-max for matching all x-axes
    global_xmax = 0.0
    for wr in well_records:
        if wr.status == "ok":
            if wr.spike_times:
                for times in wr.spike_times.values():
                    if len(times) > 0:
                        global_xmax = max(global_xmax, float(np.max(times)))
            if wr.plot_signals:
                t = wr.plot_signals.get("t")
                if t is not None and len(t) > 0:
                    global_xmax = max(global_xmax, float(np.max(t)))

    # Color scheme per group
    group_colors = _get_group_colors([wr.groupname for wr in well_records if wr.status == "ok"])

    # Add traces per well
    for wr in well_records:
        row, col = _well_id_to_position(wr.well_id)
        row += 1  # Plotly uses 1-based indexing
        col += 1

        if wr.status != "ok":
            # Missing/error well: add grey annotation
            status_text = wr.status if wr.status != "missing" else "N/A"
            fig.add_annotation(
                text=status_text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=12, color="grey"),
                row=row,
                col=col,
            )
            continue

        # Raster traces (primary Y)
        if wr.spike_times:
            raster_traces, _ = _raster_payload_for_well(
                wr.spike_times, config.max_raster_points_per_well
            )
            for trace_data in raster_traces:
                fig.add_trace(
                    go.Scattergl(
                        x=trace_data["x"],
                        y=trace_data["y"],
                        mode="markers",
                        marker=dict(
                            size=config.marker_size,
                            color="rgba(90, 90, 90, 0.75)",
                            symbol="line-ns-open",
                        ),
                        hovertemplate=f"{trace_data['hover_label']}<br>t=%{{x:.3f}}s<extra></extra>",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                    secondary_y=False,
                )

        # Synchrony signals (secondary Y)
        if wr.plot_signals:
            sync_payload, _ = _synchrony_payload_for_well(
                wr.plot_signals, config.max_synchrony_points
            )

            # Sharp synchrony (dark red)
            if sync_payload["signal"]:
                fig.add_trace(
                    go.Scattergl(
                        x=sync_payload["signal"]["x"],
                        y=sync_payload["signal"]["y"],
                        mode="lines",
                        line=dict(color="#b22222", width=config.line_width),
                        hovertemplate="Sharp sync: %{y:.3f}<br>t=%{x:.3f}s<extra></extra>",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                    secondary_y=True,
                )

            # Smooth synchrony (orange)
            if sync_payload["smooth"]:
                fig.add_trace(
                    go.Scattergl(
                        x=sync_payload["smooth"]["x"],
                        y=sync_payload["smooth"]["y"],
                        mode="lines",
                        line=dict(color="rgba(255, 140, 0, 0.95)", width=config.line_width * 0.9),
                        hovertemplate="Smooth sync: %{y:.3f}<br>t=%{x:.3f}s<extra></extra>",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                    secondary_y=True,
                )

            # Burst peaks (red dots)
            if sync_payload["peaks"]:
                fig.add_trace(
                    go.Scattergl(
                        x=sync_payload["peaks"]["x"],
                        y=sync_payload["peaks"]["y"],
                        mode="markers",
                        marker=dict(size=4, color="red"),
                        hovertemplate="Peak: %{y:.3f}<br>t=%{x:.3f}s<extra></extra>",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                    secondary_y=True,
                )

            # Baseline (dashed line)
            if sync_payload["baseline"] is not None:
                baseline_val = sync_payload["baseline"]
                fig.add_hline(
                    y=baseline_val,
                    line_dash="dash",
                    line_color="rgba(255, 102, 0, 0.7)",
                    annotation_text=None,
                    row=row,
                    col=col,
                    secondary_y=True,
                    showlegend=False,
                )

            # Threshold (dashed line)
            if sync_payload["threshold"] is not None:
                threshold_val = sync_payload["threshold"]
                fig.add_hline(
                    y=threshold_val,
                    line_dash="dash",
                    line_color="rgba(192, 57, 43, 0.8)",
                    annotation_text=None,
                    row=row,
                    col=col,
                    secondary_y=True,
                    showlegend=False,
                )

    # Add group legend traces (invisible, only for legend)
    for group_name, color in group_colors.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=8, color=color),
                name=group_name,
                hoverinfo="none",
                showlegend=True,
            )
        )

    # Configure layout
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="Unit", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Sync", row=1, col=6, secondary_y=True)

    fig.update_layout(
        title="24-Well Plate Raster + Synchrony Viewer",
        height=max(800, config.width_px // 3),
        width=config.width_px,
        showlegend=True,
        hovermode="closest",
        template="plotly_white",
        font=dict(size=10),
    )

    return fig


def _get_group_colors(group_names: list[str]) -> dict[str, str]:
    """Assign distinct colors to each unique group name."""
    color_palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
    ]
    unique_groups = sorted(set(group_names))
    return {group: color_palette[i % len(color_palette)] for i, group in enumerate(unique_groups)}
