"""Core visualization logic for plate raster + synchrony viewer.

No pipeline coupling — can be used standalone or by PlateViewerTask.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from plotly.subplots import make_subplots


BURST_EVENT_TYPES = {
    "burstlets": {
        "label": "Burstlet",
        "color": "rgba(31, 119, 180, 0.42)",
        "line_color": "rgba(31, 119, 180, 0.80)",
        "default_visible": False,
        "lane": 0,
    },
    "network_bursts": {
        "label": "Burst",
        "color": "rgba(255, 127, 14, 0.46)",
        "line_color": "rgba(214, 90, 0, 0.85)",
        "default_visible": True,
        "lane": 1,
    },
    "superbursts": {
        "label": "Superburst",
        "color": "rgba(148, 103, 189, 0.46)",
        "line_color": "rgba(112, 67, 160, 0.85)",
        "default_visible": False,
        "lane": 2,
    },
}
EVENT_MARGINAL_FRACTION = 0.22
EVENT_MARGINAL_GAP_FRACTION = 0.035


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
    event_intervals: Optional[dict[str, list[dict]]] = None
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


def _axis_name_to_ref(axis_name: str) -> str:
    """Convert Plotly layout axis names to refs used by traces and shapes."""
    if axis_name == "xaxis":
        return "x"
    if axis_name == "yaxis":
        return "y"
    if axis_name.startswith("xaxis"):
        return f"x{axis_name.removeprefix('xaxis')}"
    if axis_name.startswith("yaxis"):
        return f"y{axis_name.removeprefix('yaxis')}"
    return axis_name


def _secondary_axis_refs(fig: go.Figure, row: int, col: int) -> tuple[str, str]:
    """Return shape refs for the subplot secondary axes."""
    subplot = fig.get_subplot(row, col, secondary_y=True)
    xref = f"{_axis_name_to_ref(subplot.xaxis._plotly_name)} domain"
    yref = _axis_name_to_ref(subplot.yaxis._plotly_name)
    return xref, yref


def _primary_axis_refs(fig: go.Figure, row: int, col: int) -> tuple[str, str]:
    """Return shape refs for the subplot primary axes."""
    subplot = fig.get_subplot(row, col, secondary_y=False)
    xref = _axis_name_to_ref(subplot.xaxis._plotly_name)
    yref = _axis_name_to_ref(subplot.yaxis._plotly_name)
    return xref, yref


def _primary_axis_names(fig: go.Figure, row: int, col: int) -> tuple[str, str]:
    """Return Plotly layout axis names for the subplot primary axes."""
    subplot = fig.get_subplot(row, col, secondary_y=False)
    return subplot.xaxis._plotly_name, subplot.yaxis._plotly_name


def _event_marginal_domain(fig: go.Figure, row: int, col: int) -> tuple[float, float]:
    """Reserve an upper marginal band over the primary plot and return its paper domain."""
    _, yaxis_name = _primary_axis_names(fig, row, col)
    yaxis = fig.layout[yaxis_name]
    domain = list(yaxis.domain)
    domain_height = domain[1] - domain[0]
    marginal_height = domain_height * EVENT_MARGINAL_FRACTION
    gap = domain_height * EVENT_MARGINAL_GAP_FRACTION
    marginal_domain = (domain[1] - marginal_height, domain[1])
    yaxis.domain = [domain[0], domain[1] - marginal_height - gap]
    return marginal_domain


def _add_secondary_hline(
    fig: go.Figure,
    row: int,
    col: int,
    y: float,
    color: str,
) -> None:
    """Add an annotation-free horizontal line on the subplot secondary y-axis."""
    xref, yref = _secondary_axis_refs(fig, row, col)
    fig.add_shape(
        type="line",
        xref=xref,
        yref=yref,
        x0=0,
        x1=1,
        y0=y,
        y1=y,
        line=dict(color=color, dash="dash"),
    )


def _add_event_zone_shapes(
    fig: go.Figure,
    row: int,
    col: int,
    event_intervals: dict[str, list[dict]] | None,
) -> None:
    """Add toggleable event interval lanes in an upper marginal band."""
    marginal_y0, marginal_y1 = _event_marginal_domain(fig, row, col)
    xref, _ = _primary_axis_refs(fig, row, col)
    band_height = marginal_y1 - marginal_y0
    lane_count = max(len(BURST_EVENT_TYPES), 1)
    lane_gap = band_height * 0.08 / lane_count
    lane_height = (band_height - lane_gap * (lane_count + 1)) / lane_count

    fig.add_shape(
        type="rect",
        xref=f"{xref} domain",
        yref="paper",
        x0=0,
        x1=1,
        y0=marginal_y0,
        y1=marginal_y1,
        fillcolor="rgba(245, 247, 250, 0.92)",
        line=dict(width=1, color="rgba(170, 180, 195, 0.65)"),
        layer="above",
        name="event-marginal-background",
    )

    if not event_intervals:
        return

    for event_key, style in BURST_EVENT_TYPES.items():
        intervals = event_intervals.get(event_key, [])
        if not intervals:
            continue
        lane = int(style["lane"])
        y0 = marginal_y0 + lane_gap + lane * (lane_height + lane_gap)
        y1 = y0 + lane_height
        for interval in intervals:
            fig.add_shape(
                type="rect",
                xref=xref,
                yref="paper",
                x0=float(interval["start"]),
                x1=float(interval["end"]),
                y0=y0,
                y1=y1,
                fillcolor=str(style["color"]),
                line=dict(width=1, color=str(style["line_color"])),
                layer="above",
                visible=bool(style["default_visible"]),
                name=f"event-zone:{event_key}",
            )


def _synchrony_y_range(sync_payload: dict) -> list[float] | None:
    """Return a padded participation-axis range for visible synchrony values."""
    values = []
    for key in ("signal", "smooth", "peaks"):
        trace = sync_payload.get(key)
        if trace:
            values.extend(trace["y"])
    for key in ("baseline", "threshold"):
        value = sync_payload.get(key)
        if value is not None:
            values.append(value)

    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return None

    ymin = min(0.0, float(np.min(finite_values)))
    ymax = max(0.0, float(np.max(finite_values)))
    padding = max((ymax - ymin) * 0.08, 0.02)
    return [ymin - padding, ymax + padding]


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
    participation_signal_smooth = plot_signals.get("participation_signal_smooth")
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

    # Smooth synchrony (slow participation). Older outputs do not have this
    # field, so smooth the sharp participation trace for display compatibility.
    if participation_signal_smooth is None and participation_signal is not None:
        participation_signal_smooth = gaussian_filter1d(
            np.asarray(participation_signal, dtype=float),
            sigma=5,
        )

    if t is not None and participation_signal_smooth is not None:
        t_down, sig_down = _downsample_xy(t, participation_signal_smooth, max_points)
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
    well_records = sorted(well_records, key=lambda wr: _well_id_to_position(wr.well_id))

    fig = make_subplots(
        rows=4,
        cols=6,
        shared_xaxes=True,
        specs=[[{"secondary_y": True} for _ in range(6)] for _ in range(4)],
        vertical_spacing=0.08,
        horizontal_spacing=0.03,
        subplot_titles=[f"{wr.well_name} / {wr.groupname}" for wr in well_records],
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

        _add_event_zone_shapes(fig, row, col, wr.event_intervals)

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

            # Smooth participation synchrony (orange)
            if sync_payload["smooth"]:
                fig.add_trace(
                    go.Scattergl(
                        x=sync_payload["smooth"]["x"],
                        y=sync_payload["smooth"]["y"],
                        mode="lines",
                        line=dict(color="rgba(255, 140, 0, 0.95)", width=config.line_width * 0.9),
                        hovertemplate="Smooth participation: %{y:.3f}<br>t=%{x:.3f}s<extra></extra>",
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
                _add_secondary_hline(
                    fig,
                    row,
                    col,
                    sync_payload["baseline"],
                    "rgba(255, 102, 0, 0.7)",
                )

            # Threshold (dashed line)
            if sync_payload["threshold"] is not None:
                _add_secondary_hline(
                    fig,
                    row,
                    col,
                    sync_payload["threshold"],
                    "rgba(192, 57, 43, 0.8)",
                )

            y_range = _synchrony_y_range(sync_payload)
            if y_range is not None:
                fig.update_yaxes(range=y_range, row=row, col=col, secondary_y=True)

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
        height=max(950, int(config.width_px * 0.42)),
        width=config.width_px,
        showlegend=True,
        hovermode="closest",
        template="plotly_white",
        font=dict(size=10),
    )

    return fig


def plate_figure_to_html(fig: go.Figure) -> str:
    """Return standalone viewer HTML with burst-zone checkbox controls."""
    html = fig.to_html(full_html=True)
    controls = _burst_zone_controls_html()
    script = _burst_zone_controls_script()
    return html.replace("<body>", f"<body>\n{controls}\n{script}", 1)


def write_plate_viewer_html(fig: go.Figure, output_path: str | Path) -> None:
    """Write a standalone plate viewer with custom burst-zone controls."""
    Path(output_path).write_text(plate_figure_to_html(fig), encoding="utf-8")


def _burst_zone_controls_html() -> str:
    controls = []
    for event_key, style in BURST_EVENT_TYPES.items():
        checked = " checked" if style["default_visible"] else ""
        color = str(style["line_color"])
        controls.append(
            (
                '<label class="burst-zone-control__item">'
                f'<input type="checkbox" data-burst-event="{event_key}"{checked}>'
                f'<span class="burst-zone-control__swatch" style="background:{color}"></span>'
                f'<span>{style["label"]}</span>'
                "</label>"
            )
        )
    controls_html = "\n".join(controls)
    return f"""
<style>
.burst-zone-control {{
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.94);
  border-bottom: 1px solid #d9dee7;
  font-family: Arial, sans-serif;
  font-size: 13px;
  color: #243042;
}}
.burst-zone-control__title {{
  font-weight: 600;
}}
.burst-zone-control__item {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
}}
.burst-zone-control__swatch {{
  width: 18px;
  height: 9px;
  border-radius: 2px;
  border: 1px solid rgba(0, 0, 0, 0.18);
}}
</style>
<div class="burst-zone-control" data-burst-zone-control>
  <span class="burst-zone-control__title">Burst zones</span>
  {controls_html}
</div>
""".strip()


def _burst_zone_controls_script() -> str:
    event_keys = list(BURST_EVENT_TYPES.keys())
    event_keys_json = json.dumps(event_keys)
    return f"""
<script>
(function() {{
  const eventKeys = {event_keys_json};
  function applyBurstZoneVisibility(graphDiv, eventKey, visible) {{
    const shapes = (graphDiv.layout && graphDiv.layout.shapes) || [];
    const update = {{}};
    shapes.forEach(function(shape, index) {{
      if (shape.name === "event-zone:" + eventKey) {{
        update["shapes[" + index + "].visible"] = visible;
      }}
    }});
    if (Object.keys(update).length > 0) {{
      Plotly.relayout(graphDiv, update);
    }}
  }}
  function initBurstZoneControls() {{
    const graphDiv = document.querySelector(".plotly-graph-div");
    const panel = document.querySelector("[data-burst-zone-control]");
    if (!graphDiv || !panel || !window.Plotly) {{
      return;
    }}
    eventKeys.forEach(function(eventKey) {{
      const input = panel.querySelector('[data-burst-event="' + eventKey + '"]');
      if (!input) {{
        return;
      }}
      applyBurstZoneVisibility(graphDiv, eventKey, input.checked);
      input.addEventListener("change", function() {{
        applyBurstZoneVisibility(graphDiv, eventKey, input.checked);
      }});
    }});
  }}
  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", initBurstZoneControls);
  }} else {{
    initBurstZoneControls();
  }}
}})();
</script>
""".strip()


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
