"""Core visualization logic for plate raster + synchrony viewer.

No pipeline coupling — used by `yuxin_mea.dashboard.pages.plate_viewer` and
callable directly from notebooks or scripts. Pre-Phase-5, this module was
consumed by `PlateViewerTask`; that task was removed and replaced by the
dashboard page when we moved plate visualization out of the pipeline DAG.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# Plate data loading
# ---------------------------------------------------------------------------
# These helpers assemble `WellRecord` objects from per-well outputs written
# by `BurstDetectionTask` (plot_signals.npy + *.pkl event tables) and
# `AutoCurationTask` (curated_spike_times.npy). They were previously private
# methods on `BasePlateViewer` and are promoted here so the dashboard's
# /plate-viewer page can call them without instantiating a task class.


_PLATE_WELL_COUNT = 24


def load_plate_data(
    burst_detection_root: Path,
    curation_output_root: Path,
    recording_key: str,
    rec_name: str = "auto",
    experiment_cache_path: Path | None = None,
    burst_terminal: str = "burst_detection",
) -> list[WellRecord]:
    """Assemble plate-level data for one recording from per-well outputs.

    Reads the 24 well-records that drive :func:`build_plate_figure`. Missing
    wells (incomplete burst_detection / curation) return placeholder records
    with ``status="missing"`` so the figure still has 24 slots.

    Args:
        burst_detection_root: directory containing per-well burst_detection outputs.
        curation_output_root: directory containing per-well curation outputs.
        recording_key: e.g. ``"CX138/260329/T003346/Network/000029"``.
        rec_name: Maxwell rec name ("rec0000" etc.) or ``"auto"``/empty for
            auto-detection from disk + experiment cache.
        experiment_cache_path: optional path to ``experiment_cache.json``;
            when present, used to look up well names / groupname. Missing or
            unreadable cache → wells default to ``well_name="?", groupname="?"``.
        burst_terminal: per-well terminal directory inside ``burst_detection_root``
            holding ``plot_signals.npy`` and the event tables. ``"burst_detection"``
            (default) reads the traditional detector; pass ``"iterative_burst_detection"``
            to read the iterative detector's outputs.

    Returns:
        list[WellRecord]: 24 entries, one per well slot.
    """
    burst_root = Path(burst_detection_root)
    curation_root = Path(curation_output_root)

    well_metadata: dict[str, dict[str, Any]] = {}
    well_rec_names: dict[str, str] = {}
    if experiment_cache_path is not None:
        well_metadata, well_rec_names = _load_recording_cache(
            Path(experiment_cache_path), recording_key
        )

    discovered = _discover_well_rec_names(
        recording_key, burst_root, curation_root, burst_terminal=burst_terminal,
    )
    for well_id_str, discovered_rec_name in discovered.items():
        well_rec_names.setdefault(well_id_str, discovered_rec_name)

    return [
        _load_well_record(
            well_id_str=f"well{well_num:03d}",
            recording_key=recording_key,
            rec_name=rec_name,
            burst_root=burst_root,
            curation_root=curation_root,
            well_metadata=well_metadata,
            well_rec_names=well_rec_names,
            burst_terminal=burst_terminal,
        )
        for well_num in range(_PLATE_WELL_COUNT)
    ]


def _load_recording_cache(
    cache_path: Path, recording_key: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load well metadata + per-well rec name map from the experiment cache.

    Returns ``({}, {})`` on any failure (missing file, malformed JSON, etc.).
    """
    metadata: dict[str, dict[str, Any]] = {}
    well_rec_names: dict[str, str] = {}
    try:
        with open(cache_path) as fh:
            cache = json.load(fh)
        recording_data = cache.get(recording_key, {})
        wells_data = recording_data.get("wells", {})

        for well_id_str, well_info in wells_data.items():
            well_meta = well_info.get("metadata", {})
            metadata[well_id_str] = {
                "well_name": well_meta.get("well_name", "?"),
                "groupname": well_meta.get("groupname", "?"),
            }

        for rec, well_ids in recording_data.get("h5_recordings", {}).items():
            for wid in well_ids:
                well_rec_names[str(wid)] = str(rec)
    except Exception as exc:  # noqa: BLE001 — best-effort cache reader
        print(f"Warning: Failed to load experiment cache: {exc}")
    return metadata, well_rec_names


def _discover_well_rec_names(
    recording_key: str,
    burst_root: Path,
    curation_root: Path,
    burst_terminal: str = "burst_detection",
) -> dict[str, str]:
    """Walk task output directories to infer per-well rec name."""
    discovered: dict[str, str] = {}
    for root, terminal_dir in (
        (burst_root, burst_terminal),
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
    well_id_str: str,
    rec_name: str,
    well_rec_names: dict[str, str] | None,
    burst_root: Path,
    curation_root: Path,
    recording_key: str,
    burst_terminal: str = "burst_detection",
) -> list[str]:
    """Order rec-name candidates; only include the user hint when its files exist."""
    candidates: list[str] = []
    rec_hint = rec_name if rec_name and rec_name.lower() != "auto" else None
    mapped_rec_name = (well_rec_names or {}).get(well_id_str)

    if rec_hint:
        hinted_burst = (
            burst_root / recording_key / rec_hint / well_id_str / burst_terminal
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
    well_id_str: str,
    recording_key: str,
    rec_name: str,
    burst_root: Path,
    curation_root: Path,
    well_metadata: dict[str, dict[str, Any]],
    well_rec_names: dict[str, str] | None = None,
    burst_terminal: str = "burst_detection",
) -> WellRecord:
    """Load spike times + plot signals for one well; return a `WellRecord`."""
    meta = well_metadata.get(well_id_str, {})
    well_name = meta.get("well_name", "?")
    groupname = meta.get("groupname", "?")
    rec_names = _rec_name_candidates(
        well_id_str, rec_name, well_rec_names, burst_root, curation_root, recording_key,
        burst_terminal=burst_terminal,
    )
    event_intervals = _load_event_intervals(
        well_id_str, recording_key, rec_names, burst_root,
        burst_terminal=burst_terminal,
    )

    plot_signals = None
    for candidate in rec_names:
        path = burst_root / recording_key / candidate / well_id_str / burst_terminal / "plot_signals.npy"
        if path.exists():
            try:
                plot_signals = np.load(path, allow_pickle=True).item()
            except Exception:  # noqa: BLE001
                return _make_well_record(
                    well_id=well_id_str, well_name=well_name,
                    groupname=groupname, status="plot_signals error",
                )
            break

    spike_times = None
    for candidate in rec_names:
        path = curation_root / recording_key / candidate / well_id_str / "auto_curation" / "curated_spike_times.npy"
        if path.exists():
            try:
                spike_times = np.load(path, allow_pickle=True).item()
            except Exception:  # noqa: BLE001
                return _make_well_record(
                    well_id=well_id_str, well_name=well_name,
                    groupname=groupname, status="spike_times error",
                )
            break

    if plot_signals is None and spike_times is None:
        return _make_well_record(
            well_id=well_id_str, well_name=well_name, groupname=groupname,
            event_intervals=event_intervals, status="missing",
        )
    return _make_well_record(
        well_id=well_id_str, well_name=well_name, groupname=groupname,
        plot_signals=plot_signals, spike_times=spike_times,
        event_intervals=event_intervals, status="ok",
    )


def _make_well_record(**kwargs: Any) -> WellRecord:
    """Instantiate `WellRecord`, tolerating older test stubs without new fields."""
    try:
        return WellRecord(**kwargs)
    except TypeError as exc:
        if "event_intervals" not in str(exc):
            raise
        kwargs.pop("event_intervals", None)
        return WellRecord(**kwargs)


def _load_event_intervals(
    well_id_str: str,
    recording_key: str,
    rec_names: list[str],
    burst_root: Path,
    burst_terminal: str = "burst_detection",
) -> dict[str, list[dict[str, Any]]]:
    """Load per-event interval tables for one well from disk."""
    event_intervals: dict[str, list[dict[str, Any]]] = {
        key: [] for key in BURST_EVENT_TYPES
    }
    for candidate in rec_names:
        burst_dir = burst_root / recording_key / candidate / well_id_str / burst_terminal
        if not burst_dir.exists():
            continue
        for event_key in event_intervals:
            event_path = burst_dir / f"{event_key}.pkl"
            if event_path.exists():
                event_intervals[event_key] = _read_event_table(event_path)
        break
    return event_intervals


def _read_event_table(event_path: Path) -> list[dict[str, Any]]:
    """Read one event table; keep numerically-valid intervals only."""
    try:
        table = pd.read_pickle(event_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Failed to load event intervals from {event_path}: {exc}")
        return []
    if table is None or table.empty or "start" not in table or "end" not in table:
        return []

    intervals: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        try:
            start = float(row["start"])
            end = float(row["end"])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
            continue
        interval: dict[str, Any] = {}
        for key, value in row.items():
            safe = _json_safe_scalar(value)
            if safe is not None:
                interval[key] = safe
        interval["start"] = start
        interval["end"] = end
        intervals.append(interval)
    return intervals


def _json_safe_scalar(value: Any) -> Any:
    """Convert common numpy/pandas scalars to JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return None
