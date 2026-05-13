"""Plate viewer page.

Renders the 4×6 plate raster + synchrony figure for one recording on
demand. Reads existing `burst_detection_data` and `curation_data` outputs
via :func:`yuxin_mea.analysis.plate_raster_synchrony.load_plate_data`, then
builds the figure via :func:`build_plate_figure`.

Phase 5 replaced the old `PlateViewerTask` (which ran as part of the
pipeline DAG and wrote an HTML file) with this page: visualization isn't
processing, so it belongs in the dashboard, not the task registry.
Display settings live in the UI with hardcoded sensible defaults; no
config-file persistence.

An "Export HTML" button preserves the lab workflow of sharing standalone
HTML reports — it writes to ``<figure_root>/<recording_key>/plate_viewer.html``
using the existing `figure_root` global.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html
from flask import current_app

from yuxin_mea.analysis.plate_raster_synchrony import (
    PlateViewerConfig,
    build_plate_figure,
    load_plate_data,
    write_plate_viewer_html,
)
from yuxin_mea.dashboard.components import no_config_banner
from yuxin_mea.dataset.cache import JsonCacheStore


dash.register_page(__name__, path="/plate-viewer", name="Plate viewer", order=3)


_EMPTY_FIG = go.Figure().update_layout(
    annotations=[{
        "text": "(pick a recording and click Load)",
        "xref": "paper", "yref": "paper",
        "x": 0.5, "y": 0.5, "showarrow": False,
        "font": {"size": 14, "color": "#888"},
    }],
    margin={"l": 20, "r": 20, "t": 20, "b": 20},
    height=300,
)


# Sensible defaults — match the old `PlateViewerTask.default_params()` values.
_DEFAULTS = {
    "display_mode": "both",
    "marker_size": 5.0,
    "line_width": 1.25,
    "width_px": 2400,
    "max_raster_points_per_well": 12000,
    "max_synchrony_points": 3000,
}


# Where the per-well outputs live, relative to `analysis_root`. Matches the
# default `output_root` values in the relevant task schemas; if a lab has
# customized them, the user can override via the UI inputs.
_DEFAULT_BURST_SUBDIR = "burst_detection_data"
_DEFAULT_CURATION_SUBDIR = "curation_data"


def _display_settings_panel() -> html.Details:
    """Collapsed panel with the 6 visualization knobs."""
    return html.Details(
        [
            html.Summary("Display settings", style={"cursor": "pointer", "color": "#1f5aa6"}),
            html.Div(
                [
                    html.Label("Display mode:"),
                    dcc.Dropdown(
                        id="plate-viewer-display-mode",
                        options=[
                            {"label": "raster + synchrony", "value": "both"},
                            {"label": "raster only", "value": "raster"},
                            {"label": "synchrony only", "value": "synchrony"},
                        ],
                        value=_DEFAULTS["display_mode"],
                        clearable=False,
                        style={"maxWidth": "240px"},
                    ),
                    html.Label("Marker size:"),
                    dcc.Input(id="plate-viewer-marker-size", type="number",
                              value=_DEFAULTS["marker_size"], step=0.1, min=0,
                              style={"maxWidth": "120px"}),
                    html.Label("Line width:"),
                    dcc.Input(id="plate-viewer-line-width", type="number",
                              value=_DEFAULTS["line_width"], step=0.05, min=0,
                              style={"maxWidth": "120px"}),
                    html.Label("Figure width (px):"),
                    dcc.Input(id="plate-viewer-width-px", type="number",
                              value=_DEFAULTS["width_px"], step=100, min=400,
                              style={"maxWidth": "120px"}),
                    html.Label("Max raster pts / well:"),
                    dcc.Input(id="plate-viewer-max-raster", type="number",
                              value=_DEFAULTS["max_raster_points_per_well"],
                              step=1000, min=100,
                              style={"maxWidth": "120px"}),
                    html.Label("Max synchrony pts:"),
                    dcc.Input(id="plate-viewer-max-sync", type="number",
                              value=_DEFAULTS["max_synchrony_points"],
                              step=500, min=100,
                              style={"maxWidth": "120px"}),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "auto 1fr",
                    "gap": "8px 12px",
                    "marginTop": "8px",
                    "maxWidth": "480px",
                },
            ),
        ],
        open=False,
    )


layout = html.Div([
    html.H2("Plate viewer", style={"marginTop": "0"}),
    html.P(
        "4×6 plate raster + synchrony view for one recording. "
        "Reads existing burst_detection and curation outputs; nothing is "
        "computed from scratch."
    ),
    html.Div(id="plate-viewer-banner-slot"),
    html.Div(
        [
            html.Label("Recording: ", style={"marginRight": "6px"}),
            dcc.Dropdown(
                id="plate-viewer-recording-dropdown",
                options=[], value=None, clearable=False,
                style={"width": "420px", "display": "inline-block",
                       "verticalAlign": "middle"},
            ),
            html.Button("Load", id="plate-viewer-load-btn", n_clicks=0,
                        style={"marginLeft": "12px"}),
            html.Button("Export HTML", id="plate-viewer-export-btn", n_clicks=0,
                        style={"marginLeft": "8px"}),
            html.Span(id="plate-viewer-status",
                      style={"marginLeft": "12px", "color": "#555"}),
        ],
        style={"marginBottom": "12px"},
    ),
    _display_settings_panel(),
    dcc.Graph(id="plate-viewer-fig", figure=_EMPTY_FIG,
              style={"height": "80vh", "marginTop": "16px"}),
])


# ---------------------------------------------------------------------------
# 1) Populate the recording dropdown on page entry
# ---------------------------------------------------------------------------


@callback(
    Output("plate-viewer-banner-slot", "children"),
    Output("plate-viewer-recording-dropdown", "options"),
    Output("plate-viewer-recording-dropdown", "value"),
    Input("plate-viewer-recording-dropdown", "id"),
)
def _populate_recordings(_id: str):
    ctx_app = current_app.config.get("YUXIN_MEA", {})
    analysis_root = ctx_app.get("analysis_root")
    if not ctx_app.get("config_exists") or analysis_root is None:
        return no_config_banner(), [], None

    entries = JsonCacheStore(analysis_root).load()
    options = [{"label": e.cache_key, "value": e.cache_key} for e in entries.values()]
    initial = options[0]["value"] if options else None
    return None, options, initial


# ---------------------------------------------------------------------------
# 2) Load button → render figure
# ---------------------------------------------------------------------------


def _resolve_burst_root(analysis_root: Path) -> Path:
    return Path(analysis_root) / _DEFAULT_BURST_SUBDIR


def _resolve_curation_root(analysis_root: Path) -> Path:
    return Path(analysis_root) / _DEFAULT_CURATION_SUBDIR


def _build_config(
    display_mode: str, marker_size: float, line_width: float,
    width_px: int, max_raster: int, max_sync: int,
) -> PlateViewerConfig:
    return PlateViewerConfig(
        display_mode=str(display_mode),
        marker_size=float(marker_size),
        line_width=float(line_width),
        width_px=int(width_px),
        max_raster_points_per_well=int(max_raster),
        max_synchrony_points=int(max_sync),
    )


def _render_figure(
    analysis_root: Path,
    recording_key: str,
    settings: dict[str, Any],
) -> go.Figure:
    burst_root = _resolve_burst_root(analysis_root)
    curation_root = _resolve_curation_root(analysis_root)
    cache_path = Path(analysis_root) / "experiment_cache.json"

    well_records = load_plate_data(
        burst_detection_root=burst_root,
        curation_output_root=curation_root,
        recording_key=recording_key,
        rec_name="auto",
        experiment_cache_path=cache_path if cache_path.exists() else None,
    )
    return build_plate_figure(well_records, _build_config(**settings))


@callback(
    Output("plate-viewer-fig", "figure"),
    Output("plate-viewer-status", "children", allow_duplicate=True),
    Input("plate-viewer-load-btn", "n_clicks"),
    State("plate-viewer-recording-dropdown", "value"),
    State("plate-viewer-display-mode", "value"),
    State("plate-viewer-marker-size", "value"),
    State("plate-viewer-line-width", "value"),
    State("plate-viewer-width-px", "value"),
    State("plate-viewer-max-raster", "value"),
    State("plate-viewer-max-sync", "value"),
    prevent_initial_call=True,
)
def _on_load(_n_clicks, recording_key, display_mode, marker_size, line_width,
             width_px, max_raster, max_sync):
    if not recording_key:
        return dash.no_update, "Pick a recording first."
    analysis_root = current_app.config.get("YUXIN_MEA", {}).get("analysis_root")
    if analysis_root is None:
        return dash.no_update, "analysis_root is not set in the config."
    settings = {
        "display_mode": display_mode, "marker_size": marker_size,
        "line_width": line_width, "width_px": width_px,
        "max_raster": max_raster, "max_sync": max_sync,
    }
    try:
        fig = _render_figure(analysis_root, recording_key, settings)
    except Exception as exc:  # noqa: BLE001 — surface user-facing
        return dash.no_update, f"❌ {exc}"
    return fig, f"✓ Rendered {recording_key}"


# ---------------------------------------------------------------------------
# 3) Export button → write HTML file
# ---------------------------------------------------------------------------


@callback(
    Output("plate-viewer-status", "children", allow_duplicate=True),
    Input("plate-viewer-export-btn", "n_clicks"),
    State("plate-viewer-recording-dropdown", "value"),
    State("plate-viewer-display-mode", "value"),
    State("plate-viewer-marker-size", "value"),
    State("plate-viewer-line-width", "value"),
    State("plate-viewer-width-px", "value"),
    State("plate-viewer-max-raster", "value"),
    State("plate-viewer-max-sync", "value"),
    prevent_initial_call=True,
)
def _on_export(_n_clicks, recording_key, display_mode, marker_size, line_width,
               width_px, max_raster, max_sync):
    if not recording_key:
        return "Pick a recording first."
    ctx_app = current_app.config.get("YUXIN_MEA", {})
    analysis_root = ctx_app.get("analysis_root")
    figure_root = ctx_app.get("figure_root")
    if analysis_root is None:
        return "analysis_root is not set in the config."
    if figure_root is None:
        return "figure_root is not set in the config — set it on the Settings page."
    settings = {
        "display_mode": display_mode, "marker_size": marker_size,
        "line_width": line_width, "width_px": width_px,
        "max_raster": max_raster, "max_sync": max_sync,
    }
    try:
        fig = _render_figure(analysis_root, recording_key, settings)
        output_path = Path(figure_root) / recording_key / "plate_viewer.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_plate_viewer_html(fig, output_path)
    except Exception as exc:  # noqa: BLE001
        return f"❌ Export failed: {exc}"
    return f"✓ Saved to {output_path}"
