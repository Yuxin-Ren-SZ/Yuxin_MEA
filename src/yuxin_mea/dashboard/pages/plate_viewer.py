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

from yuxin_mea.analysis.burst_inspector import output_root_from_cache
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
        "font": {"size": 14, "color": "#84807a"},
    }],
    xaxis={"visible": False},
    yaxis={"visible": False},
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
_DEFAULT_CURATION_SUBDIR = "curation_data"

# Mapping: data-source key -> (analysis_root subdir, per-well terminal dir).
# "traditional" reads the original burst detector's output (default, back-compat).
_SOURCE_OPTIONS: dict[str, tuple[str, str]] = {
    "traditional": ("burst_detection_data", "burst_detection"),
    "ml": ("ml_burst_data", "ml_burst_detection"),
}


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
    html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("workspace"),
                            html.Span("analysis"),
                            html.Span("plate_viewer"),
                        ],
                        className="breadcrumb",
                    ),
                    html.H1("Plate viewer"),
                    html.Div(
                        "4×6 plate raster + synchrony. Reads burst_detection "
                        "and curation outputs; nothing computed from scratch.",
                        className="subtitle",
                    ),
                ]
            ),
        ],
        className="view-head",
    ),
    html.Div(id="plate-viewer-banner-slot"),
    html.Div(
        [
            html.Label("Recording", className="section-label",
                       style={"marginBottom": "0"}),
            html.Div(
                dcc.Dropdown(
                    id="plate-viewer-recording-dropdown",
                    options=[], value=None, clearable=False,
                ),
                style={"flex": "1 1 420px"},
            ),
            html.Label("Source:", className="section-label",
                       style={"marginBottom": "0", "marginLeft": "8px"}),
            dcc.RadioItems(
                id="plate-viewer-source",
                options=[
                    {"label": "traditional", "value": "traditional"},
                    {"label": "ml", "value": "ml"},
                ],
                value="traditional",
                inline=True,
                labelStyle={"marginRight": "10px"},
                inputStyle={"marginRight": "4px"},
            ),
            html.Button([html.Span("↻", className="glyph"), "Load"],
                        id="plate-viewer-load-btn", n_clicks=0,
                        className="btn primary"),
            html.Button([html.Span("⤓", className="glyph"), "Export HTML"],
                        id="plate-viewer-export-btn", n_clicks=0,
                        className="btn"),
            html.Span(id="plate-viewer-status",
                      style={"color": "var(--ink-3)",
                             "fontFamily": "var(--font-mono)", "fontSize": "11px"}),
        ],
        style={"display": "flex", "alignItems": "center", "gap": "8px",
               "marginBottom": "12px", "flexWrap": "wrap"},
    ),
    _display_settings_panel(),
    dcc.Graph(id="plate-viewer-fig", figure=_EMPTY_FIG,
              style={"height": "80vh", "marginTop": "16px"}),
], className="page")


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


def _resolve_source(source: str | None) -> tuple[str, str]:
    """Map a UI source key to (burst_subdir, burst_terminal). Unknown → traditional."""
    return _SOURCE_OPTIONS.get(str(source or "traditional"), _SOURCE_OPTIONS["traditional"])


def _resolve_burst_root(analysis_root: Path, source: str | None) -> Path:
    # Prefer the pipeline cache (source of truth for where the detector actually
    # wrote), so the viewer follows re-runs / changed output_root. Fall back to
    # the conventional subdir only when the cache has nothing for this method.
    from_cache = output_root_from_cache(analysis_root, str(source or "traditional"))
    if from_cache is not None:
        return from_cache
    burst_subdir, _ = _resolve_source(source)
    return Path(analysis_root) / burst_subdir


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
    source: str | None,
) -> go.Figure:
    burst_root = _resolve_burst_root(analysis_root, source)
    curation_root = _resolve_curation_root(analysis_root)
    _, burst_terminal = _resolve_source(source)
    cache_path = Path(analysis_root) / "experiment_cache.json"

    well_records = load_plate_data(
        burst_detection_root=burst_root,
        curation_output_root=curation_root,
        recording_key=recording_key,
        rec_name="auto",
        experiment_cache_path=cache_path if cache_path.exists() else None,
        burst_terminal=burst_terminal,
    )
    return build_plate_figure(well_records, _build_config(**settings))


@callback(
    Output("plate-viewer-fig", "figure"),
    Output("plate-viewer-status", "children", allow_duplicate=True),
    Input("plate-viewer-load-btn", "n_clicks"),
    State("plate-viewer-recording-dropdown", "value"),
    State("plate-viewer-source", "value"),
    State("plate-viewer-display-mode", "value"),
    State("plate-viewer-marker-size", "value"),
    State("plate-viewer-line-width", "value"),
    State("plate-viewer-width-px", "value"),
    State("plate-viewer-max-raster", "value"),
    State("plate-viewer-max-sync", "value"),
    prevent_initial_call=True,
)
def _on_load(_n_clicks, recording_key, source, display_mode, marker_size, line_width,
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
        fig = _render_figure(analysis_root, recording_key, settings, source)
    except Exception as exc:  # noqa: BLE001 — surface user-facing
        return dash.no_update, f"❌ {exc}"
    return fig, f"✓ Rendered {recording_key} ({source})"


# ---------------------------------------------------------------------------
# 3) Export button → write HTML file
# ---------------------------------------------------------------------------


@callback(
    Output("plate-viewer-status", "children", allow_duplicate=True),
    Input("plate-viewer-export-btn", "n_clicks"),
    State("plate-viewer-recording-dropdown", "value"),
    State("plate-viewer-source", "value"),
    State("plate-viewer-display-mode", "value"),
    State("plate-viewer-marker-size", "value"),
    State("plate-viewer-line-width", "value"),
    State("plate-viewer-width-px", "value"),
    State("plate-viewer-max-raster", "value"),
    State("plate-viewer-max-sync", "value"),
    prevent_initial_call=True,
)
def _on_export(_n_clicks, recording_key, source, display_mode, marker_size, line_width,
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
    source_key = str(source or "traditional")
    try:
        fig = _render_figure(analysis_root, recording_key, settings, source_key)
        output_path = Path(figure_root) / recording_key / f"plate_viewer_{source_key}.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_plate_viewer_html(fig, output_path)
    except Exception as exc:  # noqa: BLE001
        return f"❌ Export failed: {exc}"
    return f"✓ Saved to {output_path}"
