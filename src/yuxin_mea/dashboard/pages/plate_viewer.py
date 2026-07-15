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
from dash import ALL, Input, Output, State, callback, ctx, dcc, html
from flask import current_app

# NOTE: yuxin_mea.analysis.* pulls in torch/spikeinterface (heavy). Imported
# lazily inside the callbacks/helpers below so the dashboard can start without
# the ML stack. `PlateViewerConfig` appears only in an annotation, which stays a
# lazy string thanks to `from __future__ import annotations`.
from yuxin_mea.dashboard.components import (
    build_filter_bar,
    filter_id,
    filter_kwargs,
    iso_to_yymmdd,
    no_config_banner,
    provenance_badge,
    yymmdd_to_iso,
)
from yuxin_mea.dashboard.data import (
    filter_recordings,
    load_recordings_detail,
    recording_provenance,
)
from yuxin_mea.dashboard.data_cache import data_sig, load_plate_data_cached


dash.register_page(__name__, path="/plate-viewer", name="Plate viewer", order=3)


# Modal (single-well pop-up) root style, fully replaced each toggle. Hidden by
# default; shown as a fixed full-screen flex-centered overlay. The translucent
# backdrop is a *sibling* of the content (see layout), so clicking the content
# never bubbles to the backdrop — backdrop `n_clicks` == "clicked outside".
_MODAL_HIDDEN = {"display": "none"}
_MODAL_SHOWN = {
    "display": "flex",
    "position": "fixed",
    "inset": 0,
    "zIndex": 1000,
    "alignItems": "center",
    "justifyContent": "center",
}


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


def _sort_recordings(recordings: list[dict]) -> list[dict]:
    """Same-sample recordings together, ordered by assay ``run_id`` then date.

    ``run_id`` is a leading-zero-preserving string, so lexical order is correct
    (no int cast). Pure — the caller sorts the *filtered* list locally so the
    shared `load_recordings_detail` loader (feeding other pages) is untouched.
    """
    return sorted(recordings, key=lambda r: (r["sample_id"], r["run_id"], r["date"]))


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
    # Filters narrow the Recording dropdown below (same bar as the other pages).
    build_filter_bar("plate-viewer"),
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
    dcc.Store(id="plate-viewer-context"),
    dcc.Store(id="plate-viewer-active-well", data=None),
    # Anchor for the one-time document keydown listener (registered clientside).
    html.Div(id="plate-viewer-keydummy", style={"display": "none"}),

    # Navigation: step the open well (←/→) or switch plate (Shift+←/→).
    # Order mirrors the physical relationship: plate ‹ well ‹ well › plate ›.
    html.Div(
        [
            html.Button([html.Span("‹‹", className="glyph"), "Previous plate"],
                        id="plate-viewer-prev-plate", n_clicks=0, className="btn",
                        title="Previous plate (Shift+←)"),
            html.Button([html.Span("‹", className="glyph"), "Previous well"],
                        id="plate-viewer-prev-well", n_clicks=0, className="btn",
                        title="Previous well (←)"),
            html.Button(["Next well", html.Span("›", className="glyph")],
                        id="plate-viewer-next-well", n_clicks=0, className="btn",
                        title="Next well (→)"),
            html.Button(["Next plate", html.Span("››", className="glyph")],
                        id="plate-viewer-next-plate", n_clicks=0, className="btn",
                        title="Next plate (Shift+→)"),
            html.Span(
                "← / → well · Shift + ← / → plate",
                style={"marginLeft": "12px", "fontFamily": "var(--font-mono)",
                       "fontSize": "10px", "color": "var(--ink-3)"},
            ),
        ],
        style={"display": "flex", "gap": "6px", "alignItems": "center",
               "flexWrap": "wrap", "margin": "12px 0 0"},
    ),

    # dcc.Loading gives an in-process spinner/skeleton while a cold Load renders
    # the overview PNGs — the "responsive first-load" the guide wants, without a
    # multiprocess background callback (which would run outside Flask context and
    # bypass the Tier 0-2 caches). See Stage C.2 note in the plan.
    dcc.Loading(
        html.Div(
            "Load a recording to see the plate overview. Click a well to open "
            "its interactive raster.",
            id="plate-viewer-overview",
            style={"marginTop": "16px", "color": "var(--ink-3)"},
        ),
        type="default",
    ),

    # Single-well modal. The backdrop is a sibling of the content div (not its
    # parent), so a click on the content never bumps the backdrop's n_clicks —
    # backdrop n_clicks therefore means "clicked outside" → close.
    html.Div(
        [
            html.Div(
                id="plate-viewer-modal-backdrop",
                n_clicks=0,
                style={"position": "absolute", "inset": 0,
                       "background": "rgba(0,0,0,0.55)"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(id="plate-viewer-modal-title",
                                      style={"fontFamily": "var(--font-mono)",
                                             "fontSize": "13px", "fontWeight": "600",
                                             "color": "var(--ink)"}),
                            html.Button("✕", id="plate-viewer-modal-close",
                                        n_clicks=0, className="btn",
                                        style={"marginLeft": "auto"},
                                        title="Close (or click outside)"),
                        ],
                        style={"display": "flex", "alignItems": "center",
                               "gap": "8px", "marginBottom": "8px"},
                    ),
                    dcc.Loading(html.Div(id="plate-viewer-modal-body"), type="default"),
                ],
                style={"position": "relative", "zIndex": 1,
                       "background": "var(--bg)", "border": "1px solid var(--line)",
                       "borderRadius": "10px", "padding": "14px", "width": "88vw",
                       "maxWidth": "1400px", "maxHeight": "92vh", "overflow": "auto",
                       "boxShadow": "0 12px 40px rgba(0,0,0,0.4)"},
            ),
        ],
        id="plate-viewer-modal",
        style=_MODAL_HIDDEN,
    ),
], className="page")


# ---------------------------------------------------------------------------
# 1) Populate the recording dropdown on page entry
# ---------------------------------------------------------------------------


@callback(
    Output("plate-viewer-banner-slot", "children"),
    Output("plate-viewer-recording-dropdown", "options"),
    Output("plate-viewer-recording-dropdown", "value"),
    Output(filter_id("plate-viewer", "sample"), "options"),
    Output(filter_id("plate-viewer", "scan-type"), "options"),
    Output(filter_id("plate-viewer", "date"), "min_date_allowed"),
    Output(filter_id("plate-viewer", "date"), "max_date_allowed"),
    Output(filter_id("plate-viewer", "group"), "options"),
    Input(filter_id("plate-viewer", "sample"), "value"),
    Input(filter_id("plate-viewer", "scan-type"), "value"),
    Input(filter_id("plate-viewer", "date"), "start_date"),
    Input(filter_id("plate-viewer", "date"), "end_date"),
    Input(filter_id("plate-viewer", "group"), "value"),
    Input(filter_id("plate-viewer", "status"), "value"),
    State("plate-viewer-recording-dropdown", "value"),
)
def _populate_recordings(f_sample, f_scan_type, f_date_from, f_date_to, f_group,
                         f_status, current_value):
    ctx_app = current_app.config.get("YUXIN_MEA", {})
    analysis_root = ctx_app.get("analysis_root")
    empty: list[dict] = []
    if not ctx_app.get("config_exists") or analysis_root is None:
        return no_config_banner(), empty, None, empty, empty, None, None, empty

    recordings, well_pipeline_status = load_recordings_detail(Path(analysis_root))

    # Filter option universes (all recordings, unaffected by current selection).
    sample_opts = sorted({r["sample_id"] for r in recordings})
    scan_opts = sorted({r["scan_type"] for r in recordings})
    date_opts = sorted({r["date"] for r in recordings})
    date_min = yymmdd_to_iso(date_opts[0]) if date_opts else None
    date_max = yymmdd_to_iso(date_opts[-1]) if date_opts else None
    group_opts = sorted({g for r in recordings for g in r.get("groups", [])})

    kwargs = filter_kwargs(
        {
            "sample": f_sample,
            "scan-type": f_scan_type,
            "date-from": iso_to_yymmdd(f_date_from),
            "date-to": iso_to_yymmdd(f_date_to),
            "group": f_group,
            "status": f_status,
        }
    )
    filtered = filter_recordings(recordings, well_pipeline_status, **kwargs)

    filtered = _sort_recordings(filtered)
    rec_options = [{"label": r["cache_key"], "value": r["cache_key"]} for r in filtered]

    valid = {o["value"] for o in rec_options}
    value = current_value if current_value in valid else (
        rec_options[0]["value"] if rec_options else None
    )

    return (
        None,
        rec_options,
        value,
        [{"label": s, "value": s} for s in sample_opts],
        [{"label": s, "value": s} for s in scan_opts],
        date_min,
        date_max,
        [{"label": g, "value": g} for g in group_opts],
    )


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
    from yuxin_mea.analysis.burst_inspector import output_root_from_cache
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
    from yuxin_mea.analysis.plate_raster_synchrony import PlateViewerConfig
    return PlateViewerConfig(
        display_mode=str(display_mode),
        marker_size=float(marker_size),
        line_width=float(line_width),
        width_px=int(width_px),
        max_raster_points_per_well=int(max_raster),
        max_synchrony_points=int(max_sync),
    )


def _load_records(
    analysis_root: Path,
    recording_key: str,
    source: str | None,
):
    """Load 24 ``WellRecord``s (memoized) + the manifests that key raster PNGs.

    Path manifests come from ``pipeline_cache.json`` (Tier 0): exact per-well
    output dirs, so the loader never globs the NAS. Empty → the loader falls
    back to legacy discovery. The burst task name equals its terminal dir.
    """
    from yuxin_mea.analysis.burst_inspector import well_output_dirs_from_cache
    source_key = str(source or "traditional")
    burst_root = _resolve_burst_root(analysis_root, source)
    curation_root = _resolve_curation_root(analysis_root)
    _, burst_terminal = _resolve_source(source)
    cache_path = Path(analysis_root) / "experiment_cache.json"
    experiment_cache_path = cache_path if cache_path.exists() else None

    burst_well_dirs = (
        well_output_dirs_from_cache(analysis_root, recording_key, burst_terminal) or None
    )
    curation_well_dirs = (
        well_output_dirs_from_cache(analysis_root, recording_key, "auto_curation") or None
    )
    records = load_plate_data_cached(
        recording_key=recording_key,
        source=source_key,
        burst_root=burst_root,
        curation_root=curation_root,
        burst_terminal=burst_terminal,
        experiment_cache_path=experiment_cache_path,
        burst_well_dirs=burst_well_dirs,
        curation_well_dirs=curation_well_dirs,
        bundle_dir=Path(analysis_root) / "viewer_bundles",
    )
    return records, burst_well_dirs, curation_well_dirs


def _render_plate_figure(
    analysis_root: Path,
    recording_key: str,
    settings: dict[str, Any],
    source: str | None,
) -> go.Figure:
    """Full interactive 24-well figure — used by Export HTML (payload unchanged)."""
    from yuxin_mea.analysis.plate_raster_synchrony import build_plate_figure
    records, _bwd, _cwd = _load_records(analysis_root, recording_key, source)
    return build_plate_figure(records, _build_config(**settings))


def _well_signatures(
    burst_well_dirs: dict[str, Any] | None,
    curation_well_dirs: dict[str, Any] | None,
) -> dict[str, str]:
    """Per-well stat signature keying each raster PNG to its own source files."""
    sigs: dict[str, str] = {}
    for well_id in set(burst_well_dirs or {}) | set(curation_well_dirs or {}):
        files = []
        if burst_well_dirs and well_id in burst_well_dirs:
            files.append(Path(burst_well_dirs[well_id]) / "plot_signals.npy")
        if curation_well_dirs and well_id in curation_well_dirs:
            files.append(Path(curation_well_dirs[well_id]) / "curated_spike_times.npy")
        sigs[well_id] = repr(data_sig(files))
    return sigs


def _well_sort_key(well_id: str) -> int:
    try:
        return int(str(well_id).replace("well", ""))
    except ValueError:
        return 0


_GRID_STYLE = {
    "display": "grid", "gridTemplateColumns": "repeat(6, 1fr)",
    "gap": "6px", "marginTop": "16px",
}
_CELL_STYLE = {
    "padding": "4px", "border": "1px solid var(--line, #e0e0e0)",
    "borderRadius": "6px", "background": "#fff", "cursor": "pointer",
    "display": "flex", "flexDirection": "column", "alignItems": "center",
}
_CELL_EMPTY_STYLE = {**_CELL_STYLE, "cursor": "default", "background": "#f4f4f6",
                     "minHeight": "90px", "justifyContent": "center"}
_CAP_STYLE = {"fontSize": "11px", "marginTop": "2px",
              "fontFamily": "var(--font-mono)", "color": "var(--ink-3)"}


def _overview_grid(records, png_map: dict[str, Any]) -> html.Div:
    """4×6 grid of clickable raster PNGs; missing wells shown as greyed cells."""
    from yuxin_mea.analysis.raster_image import png_to_data_uri
    cells = []
    for wr in sorted(records, key=lambda r: _well_sort_key(r.well_id)):
        caption = html.Div(wr.well_name, style=_CAP_STYLE)
        if wr.status == "ok" and wr.well_id in png_map:
            cells.append(
                html.Button(
                    [
                        html.Img(src=png_to_data_uri(png_map[wr.well_id]),
                                 style={"width": "100%", "height": "auto", "display": "block"}),
                        caption,
                    ],
                    id={"type": "pv-well-cell", "index": wr.well_id},
                    n_clicks=0, style=_CELL_STYLE, title=f"Open {wr.well_name}",
                )
            )
        else:
            label = "N/A" if wr.status == "missing" else str(wr.status)
            cells.append(
                html.Div(
                    [html.Div(label, style={"fontSize": "12px", "color": "#9a9a9a"}), caption],
                    style=_CELL_EMPTY_STYLE,
                )
            )
    return html.Div(cells, style=_GRID_STYLE)


def _settings_from(display_mode, marker_size, line_width, width_px, max_raster, max_sync):
    return {
        "display_mode": display_mode, "marker_size": marker_size,
        "line_width": line_width, "width_px": width_px,
        "max_raster": max_raster, "max_sync": max_sync,
    }


def _render_overview_outputs(analysis_root, cache_root, recording_key, source_key, settings):
    """Build the 24-well overview + context + status for a recording.

    Shared by the Load button and the prev/next-plate nav. May raise; callers
    wrap it and surface the message. Returns ``(overview, context, status)``.
    """
    from yuxin_mea.analysis.plate_raster_synchrony import build_plate_figure
    from yuxin_mea.analysis.raster_image import render_overview_pngs
    records, bwd, cwd = _load_records(analysis_root, recording_key, source_key)
    if cache_root is None:
        # No local cache dir configured → keep the page usable by rendering the
        # full interactive figure inline (no pre-rasterization).
        overview = dcc.Graph(
            figure=build_plate_figure(records, _build_config(**settings)),
            style={"height": "78vh"},
        )
    else:
        png_map = render_overview_pngs(
            records, cache_root,
            well_sigs=_well_signatures(bwd, cwd),
            recording_key=recording_key, source=source_key,
        )
        overview = _overview_grid(records, png_map)

    ok_wells = [wr.well_id for wr in sorted(records, key=lambda r: _well_sort_key(r.well_id))
                if wr.status == "ok"]
    # Provenance status for this recording (computed once per Load, shown in the
    # single-well modal). Advisory — never blocks the view.
    config_path = current_app.config.get("YUXIN_MEA", {}).get("config_path")
    prov_status = recording_provenance(Path(analysis_root), config_path).get(recording_key)
    context = {
        "recording_key": recording_key, "source": source_key,
        "settings": settings, "ok_wells": ok_wells, "provenance": prov_status,
    }
    status = f"✓ {recording_key} ({source_key}) — {len(ok_wells)}/24 wells"
    return overview, context, status


@callback(
    Output("plate-viewer-overview", "children"),
    Output("plate-viewer-context", "data"),
    Output("plate-viewer-status", "children", allow_duplicate=True),
    Output("plate-viewer-modal", "style", allow_duplicate=True),
    Output("plate-viewer-active-well", "data", allow_duplicate=True),
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
    nu = dash.no_update
    if not recording_key:
        return nu, nu, "Pick a recording first.", nu, nu
    ctx_app = current_app.config.get("YUXIN_MEA", {})
    analysis_root = ctx_app.get("analysis_root")
    cache_root = ctx_app.get("cache_root")
    if analysis_root is None:
        return nu, nu, "analysis_root is not set in the config.", nu, nu
    source_key = str(source or "traditional")
    settings = _settings_from(display_mode, marker_size, line_width,
                              width_px, max_raster, max_sync)
    try:
        overview, context, status = _render_overview_outputs(
            analysis_root, cache_root, recording_key, source_key, settings,
        )
    except Exception as exc:  # noqa: BLE001 — surface user-facing
        return nu, nu, f"❌ {exc}", nu, nu
    # A fresh Load closes any open modal and clears the active well.
    return overview, context, status, _MODAL_HIDDEN, None


@callback(
    Output("plate-viewer-recording-dropdown", "value", allow_duplicate=True),
    Output("plate-viewer-overview", "children", allow_duplicate=True),
    Output("plate-viewer-context", "data", allow_duplicate=True),
    Output("plate-viewer-status", "children", allow_duplicate=True),
    Output("plate-viewer-modal", "style", allow_duplicate=True),
    Output("plate-viewer-active-well", "data", allow_duplicate=True),
    Input("plate-viewer-prev-plate", "n_clicks"),
    Input("plate-viewer-next-plate", "n_clicks"),
    State("plate-viewer-recording-dropdown", "value"),
    State("plate-viewer-recording-dropdown", "options"),
    State("plate-viewer-source", "value"),
    State("plate-viewer-display-mode", "value"),
    State("plate-viewer-marker-size", "value"),
    State("plate-viewer-line-width", "value"),
    State("plate-viewer-width-px", "value"),
    State("plate-viewer-max-raster", "value"),
    State("plate-viewer-max-sync", "value"),
    prevent_initial_call=True,
)
def _on_plate_nav(_p, _n, current, options, source, display_mode, marker_size,
                  line_width, width_px, max_raster, max_sync):
    """Step the recording dropdown ±1 within its filtered options and re-render."""
    nu = dash.no_update
    trig = ctx.triggered[0] if ctx.triggered else None
    if not trig or not trig.get("value"):
        return nu, nu, nu, nu, nu, nu
    opt_values = [o["value"] for o in (options or [])]
    if not opt_values:
        return nu, nu, nu, "No recordings to navigate.", nu, nu
    idx = opt_values.index(current) if current in opt_values else 0
    idx = max(0, idx - 1) if ctx.triggered_id == "plate-viewer-prev-plate" \
        else min(len(opt_values) - 1, idx + 1)
    new_key = opt_values[idx]
    if new_key == current:
        return nu, nu, nu, nu, nu, nu  # already at the end — nothing to do

    ctx_app = current_app.config.get("YUXIN_MEA", {})
    analysis_root = ctx_app.get("analysis_root")
    cache_root = ctx_app.get("cache_root")
    if analysis_root is None:
        return nu, nu, nu, "analysis_root is not set in the config.", nu, nu
    source_key = str(source or "traditional")
    settings = _settings_from(display_mode, marker_size, line_width,
                              width_px, max_raster, max_sync)
    try:
        overview, context, status = _render_overview_outputs(
            analysis_root, cache_root, new_key, source_key, settings,
        )
    except Exception as exc:  # noqa: BLE001
        return new_key, nu, nu, f"❌ {exc}", nu, nu
    return new_key, overview, context, status, _MODAL_HIDDEN, None


def _well_meta_row(recording_key: str, groupname: str | None,
                   prov_status: str | None = None) -> html.Div:
    """Pill row of recording metadata shown atop the single-well modal.

    Recording-level fields come from splitting ``recording_key``
    (``sample_id/date/plate_id/scan_type/run_id``); ``groupname`` is per-well.
    A provenance badge is appended when the recording has a verified task.
    """
    parts = str(recording_key).split("/")
    keys = ("sample_id", "date", "plate_id", "scan_type", "run_id")
    meta = dict(zip(keys, parts))
    date_disp = yymmdd_to_iso(meta.get("date")) or meta.get("date", "?")
    fields = [
        ("sample", meta.get("sample_id", "?")),
        ("run", meta.get("run_id", "?")),
        ("date", date_disp),
        ("plate", meta.get("plate_id", "?")),
        ("type", meta.get("scan_type", "?")),
        ("group", groupname or "?"),
    ]
    pills = [
        html.Span(
            [
                html.Span(f"{label} ", style={"color": "var(--ink-3)"}),
                html.Span(str(val), style={"color": "var(--ink)", "fontWeight": "600"}),
            ],
            style={"fontFamily": "var(--font-mono)", "fontSize": "11px",
                   "padding": "2px 8px", "border": "1px solid var(--line)",
                   "borderRadius": "999px", "background": "var(--bg)",
                   "whiteSpace": "nowrap"},
        )
        for label, val in fields
    ]
    badge = provenance_badge(prov_status)
    if badge is not None:
        pills.append(badge)
    return html.Div(pills, style={"display": "flex", "gap": "6px",
                                  "flexWrap": "wrap", "marginBottom": "10px"})


@callback(
    Output("plate-viewer-modal-body", "children"),
    Output("plate-viewer-modal", "style", allow_duplicate=True),
    Output("plate-viewer-modal-title", "children"),
    Output("plate-viewer-active-well", "data", allow_duplicate=True),
    Input({"type": "pv-well-cell", "index": ALL}, "n_clicks"),
    Input("plate-viewer-prev-well", "n_clicks"),
    Input("plate-viewer-next-well", "n_clicks"),
    State("plate-viewer-active-well", "data"),
    State("plate-viewer-context", "data"),
    prevent_initial_call=True,
)
def _open_well(_cell_clicks, _prev, _next, active_well, context):
    """Open (or step) the single-well modal, building the figure fresh.

    Triggered by a well-cell click or the prev/next-well buttons. Mounts the
    ``dcc.Graph`` into the now-visible modal so Plotly sizes it correctly.
    """
    nu = dash.no_update
    trig = ctx.triggered[0] if ctx.triggered else None
    tid = ctx.triggered_id
    # Ignore the all-zero fire when the overview grid is (re)created.
    if not trig or not trig.get("value") or not context:
        return nu, nu, nu, nu

    analysis_root = current_app.config.get("YUXIN_MEA", {}).get("analysis_root")
    if analysis_root is None:
        return nu, nu, nu, nu

    ok_wells = context.get("ok_wells", [])
    if isinstance(tid, dict):  # a cell was clicked
        target = tid["index"]
    else:  # prev/next-well button — step from the active well
        if not ok_wells:
            return nu, nu, nu, nu
        cur = active_well if active_well in ok_wells else ok_wells[0]
        i = ok_wells.index(cur)
        i = max(0, i - 1) if tid == "plate-viewer-prev-well" \
            else min(len(ok_wells) - 1, i + 1)
        target = ok_wells[i]

    from yuxin_mea.analysis.plate_raster_synchrony import build_single_well_figure
    try:
        records, _bwd, _cwd = _load_records(
            analysis_root, context["recording_key"], context["source"],
        )
        wr = next((r for r in records if r.well_id == target), None)
        if wr is None:
            return nu, nu, nu, nu
        fig = build_single_well_figure(wr, _build_config(**context["settings"]))
    except Exception:  # noqa: BLE001
        return nu, nu, nu, nu

    graph = dcc.Graph(figure=fig, style={"height": "72vh"}, config={"responsive": True})
    groupname = getattr(wr, "groupname", None)
    body = html.Div([
        _well_meta_row(context["recording_key"], groupname, context.get("provenance")),
        graph,
    ])
    return body, _MODAL_SHOWN, f"{wr.well_name} ({wr.well_id})", target


@callback(
    Output("plate-viewer-modal", "style", allow_duplicate=True),
    Input("plate-viewer-modal-backdrop", "n_clicks"),
    Input("plate-viewer-modal-close", "n_clicks"),
    prevent_initial_call=True,
)
def _close_modal(_backdrop, _close):
    """Close the modal on an outside (backdrop) click or the ✕ button."""
    trig = ctx.triggered[0] if ctx.triggered else None
    if not trig or not trig.get("value"):
        return dash.no_update
    return _MODAL_HIDDEN


# One-time document keydown listener: ←/→ step the open well, Shift+←/→ switch
# plate. It just synthesises clicks on the nav buttons, so all logic stays in the
# server callbacks above. Guarded so typing in a filter input isn't hijacked.
dash.clientside_callback(
    """
    function(_id) {
        if (window.__pvKeysBound) { return ''; }
        window.__pvKeysBound = true;
        document.addEventListener('keydown', function(e) {
            var t = e.target;
            if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) { return; }
            if (t && t.closest && t.closest('.Select, .dash-dropdown, [role=combobox]')) { return; }
            if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') { return; }
            var modal = document.getElementById('plate-viewer-modal');
            var open = modal && modal.style.display && modal.style.display !== 'none';
            function click(id) { var el = document.getElementById(id); if (el) { el.click(); } }
            if (e.key === 'ArrowLeft') {
                if (e.shiftKey) { e.preventDefault(); click('plate-viewer-prev-plate'); }
                else if (open) { e.preventDefault(); click('plate-viewer-prev-well'); }
            } else {
                if (e.shiftKey) { e.preventDefault(); click('plate-viewer-next-plate'); }
                else if (open) { e.preventDefault(); click('plate-viewer-next-well'); }
            }
        });
        return '';
    }
    """,
    Output("plate-viewer-keydummy", "children"),
    Input("plate-viewer-keydummy", "id"),
)


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
    from yuxin_mea.analysis.plate_raster_synchrony import write_plate_viewer_html
    try:
        fig = _render_plate_figure(analysis_root, recording_key, settings, source_key)
        output_path = Path(figure_root) / recording_key / f"plate_viewer_{source_key}.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_plate_viewer_html(fig, output_path)
    except Exception as exc:  # noqa: BLE001
        return f"❌ Export failed: {exc}"
    return f"✓ Saved to {output_path}"
