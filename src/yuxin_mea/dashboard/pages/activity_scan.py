"""Activity scan page — per-electrode activity maps for an ActivityScan.

Navigation mirrors the Plate viewer / Network inspector: pick an ActivityScan
recording, the 4×6 plate grid paints each well's chosen map as a thumbnail, and
clicking a well zooms into full-resolution maps + histograms + stats.

Unlike a Network recording, an ActivityScan carries no spike sorting — activity
is measured directly from the raw traces by MAD-threshold peak detection and a
well is *assembled* across its 14–29 electrode configurations. All data comes
from :mod:`yuxin_mea.analysis.activity_scan_inspector` (plain
``pd.read_parquet`` / ``np.load`` / ``json`` — no SpikeInterface, no torch). The
dashboard never computes: a well with no output renders an empty state that
names the ``run_activity_scan.py`` command that would produce it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import dash
from dash import ALL, Input, Output, State, callback, ctx, dcc, html
from flask import current_app

from yuxin_mea.dashboard.components import (
    build_filter_bar,
    filter_id,
    filter_kwargs,
    iso_to_yymmdd,
    yymmdd_to_iso,
)

dash.register_page(__name__, path="/activity-scan", name="Activity scan", order=8)

_FILTER_FIELDS = ("sample", "date", "group")

_MONO = {"fontFamily": "var(--font-mono)", "fontSize": "11px", "color": "var(--ink-3)"}

_MODAL_HIDDEN = {"display": "none"}
_MODAL_SHOWN = {
    "display": "flex", "position": "fixed", "inset": 0, "zIndex": 1000,
    "alignItems": "center", "justifyContent": "center",
}

_GRID_STYLE = {"display": "grid", "gridTemplateColumns": "repeat(6, 1fr)",
               "gap": "6px", "marginTop": "16px"}
_CELL_BASE = {
    "padding": "6px 4px", "border": "1px solid var(--line)", "borderRadius": "6px",
    "cursor": "pointer", "display": "flex", "flexDirection": "column",
    "alignItems": "center", "gap": "3px", "background": "var(--bg)",
}
_CELL_EMPTY = {**_CELL_BASE, "cursor": "default", "background": "var(--bg-deep)",
               "color": "var(--ink-4)", "minHeight": "104px",
               "justifyContent": "center"}


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def _threshold_slider() -> html.Div:
    marks = {0.05: "0.05", 0.1: "0.1", 0.25: "0.25", 0.5: "0.5", 1.0: "1", 2.0: "2"}
    return html.Div([
        html.Label("Active cut (Hz)", className="section-label",
                   style={"marginBottom": "0"}),
        html.Div(
            dcc.Slider(id="as-threshold", min=0.05, max=2.0, step=None, value=0.1,
                       marks={k: {"label": v, "style": {"fontSize": "9px"}}
                              for k, v in marks.items()},
                       included=False),
            style={"flex": "1 1 300px", "paddingTop": "6px"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "10px",
              "flexWrap": "wrap", "margin": "4px 0"})


def _root_panel() -> html.Details:
    return html.Details([
        html.Summary("output root", className="section-label",
                     style={"cursor": "pointer"}),
        html.Div([
            html.Label("activity_scan_data", style={**_MONO, "flex": "0 0 130px"}),
            dcc.Input(id="as-root", type="text", debounce=True, value="",
                      placeholder="<analysis_root>/activity_scan_data",
                      style={"flex": "1 1 420px", "fontFamily": "var(--font-mono)",
                             "fontSize": "11px", "padding": "4px 8px",
                             "background": "var(--bg)", "color": "var(--ink)",
                             "border": "1px solid var(--line)", "borderRadius": "4px"}),
            html.Span(id="as-root-status", style={**_MONO, "flex": "0 0 150px"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px",
                  "padding": "10px 4px 2px", "flexWrap": "wrap"}),
    ], open=False)


layout = html.Div([
    html.Div(
        html.Div([
            html.Div([
                html.Span("workspace"), html.Span("analysis"),
                html.Span("activity scan"),
            ], className="breadcrumb"),
            html.H1("Activity scan"),
            html.Div(
                "Per-electrode firing rate, spike amplitude and noise from the "
                "raw checkerboard sweep — MAD-threshold peak detection, assembled "
                "across each well's configurations. Reads precomputed outputs; "
                "nothing is computed here.",
                className="subtitle"),
        ]),
        className="view-head",
    ),

    html.Div(id="as-banner"),
    build_filter_bar("activity", show=_FILTER_FIELDS),

    html.Div([
        html.Label("Recording", className="section-label", style={"marginBottom": "0"}),
        html.Div(dcc.Dropdown(id="as-rec", options=[], value=None, clearable=False),
                 style={"flex": "1 1 380px"}),
        html.Label("Map", className="section-label",
                   style={"marginBottom": "0", "marginLeft": "8px"}),
        html.Div(dcc.Dropdown(id="as-map", options=[], value="firing_rate",
                              clearable=False), style={"flex": "0 1 220px"}),
        html.Button([html.Span("↻", className="glyph"), "Load"], id="as-load",
                    n_clicks=0, className="btn primary"),
        html.Button("Rescan", id="as-rescan", n_clicks=0, className="btn"),
        html.Span(id="as-status", style=_MONO),
    ], style={"display": "flex", "alignItems": "center", "gap": "8px",
              "marginBottom": "8px", "flexWrap": "wrap"}),

    _threshold_slider(),
    _root_panel(),

    dcc.Store(id="as-context"),
    dcc.Store(id="as-active-well", data=None),

    html.Div([
        html.Button([html.Span("‹", className="glyph"), "Previous well"],
                    id="as-prev-well", n_clicks=0, className="btn"),
        html.Button(["Next well", html.Span("›", className="glyph")],
                    id="as-next-well", n_clicks=0, className="btn"),
        html.Span(id="as-legend", style={**_MONO, "marginLeft": "12px"}),
    ], style={"display": "flex", "gap": "6px", "alignItems": "center",
              "flexWrap": "wrap", "margin": "12px 0 0"}),

    dcc.Loading(
        html.Div("Load an ActivityScan recording to paint the plate. Click a "
                 "well to zoom in.",
                 id="as-grid", style={"marginTop": "16px", "color": "var(--ink-3)"}),
        type="default",
    ),

    html.Div([
        html.Div(id="as-modal-backdrop", n_clicks=0,
                 style={"position": "absolute", "inset": 0,
                        "background": "rgba(0,0,0,0.55)"}),
        html.Div([
            html.Div([
                html.Span(id="as-modal-title",
                          style={"fontFamily": "var(--font-mono)", "fontSize": "13px",
                                 "fontWeight": "600", "color": "var(--ink)"}),
                html.Button("✕", id="as-modal-close", n_clicks=0, className="btn",
                            style={"marginLeft": "auto"},
                            title="Close (or click outside)"),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px",
                      "marginBottom": "8px"}),
            dcc.Loading(html.Div(id="as-modal-body"), type="default"),
        ], style={"position": "relative", "zIndex": 1, "background": "var(--bg)",
                  "border": "1px solid var(--line)", "borderRadius": "10px",
                  "padding": "14px", "width": "92vw", "maxWidth": "1400px",
                  "maxHeight": "92vh", "overflow": "auto",
                  "boxShadow": "0 12px 40px rgba(0,0,0,0.4)"}),
    ], id="as-modal", style=_MODAL_HIDDEN),
], className="page")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ctx() -> dict[str, Any]:
    return current_app.config.get("YUXIN_MEA", {})


def _root_override(raw: str | None) -> str | None:
    raw = (raw or "").strip()
    return raw or None


def _graph(fig, flex: str = "1 1 440px") -> html.Div:
    return html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False}),
                    style={"flex": flex})


def _row(*children) -> html.Div:
    return html.Div(list(children),
                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})


def _stats_card(rows: list[dict[str, str]]) -> html.Div:
    if not rows:
        return html.Div()
    trs = [html.Tr([
        html.Td(r["metric"], style={"padding": "3px 14px 3px 0",
                                    "color": "var(--ink-3)",
                                    "fontFamily": "var(--font-mono)",
                                    "fontSize": "11px", "whiteSpace": "nowrap"}),
        html.Td(r["value"], style={"padding": "3px 0",
                                   "fontFamily": "var(--font-mono)",
                                   "fontSize": "11px"}),
    ]) for r in rows]
    half = (len(trs) + 1) // 2
    return html.Div([
        html.Div(html.Span("well statistics", className="h-title"),
                 className="card-head"),
        html.Div([
            html.Table(html.Tbody(trs[:half]), style={"borderCollapse": "collapse"}),
            html.Table(html.Tbody(trs[half:]), style={"borderCollapse": "collapse"}),
        ], className="card-body",
            style={"display": "flex", "gap": "28px", "flexWrap": "wrap"}),
    ], className="card", style={"marginBottom": "12px"})


def _empty_command(rec_key: str) -> str:
    return ("python scripts/run_activity_scan.py --config pipeline_config_local.json "
            f"--recordings {rec_key}")


def _modal_body(vi, results, rec_key: str, well_id: str, threshold: float,
                map_metric: str):
    if results is None:
        return html.Div([
            html.Div(f"No activity-scan output for {well_id}.", style=_MONO),
            html.Div("Compute it with:", style={**_MONO, "marginTop": "8px"}),
            html.Code(_empty_command(rec_key),
                      style={"display": "block", "marginTop": "4px",
                             "fontSize": "11px", "color": "var(--ink)",
                             "background": "var(--bg-deep)", "padding": "8px",
                             "borderRadius": "6px", "wordBreak": "break-all"}),
        ])
    return html.Div([
        _stats_card(vi.well_stats_table(results)),
        _row(_graph(vi.fig_map(results, map_metric, threshold), flex="2 1 560px"),
             _graph(vi.fig_active_area_curve(results), flex="1 1 320px")),
        _row(_graph(vi.fig_histogram(results, "firing_rate")),
             _graph(vi.fig_histogram(results, "amplitude")),
             _graph(vi.fig_histogram(results, "noise"))),
        _graph(vi.fig_config_coverage(results), flex="1 1 100%"),
    ])


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
@callback(
    Output("as-banner", "children"),
    Output("as-rec", "options"),
    Output("as-rec", "value"),
    Output("as-status", "children"),
    Output(filter_id("activity", "sample"), "options"),
    Output(filter_id("activity", "date"), "min_date_allowed"),
    Output(filter_id("activity", "date"), "max_date_allowed"),
    Output(filter_id("activity", "group"), "options"),
    Input("as-rescan", "n_clicks"),
    Input(filter_id("activity", "sample"), "value"),
    Input(filter_id("activity", "date"), "start_date"),
    Input(filter_id("activity", "date"), "end_date"),
    Input(filter_id("activity", "group"), "value"),
    # `as-root` is a State, not an Input: the prefill callback writes it whenever
    # the recording changes, so making it an Input would form a two-callback
    # cycle (rec → root → rec). Editing the root then hitting Rescan re-discovers.
    State("as-root", "value"),
    State("as-rec", "value"),
)
def _populate_recordings(_n, f_sample, f_from, f_to, f_group, root, current_rec):
    from yuxin_mea.analysis import activity_scan_inspector as vi
    from yuxin_mea.dashboard.components import no_config_banner
    from yuxin_mea.dashboard.data import filter_recordings, load_recordings_detail

    ctx_app = _ctx()
    banner = None if ctx_app.get("config_exists") else no_config_banner()
    analysis_root = ctx_app.get("analysis_root")
    empty: list[dict] = []
    if analysis_root is None:
        return (banner, [], None, "analysis_root not set.", empty, None, None, empty)

    override = _root_override(root)
    computed = set(vi.recordings_with_activity_scan(analysis_root, override))
    recordings, well_status = load_recordings_detail(Path(analysis_root))
    with_out = [r for r in recordings if r["cache_key"] in computed]
    sample_opts = sorted({r["sample_id"] for r in with_out})
    date_opts = sorted({r["date"] for r in with_out})
    group_opts = sorted({g for r in with_out for g in r.get("groups", [])})

    kwargs = filter_kwargs({
        "sample": f_sample,
        "date-from": iso_to_yymmdd(f_from), "date-to": iso_to_yymmdd(f_to),
        "group": f_group,
    })
    filtered = filter_recordings(recordings, well_status, scan_types=["ActivityScan"],
                                 **kwargs)
    rec_keys = sorted(r["cache_key"] for r in filtered if r["cache_key"] in computed)
    options = [{"label": k, "value": k} for k in rec_keys]
    value = current_rec if current_rec in rec_keys else (rec_keys[0] if rec_keys else None)
    status = ("No activity-scan output found — run scripts/run_activity_scan.py."
              if not computed else
              f"{len(rec_keys)} of {len(computed)} computed recording(s) match.")
    return (
        banner, options, value, status,
        [{"label": s, "value": s} for s in sample_opts],
        yymmdd_to_iso(date_opts[0]) if date_opts else None,
        yymmdd_to_iso(date_opts[-1]) if date_opts else None,
        [{"label": g, "value": g} for g in group_opts],
    )


@callback(
    Output("as-map", "options"),
    Output("as-root", "value"),
    Output("as-root-status", "children"),
    Input("as-rec", "value"),
    State("as-root", "value"),
)
def _map_options_and_root(_rec, root):
    from yuxin_mea.analysis import activity_scan_inspector as vi

    opts = [{"label": vi.MAP_SPECS[m]["label"], "value": m} for m in vi.MAP_ORDER]
    analysis_root = _ctx().get("analysis_root")
    override = _root_override(root)
    if analysis_root is None:
        return opts, root or "", "analysis_root not set"
    resolved = vi.output_root(analysis_root, override)
    note = ("custom" if override else "default") + \
           (" · exists" if resolved.exists() else " · missing")
    return opts, str(resolved), note


@callback(
    Output("as-grid", "children"),
    Output("as-context", "data"),
    Output("as-legend", "children"),
    Input("as-load", "n_clicks"),
    Input("as-rec", "value"),
    Input("as-map", "value"),
    Input("as-threshold", "value"),
    State("as-root", "value"),
)
def _render_grid(_n, rec_key, map_metric, threshold, root):
    from yuxin_mea.analysis import activity_scan_inspector as vi

    analysis_root = _ctx().get("analysis_root")
    if not (rec_key and analysis_root and map_metric):
        return ("Load an ActivityScan recording to paint the plate.", None, "")

    override = _root_override(root)
    cache_root = _ctx().get("cache_root")
    try:
        wells = vi.wells_in_recording(analysis_root, rec_key, override)
    except Exception as exc:  # noqa: BLE001 — surface, never 500 the page
        return (html.Div(f"Failed to read outputs: {exc}", style=_MONO), None, "")
    if not wells:
        return (html.Div("No activity-scan output for this recording.", style=_MONO),
                None, "")

    spec = vi.MAP_SPECS[map_metric]
    # Load every well once, then paint all thumbnails on ONE shared colour scale
    # so the plate reads as a cross-well comparison (a quiet well must look dim
    # next to an active one, not self-normalised to the same brightness).
    loaded = {well: vi.load_well(analysis_root, rec_key, well, override)
              for well in wells}
    vrange = vi.plate_value_range(list(loaded.values()), map_metric)

    cells = []
    for well in wells:
        rdir = vi.well_dir(analysis_root, rec_key, well, override)
        results = loaded[well]
        uri = ""
        subtitle = "n/a"
        if results is not None:
            if cache_root:
                uri = vi.well_png_data_uri(results, map_metric, Path(cache_root),
                                           rdir, threshold, vrange)
            subtitle = vi.well_subtitle(results, map_metric, threshold)
        thumb = (html.Img(src=uri, style={"width": "100%", "borderRadius": "4px",
                                          "background": "var(--bg-deep)"})
                 if uri else
                 html.Div("no map", style={"height": "70px", "display": "flex",
                                           "alignItems": "center",
                                           "justifyContent": "center",
                                           **_MONO}))
        cells.append(html.Button(
            [html.Div([html.Span(vi.well_name(well),
                                 style={"fontWeight": "600", "fontSize": "11px"}),
                       html.Span(well, style={"fontSize": "9px",
                                              "color": "var(--ink-3)",
                                              "marginLeft": "4px"})],
                      style={"display": "flex", "alignItems": "baseline",
                             "gap": "2px"}),
             thumb,
             html.Div(subtitle, style={"fontSize": "10px", "color": "var(--ink-3)"})],
            id={"type": "as-well-cell", "index": well}, n_clicks=0,
            style=_CELL_BASE, title=f"Open {vi.well_name(well)} ({well})",
        ))

    if map_metric == "active":
        legend = f"{spec['label']} · {len(wells)} well(s) · active cut {threshold:g} Hz"
    elif vrange is not None:
        legend = (f"{spec['label']} · {len(wells)} well(s) · shared scale "
                  f"{vrange[0]:.3g}–{vrange[1]:.3g} {spec['unit']}".rstrip())
    else:
        legend = f"{spec['label']} · {len(wells)} well(s)"
    context = {"recording_key": rec_key, "root": override or "",
               "wells": wells, "threshold": threshold, "map_metric": map_metric}
    return html.Div(cells, style=_GRID_STYLE), context, legend


@callback(
    Output("as-modal-body", "children"),
    Output("as-modal", "style", allow_duplicate=True),
    Output("as-modal-title", "children"),
    Output("as-active-well", "data", allow_duplicate=True),
    Input({"type": "as-well-cell", "index": ALL}, "n_clicks"),
    Input("as-prev-well", "n_clicks"),
    Input("as-next-well", "n_clicks"),
    State("as-active-well", "data"),
    State("as-context", "data"),
    prevent_initial_call=True,
)
def _open_well(_cells, _prev, _next, active_well, context):
    from yuxin_mea.analysis import activity_scan_inspector as vi

    nu = dash.no_update
    trig = ctx.triggered[0] if ctx.triggered else None
    tid = ctx.triggered_id
    if not trig or not trig.get("value") or not context:
        return nu, nu, nu, nu

    analysis_root = _ctx().get("analysis_root")
    if analysis_root is None:
        return nu, nu, nu, nu

    wells = context.get("wells") or []
    if isinstance(tid, dict):
        target = tid["index"]
    else:
        if not wells:
            return nu, nu, nu, nu
        cur = active_well if active_well in wells else wells[0]
        i = wells.index(cur)
        i = max(0, i - 1) if tid == "as-prev-well" else min(len(wells) - 1, i + 1)
        target = wells[i]

    override = _root_override(context.get("root"))
    threshold = context.get("threshold", 0.1)
    map_metric = context.get("map_metric", "firing_rate")
    rec_key = context["recording_key"]
    try:
        results = vi.load_well(analysis_root, rec_key, target, override)
    except Exception as exc:  # noqa: BLE001
        return (html.Div(f"Failed to load well: {exc}", style=_MONO),
                _MODAL_SHOWN, target, target)

    body = _modal_body(vi, results, rec_key, target, threshold, map_metric)
    title = f"{rec_key} · {vi.well_name(target)} ({target})"
    return body, _MODAL_SHOWN, title, target


@callback(
    Output("as-modal", "style", allow_duplicate=True),
    Input("as-modal-backdrop", "n_clicks"),
    Input("as-modal-close", "n_clicks"),
    prevent_initial_call=True,
)
def _close_modal(_backdrop, _close):
    trig = ctx.triggered[0] if ctx.triggered else None
    if not trig or not trig.get("value"):
        return dash.no_update
    return _MODAL_HIDDEN
