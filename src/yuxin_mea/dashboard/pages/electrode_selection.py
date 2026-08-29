"""Electrode selection QC — does a well keep recording the same sites over time?

A MaxTwo Network run routes only ~1,000 of 26,400 electrodes and re-picks them
every time, so "the same well" on two dates need not mean the same recording
sites. This page paints, per well, how many scans selected each electrode, and
drills into the stability metrics behind that.

Read-only, like every other dashboard page: it loads what
``scripts/run_electrode_selection_qc.py`` wrote and computes nothing. A well
with no output — or with no animation rendered — shows the command that would
produce it rather than shelling out.

Figures follow the house two-tier split: server-rendered PNG thumbnails for the
4x6 plate grid (cheap, cached by content hash), a live Plotly heatmap in the
drill-in modal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import dash
from dash import ALL, Input, Output, State, callback, ctx, dcc, html
from flask import current_app

from yuxin_mea.dashboard.components.layout import no_config_banner

dash.register_page(__name__, path="/electrode-selection",
                   name="Electrode selection", order=9)

_MONO = {"fontFamily": "var(--font-mono)", "fontSize": "11px", "color": "var(--ink-3)"}

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

_MODAL_HIDDEN = {"display": "none"}
_MODAL_SHOWN = {
    "display": "flex", "position": "fixed", "inset": 0, "zIndex": 1000,
    "alignItems": "center", "justifyContent": "center",
    "background": "rgba(0,0,0,0.35)",
}

_VERDICT_COLOR = {
    "stable": "#1f5a3f", "partial": "#a17a2a", "weak": "#cf7d54",
    "unstable": "#b2182b", "insufficient": "var(--ink-3)",
}


def _ctx() -> dict[str, Any]:
    return current_app.config.get("YUXIN_MEA", {})


def _root_override(raw: str | None) -> str | None:
    raw = (raw or "").strip()
    return raw or None


def _resolved_root(root: str | None):
    from yuxin_mea.analysis import electrode_selection_inspector as VI

    analysis_root = _ctx().get("analysis_root")
    if analysis_root is None:
        return None
    return VI.output_root(analysis_root, _root_override(root))


def _root_panel() -> html.Details:
    return html.Details([
        html.Summary("output root", className="section-label",
                     style={"cursor": "pointer"}),
        html.Div([
            html.Label("electrode_selection_qc", style={**_MONO, "flex": "0 0 160px"}),
            dcc.Input(id="es-root", type="text", debounce=True, value="",
                      placeholder="<analysis_root>/electrode_selection_qc",
                      style={"flex": "1 1 420px", "fontFamily": "var(--font-mono)",
                             "fontSize": "11px", "padding": "4px 8px",
                             "background": "var(--bg)", "color": "var(--ink)",
                             "border": "1px solid var(--line)", "borderRadius": "4px"}),
            html.Span(id="es-root-status", style={**_MONO, "flex": "0 0 150px"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px",
                  "padding": "10px 4px 2px", "flexWrap": "wrap"}),
    ], open=False)


layout = html.Div([
    html.Div(
        html.Div([
            html.Div([html.Span("workspace"), html.Span("analysis"),
                      html.Span("electrode selection")], className="breadcrumb"),
            html.H1("Electrode selection"),
            html.Div(
                "How often each electrode was routed across a well's Network "
                "scans. Only ~1,000 of 26,400 electrodes record at once and "
                "every run re-picks them, so this is the QC for whether "
                "longitudinal comparisons are looking at the same sites. Reads "
                "precomputed outputs; nothing is computed here.",
                className="subtitle"),
        ]),
        className="view-head",
    ),

    html.Div(id="es-banner"),

    html.Div([
        html.Label("Sample", className="section-label", style={"marginBottom": "0"}),
        html.Div(dcc.Dropdown(id="es-sample", options=[], value=None,
                              clearable=False), style={"flex": "1 1 200px"}),
        html.Label("Chip", className="section-label",
                   style={"marginBottom": "0", "marginLeft": "8px"}),
        html.Div(dcc.Dropdown(id="es-plate", options=[], value=None,
                              clearable=False), style={"flex": "1 1 200px"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "8px",
              "flexWrap": "wrap", "margin": "10px 0"}),

    _root_panel(),
    html.Div(id="es-legend", style={**_MONO, "marginTop": "10px"}),
    html.Div(id="es-grid"),
    dcc.Store(id="es-context"),

    html.Div(
        html.Div([
            html.Div([
                html.Strong(id="es-modal-title"),
                html.Button("close", id="es-close", n_clicks=0,
                            style={"marginLeft": "auto", "cursor": "pointer"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px",
                      "marginBottom": "8px"}),
            html.Div(id="es-modal-body"),
        ], style={"background": "var(--bg)", "padding": "16px",
                  "borderRadius": "8px", "width": "min(1100px, 94vw)",
                  "maxHeight": "92vh", "overflowY": "auto",
                  "border": "1px solid var(--line)"}),
        id="es-modal", style=_MODAL_HIDDEN),
], className="view")


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
@callback(
    Output("es-banner", "children"),
    Output("es-sample", "options"),
    Output("es-sample", "value"),
    Output("es-root-status", "children"),
    Input("es-root", "value"),
)
def _populate(root):
    from yuxin_mea.analysis import electrode_selection_inspector as VI

    app_ctx = _ctx()
    banner = None if app_ctx.get("config_exists") else no_config_banner()
    resolved = _resolved_root(root)
    if resolved is None:
        return banner, [], None, "analysis_root not set"
    note = ("custom" if _root_override(root) else "default") + \
           (" · exists" if resolved.exists() else " · missing")
    found = VI.samples(resolved)
    options = [{"label": s, "value": s} for s in found]
    return banner, options, (found[0] if found else None), note


@callback(
    Output("es-plate", "options"),
    Output("es-plate", "value"),
    Input("es-sample", "value"),
    State("es-root", "value"),
)
def _plates(sample_id, root):
    from yuxin_mea.analysis import electrode_selection_inspector as VI

    resolved = _resolved_root(root)
    if not (sample_id and resolved):
        return [], None
    found = VI.plates(resolved, sample_id)
    return [{"label": p, "value": p} for p in found], (found[0] if found else None)


@callback(
    Output("es-grid", "children"),
    Output("es-legend", "children"),
    Output("es-context", "data"),
    Input("es-sample", "value"),
    Input("es-plate", "value"),
    State("es-root", "value"),
)
def _render_grid(sample_id, plate_id, root):
    from yuxin_mea.analysis import electrode_selection_inspector as VI

    resolved = _resolved_root(root)
    if not (sample_id and plate_id and resolved):
        return (html.Div("Pick a sample and chip.", style=_MONO), "", None)

    found = VI.wells(resolved, sample_id, plate_id)
    if not found:
        return (html.Div("No electrode-selection QC for this chip. Run "
                         "scripts/run_electrode_selection_qc.py first.",
                         style=_MONO), "", None)

    cache_root = _ctx().get("cache_root")
    cells, windows = [], set()
    for well in found:
        payload = VI.load_well(resolved, sample_id, plate_id, well)
        wdir = VI.well_dir(resolved, sample_id, plate_id, well)
        uri = ""
        if payload is not None and cache_root:
            uri = VI.count_png_data_uri(payload, Path(cache_root), wdir)
        stab = (payload or {}).get("stability") or {}
        verdict = str(stab.get("verdict") or "—")
        windows.add(VI.window_label(payload))

        thumb = (html.Img(src=uri, style={"width": "100%", "borderRadius": "4px",
                                          "background": "#ffffff"})
                 if uri else
                 html.Div("no map", style={"height": "66px", "display": "flex",
                                           "alignItems": "center",
                                           "justifyContent": "center", **_MONO}))
        cells.append(html.Button(
            [html.Div([html.Span(VI.well_name(well),
                                 style={"fontWeight": "600", "fontSize": "11px"}),
                       html.Span(well, style={"fontSize": "9px",
                                              "color": "var(--ink-3)",
                                              "marginLeft": "4px"})],
                      style={"display": "flex", "alignItems": "baseline",
                             "gap": "2px"}),
             thumb,
             html.Div(f"{stab.get('n_scans', 0)} scans · {verdict}",
                      style={"fontSize": "10px",
                             "color": _VERDICT_COLOR.get(verdict, "var(--ink-3)")})],
            id={"type": "es-well-cell", "index": well}, n_clicks=0,
            style=_CELL_BASE if payload else _CELL_EMPTY,
            title=f"Open {VI.well_name(well)} ({well})"))

    window = windows.pop() if len(windows) == 1 else "mixed windows"
    legend = (f"{len(found)} well(s) · window {window} · thumbnails show the "
              f"fraction of each well's own scans (0–1), so wells with "
              f"different scan counts stay comparable")
    context = {"sample_id": sample_id, "plate_id": plate_id,
               "root": _root_override(root) or "", "wells": found}
    return html.Div(cells, style=_GRID_STYLE), legend, context


@callback(
    Output("es-modal-body", "children"),
    Output("es-modal", "style"),
    Output("es-modal-title", "children"),
    Input({"type": "es-well-cell", "index": ALL}, "n_clicks"),
    Input("es-close", "n_clicks"),
    State("es-context", "data"),
    prevent_initial_call=True,
)
def _open_well(_cells, _close, context):
    from yuxin_mea.analysis import electrode_selection_inspector as VI

    nu = dash.no_update
    trig = ctx.triggered[0] if ctx.triggered else None
    if not trig or not trig.get("value"):
        return nu, nu, nu
    if ctx.triggered_id == "es-close" or not context:
        return nu, _MODAL_HIDDEN, nu

    well = ctx.triggered_id["index"]
    sample_id, plate_id = context["sample_id"], context["plate_id"]
    resolved = _resolved_root(context.get("root"))
    payload = VI.load_well(resolved, sample_id, plate_id, well)
    if payload is None:
        return (html.Div("No output for this well.", style=_MONO),
                _MODAL_SHOWN, f"{sample_id} · {plate_id} · {well}")

    stab = payload.get("stability") or {}
    verdict = str(stab.get("verdict") or "—")
    wdir = VI.well_dir(resolved, sample_id, plate_id, well)
    anim = next((p for p in (wdir / "selection_history.mp4",
                             wdir / "selection_history.gif") if p.exists()), None)

    body = [
        html.Div([
            html.Span(verdict.upper(),
                      style={"fontWeight": "700",
                             "color": _VERDICT_COLOR.get(verdict, "var(--ink)")}),
            html.Span(stab.get("verdict_text", ""),
                      style={"marginLeft": "8px", "fontSize": "12px"}),
        ], style={"marginBottom": "10px"}),
        dcc.Graph(figure=VI.fig_count_map(payload), config={"displaylogo": False}),
        dcc.Graph(figure=VI.fig_lag_decay(payload), config={"displaylogo": False}),
        html.Div("Stability", className="section-label"),
        html.Table([html.Tbody([
            html.Tr([html.Td(k, style={"paddingRight": "14px", **_MONO}),
                     html.Td(v, style={"fontSize": "12px"})])
            for k, v in VI.stability_rows(payload)])],
            style={"borderCollapse": "collapse", "marginBottom": "12px"}),
        html.Div("Scans", className="section-label"),
    ]

    rows = VI.scan_table(payload)
    if rows:
        headers = list(rows[0].keys())
        body.append(html.Table([
            html.Thead(html.Tr([html.Th(h, style={"textAlign": "left", **_MONO})
                                for h in headers])),
            html.Tbody([html.Tr([html.Td(r[h], style={"fontSize": "11px",
                                                      "paddingRight": "12px"})
                                 for h in headers]) for r in rows]),
        ], style={"borderCollapse": "collapse"}))

    if anim is None:
        # The dashboard never computes — surface the command instead.
        body += [
            html.Div("Animation", className="section-label",
                     style={"marginTop": "12px"}),
            html.Div("Not rendered for this well. To produce it:", style=_MONO),
            html.Pre(
                "python scripts/run_electrode_selection_qc.py "
                f"--sample {sample_id} --plate {plate_id} --well {well} --mode anim",
                style={**_MONO, "whiteSpace": "pre-wrap",
                       "background": "var(--bg-deep)", "padding": "8px",
                       "borderRadius": "4px"}),
        ]
    else:
        body += [
            html.Div("Animation", className="section-label",
                     style={"marginTop": "12px"}),
            html.Div(str(anim), style=_MONO),
        ]

    return (html.Div(body), _MODAL_SHOWN,
            f"{sample_id} · {plate_id} · {VI.well_name(well)} ({well})")
