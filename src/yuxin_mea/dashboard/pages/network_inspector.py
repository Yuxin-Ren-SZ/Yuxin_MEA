"""Network inspector page — per-well view of the four network-analysis stages.

Navigation mirrors the Plate viewer: pick a recording, paint the 4x6 plate grid
with a well-level metric, click a well to open its detail modal. The modal is
tabbed over the stages that actually ran for that well:

    Connectivity · CCG · Nodes · Criticality · Directed · Spatial

All data comes from :mod:`yuxin_mea.analysis.network_inspector` (plain
``np.load`` / ``pd.read_parquet`` / ``json`` — no SpikeInterface, no torch). The
dashboard **never runs compute**: a stage that hasn't been computed for a well
simply renders its empty state.

Two absences are normal here and are surfaced rather than hidden:

* ``criticality`` / ``directed_connectivity`` are not in the shipped pipeline
  config and not recorded in ``pipeline_cache.json``, so their roots are resolved
  by convention and shown in the "stage roots" panel, where they can be
  overridden.
* ``node_metrics.parquet`` has no committed producer, so the Nodes tab is empty
  for wells that predate the ad-hoc backfill.
* ``ccg.npz`` postdates the first connectivity pass, so the CCG tab is empty for
  wells that have not been re-run or backfilled (``scripts/backfill_ccg.py``).
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


dash.register_page(__name__, path="/network-inspector", name="Network inspector",
                   order=7)

_FILTER_FIELDS = ("sample", "scan-type", "date", "group")

# Stage key -> (label, root-input id). Order matches STAGE_ORDER.
_ROOT_INPUTS = (
    ("connectivity", "Connectivity"),
    ("spatial_map", "Spatial map"),
    ("criticality", "Criticality"),
    ("directed", "Directed TE"),
)

_TABS = (
    ("connectivity", "Connectivity"),
    ("ccg", "CCG"),
    ("nodes", "Nodes"),
    ("criticality", "Criticality"),
    ("directed", "Directed"),
    ("spatial_map", "Spatial"),
)

_MODAL_HIDDEN = {"display": "none"}
_MODAL_SHOWN = {
    "display": "flex", "position": "fixed", "inset": 0, "zIndex": 1000,
    "alignItems": "center", "justifyContent": "center",
}

_GRID_STYLE = {"display": "grid", "gridTemplateColumns": "repeat(6, 1fr)",
               "gap": "6px", "marginTop": "16px"}
_CELL_BASE = {
    "padding": "10px 4px", "border": "1px solid var(--line)",
    "borderRadius": "6px", "cursor": "pointer", "display": "flex",
    "flexDirection": "column", "alignItems": "center", "justifyContent": "center",
    "minHeight": "76px", "fontFamily": "var(--font-mono)",
}
_CELL_EMPTY = {**_CELL_BASE, "cursor": "default", "background": "var(--bg-deep)",
               "color": "var(--ink-4)"}
_MONO = {"fontFamily": "var(--font-mono)", "fontSize": "11px",
         "color": "var(--ink-3)"}


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def _root_panel() -> html.Details:
    """Overridable per-stage output roots.

    Criticality and directed connectivity are absent from the config, so this is
    where a user points the page at a non-default location instead of the page
    silently finding nothing.
    """
    rows = []
    for stage, label in _ROOT_INPUTS:
        rows.append(html.Div([
            html.Label(label, style={**_MONO, "flex": "0 0 130px"}),
            dcc.Input(id=f"ni-root-{stage}", type="text", debounce=True, value="",
                      placeholder=f"<analysis_root>/{stage}_data",
                      style={"flex": "1 1 420px", "fontFamily": "var(--font-mono)",
                             "fontSize": "11px", "padding": "4px 8px",
                             "background": "var(--bg)", "color": "var(--ink)",
                             "border": "1px solid var(--line)", "borderRadius": "4px"}),
            html.Span(id=f"ni-root-status-{stage}", style={**_MONO, "flex": "0 0 150px"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px",
                  "marginBottom": "6px", "flexWrap": "wrap"}))
    return html.Details([
        html.Summary("stage roots", className="section-label",
                     style={"cursor": "pointer"}),
        html.Div(rows, style={"padding": "10px 4px 2px"}),
    ], open=False)


layout = html.Div([
    html.Div(
        html.Div([
            html.Div([
                html.Span("workspace"),
                html.Span("analysis"),
                html.Span("network"),
            ], className="breadcrumb"),
            html.H1("Network inspector"),
            html.Div(
                "STTC connectivity, node cartography, avalanche criticality, "
                "directed transfer entropy and spatial activity — per well. "
                "Reads stage outputs; nothing is computed here.",
                className="subtitle"),
        ]),
        className="view-head",
    ),

    html.Div(id="ni-banner"),
    build_filter_bar("network", show=_FILTER_FIELDS),

    html.Div([
        html.Label("Recording", className="section-label", style={"marginBottom": "0"}),
        html.Div(dcc.Dropdown(id="ni-rec", options=[], value=None, clearable=False),
                 style={"flex": "1 1 380px"}),
        html.Label("Metric", className="section-label",
                   style={"marginBottom": "0", "marginLeft": "8px"}),
        html.Div(dcc.Dropdown(id="ni-metric", options=[], value="mean_sttc",
                              clearable=False), style={"flex": "0 1 250px"}),
        html.Button([html.Span("↻", className="glyph"), "Load"], id="ni-load",
                    n_clicks=0, className="btn primary"),
        html.Button("Rescan", id="ni-rescan", n_clicks=0, className="btn"),
        html.Span(id="ni-status", style=_MONO),
    ], style={"display": "flex", "alignItems": "center", "gap": "8px",
              "marginBottom": "10px", "flexWrap": "wrap"}),

    _root_panel(),

    dcc.Store(id="ni-context"),
    dcc.Store(id="ni-active-well", data=None),

    html.Div([
        html.Button([html.Span("‹", className="glyph"), "Previous well"],
                    id="ni-prev-well", n_clicks=0, className="btn"),
        html.Button(["Next well", html.Span("›", className="glyph")],
                    id="ni-next-well", n_clicks=0, className="btn"),
        html.Span(id="ni-scale-legend", style={**_MONO, "marginLeft": "12px"}),
    ], style={"display": "flex", "gap": "6px", "alignItems": "center",
              "flexWrap": "wrap", "margin": "12px 0 0"}),

    dcc.Loading(
        html.Div("Load a recording to paint the plate. Click a well to open its "
                 "network detail.",
                 id="ni-grid", style={"marginTop": "16px", "color": "var(--ink-3)"}),
        type="default",
    ),

    # Backdrop is a sibling of the content div, so a click inside the content
    # never bumps the backdrop's n_clicks -> backdrop click means "outside".
    html.Div([
        html.Div(id="ni-modal-backdrop", n_clicks=0,
                 style={"position": "absolute", "inset": 0,
                        "background": "rgba(0,0,0,0.55)"}),
        html.Div([
            html.Div([
                html.Span(id="ni-modal-title",
                          style={"fontFamily": "var(--font-mono)", "fontSize": "13px",
                                 "fontWeight": "600", "color": "var(--ink)"}),
                html.Button("✕", id="ni-modal-close", n_clicks=0, className="btn",
                            style={"marginLeft": "auto"},
                            title="Close (or click outside)"),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px",
                      "marginBottom": "8px"}),
            dcc.Loading(html.Div(id="ni-modal-body"), type="default"),
        ], style={"position": "relative", "zIndex": 1, "background": "var(--bg)",
                  "border": "1px solid var(--line)", "borderRadius": "10px",
                  "padding": "14px", "width": "90vw", "maxWidth": "1400px",
                  "maxHeight": "92vh", "overflow": "auto",
                  "boxShadow": "0 12px 40px rgba(0,0,0,0.4)"}),
    ], id="ni-modal", style=_MODAL_HIDDEN),
], className="page")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ctx() -> dict[str, Any]:
    return current_app.config.get("YUXIN_MEA", {})


def _config_roots() -> dict[str, str]:
    """Per-stage ``output_root`` from the pipeline config, when present."""
    from yuxin_mea.analysis.network_inspector import STAGES
    from yuxin_mea.config import ConfigManager

    config_path = _ctx().get("config_path")
    if not config_path or not Path(config_path).exists():
        return {}
    cm = ConfigManager()
    try:
        cm.load(config_path)
    except Exception:  # noqa: BLE001 — a bad config must not break the page
        return {}
    out: dict[str, str] = {}
    for stage, spec in STAGES.items():
        params = cm.get_task_params(spec.task_name) or {}
        root = params.get("output_root")
        if root:
            out[stage] = str(root)
    return out


def _well_sort_key(well_id: str) -> int:
    try:
        return int(str(well_id).replace("well", ""))
    except ValueError:
        return 0


def _kv_card(title: str, data: dict[str, str]) -> html.Div:
    if not data:
        return html.Div()
    rows = [
        html.Tr([
            html.Td(k, style={"padding": "3px 14px 3px 0", "color": "var(--ink-3)",
                              "fontFamily": "var(--font-mono)", "fontSize": "11px",
                              "whiteSpace": "nowrap", "verticalAlign": "top"}),
            html.Td(v, style={"padding": "3px 0", "fontFamily": "var(--font-mono)",
                              "fontSize": "11px"}),
        ]) for k, v in data.items()
    ]
    half = (len(rows) + 1) // 2
    return html.Div([
        html.Div(html.Span(title, className="h-title"), className="card-head"),
        html.Div(
            [html.Table(html.Tbody(rows[:half]), style={"borderCollapse": "collapse"}),
             html.Table(html.Tbody(rows[half:]), style={"borderCollapse": "collapse"})],
            className="card-body",
            style={"display": "flex", "gap": "28px", "flexWrap": "wrap"},
        ),
    ], className="card", style={"marginBottom": "12px"})


def _graph(fig, flex: str = "1 1 460px") -> html.Div:
    return html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False}),
                    style={"flex": flex})


def _row(*children) -> html.Div:
    return html.Div(list(children),
                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})


def _node_table(df) -> html.Div:
    """Table view for the Nodes tab.

    Also the relief the cartography palette's contrast warning requires: every
    role is legible as text here, not only as a colour in the scatter.
    """
    if df is None or not len(df):
        return html.Div()
    head = html.Tr([html.Th(c, style={"textAlign": "left", "padding": "4px 10px 4px 0",
                                      "color": "var(--ink-3)", "fontSize": "10px",
                                      "textTransform": "uppercase"})
                    for c in df.columns])
    body = []
    for _, r in df.iterrows():
        body.append(html.Tr([
            html.Td(f"{v:.4g}" if isinstance(v, float) else str(v),
                    style={"padding": "3px 10px 3px 0", "fontSize": "11px"})
            for v in r
        ]))
    return html.Div([
        html.Div(html.Span("node table", className="h-title"), className="card-head"),
        html.Div(html.Table([html.Thead(head), html.Tbody(body)],
                            style={"borderCollapse": "collapse", "width": "100%",
                                   "fontFamily": "var(--font-mono)"}),
                 className="card-body",
                 style={"maxHeight": "340px", "overflowY": "auto"}),
    ], className="card", style={"marginTop": "12px"})


def _tab_body(ni, bundle, tab: str):
    """Render one tab's figures + metrics card."""
    if tab == "connectivity":
        if not bundle.has("connectivity"):
            return html.Div("No connectivity output for this well.", style=_MONO)
        return html.Div([
            _kv_card("graph metrics", ni.metrics_kv(bundle, "connectivity")),
            _row(_graph(ni.fig_sttc_matrix(bundle)),
                 _graph(ni.fig_sttc_graph(bundle))),
            _row(_graph(ni.fig_sttc_distance(bundle)),
                 _graph(ni.fig_dt_sweep(bundle))),
        ])
    if tab == "ccg":
        if bundle.ccg_counts is None or not len(bundle.ccg_counts):
            return html.Div([
                html.Div("No correlograms for this well.", style=_MONO),
                html.Div("ccg.npz is written by the connectivity stage; wells "
                         "computed before CCGs landed need a re-run or "
                         "scripts/backfill_ccg.py.",
                         style={**_MONO, "marginTop": "6px"}),
            ])
        return html.Div([
            _row(_graph(ni.fig_ccg_heatmap(bundle)),
                 _graph(ni.fig_ccg_small_multiples(bundle))),
            html.Div("Reference = u, target = v: a peak at positive lag means v "
                     "fires after u, i.e. u leads. MEA activity is network-burst "
                     "dominated, so every CCG carries a broad central "
                     "co-activation bump — the hollow-Gaussian baseline (grey "
                     "line) strips it, and only the shaded causal window decides "
                     "ccg_sig. A peak sitting at zero means co-activation, not a "
                     "directed interaction.",
                     style={**_MONO, "marginTop": "8px"}),
        ])
    if tab == "nodes":
        if bundle.node_metrics is None or not len(bundle.node_metrics):
            return html.Div([
                html.Div("No node metrics for this well.", style=_MONO),
                html.Div("node_metrics.parquet is written by code that was never "
                         "committed, so it exists for only part of the dataset.",
                         style={**_MONO, "marginTop": "6px"}),
            ])
        return html.Div([
            _kv_card("node scalars", ni.metrics_kv(bundle, "nodes")),
            _row(_graph(ni.fig_cartography(bundle)),
                 _graph(ni.fig_node_degree(bundle))),
            _node_table(ni.node_table(bundle)),
        ])
    if tab == "criticality":
        if not bundle.has("criticality"):
            return html.Div("No criticality output for this well.", style=_MONO)
        return html.Div([
            _kv_card("criticality metrics", ni.metrics_kv(bundle, "criticality")),
            _row(_graph(ni.fig_avalanche_dist(bundle)),
                 _graph(ni.fig_crackling(bundle))),
        ])
    if tab == "directed":
        if not bundle.has("directed"):
            return html.Div("No directed-connectivity output for this well.",
                            style=_MONO)
        return html.Div([
            _kv_card("directed metrics", ni.metrics_kv(bundle, "directed")),
            _row(_graph(ni.fig_te_matrix(bundle)), _graph(ni.fig_te_graph(bundle))),
            html.Div("Binary spike-train TE is inflated by network-burst "
                     "co-activation, so flow hierarchy and degree asymmetry are the "
                     "interpretable readouts — not edge density.",
                     style={**_MONO, "marginTop": "8px"}),
        ])
    if tab == "spatial_map":
        if not bundle.has("spatial_map"):
            return html.Div("No spatial-map output for this well.", style=_MONO)
        return html.Div([
            _kv_card("spatial metrics", ni.metrics_kv(bundle, "spatial_map")),
            _row(_graph(ni.fig_activity_field(bundle)),
                 _graph(ni.fig_burst_propagation(bundle))),
        ])
    return html.Div()


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
@callback(
    Output("ni-banner", "children"),
    Output("ni-rec", "options"),
    Output("ni-rec", "value"),
    Output("ni-status", "children"),
    Output(filter_id("network", "sample"), "options"),
    Output(filter_id("network", "scan-type"), "options"),
    Output(filter_id("network", "date"), "min_date_allowed"),
    Output(filter_id("network", "date"), "max_date_allowed"),
    Output(filter_id("network", "group"), "options"),
    Input("ni-rescan", "n_clicks"),
    Input(filter_id("network", "sample"), "value"),
    Input(filter_id("network", "scan-type"), "value"),
    Input(filter_id("network", "date"), "start_date"),
    Input(filter_id("network", "date"), "end_date"),
    Input(filter_id("network", "group"), "value"),
    State("ni-rec", "value"),
)
def _populate_recordings(_n, f_sample, f_scan, f_from, f_to, f_group, current_rec):
    from yuxin_mea.analysis import network_inspector as ni
    from yuxin_mea.dashboard.components import no_config_banner
    from yuxin_mea.dashboard.data import filter_recordings, load_recordings_detail

    ctx_app = _ctx()
    banner = None if ctx_app.get("config_exists") else no_config_banner()
    analysis_root = ctx_app.get("analysis_root")
    empty: list[dict] = []
    if analysis_root is None:
        return banner, [], None, "analysis_root not set.", empty, empty, None, None, empty

    if ctx.triggered_id == "ni-rescan":
        ni.clear_cache()

    net_set = set(ni.list_network_recordings(analysis_root))
    recordings, well_status = load_recordings_detail(Path(analysis_root))
    with_net = [r for r in recordings if r["cache_key"] in net_set]
    sample_opts = sorted({r["sample_id"] for r in with_net})
    scan_opts = sorted({r["scan_type"] for r in with_net})
    date_opts = sorted({r["date"] for r in with_net})
    group_opts = sorted({g for r in with_net for g in r.get("groups", [])})

    kwargs = filter_kwargs({
        "sample": f_sample, "scan-type": f_scan,
        "date-from": iso_to_yymmdd(f_from), "date-to": iso_to_yymmdd(f_to),
        "group": f_group,
    })
    filtered = filter_recordings(recordings, well_status, **kwargs)
    rec_keys = sorted(r["cache_key"] for r in filtered if r["cache_key"] in net_set)
    options = [{"label": k, "value": k} for k in rec_keys]
    value = current_rec if current_rec in rec_keys else (rec_keys[0] if rec_keys else None)
    status = ("No connectivity output found in pipeline_cache."
              if not net_set else
              f"{len(rec_keys)} of {len(net_set)} network recording(s) match.")
    return (
        banner, options, value, status,
        [{"label": s, "value": s} for s in sample_opts],
        [{"label": s, "value": s} for s in scan_opts],
        yymmdd_to_iso(date_opts[0]) if date_opts else None,
        yymmdd_to_iso(date_opts[-1]) if date_opts else None,
        [{"label": g, "value": g} for g in group_opts],
    )


@callback(
    Output("ni-metric", "options"),
    Input("ni-rec", "value"),
)
def _metric_options(_rec):
    from yuxin_mea.analysis.network_inspector import METRIC_LABELS, PLATE_METRICS

    return [{"label": f"{METRIC_LABELS.get(m, m)}  ({stage})", "value": m}
            for m, stage in PLATE_METRICS.items()]


@callback(
    *[Output(f"ni-root-{stage}", "value") for stage, _ in _ROOT_INPUTS],
    *[Output(f"ni-root-status-{stage}", "children") for stage, _ in _ROOT_INPUTS],
    Input("ni-rec", "value"),
)
def _prefill_roots(_rec):
    """Fill each stage root from config, else convention; report which applied."""
    from yuxin_mea.analysis.network_inspector import stage_output_root

    analysis_root = _ctx().get("analysis_root")
    if analysis_root is None:
        return [""] * len(_ROOT_INPUTS) + ["analysis_root not set"] * len(_ROOT_INPUTS)

    cfg = _config_roots()
    values, notes = [], []
    for stage, _label in _ROOT_INPUTS:
        root = stage_output_root(analysis_root, stage, cfg)
        values.append(str(root))
        if stage in cfg:
            note = "from config"
        else:
            note = "by convention"
        notes.append(f"{note} · {'exists' if root.exists() else 'missing'}")
    return values + notes


@callback(
    Output("ni-grid", "children"),
    Output("ni-context", "data"),
    Output("ni-scale-legend", "children"),
    Input("ni-load", "n_clicks"),
    Input("ni-rec", "value"),
    Input("ni-metric", "value"),
    *[Input(f"ni-root-{stage}", "value") for stage, _ in _ROOT_INPUTS],
)
def _render_grid(_n, rec_key, metric, *roots):
    from yuxin_mea.analysis import network_inspector as ni

    analysis_root = _ctx().get("analysis_root")
    if not (rec_key and analysis_root and metric):
        return ("Load a recording to paint the plate.", None, "")

    overrides = {stage: (roots[i] or "").strip()
                 for i, (stage, _l) in enumerate(_ROOT_INPUTS) if roots[i]}
    try:
        stage_dirs = ni.all_stage_dirs(analysis_root, rec_key, overrides)
        scalars = ni.plate_scalars(stage_dirs)
    except Exception as exc:  # noqa: BLE001 — surface, never 500 the page
        return (html.Div(f"Failed to read stage outputs: {exc}", style=_MONO),
                None, "")

    # Well universe = wells with *any* network-stage output. auto_curation is
    # excluded deliberately: it exists for wells that never got a network stage,
    # and those would be empty cells opening onto five empty tabs.
    wells = sorted({w for stage in ni.STAGE_ORDER for w in (stage_dirs.get(stage) or {})},
                   key=_well_sort_key)
    if not wells:
        return (html.Div("No network output found for this recording.", style=_MONO),
                None, "")

    values = {w: scalars.get(w, {}).get(metric) for w in wells}
    colors, vmin, vmax = ni.plate_colors(values, metric)

    # Every well stays clickable, whatever the selected metric. A well can lack
    # `dcc` and still have a full connectivity + spatial graph worth opening, so
    # "no value for this metric" greys the fill only — it never removes the well
    # from the grid or from prev/next.
    cells = []
    for well in wells:
        v = values.get(well)
        if well in colors:
            bg, ink = colors[well]
            value_text, title = f"{v:.4g}", f"Open {well}"
        else:
            bg, ink = _CELL_EMPTY["background"], _CELL_EMPTY["color"]
            value_text = "n/a"
            title = f"Open {well} (no {ni.METRIC_LABELS.get(metric, metric)})"
        cells.append(html.Button(
            [html.Div(well, style={"fontSize": "11px", "opacity": 0.85}),
             html.Div(value_text, style={"fontSize": "14px", "fontWeight": "600",
                                         "marginTop": "2px"})],
            id={"type": "ni-well-cell", "index": well}, n_clicks=0,
            style={**_CELL_BASE, "background": bg, "color": ink}, title=title,
        ))

    label = ni.METRIC_LABELS.get(metric, metric)
    n_have = len(colors)
    legend = (f"{label}: {vmin:.4g} → {vmax:.4g} · {n_have}/{len(wells)} wells"
              if n_have else f"{label}: no well has this metric")
    context = {"recording_key": rec_key, "roots": overrides, "ok_wells": wells}
    return html.Div(cells, style=_GRID_STYLE), context, legend


@callback(
    Output("ni-modal-body", "children"),
    Output("ni-modal", "style", allow_duplicate=True),
    Output("ni-modal-title", "children"),
    Output("ni-active-well", "data", allow_duplicate=True),
    Input({"type": "ni-well-cell", "index": ALL}, "n_clicks"),
    Input("ni-prev-well", "n_clicks"),
    Input("ni-next-well", "n_clicks"),
    State("ni-active-well", "data"),
    State("ni-context", "data"),
    prevent_initial_call=True,
)
def _open_well(_cells, _prev, _next, active_well, context):
    """Open (or step) the per-well modal, building each tab fresh."""
    nu = dash.no_update
    trig = ctx.triggered[0] if ctx.triggered else None
    tid = ctx.triggered_id
    # Ignore the all-zero fire when the grid is (re)created.
    if not trig or not trig.get("value") or not context:
        return nu, nu, nu, nu

    analysis_root = _ctx().get("analysis_root")
    if analysis_root is None:
        return nu, nu, nu, nu

    ok_wells = context.get("ok_wells") or []
    if isinstance(tid, dict):
        target = tid["index"]
    else:
        if not ok_wells:
            return nu, nu, nu, nu
        cur = active_well if active_well in ok_wells else ok_wells[0]
        i = ok_wells.index(cur)
        i = max(0, i - 1) if tid == "ni-prev-well" else min(len(ok_wells) - 1, i + 1)
        target = ok_wells[i]

    from yuxin_mea.analysis import network_inspector as ni

    try:
        stage_dirs = ni.all_stage_dirs(
            analysis_root, context["recording_key"], context.get("roots") or {},
        )
        dirs = {stage: d.get(target) for stage, d in stage_dirs.items()}
        rec_name = ""
        for rn, wid in ni.wells_in_recording(analysis_root, context["recording_key"]):
            if wid == target:
                rec_name = rn
                break
        bundle = ni.load_network_bundle_cached(
            dirs, recording_key=context["recording_key"], rec_name=rec_name,
            well_id=target,
        )
    except Exception as exc:  # noqa: BLE001
        return (html.Div(f"Failed to load well: {exc}", style=_MONO),
                _MODAL_SHOWN, target, target)

    present = set(bundle.stages_present)
    tabs = []
    for key, label in _TABS:
        stage = "connectivity" if key in ("nodes", "ccg") else key
        mark = "" if stage in present else " ·"
        tabs.append(dcc.Tab(
            label=f"{label}{mark}", value=key, className="tab--regular",
            selected_className="tab--selected",
            children=html.Div(_tab_body(ni, bundle, key),
                              style={"marginTop": "12px"}),
        ))

    missing = [s for s in ni.STAGE_ORDER if s not in present]
    header = html.Div(
        ("stages present: " + ", ".join(sorted(present)) if present else "no stage output")
        + (f"  ·  missing: {', '.join(missing)}" if missing else ""),
        style={**_MONO, "marginBottom": "8px"},
    )
    body = html.Div([
        header,
        dcc.Tabs(id="ni-modal-tabs", value=_TABS[0][0], parent_className="tab-strip",
                 children=tabs),
    ])
    title = f"{context['recording_key']} · {rec_name}/{target}" if rec_name \
        else f"{context['recording_key']} · {target}"
    return body, _MODAL_SHOWN, title, target


@callback(
    Output("ni-modal", "style", allow_duplicate=True),
    Input("ni-modal-backdrop", "n_clicks"),
    Input("ni-modal-close", "n_clicks"),
    prevent_initial_call=True,
)
def _close_modal(_backdrop, _close):
    trig = ctx.triggered[0] if ctx.triggered else None
    if not trig or not trig.get("value"):
        return dash.no_update
    return _MODAL_HIDDEN
