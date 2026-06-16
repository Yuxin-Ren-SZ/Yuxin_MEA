"""Per-well Burst inspector page.

Single-well inspector supporting two burst detection methods:
- **traditional** — standard BurstResults only (no debug trace)
- **ml** — standard BurstResults + optional MLBurstTrace

Loads the standard pickle output (burstlets.pkl, plot_signals.npy, etc.) and
shows raster + composite diagnostic views.

State held in two server-side stores:
- ``burst-insp-bundle-key`` (Store): hash key into ``_BUNDLE_CACHE``.
- ``_BUNDLE_CACHE``: module-level dict, ``key -> InspectorBundle``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html
from flask import current_app

from yuxin_mea.analysis.burst_inspector import (
    METHOD_SUBDIRS,
    METHOD_TASK_NAMES,
    METHOD_TERMINALS,
    InspectorBundle,
    fig_composite_basic,
    fig_raster_basic,
    load_generic_bundle,
    summary_card,
)
from yuxin_mea.config import ConfigManager


dash.register_page(__name__, path="/burst-inspector", name="Burst inspector", order=5)


_BUNDLE_CACHE: dict[str, InspectorBundle] = {}


_EMPTY_FIGURE = go.Figure().update_layout(
    annotations=[{
        "text": "(no well selected)",
        "xref": "paper", "yref": "paper",
        "x": 0.5, "y": 0.5, "showarrow": False,
        "font": {"size": 14, "color": "#84807a"},
    }],
    xaxis={"visible": False}, yaxis={"visible": False},
    margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=240,
)

_METHOD_OPTIONS = [
    {"label": "traditional", "value": "traditional"},
    {"label": "ml", "value": "ml"},
]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _input_style() -> dict[str, str]:
    return {
        "fontFamily": "var(--font-mono)", "fontSize": "12px",
        "padding": "4px 10px",
        "background": "var(--bg)", "color": "var(--ink)",
        "border": "1px solid var(--line)", "borderRadius": "4px",
        "minHeight": "28px",
    }


layout = html.Div([
    html.Div(
        [
            html.Div([
                html.Div([
                    html.Span("workspace"),
                    html.Span("burst_detection"),
                    html.Span("inspector"),
                ], className="breadcrumb"),
                html.H1("Burst inspector"),
                html.Div(
                    "Single-well diagnostic for burst detectors. "
                    "Select a method, then a recording and well.",
                    className="subtitle",
                ),
            ]),
        ],
        className="view-head",
    ),

    html.Div(id="burst-insp-root-status",
             style={"color": "var(--ink-3)", "fontFamily": "var(--font-mono)",
                    "fontSize": "11px", "marginBottom": "8px"}),

    # Method selector + root input + rescan
    html.Div([
        html.Label("Method:", className="section-label",
                   style={"marginBottom": "0"}),
        html.Div(dcc.Dropdown(
            id="burst-insp-method-dropdown",
            options=_METHOD_OPTIONS,
            value="ml",
            clearable=False,
        ), style={"width": "140px"}),
        dcc.Input(
            id="burst-insp-root-input", value="", type="text",
            placeholder="output_root",
            style={**_input_style(), "flex": "1 1 480px"},
        ),
        html.Button("Rescan", id="burst-insp-rescan-btn",
                    n_clicks=0, className="btn primary"),
        html.Span(id="burst-insp-status-badge",
                  style={"marginLeft": "12px", "padding": "2px 10px",
                         "borderRadius": "10px", "fontFamily": "var(--font-mono)",
                         "fontSize": "11px",
                         "background": "var(--bg-2)", "color": "var(--ink-3)"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "6px",
              "marginBottom": "12px"}),

    # Recording / well controls
    html.Div([
        html.Label("Recording", className="section-label",
                   style={"marginRight": "6px"}),
        html.Div(dcc.Dropdown(id="burst-insp-rec-dropdown", options=[],
                              value=None, clearable=False),
                 style={"width": "320px"}),
        html.Span(style={"display": "inline-block", "width": "16px"}),
        html.Label("Well", className="section-label",
                   style={"marginRight": "6px"}),
        html.Div(dcc.Dropdown(id="burst-insp-well-dropdown", options=[],
                              value=None, clearable=False),
                 style={"width": "160px"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "8px",
              "marginBottom": "16px"}),

    dcc.Store(id="burst-insp-bundle-key", data=None),
    dcc.Store(id="burst-insp-source-store", data="none"),

    dcc.Tabs(id="burst-insp-tabs", value="algo",
             parent_className="tab-strip", children=[
        dcc.Tab(label="Algorithm", value="algo",
                className="tab--regular", selected_className="tab--selected",
                children=[
            html.Div(id="burst-insp-summary-card",
                     style={"marginTop": "12px"}),
        ]),
        dcc.Tab(label="Raster + composite", value="raster",
                className="tab--regular", selected_className="tab--selected",
                children=[
            dcc.Graph(id="burst-insp-fig-composite", figure=_EMPTY_FIGURE),
            dcc.Graph(id="burst-insp-fig-raster", figure=_EMPTY_FIGURE),
        ]),
    ]),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yuxin_ctx() -> dict[str, Any]:
    return current_app.config.get("YUXIN_MEA", {})


def _load_config_manager() -> ConfigManager:
    cm = ConfigManager()
    config_path = _yuxin_ctx().get("config_path")
    if config_path is not None:
        cm.load(config_path)
    return cm


def _resolve_under_analysis_root(raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    root = _yuxin_ctx().get("analysis_root")
    if root is None:
        return None
    return Path(root) / p


def _discover_wells(
    output_root: Path,
    terminal: str = "ml_burst_detection",
) -> dict[str, list[tuple[str, str, str]]]:
    """Walk ``output_root`` and group ``(recording_key, rec_name, well_id)`` by recording."""
    if not output_root.exists() or not output_root.is_dir():
        return {}
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for ibd in output_root.rglob(terminal):
        if not ibd.is_dir():
            continue
        try:
            rel = ibd.parent.relative_to(output_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 3:
            continue
        well_id = parts[-1]
        rec_name = parts[-2]
        recording_key = "/".join(parts[:-2])
        grouped.setdefault(recording_key, []).append((recording_key, rec_name, well_id))
    for k in grouped:
        grouped[k].sort(key=lambda t: (t[1], t[2]))
    return grouped


def _bundle_key(root: str, rec_key: str, rec_name: str, well_id: str) -> str:
    return f"{root}|{rec_key}|{rec_name}|{well_id}"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("burst-insp-root-input", "value"),
    Output("burst-insp-root-status", "children"),
    Input("burst-insp-method-dropdown", "value"),
)
def _prefill_root(method: str | None):
    """Auto-fill output_root from config based on selected method."""
    method = method or "ml"
    ctx_app = _yuxin_ctx()
    analysis_root = ctx_app.get("analysis_root")
    if analysis_root is None:
        return "", "(analysis_root not set in config — set output_root manually.)"

    task_name = METHOD_TASK_NAMES.get(method, "ml_burst_detection")
    subdir = METHOD_SUBDIRS.get(method, "ml_burst_data")

    cm = _load_config_manager()
    params = cm.get_task_params(task_name) or {}
    raw = params.get("output_root")
    p = _resolve_under_analysis_root(raw) if raw else None
    if p is not None and p.exists():
        return str(p), f"Default from `{task_name}.output_root`: {p}"
    fallback = Path(analysis_root) / subdir
    if fallback.exists():
        return str(fallback), f"Default from convention: {fallback}"
    return "", f"No {task_name} output directory found. Set output_root manually."


@callback(
    Output("burst-insp-rec-dropdown", "options"),
    Output("burst-insp-rec-dropdown", "value"),
    Input("burst-insp-root-input", "value"),
    Input("burst-insp-rescan-btn", "n_clicks"),
    State("burst-insp-method-dropdown", "value"),
)
def _populate_recordings(root: str, _n: int, method: str | None):
    if ctx.triggered_id == "burst-insp-rescan-btn":
        _BUNDLE_CACHE.clear()
    if not root:
        return [], None
    terminal = METHOD_TERMINALS.get(method or "ml", "ml_burst_detection")
    grouped = _discover_wells(Path(root), terminal=terminal)
    if not grouped:
        return [], None
    rec_keys = sorted(grouped.keys())
    options = [{"label": k, "value": k} for k in rec_keys]
    return options, rec_keys[0]


@callback(
    Output("burst-insp-well-dropdown", "options"),
    Output("burst-insp-well-dropdown", "value"),
    Input("burst-insp-rec-dropdown", "value"),
    State("burst-insp-root-input", "value"),
    State("burst-insp-method-dropdown", "value"),
)
def _populate_wells(rec_key: str | None, root: str | None, method: str | None):
    if not (rec_key and root):
        return [], None
    terminal = METHOD_TERMINALS.get(method or "ml", "ml_burst_detection")
    grouped = _discover_wells(Path(root), terminal=terminal)
    triples = grouped.get(rec_key, [])
    options = [
        {"label": f"{rec_name}/{well_id}",
         "value": f"{rec_name}|{well_id}"}
        for _, rec_name, well_id in triples
    ]
    value = options[0]["value"] if options else None
    return options, value


@callback(
    Output("burst-insp-bundle-key", "data"),
    Output("burst-insp-source-store", "data"),
    Output("burst-insp-status-badge", "children"),
    Output("burst-insp-status-badge", "style"),
    Input("burst-insp-rec-dropdown", "value"),
    Input("burst-insp-well-dropdown", "value"),
    State("burst-insp-root-input", "value"),
    State("burst-insp-method-dropdown", "value"),
)
def _load_bundle(
    rec_key: str | None,
    well_value: str | None,
    root: str | None,
    method: str | None,
):
    badge_base = {
        "marginLeft": "12px", "padding": "2px 10px",
        "borderRadius": "10px", "fontFamily": "var(--font-mono)",
        "fontSize": "11px",
    }
    if not (rec_key and well_value and root):
        return None, "none", "no well", {**badge_base,
                                          "background": "var(--bg-2)",
                                          "color": "var(--ink-3)"}
    method = method or "ml"
    rec_name, well_id = well_value.split("|", 1)

    try:
        bundle = load_generic_bundle(
            Path(root), rec_key, rec_name, well_id, method=method,
        )
    except FileNotFoundError as exc:
        return None, "none", f"error: {exc}", {
            **badge_base, "background": "#fdecec", "color": "#c62828"
        }

    key = _bundle_key(root, rec_key, rec_name, well_id)
    _BUNDLE_CACHE[key] = bundle

    if bundle.source == "disk":
        badge = ("disk", "#e7f5e9", "#2e7d32")
    else:
        badge = ("on-demand", "#fff3e0", "#ef6c00")
    return key, bundle.source, badge[0], {
        **badge_base, "background": badge[1], "color": badge[2]
    }


def _render_summary(bundle: InspectorBundle) -> Any:
    """Render the summary_card() dict as a KV card."""
    data = summary_card(bundle)
    rows = []
    for k, v in data.items():
        if isinstance(v, dict):
            v = ", ".join(f"{kk}={vv}" for kk, vv in v.items()) or "---"
        elif isinstance(v, float):
            v = f"{v:.4g}"
        rows.append(html.Tr([
            html.Td(k, style={"padding": "4px 12px 4px 0",
                              "color": "var(--ink-3)",
                              "fontFamily": "var(--font-mono)",
                              "fontSize": "11px",
                              "verticalAlign": "top",
                              "whiteSpace": "nowrap"}),
            html.Td(str(v), style={"padding": "4px 0",
                                   "fontFamily": "var(--font-mono)",
                                   "fontSize": "11px"}),
        ]))
    return html.Div(
        html.Table(html.Tbody(rows),
                   style={"borderCollapse": "collapse"}),
        className="card",
        style={"padding": "12px 16px"},
    )


@callback(
    Output("burst-insp-summary-card", "children"),
    Output("burst-insp-fig-composite", "figure"),
    Output("burst-insp-fig-raster", "figure"),
    Input("burst-insp-bundle-key", "data"),
)
def _render_tabs(bundle_key: str | None):
    if not bundle_key or bundle_key not in _BUNDLE_CACHE:
        empty = _EMPTY_FIGURE
        return ("(no well selected)", empty, empty)

    bundle = _BUNDLE_CACHE[bundle_key]
    return (
        _render_summary(bundle),
        fig_composite_basic(bundle),
        fig_raster_basic(bundle),
    )
