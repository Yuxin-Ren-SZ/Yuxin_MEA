"""Per-well Burst inspector page.

Single-well, disk-first inspector for the iterative burst detector.
Reads ``debug_trace.pkl`` written by ``IterativeBurstDetectionTask`` (when
``debug=True``); falls back to an on-demand in-process recompute when the
debug artifacts are missing — same UI either way.

State held in two server-side stores:
- ``burst-insp-bundle-key`` (Store): hash key into ``_BUNDLE_CACHE``.
- ``_BUNDLE_CACHE``: module-level dict, ``key -> InspectorBundle``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html
from flask import current_app

from yuxin_mea.analysis.burst_inspector import (
    InspectorBundle,
    fig_composite_with_threshold,
    fig_event_gmm_clusters,
    fig_iteration_trajectory,
    fig_label_comparison_table,
    fig_pca_feature_space,
    fig_raster,
    load_inspector_bundle,
    summary_card,
)
from yuxin_mea.analysis.iterative_burst_detector import IterativeBurstConfig
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
                    html.Span("iterative_burst_detection"),
                    html.Span("inspector"),
                ], className="breadcrumb"),
                html.H1("Burst inspector"),
                html.Div(
                    "Single-well diagnostic for the iterative burst detector. "
                    "Reads `debug_trace.pkl` when present; otherwise recomputes "
                    "on demand (slower).",
                    className="subtitle",
                ),
            ]),
        ],
        className="view-head",
    ),

    # Control strip: root, recording, well, iteration, status
    html.Div(id="burst-insp-root-status",
             style={"color": "var(--ink-3)", "fontFamily": "var(--font-mono)",
                    "fontSize": "11px", "marginBottom": "8px"}),
    html.Div([
        dcc.Input(
            id="burst-insp-root-input", value="", type="text",
            placeholder="iterative_burst_detection output_root",
            style={**_input_style(), "flex": "1 1 480px", "marginRight": "8px"},
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
        html.Span(style={"display": "inline-block", "width": "16px"}),
        html.Label("Iteration", className="section-label",
                   style={"marginRight": "6px"}),
        html.Div(dcc.Slider(id="burst-insp-iter-slider",
                            min=0, max=19, step=1, value=19,
                            tooltip={"placement": "bottom",
                                     "always_visible": False}),
                 style={"width": "260px"}),
        html.Span(id="burst-insp-iter-label",
                  style={"marginLeft": "8px", "color": "var(--ink-3)",
                         "fontFamily": "var(--font-mono)", "fontSize": "11px"}),
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
        dcc.Tab(label="Iteration & PCA", value="iter",
                className="tab--regular", selected_className="tab--selected",
                children=[
            dcc.Graph(id="burst-insp-fig-trajectory", figure=_EMPTY_FIGURE),
            dcc.Graph(id="burst-insp-fig-pca", figure=_EMPTY_FIGURE),
        ]),
        dcc.Tab(label="Clusters & labels", value="clusters",
                className="tab--regular", selected_className="tab--selected",
                children=[
            dcc.Graph(id="burst-insp-fig-gmm", figure=_EMPTY_FIGURE),
            dcc.Graph(id="burst-insp-fig-table", figure=_EMPTY_FIGURE),
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


def _discover_wells(output_root: Path) -> dict[str, list[tuple[str, str, str]]]:
    """Walk ``output_root`` and group ``(recording_key, rec_name, well_id)`` by recording.

    Recording key may itself contain slashes (e.g. ``Sample/Date/Plate/Type/RunID``).
    We pin to ``.../iterative_burst_detection`` directories and back out the
    last two parts (rec_name, well_id), treating everything in between as
    the multi-segment recording key.
    """
    if not output_root.exists() or not output_root.is_dir():
        return {}
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for ibd in output_root.rglob("iterative_burst_detection"):
        if not ibd.is_dir():
            continue
        # parent layout: <output_root>/<recording_key>/<rec_name>/<well_id>/iterative_burst_detection
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


def _config_for(rec_key: str, rec_name: str, well_id: str) -> IterativeBurstConfig:
    """Build an IterativeBurstConfig from the user's pipeline config.

    Falls back to defaults if the task block is missing. On-demand fallback
    needs this so the recompute matches what the task would have produced.
    """
    cm = _load_config_manager()
    params = cm.get_task_params("iterative_burst_detection") or {}
    defaults = IterativeBurstConfig()

    def _get(name: str, cast=float):
        val = params.get(name)
        if val is None:
            return getattr(defaults, name)
        try:
            if cast is tuple:
                return tuple(float(x) for x in val)
            return cast(val)
        except (TypeError, ValueError):
            return getattr(defaults, name)

    return IterativeBurstConfig(
        permissive_mad_scale=_get("permissive_mad_scale"),
        permissive_percentile=_get("permissive_percentile"),
        mad_fallback_threshold=_get("mad_fallback_threshold"),
        composite_mad_scale=_get("composite_mad_scale"),
        extent_frac=_get("extent_frac"),
        merge_floor_frac=_get("merge_floor_frac"),
        network_merge_gap_min_s=_get("network_merge_gap_min_s"),
        max_iterations=_get("max_iterations", int),
        convergence_eps=_get("convergence_eps"),
        fisher_alpha_frac=_get("fisher_alpha_frac"),
        ff_scale_multipliers=_get("ff_scale_multipliers", tuple),
        min_burst_modulation=_get("min_burst_modulation"),
        cluster_events=_get("cluster_events", bool),
        cluster_initial_components=_get("cluster_initial_components", int),
        cluster_min_events=_get("cluster_min_events", int),
        cluster_min_separation=_get("cluster_min_separation"),
    )


def _curated_spike_times_path(rec_key: str, rec_name: str, well_id: str) -> Path | None:
    """Locate ``curated_spike_times.npy`` for the on-demand fallback path."""
    cm = _load_config_manager()
    params = cm.get_task_params("auto_curation") or {}
    raw = params.get("output_root")
    cur_root = _resolve_under_analysis_root(raw) if raw else None
    if cur_root is None:
        root = _yuxin_ctx().get("analysis_root")
        if root is None:
            return None
        cur_root = Path(root) / "curation_data"
    path = cur_root / rec_key / rec_name / well_id / "auto_curation" / "curated_spike_times.npy"
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("burst-insp-root-input", "value"),
    Output("burst-insp-root-status", "children"),
    Input("burst-insp-root-input", "id"),
)
def _prefill_root(_id: str):
    """Auto-fill output_root from config (then conventional fallback)."""
    ctx = _yuxin_ctx()
    analysis_root = ctx.get("analysis_root")
    if analysis_root is None:
        return "", "(analysis_root not set in config — set output_root manually.)"
    cm = _load_config_manager()
    params = cm.get_task_params("iterative_burst_detection") or {}
    raw = params.get("output_root")
    p = _resolve_under_analysis_root(raw) if raw else None
    if p is not None and p.exists():
        return str(p), f"Default from `iterative_burst_detection.output_root`: {p}"
    fallback = Path(analysis_root) / "iterative_burst_data"
    if fallback.exists():
        return str(fallback), f"Default from convention: {fallback}"
    return "", (
        "No iterative_burst_detection output directory found. "
        "Re-run the task with debug=True, or set output_root manually."
    )


@callback(
    Output("burst-insp-rec-dropdown", "options"),
    Output("burst-insp-rec-dropdown", "value"),
    Input("burst-insp-root-input", "value"),
    Input("burst-insp-rescan-btn", "n_clicks"),
)
def _populate_recordings(root: str, _n: int):
    # Rescan should invalidate stale bundles so the user sees fresh debug
    # artifacts after a task re-run with debug=True.
    if ctx.triggered_id == "burst-insp-rescan-btn":
        _BUNDLE_CACHE.clear()
    if not root:
        return [], None
    grouped = _discover_wells(Path(root))
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
)
def _populate_wells(rec_key: str | None, root: str | None):
    if not (rec_key and root):
        return [], None
    grouped = _discover_wells(Path(root))
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
)
def _load_bundle(rec_key: str | None, well_value: str | None, root: str | None):
    badge_base = {
        "marginLeft": "12px", "padding": "2px 10px",
        "borderRadius": "10px", "fontFamily": "var(--font-mono)",
        "fontSize": "11px",
    }
    if not (rec_key and well_value and root):
        return None, "none", "no well", {**badge_base,
                                          "background": "var(--bg-2)",
                                          "color": "var(--ink-3)"}
    rec_name, well_id = well_value.split("|", 1)
    config = _config_for(rec_key, rec_name, well_id)

    # Probe for the disk pickle first so we only pay the curated spike-time
    # load on the actual fallback path. Saves a multi-MB read per well
    # switch on the common (debug=True ran) case.
    output_dir = (
        Path(root) / rec_key / rec_name / well_id / "iterative_burst_detection"
    )
    disk_hit = (output_dir / "debug_trace.pkl").exists()

    spike_times_for_fallback = None
    if not disk_hit:
        sp_path = _curated_spike_times_path(rec_key, rec_name, well_id)
        if sp_path is not None:
            try:
                spike_times_for_fallback = np.load(sp_path, allow_pickle=True).item()
            except Exception:
                spike_times_for_fallback = None

    try:
        bundle = load_inspector_bundle(
            Path(root), rec_key, rec_name, well_id,
            on_demand_spike_times=spike_times_for_fallback,
            on_demand_config=config,
        )
    except FileNotFoundError as exc:
        return None, "none", f"❌ {exc}", {
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


@callback(
    Output("burst-insp-iter-label", "children"),
    Input("burst-insp-bundle-key", "data"),
    Input("burst-insp-iter-slider", "value"),
)
def _iter_label(bundle_key: str | None, slider_value: int):
    if not bundle_key or bundle_key not in _BUNDLE_CACHE:
        return ""
    bundle = _BUNDLE_CACHE[bundle_key]
    n = len(bundle.trace.iterations)
    if n == 0:
        return "(no iterations)"
    eff = min(int(slider_value or 0), n - 1)
    suffix = " (final)" if eff == n - 1 else ""
    return f"iter {eff} / {n - 1}{suffix}"


def _render_summary(bundle: InspectorBundle) -> Any:
    """Render the summary_card() dict as a KV card."""
    data = summary_card(bundle)
    rows = []
    for k, v in data.items():
        if isinstance(v, dict):
            v = ", ".join(f"{kk}={vv}" for kk, vv in v.items()) or "—"
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
    Output("burst-insp-fig-trajectory", "figure"),
    Output("burst-insp-fig-pca", "figure"),
    Output("burst-insp-fig-gmm", "figure"),
    Output("burst-insp-fig-table", "figure"),
    Input("burst-insp-bundle-key", "data"),
    Input("burst-insp-iter-slider", "value"),
)
def _render_tabs(bundle_key: str | None, iteration: int | None):
    if not bundle_key or bundle_key not in _BUNDLE_CACHE:
        empty = _EMPTY_FIGURE
        return ("(no well selected)", empty, empty, empty, empty, empty, empty)

    bundle = _BUNDLE_CACHE[bundle_key]
    it: int | str = "final"
    if iteration is not None:
        n = len(bundle.trace.iterations)
        if n > 0:
            it = min(int(iteration), n - 1)

    return (
        _render_summary(bundle),
        fig_composite_with_threshold(bundle, iteration=it),
        fig_raster(bundle, iteration=it),
        fig_iteration_trajectory(bundle),
        fig_pca_feature_space(bundle, iteration=it),
        fig_event_gmm_clusters(bundle),
        fig_label_comparison_table(bundle),
    )
