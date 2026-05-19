"""Burst diagnostic page.

Wraps the figures in :mod:`yuxin_mea.analysis.burst_diagnostic` as a Dash
page. The page state is held in two server-side stores:

- A ``dcc.Store`` (``burst-diag-batch-key``) tracks which batch the user
  has currently loaded, by its cache key.
- A module-level dict ``_LOADED_BATCHES`` maps cache key → ``BatchResults``.
  BatchResults contains numpy arrays and custom dataclasses, so it cannot
  live in a Store. The module-global is single-user; if the dashboard ever
  needs multi-tenant access, swap for a flask-caching server-side Cache.
"""

from __future__ import annotations

from pathlib import Path

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html
from flask import current_app

from yuxin_mea.analysis.burst_diagnostic import (
    BatchResults,
    cache_key,
    fig_cross_stage_flow,
    fig_kill_attribution,
    fig_section_c_lda_pca,
    fig_section_d_boundary_shift,
    fig_section_e_3d_pca,
    fig_section_f_gmm_bic_sweep,
    fig_section_g_time_strip,
    fig_stage1_composite_slider,
    fig_stage2_participation,
    fig_stage3_bmi,
    fig_stage4_gmm_pca,
    load_or_run_batch,
)
from yuxin_mea.config import ConfigManager


dash.register_page(__name__, path="/burst-diagnostic", name="Burst diagnostic", order=4)


_LOADED_BATCHES: dict[str, BatchResults] = {}

_EMPTY_FIGURE = go.Figure().update_layout(
    annotations=[{
        "text": "(no batch loaded)",
        "xref": "paper", "yref": "paper",
        "x": 0.5, "y": 0.5,
        "showarrow": False,
        "font": {"size": 14, "color": "#84807a"},
    }],
    xaxis={"visible": False},
    yaxis={"visible": False},
    margin={"l": 20, "r": 20, "t": 20, "b": 20},
    height=200,
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
                            html.Span("burst_diagnostic"),
                        ],
                        className="breadcrumb",
                    ),
                    html.H1("Burst diagnostic"),
                    html.Div(
                        "Iterative burst detector per-stage diagnostics. First "
                        "load may take a few minutes; cached afterwards.",
                        className="subtitle",
                    ),
                ]
            ),
        ],
        className="view-head",
    ),
    html.Div(id="burst-diag-default-status",
             style={"color": "var(--ink-3)", "fontFamily": "var(--font-mono)",
                    "fontSize": "11px", "marginBottom": "8px"}),
    html.Div([
        dcc.Input(
            id="burst-diag-root-input", value="", type="text",
            placeholder="Kilosort root directory",
            style={"flex": "1 1 480px", "marginRight": "8px",
                   "fontFamily": "var(--font-mono)", "fontSize": "12px",
                   "padding": "4px 10px",
                   "background": "var(--bg)", "color": "var(--ink)",
                   "border": "1px solid var(--line)", "borderRadius": "4px",
                   "minHeight": "28px"},
        ),
        html.Button([html.Span("↻", className="glyph"), "Load"],
                    id="burst-diag-load-btn", n_clicks=0, className="btn primary"),
        html.Button([html.Span("⟳", className="glyph"), "Recompute"],
                    id="burst-diag-recompute-btn", n_clicks=0,
                    className="btn", style={"marginLeft": "8px"}),
        html.Span(id="burst-diag-load-status",
                  style={"marginLeft": "12px", "color": "var(--ink-3)",
                         "fontFamily": "var(--font-mono)", "fontSize": "11px"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "6px",
              "marginBottom": "12px"}),
    html.Div([
        html.Label("Recording", className="section-label",
                   style={"marginRight": "6px", "marginBottom": "0"}),
        html.Div(
            dcc.Dropdown(
                id="burst-diag-recording-dropdown", options=[], value=None,
                clearable=False,
            ),
            style={"width": "260px"},
        ),
        html.Span(style={"display": "inline-block", "width": "24px"}),
        html.Label("Trace", className="section-label",
                   style={"marginRight": "6px", "marginBottom": "0"}),
        html.Div(
            dcc.Dropdown(
                id="burst-diag-trace-dropdown",
                options=[{"label": "default", "value": "default"},
                         {"label": "no_gate", "value": "no_gate"}],
                value="default", clearable=False,
            ),
            style={"width": "150px"},
        ),
    ], style={"display": "flex", "alignItems": "center", "gap": "8px",
              "marginBottom": "12px"}),
    dcc.Store(id="burst-diag-batch-key", data=None),
    dcc.Tabs(id="burst-diag-tabs", value="summary",
             parent_className="tab-strip", children=[
        dcc.Tab(label="Summary", value="summary",
                className="tab--regular", selected_className="tab--selected",
                children=[
            dcc.Graph(id="burst-diag-fig-kill", figure=_EMPTY_FIGURE),
            dcc.Graph(id="burst-diag-fig-xflow", figure=_EMPTY_FIGURE),
        ]),
        dcc.Tab(label="Kill stages", value="kill",
                className="tab--regular", selected_className="tab--selected",
                children=[
            html.H4("Stage 1 — Composite signal"),
            dcc.Graph(id="burst-diag-fig-stage1", figure=_EMPTY_FIGURE),
            html.H4("Stage 2 — Participation floor"),
            dcc.Graph(id="burst-diag-fig-part", figure=_EMPTY_FIGURE),
            html.H4("Stage 3 — BMI / LLR gate"),
            dcc.Graph(id="burst-diag-fig-bmi", figure=_EMPTY_FIGURE),
            html.H4("Stage 4 — GMM event clustering"),
            dcc.Graph(id="burst-diag-fig-stage4", figure=_EMPTY_FIGURE),
        ]),
        dcc.Tab(label="LDA deep-dive", value="lda",
                className="tab--regular", selected_className="tab--selected",
                children=[
            html.H4("Section C — LDA PCA per iteration"),
            dcc.Graph(id="burst-diag-fig-c", figure=_EMPTY_FIGURE),
            html.H4("Section D — Boundary shift"),
            dcc.Graph(id="burst-diag-fig-d", figure=_EMPTY_FIGURE),
            html.H4("Section E — 3D PCA"),
            dcc.Graph(id="burst-diag-fig-e", figure=_EMPTY_FIGURE),
            html.H4("Section F — GMM BIC sweep"),
            dcc.Graph(id="burst-diag-fig-f", figure=_EMPTY_FIGURE),
            html.H4("Section G — Cluster time strip"),
            dcc.Graph(id="burst-diag-fig-g", figure=_EMPTY_FIGURE),
        ]),
    ]),
])


# ---------------------------------------------------------------------------
# 1) Pre-fill default Kilosort root on first render of the page
# ---------------------------------------------------------------------------


@callback(
    Output("burst-diag-root-input", "value"),
    Output("burst-diag-default-status", "children"),
    Input("burst-diag-root-input", "id"),
)
def _prefill_default_root(_id: str):
    """Try a chain of candidate locations and pick the first that exists.

    The burst diagnostic prefers `curated_spike_times.npy` over raw Kilosort
    triples, so curation outputs come first in the chain. Falls back to
    conventional analysis_root subdirs when the config doesn't declare them.
    """
    yuxin_ctx = current_app.config.get("YUXIN_MEA", {})
    analysis_root = yuxin_ctx.get("analysis_root")
    config_path = yuxin_ctx.get("config_path")
    if analysis_root is None:
        return "", "(analysis_root not set in config — set the Kilosort root manually.)"

    cm = ConfigManager()
    if config_path is not None:
        cm.load(config_path)

    def _resolve(raw: str | None) -> Path | None:
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = Path(analysis_root) / p
        return p

    # Ordered candidates: (label, path)
    candidates: list[tuple[str, Path | None]] = [
        ("auto_curation.output_root",
         _resolve(cm.get_task_params("auto_curation").get("output_root") if cm.get_task_params("auto_curation") else None)),
        ("sorting.output_root",
         _resolve(cm.get_task_params("sorting").get("output_root") if cm.get_task_params("sorting") else None)),
        ("analysis_root/curation_data", Path(analysis_root) / "curation_data"),
        ("analysis_root/spikesorted_data", Path(analysis_root) / "spikesorted_data"),
    ]

    for label, path in candidates:
        if path is not None and path.exists():
            return str(path), f"Default from `{label}`: {path}"

    return "", (
        "No spike-source directory found via config or convention. "
        "Tried: " + ", ".join(label for label, _ in candidates) + "."
    )


# ---------------------------------------------------------------------------
# 2) Load / Recompute → batch_key
# ---------------------------------------------------------------------------


@callback(
    Output("burst-diag-batch-key", "data"),
    Output("burst-diag-load-status", "children"),
    Input("burst-diag-load-btn", "n_clicks"),
    Input("burst-diag-recompute-btn", "n_clicks"),
    State("burst-diag-root-input", "value"),
    prevent_initial_call=True,
)
def _load_or_recompute(_l: int, _r: int, root: str):
    if not root:
        return dash.no_update, "Please set a Kilosort root first."

    force = ctx.triggered_id == "burst-diag-recompute-btn"
    analysis_root = current_app.config.get("YUXIN_MEA", {}).get("analysis_root")
    try:
        batch, from_cache = load_or_run_batch(
            Path(root), analysis_root, force_recompute=force,
        )
    except FileNotFoundError as exc:
        return dash.no_update, f"❌ {exc}"

    key = cache_key(Path(root))
    _LOADED_BATCHES[key] = batch
    if analysis_root is None:
        suffix = "(fresh run — no cache; analysis_root not set)"
    else:
        suffix = "(from cache)" if from_cache else "(fresh run)"
    return key, f"✓ Loaded {len(batch.recording_names)} recording(s) {suffix}"


# ---------------------------------------------------------------------------
# 3) batch_key → recording dropdown + cross-recording figures
# ---------------------------------------------------------------------------


@callback(
    Output("burst-diag-recording-dropdown", "options"),
    Output("burst-diag-recording-dropdown", "value"),
    Output("burst-diag-fig-kill", "figure"),
    Output("burst-diag-fig-xflow", "figure"),
    Output("burst-diag-fig-part", "figure"),
    Output("burst-diag-fig-bmi", "figure"),
    Input("burst-diag-batch-key", "data"),
)
def _populate_cross_recording(batch_key: str | None):
    if not batch_key or batch_key not in _LOADED_BATCHES:
        return [], None, _EMPTY_FIGURE, _EMPTY_FIGURE, _EMPTY_FIGURE, _EMPTY_FIGURE
    batch = _LOADED_BATCHES[batch_key]
    options = [{"label": n, "value": n} for n in batch.recording_names]
    first = batch.recording_names[0] if batch.recording_names else None
    return (
        options, first,
        fig_kill_attribution(batch),
        fig_cross_stage_flow(batch),
        fig_stage2_participation(batch),
        fig_stage3_bmi(batch),
    )


# ---------------------------------------------------------------------------
# 4) (batch_key, recording, trace) → per-recording figures
# ---------------------------------------------------------------------------


@callback(
    Output("burst-diag-fig-stage1", "figure"),
    Output("burst-diag-fig-stage4", "figure"),
    Output("burst-diag-fig-c", "figure"),
    Output("burst-diag-fig-d", "figure"),
    Output("burst-diag-fig-e", "figure"),
    Output("burst-diag-fig-f", "figure"),
    Output("burst-diag-fig-g", "figure"),
    Input("burst-diag-batch-key", "data"),
    Input("burst-diag-recording-dropdown", "value"),
    Input("burst-diag-trace-dropdown", "value"),
)
def _per_recording(batch_key: str | None, recording: str | None, trace_kind: str):
    if not batch_key or batch_key not in _LOADED_BATCHES or not recording:
        return (_EMPTY_FIGURE,) * 7
    batch = _LOADED_BATCHES[batch_key]
    return (
        fig_stage1_composite_slider(batch, recording, trace_kind),
        fig_stage4_gmm_pca(batch, recording),
        fig_section_c_lda_pca(batch, recording, trace_kind, False),
        fig_section_d_boundary_shift(batch, recording, trace_kind),
        fig_section_e_3d_pca(batch, recording, trace_kind),
        fig_section_f_gmm_bic_sweep(batch, recording, trace_kind),
        fig_section_g_time_strip(batch, recording, trace_kind),
    )
