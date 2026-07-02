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
    fig_generic_summary,
    load_or_run_batch,
)
from yuxin_mea.analysis.ml_burst_detector import MLBurstConfig
from yuxin_mea.config import ConfigManager
from yuxin_mea.dashboard.cache import cache_get, cache_set, make_key


dash.register_page(__name__, path="/burst-diagnostic", name="Burst diagnostic", order=4)


# L1 in-process (fast, same-session) backed by the persistent flask-caching
# layer (L2, survives restarts) — the "swap for a flask-caching Cache" the old
# comment anticipated. BatchResults is assumed picklable (numpy + dataclasses);
# if a pickle ever fails, cache_set swallows it and the page runs L1-only.
_LOADED_BATCHES: dict[str, BatchResults] = {}


def _store_batch(key: str, batch: BatchResults) -> None:
    _LOADED_BATCHES[key] = batch
    cache_set(make_key("burst_diag", key), batch)


def _get_batch(key: str | None) -> BatchResults | None:
    if not key:
        return None
    batch = _LOADED_BATCHES.get(key)
    if batch is not None:
        return batch
    batch = cache_get(make_key("burst_diag", key))
    if batch is not None:
        _LOADED_BATCHES[key] = batch
    return batch

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
                        "Burst detector event-count summary across recordings "
                        "for traditional/ML. First load may take a few minutes; "
                        "cached afterwards.",
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
        html.Label("Method:", className="section-label",
                   style={"marginBottom": "0"}),
        html.Div(dcc.Dropdown(
            id="burst-diag-method-dropdown",
            options=[
                {"label": "traditional", "value": "traditional"},
                {"label": "ml", "value": "ml"},
            ],
            value="ml",
            clearable=False,
        ), style={"width": "140px"}),
        dcc.Input(
            id="burst-diag-root-input", value="", type="text",
            placeholder="Spike-source root directory",
            style={"flex": "1 1 480px",
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
    dcc.Store(id="burst-diag-batch-key", data=None),
    dcc.Graph(id="burst-diag-fig-summary", figure=_EMPTY_FIGURE),
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
    State("burst-diag-method-dropdown", "value"),
    prevent_initial_call=True,
)
def _load_or_recompute(_l: int, _r: int, root: str, method: str | None):
    if not root:
        return dash.no_update, "Please set a spike-source root first."

    method = method or "ml"
    force = ctx.triggered_id == "burst-diag-recompute-btn"
    yuxin_ctx = current_app.config.get("YUXIN_MEA", {})
    analysis_root = yuxin_ctx.get("analysis_root")
    config_path = yuxin_ctx.get("config_path")

    # Honor the pipeline's ML params on recompute so the diagnostic matches the
    # configured detector (e.g. UMAP embedding) instead of bare defaults.
    ml_config = None
    if method == "ml" and config_path is not None:
        cm = ConfigManager()
        cm.load(config_path)
        params = cm.get_task_params("ml_burst_detection") or {}
        if params:
            ml_config = MLBurstConfig.from_task_params(params)

    try:
        batch, from_cache = load_or_run_batch(
            Path(root), analysis_root,
            force_recompute=force, method=method, ml_config=ml_config,
        )
    except FileNotFoundError as exc:
        return dash.no_update, f"error: {exc}"

    key = f"{cache_key(Path(root))}_{method}"
    _store_batch(key, batch)
    if analysis_root is None:
        suffix = "(fresh run — no cache; analysis_root not set)"
    else:
        suffix = "(from cache)" if from_cache else "(fresh run)"
    return key, f"Loaded {len(batch.recording_names)} recording(s) [{method}] {suffix}"


# ---------------------------------------------------------------------------
# 3) batch_key → summary figure
# ---------------------------------------------------------------------------


@callback(
    Output("burst-diag-fig-summary", "figure"),
    Input("burst-diag-batch-key", "data"),
)
def _render_summary(batch_key: str | None):
    batch = _get_batch(batch_key)
    if batch is None:
        return _EMPTY_FIGURE
    return fig_generic_summary(batch)
