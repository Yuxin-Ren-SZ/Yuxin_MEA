"""Home page — confirms which config is loaded and shows entry/status counts."""

from __future__ import annotations

from pathlib import Path

import dash
from dash import Input, Output, callback, html
from flask import current_app

from yuxin_mea.dashboard.components import no_config_banner
from yuxin_mea.dashboard.data import load_pipeline_df, load_recordings_df


dash.register_page(__name__, path="/", name="Home", order=0)


_STATUSES = ("not_run", "running", "complete", "failed")
# Map each status to its pill modifier so KPI tiles pick up the matching
# soft background + accent color.
_STATUS_PILL = {
    "complete": "ok",
    "running": "run",
    "failed": "fail",
    "not_run": "idle",
}


def _kv(label: str, value: object) -> list:
    return [html.Dt(label), html.Dd(str(value), className="path")]


def _kpi(label: str, value: int, pill_kind: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="label"),
            html.Div(str(value), className="value"),
            html.Div(
                [html.Span(pill_kind, className=f"pill {pill_kind}")],
                className="sub",
            ),
        ],
        className="kpi",
    )


layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("workspace"),
                                html.Span("analysis_root"),
                                html.Span("home"),
                            ],
                            className="breadcrumb",
                        ),
                        html.H1("Analysis operations"),
                        html.Div(
                            "Read-only view of dataset and pipeline caches.",
                            className="subtitle",
                        ),
                    ]
                ),
            ],
            className="view-head",
        ),
        html.Div(id="home-banner-slot"),
        html.Div(
            [
                html.Div(
                    [html.Span("loaded configuration", className="h-title")],
                    className="card-head",
                ),
                html.Div(
                    html.Dl(id="home-config-summary", className="kv"),
                    className="card-body",
                ),
            ],
            className="card",
            style={"marginBottom": "16px"},
        ),
        html.Div(
            [
                html.Div(
                    [html.Span("cache contents", className="h-title")],
                    className="card-head",
                ),
                html.Div(
                    html.Dl(id="home-cache-summary", className="kv"),
                    className="card-body",
                ),
            ],
            className="card",
            style={"marginBottom": "16px"},
        ),
        html.Div(
            "pipeline status",
            className="section-label",
        ),
        html.Div(id="home-status-chips", className="kpi-grid"),
    ],
    className="page",
)


@callback(
    Output("home-banner-slot", "children"),
    Output("home-config-summary", "children"),
    Output("home-cache-summary", "children"),
    Output("home-status-chips", "children"),
    Input("home-config-summary", "id"),
)
def _render(_id: str):
    """Render banner + config + cache + status summaries on page load."""
    ctx = current_app.config["YUXIN_MEA"]
    banner = None if ctx.get("config_exists") else no_config_banner()
    config_block: list = []
    for label, value in [
        ("config_path", ctx["config_path"]),
        ("config_exists", ctx.get("config_exists", False)),
        ("data_root", ctx["data_root"] or "(not set)"),
        ("analysis_root", ctx["analysis_root"] or "(not set)"),
    ]:
        config_block.extend(_kv(label, value))

    analysis_root = ctx["analysis_root"]
    if analysis_root is None:
        cache_block = _kv("status", "analysis_root not set — no caches to summarize.")
        empty_kpis = [_kpi(s, 0, _STATUS_PILL[s]) for s in _STATUSES]
        return banner, config_block, cache_block, empty_kpis

    n_rec = len(load_recordings_df(Path(analysis_root)))
    pipe_df, task_names = load_pipeline_df(Path(analysis_root))
    cache_block: list = []
    for label, value in [
        ("recordings (experiment_cache.json)", n_rec),
        ("pipeline entries (pipeline_cache.json)", len(pipe_df)),
        ("registered task columns", ", ".join(task_names) if task_names else "—"),
    ]:
        cache_block.extend(_kv(label, value))

    counts = {s: 0 for s in _STATUSES}
    for col in task_names:
        for value in pipe_df[col]:
            if value in counts:
                counts[value] += 1
    chips = [_kpi(s, counts[s], _STATUS_PILL[s]) for s in _STATUSES]
    return banner, config_block, cache_block, chips
