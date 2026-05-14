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
_STATUS_COLORS = {
    "complete": "#c8e6c9",
    "running":  "#bbdefb",
    "failed":   "#ffcdd2",
    "not_run":  "#eceff1",
}


def _row(label: str, value: object) -> html.Div:
    return html.Div(
        [
            html.Span(f"{label}: ", style={"fontWeight": "600", "color": "#555"}),
            html.Span(str(value), style={"fontFamily": "monospace"}),
        ],
        style={"margin": "4px 0"},
    )


def _status_chip(status: str, count: int) -> html.Div:
    return html.Div(
        [
            html.Div(
                str(count),
                style={"fontSize": "22px", "fontWeight": "600", "fontFamily": "monospace"},
            ),
            html.Div(status, style={"fontSize": "12px", "color": "#555"}),
        ],
        style={
            "backgroundColor": _STATUS_COLORS[status],
            "border": "1px solid #bbb",
            "borderRadius": "4px",
            "padding": "10px 14px",
            "minWidth": "84px",
            "textAlign": "center",
        },
    )


layout = html.Div(
    [
        html.H2("Home", style={"marginTop": "0"}),
        html.P(
            "Read-only view of the dataset and pipeline caches. Use the "
            "Recordings page to queue wells; the Run page builds the CLI "
            "command for `yuxin-mea-run` to drain the queue."
        ),
        html.Div(id="home-banner-slot"),
        html.H4("Loaded configuration"),
        html.Div(id="home-config-summary"),
        html.H4("Cache contents", style={"marginTop": "24px"}),
        html.Div(id="home-cache-summary"),
        html.H4("Pipeline status", style={"marginTop": "24px"}),
        html.Div(
            id="home-status-chips",
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
    ]
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
    config_block = [
        _row("config_path", ctx["config_path"]),
        _row("config_exists", ctx.get("config_exists", False)),
        _row("data_root", ctx["data_root"] or "(not set)"),
        _row("analysis_root", ctx["analysis_root"] or "(not set)"),
    ]

    analysis_root = ctx["analysis_root"]
    if analysis_root is None:
        cache_block = [html.P(
            "analysis_root is not set in the config — no caches to summarize.",
            style={"color": "#888"},
        )]
        chips: list = [html.Span("—", style={"color": "#888"})]
        return banner, config_block, cache_block, chips

    n_rec = len(load_recordings_df(Path(analysis_root)))
    pipe_df, task_names = load_pipeline_df(Path(analysis_root))
    cache_block = [
        _row("recordings (experiment_cache.json)", n_rec),
        _row("pipeline entries (pipeline_cache.json)", len(pipe_df)),
        _row("registered task columns", ", ".join(task_names) if task_names else "—"),
    ]

    counts = {s: 0 for s in _STATUSES}
    for col in task_names:
        for value in pipe_df[col]:
            if value in counts:
                counts[value] += 1
    chips = [_status_chip(s, counts[s]) for s in _STATUSES]
    return banner, config_block, cache_block, chips
