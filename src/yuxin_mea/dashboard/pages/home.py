"""Home page — confirms which config is loaded and shows entry counts."""

from __future__ import annotations

import dash
from dash import Input, Output, callback, html
from flask import current_app

from yuxin_mea.dashboard.data import load_pipeline_df, load_recordings_df


dash.register_page(__name__, path="/", name="Home", order=0)


def _row(label: str, value: object) -> html.Div:
    return html.Div(
        [
            html.Span(f"{label}: ", style={"fontWeight": "600", "color": "#555"}),
            html.Span(str(value), style={"fontFamily": "monospace"}),
        ],
        style={"margin": "4px 0"},
    )


layout = html.Div(
    [
        html.H2("Home", style={"marginTop": "0"}),
        html.P(
            "Read-only view of the dataset and pipeline caches. "
            "Use the Refresh button on each data page to reload after a "
            "pipeline run."
        ),
        html.H4("Loaded configuration"),
        html.Div(id="home-config-summary"),
        html.H4("Cache contents", style={"marginTop": "24px"}),
        html.Div(id="home-cache-summary"),
    ]
)


@callback(
    Output("home-config-summary", "children"),
    Output("home-cache-summary", "children"),
    Input("home-config-summary", "id"),
)
def _render(_id: str) -> tuple[list, list]:
    """Render config + cache summaries on initial page load."""
    ctx = current_app.config["YUXIN_MEA"]
    config_block = [
        _row("config_path", ctx["config_path"]),
        _row("data_root", ctx["data_root"] or "(not set)"),
        _row("analysis_root", ctx["analysis_root"] or "(not set)"),
    ]

    analysis_root = ctx["analysis_root"]
    if analysis_root is None:
        cache_block = [html.P(
            "analysis_root is not set in the config — no caches to summarize.",
            style={"color": "#888"},
        )]
    else:
        n_rec = len(load_recordings_df(analysis_root))
        pipe_df, task_names = load_pipeline_df(analysis_root)
        cache_block = [
            _row("recordings (experiment_cache.json)", n_rec),
            _row("pipeline entries (pipeline_cache.json)", len(pipe_df)),
            _row("registered task columns", ", ".join(task_names) if task_names else "—"),
        ]
    return config_block, cache_block
