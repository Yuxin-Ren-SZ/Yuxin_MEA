"""Pipeline status page — task matrix from `pipeline_cache.json`."""

from __future__ import annotations

import dash
from dash import Input, Output, callback, dash_table, html
from flask import current_app

from yuxin_mea.dashboard.components import no_config_banner
from yuxin_mea.dashboard.data import load_pipeline_df


dash.register_page(__name__, path="/pipeline", name="Pipeline", order=2)


# Status → cell background color. Statuses match yuxin_mea.pipeline.task_record.TaskStatus.
_STATUS_COLORS = {
    "complete": "#c8e6c9",  # green 100
    "running":  "#bbdefb",  # blue 100
    "failed":   "#ffcdd2",  # red 100
    "not_run":  "#eceff1",  # blue grey 50
}


def _legend_swatch(status: str, color: str) -> html.Div:
    return html.Div(
        [
            html.Span(
                style={
                    "display": "inline-block", "width": "14px", "height": "14px",
                    "backgroundColor": color, "marginRight": "6px",
                    "border": "1px solid #bbb", "verticalAlign": "middle",
                }
            ),
            html.Span(status, style={"fontSize": "13px", "verticalAlign": "middle"}),
        ]
    )


layout = html.Div(
    [
        html.H2("Pipeline status", style={"marginTop": "0"}),
        html.P(
            "One row per (recording, well). One column per task name seen in "
            "the cache. Em-dash cells (—) mean that task was never "
            "registered for the entry. Filter and sort using the column headers."
        ),
        html.Div(id="pipeline-banner-slot"),
        html.Div(
            [
                html.Button("Refresh", id="pipeline-refresh", n_clicks=0),
                html.Span(id="pipeline-status", style={"marginLeft": "12px", "color": "#555"}),
            ],
            style={"marginBottom": "12px"},
        ),
        html.Div(
            [_legend_swatch(status, color) for status, color in _STATUS_COLORS.items()],
            style={"display": "flex", "gap": "16px", "marginBottom": "12px"},
        ),
        dash_table.DataTable(
            id="pipeline-table",
            columns=[],
            data=[],
            filter_action="native",
            sort_action="native",
            page_size=25,
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "monospace", "fontSize": "13px", "padding": "4px 8px"},
            style_header={"fontWeight": "600", "backgroundColor": "#f4f6f8"},
        ),
    ]
)


@callback(
    Output("pipeline-banner-slot", "children"),
    Output("pipeline-table", "data"),
    Output("pipeline-table", "columns"),
    Output("pipeline-table", "style_data_conditional"),
    Output("pipeline-status", "children"),
    Input("pipeline-refresh", "n_clicks"),
)
def _refresh(_n_clicks: int):
    ctx = current_app.config["YUXIN_MEA"]
    banner = None if ctx.get("config_exists") else no_config_banner()
    analysis_root = ctx["analysis_root"]
    if analysis_root is None:
        return banner, [], [], [], "analysis_root is not set in the config."

    df, task_names = load_pipeline_df(analysis_root)
    columns = [{"name": c, "id": c} for c in df.columns]
    style = _build_conditional_style(task_names)
    return banner, df.to_dict("records"), columns, style, f"{len(df)} pipeline entr(ies)"


def _build_conditional_style(task_names: list[str]) -> list[dict]:
    """One color rule per (task column, status) pair."""
    return [
        {
            "if": {"filter_query": f'{{{col}}} = "{status}"', "column_id": col},
            "backgroundColor": color,
        }
        for col in task_names
        for status, color in _STATUS_COLORS.items()
    ]
