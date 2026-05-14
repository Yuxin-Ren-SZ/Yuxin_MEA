"""Pipeline status page — task matrix from `pipeline_cache.json`.

Select a single row + a task → click Reset to flip that task (and its
transitive dependents) to NOT_RUN. Done via `PipelineManager.refresh()`.
"""

from __future__ import annotations

from pathlib import Path

import dash
from dash import Input, Output, State, callback, dash_table, dcc, html
from flask import current_app

from yuxin_mea.dashboard.components import no_config_banner
from yuxin_mea.dashboard.context import load_pipeline_mgr
from yuxin_mea.dashboard.data import load_pipeline_df
from yuxin_mea.tasks import TASK_CLASSES


dash.register_page(__name__, path="/pipeline", name="Pipeline", order=2)


_STATUS_COLORS = {
    "complete": "#c8e6c9",
    "running":  "#bbdefb",
    "failed":   "#ffcdd2",
    "not_run":  "#eceff1",
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


_TASK_NAMES = [cls.task_name for cls in TASK_CLASSES]


layout = html.Div(
    [
        html.H2("Pipeline status", style={"marginTop": "0"}),
        html.P(
            "One row per (recording, well). One column per task name seen in "
            "the cache. Em-dash cells (—) mean that task was never "
            "registered for the entry. Filter and sort using the column "
            "headers. Select a row, pick a task, and click Reset to flip "
            "that task — and every downstream task — back to NOT_RUN."
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
            row_selectable="single",
            selected_rows=[],
            filter_action="native",
            sort_action="native",
            page_size=25,
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "monospace", "fontSize": "13px", "padding": "4px 8px"},
            style_header={"fontWeight": "600", "backgroundColor": "#f4f6f8"},
        ),
        html.Div(
            [
                html.Span(
                    "Reset task on selected row:",
                    style={"marginRight": "8px", "fontSize": "13px"},
                ),
                dcc.Dropdown(
                    id="pipeline-reset-task",
                    options=[{"label": n, "value": n} for n in _TASK_NAMES],
                    value=None,
                    placeholder="pick a task",
                    style={"width": "260px", "display": "inline-block", "verticalAlign": "middle"},
                ),
                html.Button(
                    "Reset (cascade dependents)",
                    id="pipeline-reset-btn",
                    n_clicks=0,
                    style={"marginLeft": "8px"},
                ),
                html.Span(
                    id="pipeline-reset-status",
                    style={"marginLeft": "12px", "color": "#555"},
                ),
            ],
            style={"marginTop": "16px"},
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
    Input("pipeline-reset-btn", "n_clicks"),
)
def _refresh(_n_refresh: int, _n_reset: int):
    ctx = current_app.config["YUXIN_MEA"]
    banner = None if ctx.get("config_exists") else no_config_banner()
    analysis_root = ctx["analysis_root"]
    if analysis_root is None:
        return banner, [], [], [], "analysis_root is not set in the config."

    df, task_names = load_pipeline_df(Path(analysis_root))
    columns = [{"name": c, "id": c} for c in df.columns]
    style = _build_conditional_style(task_names)
    return banner, df.to_dict("records"), columns, style, f"{len(df)} pipeline entr(ies)"


@callback(
    Output("pipeline-reset-status", "children"),
    Input("pipeline-reset-btn", "n_clicks"),
    State("pipeline-table", "data"),
    State("pipeline-table", "selected_rows"),
    State("pipeline-reset-task", "value"),
    prevent_initial_call=True,
)
def _reset(_n_clicks: int, table_data, selected_rows, task_name: str | None):
    if not task_name:
        return "Reset: pick a task first."
    if not selected_rows:
        return "Reset: select a row first."
    if not table_data:
        return "Reset: table is empty."

    pm = load_pipeline_mgr()
    if pm is None:
        return "Reset: analysis_root not configured."

    row = table_data[selected_rows[0]]
    recording_key = row.get("recording_key")
    well_id = row.get("well_id")
    try:
        pm.refresh(task_name, recording_key=recording_key, well_id=well_id)
    except (ValueError, KeyError) as exc:
        return f"Reset failed: {exc}"
    return (
        f"Reset {task_name} (+ dependents) for "
        f"{recording_key}/{well_id}. Click Refresh to redraw."
    )


def _build_conditional_style(task_names: list[str]) -> list[dict]:
    return [
        {
            "if": {"filter_query": f'{{{col}}} = "{status}"', "column_id": col},
            "backgroundColor": color,
        }
        for col in task_names
        for status, color in _STATUS_COLORS.items()
    ]
