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


# Soft accent backgrounds + matching text colors mirror the `mea-chip`
# `.cell-btn.<status>` palette in styles.css. Plain hex (oklch) so
# `style_data_conditional` can hand them straight to Dash's renderer.
_STATUS_STYLE = {
    "complete": {"bg": "oklch(0.62 0.1 155 / 0.16)", "fg": "oklch(0.42 0.1 155)"},
    "running":  {"bg": "oklch(0.7 0.13 80 / 0.18)",  "fg": "oklch(0.45 0.13 80)"},
    "failed":   {"bg": "oklch(0.6 0.16 28 / 0.16)",  "fg": "oklch(0.43 0.16 28)"},
    "not_run":  {"bg": "rgba(132,128,122,0.14)",     "fg": "#84807a"},
}


def _legend_swatch(status: str) -> html.Div:
    s = _STATUS_STYLE[status]
    return html.Span(
        [
            html.Span(
                className="swatch",
                style={"background": s["bg"], "borderColor": s["fg"]},
            ),
            status,
        ],
        style={"display": "inline-flex", "alignItems": "center"},
    )


_TASK_NAMES = [cls.task_name for cls in TASK_CLASSES]


_TABLE_STYLE_CELL = {
    "fontFamily": "var(--font-mono)",
    "fontSize": "12px",
    "padding": "8px 12px",
    "backgroundColor": "var(--bg-elev)",
    "color": "var(--ink)",
    "border": "0",
    "borderBottom": "1px solid var(--line-soft)",
    "textAlign": "left",
}
_TABLE_STYLE_HEADER = {
    "fontFamily": "var(--font-mono)",
    "fontSize": "10px",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.06em",
    "backgroundColor": "var(--bg)",
    "color": "var(--ink-3)",
    "border": "0",
    "borderBottom": "1px solid var(--line)",
    "padding": "8px 12px",
}


layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("workspace"),
                                html.Span("pipeline_manager"),
                                html.Span("pipeline_cache.json"),
                            ],
                            className="breadcrumb",
                        ),
                        html.H1("Pipeline status"),
                        html.Div(
                            "One row per (recording, well); one column per task. "
                            "Reset cascades to dependents.",
                            className="subtitle",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Button(
                            [html.Span("↻", className="glyph"), "Refresh"],
                            id="pipeline-refresh",
                            n_clicks=0,
                            className="btn",
                        ),
                    ],
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),
        html.Div(id="pipeline-banner-slot"),
        html.Div(
            [
                html.Div(
                    [
                        html.Span("legend", className="h-title"),
                        html.Span(
                            id="pipeline-status",
                            className="h-actions",
                            style={"color": "var(--ink-3)",
                                   "fontFamily": "var(--font-mono)",
                                   "fontSize": "11px",
                                   "textTransform": "none",
                                   "letterSpacing": "0"},
                        ),
                    ],
                    className="card-head",
                ),
                html.Div(
                    html.Div(
                        [_legend_swatch(s) for s in _STATUS_STYLE],
                        className="legend",
                    ),
                    className="card-body",
                ),
            ],
            className="card",
            style={"marginBottom": "12px"},
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
            style_cell=_TABLE_STYLE_CELL,
            style_header=_TABLE_STYLE_HEADER,
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Span("reset task on selected row", className="h-title"),
                    ],
                    className="card-head",
                ),
                html.Div(
                    [
                        html.Div(
                            dcc.Dropdown(
                                id="pipeline-reset-task",
                                options=[{"label": n, "value": n} for n in _TASK_NAMES],
                                value=None,
                                placeholder="pick a task",
                            ),
                            style={"flex": "1 1 260px"},
                        ),
                        html.Button(
                            [html.Span("⟲", className="glyph"),
                             "Reset (cascade dependents)"],
                            id="pipeline-reset-btn",
                            n_clicks=0,
                            className="btn primary",
                        ),
                        html.Span(
                            id="pipeline-reset-status",
                            style={"color": "var(--ink-3)",
                                   "fontFamily": "var(--font-mono)",
                                   "fontSize": "11px"},
                        ),
                    ],
                    className="card-body",
                    style={"display": "flex", "alignItems": "center", "gap": "12px"},
                ),
            ],
            className="card",
            style={"marginTop": "16px"},
        ),
    ],
    className="page",
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
    rules: list[dict] = []
    for col in task_names:
        for status, style in _STATUS_STYLE.items():
            rules.append(
                {
                    "if": {"filter_query": f'{{{col}}} = "{status}"', "column_id": col},
                    "backgroundColor": style["bg"],
                    "color": style["fg"],
                }
            )
    return rules
