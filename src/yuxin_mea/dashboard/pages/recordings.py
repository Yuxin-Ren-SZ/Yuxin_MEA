"""Recordings explorer — table view of `experiment_cache.json`."""

from __future__ import annotations

import dash
from dash import Input, Output, callback, dash_table, html
from flask import current_app

from yuxin_mea.dashboard.data import load_recordings_df


dash.register_page(__name__, path="/recordings", name="Recordings", order=1)


layout = html.Div(
    [
        html.H2("Recordings", style={"marginTop": "0"}),
        html.P(
            "Every recording known to the dataset cache. "
            "Click column headers to sort; the row of filter boxes under each "
            "header filters on substring match."
        ),
        html.Div(
            [
                html.Button("Refresh", id="recordings-refresh", n_clicks=0),
                html.Span(id="recordings-status", style={"marginLeft": "12px", "color": "#555"}),
            ],
            style={"marginBottom": "12px"},
        ),
        dash_table.DataTable(
            id="recordings-table",
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
    Output("recordings-table", "data"),
    Output("recordings-table", "columns"),
    Output("recordings-status", "children"),
    Input("recordings-refresh", "n_clicks"),
)
def _refresh(_n_clicks: int) -> tuple[list[dict], list[dict], str]:
    analysis_root = current_app.config["YUXIN_MEA"]["analysis_root"]
    if analysis_root is None:
        return [], [], "analysis_root is not set in the config."
    df = load_recordings_df(analysis_root)
    columns = [{"name": c, "id": c} for c in df.columns]
    return df.to_dict("records"), columns, f"{len(df)} recording(s)"
