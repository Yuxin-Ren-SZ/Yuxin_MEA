"""Recordings explorer — filter, queue, and (re)scan the dataset cache.

The page reads `experiment_cache.json` for display (via `data.load_recordings_df`,
which never mutates state) and uses the per-callback `DatasetManager` /
`PipelineManager` constructors in `dashboard.context` only when the user
actively clicks "Scan disk" or "Queue selected wells".
"""

from __future__ import annotations

from pathlib import Path

import dash
from dash import Input, Output, State, callback, dash_table, dcc, html
from flask import current_app

from yuxin_mea.dashboard.components import no_config_banner
from yuxin_mea.dashboard.context import load_dataset_mgr, load_pipeline_mgr
from yuxin_mea.dashboard.data import load_recordings_df


dash.register_page(__name__, path="/recordings", name="Recordings", order=1)


# Columns the filter dropdowns operate on (must be present in load_recordings_df).
_FILTER_COLUMNS = ["sample_id", "date", "plate_id", "scan_type", "run_id"]


# Shared DataTable styling — design tokens via inline CSS-var references
# (DataTable doesn't honor stylesheet classes for its internal cells, but
# CSS variables resolve through `inherit`).
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


def _filter_row() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Label(col, className="section-label"),
                    dcc.Dropdown(
                        id={"recordings-filter": col},
                        options=[],
                        value=None,
                        multi=True,
                        placeholder=f"any {col}",
                    ),
                ],
                style={"flex": "1 1 140px"},
            )
            for col in _FILTER_COLUMNS
        ],
        style={"display": "flex", "gap": "12px", "alignItems": "flex-end"},
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
                                html.Span("dataset_manager"),
                                html.Span("recordings"),
                            ],
                            className="breadcrumb",
                        ),
                        html.H1("Recordings"),
                        html.Div(
                            "Every recording known to the dataset cache. "
                            "Filter, then queue selected wells onto the pipeline.",
                            className="subtitle",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Button(
                            [html.Span("↻", className="glyph"), "Refresh"],
                            id="recordings-refresh",
                            n_clicks=0,
                            className="btn",
                        ),
                        html.Button(
                            [html.Span("⇣", className="glyph"), "Scan disk"],
                            id="recordings-scan",
                            n_clicks=0,
                            title="Walk data_root, rebuild experiment_cache.json",
                            className="btn",
                        ),
                        html.Button(
                            [html.Span("▸", className="glyph"), "Queue selected"],
                            id="recordings-queue",
                            n_clicks=0,
                            title="Register every well under selected recordings with the pipeline",
                            className="btn primary",
                        ),
                    ],
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),
        html.Div(id="recordings-banner-slot"),
        html.Div(
            [
                html.Div(
                    [html.Span("filters", className="h-title")],
                    className="card-head",
                ),
                html.Div(_filter_row(), className="card-body"),
            ],
            className="card",
            style={"marginBottom": "12px"},
        ),
        html.Div(
            html.Span(
                id="recordings-status",
                style={"color": "var(--ink-3)", "fontFamily": "var(--font-mono)",
                       "fontSize": "11px"},
            ),
            style={"marginBottom": "12px"},
        ),
        dash_table.DataTable(
            id="recordings-table",
            columns=[],
            data=[],
            row_selectable="multi",
            selected_rows=[],
            filter_action="native",
            sort_action="native",
            page_size=25,
            style_table={"overflowX": "auto"},
            style_cell=_TABLE_STYLE_CELL,
            style_header=_TABLE_STYLE_HEADER,
        ),
    ],
    className="page",
)


# ---------------------------------------------------------------------------
# Refresh / scan: the table content + the filter dropdown choices.
# ---------------------------------------------------------------------------


@callback(
    Output("recordings-banner-slot", "children"),
    Output("recordings-table", "data"),
    Output("recordings-table", "columns"),
    Output("recordings-status", "children"),
    Output({"recordings-filter": "sample_id"}, "options"),
    Output({"recordings-filter": "date"}, "options"),
    Output({"recordings-filter": "plate_id"}, "options"),
    Output({"recordings-filter": "scan_type"}, "options"),
    Output({"recordings-filter": "run_id"}, "options"),
    Input("recordings-refresh", "n_clicks"),
    Input("recordings-scan", "n_clicks"),
    Input({"recordings-filter": "sample_id"}, "value"),
    Input({"recordings-filter": "date"}, "value"),
    Input({"recordings-filter": "plate_id"}, "value"),
    Input({"recordings-filter": "scan_type"}, "value"),
    Input({"recordings-filter": "run_id"}, "value"),
)
def _refresh(
    _refresh_clicks: int,
    _scan_clicks: int,
    f_sample, f_date, f_plate, f_scan, f_run,
):
    ctx = current_app.config["YUXIN_MEA"]
    banner = None if ctx.get("config_exists") else no_config_banner()
    analysis_root = ctx["analysis_root"]
    if analysis_root is None:
        return banner, [], [], "analysis_root is not set in the config.", [], [], [], [], []

    triggered = dash.ctx.triggered_id
    scan_msg = ""
    if isinstance(triggered, str) and triggered == "recordings-scan":
        scan_msg = _do_scan() + "; "

    full = load_recordings_df(Path(analysis_root))
    full_count = len(full)

    def _opts(col: str):
        if full.empty:
            return []
        return [{"label": str(v), "value": str(v)} for v in sorted(full[col].dropna().unique())]
    options = [_opts(col) for col in _FILTER_COLUMNS]

    df = full
    for col, selected in zip(_FILTER_COLUMNS, [f_sample, f_date, f_plate, f_scan, f_run]):
        if selected:
            df = df[df[col].isin(selected)]

    columns = [{"name": c, "id": c} for c in df.columns]
    status = f"{scan_msg}{len(df)} of {full_count} recording(s) after filters"

    return (banner, df.to_dict("records"), columns, status, *options)


def _do_scan() -> str:
    """Run DatasetManager.refresh() and return a one-line summary."""
    dataset_mgr = load_dataset_mgr()
    if dataset_mgr is None:
        return "Scan skipped: data_root or analysis_root not configured"
    before = len(dataset_mgr.recordings)
    dataset_mgr.refresh()
    after = len(dataset_mgr.recordings)
    delta = after - before
    return f"Scanned disk: {after} recording(s) ({delta:+d})"


# ---------------------------------------------------------------------------
# Queue selected wells onto the pipeline.
# ---------------------------------------------------------------------------


@callback(
    Output("recordings-status", "children", allow_duplicate=True),
    Input("recordings-queue", "n_clicks"),
    State("recordings-table", "data"),
    State("recordings-table", "selected_rows"),
    prevent_initial_call=True,
)
def _queue_selected(_n_clicks: int, table_data, selected_rows):
    if not selected_rows:
        return "Queue: select at least one row first."
    if not table_data:
        return "Queue: table is empty."

    pm = load_pipeline_mgr()
    dataset_mgr = load_dataset_mgr()
    if pm is None or dataset_mgr is None:
        return "Queue: analysis_root or data_root not configured."

    selected_keys = [
        table_data[i]["cache_key"]
        for i in selected_rows
        if i < len(table_data)
    ]
    n_wells = 0
    n_recs = 0
    n_skipped = 0
    for cache_key in selected_keys:
        matches = dataset_mgr.get_recording_by([("cache_key", "==", cache_key)])
        if not matches:
            n_skipped += 1
            continue
        recording = matches[0]
        if not recording.h5_recordings:
            n_skipped += 1
            continue
        n_recs += 1
        for rec_name, well_ids in recording.h5_recordings.items():
            for well_id in well_ids:
                pm.add_well(cache_key, f"{rec_name}/{well_id}")
                n_wells += 1

    return (
        f"Queued {n_wells} well(s) across {n_recs} recording(s) "
        f"({n_skipped} skipped)."
    )
