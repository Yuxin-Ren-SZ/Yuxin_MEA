"""Recordings page — master/detail browser matching the mea-chip design.

Left rail: recording cards grouped by sample_id, each with a pipeline
progress bar and a checkbox for bulk-queue.  Right pane: metadata KV grid +
wells table with per-task cell-btn status dots.

Selection state lives in a dcc.Store; a pattern-matched callback on the
card buttons writes to it; a second callback renders the detail pane.
Checked-keys (for bulk queue) live in a separate store.
"""

from __future__ import annotations

from pathlib import Path

import dash
from dash import ALL, Input, Output, State, callback, dcc, html
from flask import current_app

from yuxin_mea.dashboard.components import no_config_banner
from yuxin_mea.dashboard.context import load_dataset_mgr, load_pipeline_mgr
from yuxin_mea.dashboard.data import filter_recordings, load_recordings_detail
from yuxin_mea.tasks import TASK_CLASSES


dash.register_page(__name__, path="/recordings", name="Recordings", order=1)

_TASK_NAMES = [tc.task_name for tc in TASK_CLASSES]
_STATUS_OK = "complete"
_STATUS_RUN = "running"
_STATUS_FAIL = "failed"


# ---------------------------------------------------------------------------
# Static layout shell — data injected by callbacks
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        dcc.Store(id="recordings-selected-key", data=""),
        dcc.Store(id="recordings-checked-keys", data=[]),
        dcc.Store(id="recordings-data-store", data={}),

        # ── view-head ────────────────────────────────────────────────────
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
                        html.H1("Datasets"),
                        html.Div(id="recordings-subtitle", className="subtitle"),
                    ]
                ),
                html.Div(
                    [
                        html.Button(
                            [html.Span("⇣", className="glyph"), "Scan disk"],
                            id="recordings-scan",
                            n_clicks=0,
                            className="btn",
                            title="Walk data_root and rebuild experiment_cache.json",
                        ),
                        html.Button(
                            [html.Span("▸", className="glyph"), "Queue selected"],
                            id="recordings-queue-btn",
                            n_clicks=0,
                            className="btn primary",
                            title="Queue all checked recordings (or the active recording if none checked)",
                        ),
                        html.Button(
                            [html.Span("↻", className="glyph"), "refresh()"],
                            id="recordings-refresh",
                            n_clicks=0,
                            className="btn",
                        ),
                    ],
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),

        html.Div(id="recordings-banner-slot"),

        # ── filter bar ──────────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [html.Span("filters", className="h-title")],
                    className="card-head",
                ),
                html.Div(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Scan type", style={
                                        "fontFamily": "var(--font-mono)",
                                        "fontSize": "10px",
                                        "color": "var(--ink-3)",
                                        "marginBottom": "4px",
                                    }),
                                    dcc.Dropdown(
                                        id="recordings-filter-scan-type",
                                        options=[],
                                        value=[],
                                        multi=True,
                                        placeholder="any",
                                    ),
                                ],
                                style={"flex": "1 1 160px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Date", style={
                                        "fontFamily": "var(--font-mono)",
                                        "fontSize": "10px",
                                        "color": "var(--ink-3)",
                                        "marginBottom": "4px",
                                    }),
                                    dcc.Dropdown(
                                        id="recordings-filter-date",
                                        options=[],
                                        value=[],
                                        multi=True,
                                        placeholder="any",
                                    ),
                                ],
                                style={"flex": "1 1 160px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Queue status", style={
                                        "fontFamily": "var(--font-mono)",
                                        "fontSize": "10px",
                                        "color": "var(--ink-3)",
                                        "marginBottom": "4px",
                                    }),
                                    dcc.Dropdown(
                                        id="recordings-filter-queue-status",
                                        options=[
                                            {"label": "all", "value": "all"},
                                            {"label": "queued", "value": "queued"},
                                            {"label": "not queued", "value": "not_queued"},
                                        ],
                                        value="all",
                                        clearable=False,
                                    ),
                                ],
                                style={"flex": "1 1 130px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "16px",
                            "alignItems": "flex-start",
                            "flexWrap": "wrap",
                        },
                    ),
                    className="card-body",
                ),
            ],
            className="card",
        ),

        # ── status strip ─────────────────────────────────────────────────
        html.Div(
            id="recordings-status",
            style={
                "fontFamily": "var(--font-mono)",
                "fontSize": "11px",
                "color": "var(--ink-3)",
                "minHeight": "16px",
            },
        ),

        # ── master/detail body ───────────────────────────────────────────
        html.Div(
            [
                # Left: recording list
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("recordings", className="h-title"),
                                html.Div(
                                    html.Span(id="recordings-count", className="badge"),
                                    className="h-actions",
                                ),
                            ],
                            className="card-head",
                        ),
                        html.Div(
                            html.Div(id="recordings-list", className="rec-list"),
                            className="card-body flush",
                            style={"padding": "0"},
                        ),
                    ],
                    className="card",
                    style={"width": "340px", "flexShrink": "0"},
                ),

                # Right: metadata + wells
                html.Div(
                    [
                        html.Div(id="recordings-detail-meta"),
                        html.Div(id="recordings-detail-wells"),
                    ],
                    className="col",
                    style={"flex": "1", "minWidth": "0", "gap": "16px"},
                ),
            ],
            style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
        ),
    ],
    className="page",
)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _cell_btn(status: str | None) -> html.Button:
    s = status or "not_run"
    return html.Button(
        html.Span(className="dot"),
        className=f"cell-btn {s}",
        title=s,
        style={"cursor": "default"},
    )


def _rec_progress_bar(wells: list[str], well_pipeline_status: dict) -> html.Div:
    statuses = [
        well_pipeline_status.get(w, {})
        for w in wells
    ]
    total = max(1, len(wells) * len(_TASK_NAMES))
    done = sum(
        1 for s in statuses for tn in _TASK_NAMES if s.get(tn) == _STATUS_OK
    )
    running = sum(
        1 for s in statuses for tn in _TASK_NAMES if s.get(tn) == _STATUS_RUN
    )
    failed = sum(
        1 for s in statuses for tn in _TASK_NAMES if s.get(tn) == _STATUS_FAIL
    )
    ok_pct = f"{100 * done / total:.1f}%"
    run_pct = f"{100 * running / total:.1f}%"
    fail_pct = f"{100 * failed / total:.1f}%"
    pct_int = int(100 * done / total)
    return html.Div(
        [
            html.Div(
                [
                    html.Span(style={"background": "var(--ok)", "width": ok_pct}),
                    html.Span(style={"background": "var(--run)", "width": run_pct}),
                    html.Span(style={"background": "var(--fail)", "width": fail_pct}),
                ],
                className="rec-progress",
            ),
            html.Div(
                f"{pct_int}% pipeline complete",
                style={
                    "marginTop": "3px",
                    "fontFamily": "var(--font-mono)",
                    "fontSize": "10px",
                    "color": "var(--ink-4)",
                },
            ),
        ]
    )


def _build_rec_card(rec: dict, well_pipeline_status: dict, selected_key: str) -> html.Button:
    cache_key = rec["cache_key"]
    wells = [f"{cache_key}/{w}" for w in rec["wells"]]
    is_active = cache_key == selected_key
    return html.Button(
        [
            html.Div(
                [
                    html.Span(
                        f"{rec['plate_id']} · {rec['run_id']}",
                        style={"fontFamily": "var(--font-mono)", "fontSize": "12px", "fontWeight": "600"},
                    ),
                    html.Span(
                        rec["date"],
                        style={"fontFamily": "var(--font-mono)", "fontSize": "10px", "color": "var(--ink-3)"},
                    ),
                ],
                className="rec-card-title",
            ),
            html.Div(
                [
                    html.Span(rec["scan_type"], className="badge"),
                    html.Span(
                        f"{rec['n_wells']} wells",
                        style={"fontFamily": "var(--font-mono)", "fontSize": "10px", "color": "var(--ink-3)"},
                    ),
                ],
                className="rec-card-meta",
            ),
            _rec_progress_bar(wells, well_pipeline_status),
        ],
        id={"rec-card": cache_key},
        n_clicks=0,
        className=f"rec-card {'active' if is_active else ''}",
        style={"flex": "1", "minWidth": "0"},
    )


def _build_rec_row(
    rec: dict,
    well_pipeline_status: dict,
    selected_key: str,
    checked_keys: list[str],
) -> html.Div:
    cache_key = rec["cache_key"]
    is_checked = cache_key in checked_keys
    return html.Div(
        [
            dcc.Checklist(
                id={"rec-check": cache_key},
                options=[{"label": "", "value": cache_key}],
                value=[cache_key] if is_checked else [],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "paddingLeft": "8px",
                    "flexShrink": "0",
                },
            ),
            _build_rec_card(rec, well_pipeline_status, selected_key),
        ],
        style={"display": "flex", "alignItems": "stretch"},
    )


def _build_group_header(
    sample_id: str,
    n_recs: int,
    checked_keys: list[str],
    group_keys: set[str],
) -> html.Div:
    all_checked = group_keys.issubset(set(checked_keys)) if group_keys else False
    return html.Div(
        [
            html.Span(f"{sample_id} · {n_recs} run(s)"),
            html.Button(
                "deselect all" if all_checked else "select all",
                id={"rec-group-select": sample_id},
                n_clicks=0,
                className="btn ghost",
                style={"height": "18px", "padding": "0 6px", "fontSize": "10px"},
            ),
        ],
        className="rec-group-header",
        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
    )


def _build_rec_list(
    recordings: list[dict],
    well_pipeline_status: dict,
    selected_key: str,
    checked_keys: list[str],
) -> list:
    groups: dict[str, list[dict]] = {}
    for rec in recordings:
        groups.setdefault(rec["sample_id"], []).append(rec)

    children: list = []
    for sample_id, recs in groups.items():
        group_keys = {r["cache_key"] for r in recs}
        children.append(_build_group_header(sample_id, len(recs), checked_keys, group_keys))
        for rec in recs:
            children.append(_build_rec_row(rec, well_pipeline_status, selected_key, checked_keys))

    if not recordings:
        children = [
            html.Div(
                "No recordings found in experiment_cache.json.",
                style={
                    "padding": "32px 16px",
                    "fontFamily": "var(--font-mono)",
                    "fontSize": "12px",
                    "color": "var(--ink-3)",
                    "textAlign": "center",
                },
            )
        ]
    return children


def _build_meta_card(rec: dict) -> html.Div:
    def kv(label: str, value: str, path: bool = False) -> list:
        return [
            html.Dt(label),
            html.Dd(value, className="path" if path else ""),
        ]

    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"recording · {rec['run_id']}", className="h-title"),
                    html.Div(
                        [
                            html.Span(rec["scan_type"], className="badge"),
                        ],
                        className="h-actions",
                    ),
                ],
                className="card-head",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Dl(
                                kv("sample_id", rec["sample_id"])
                                + kv("date", rec["date"])
                                + kv("plate_id", rec["plate_id"])
                                + kv("scan_type", rec["scan_type"])
                                + kv("run_id", rec["run_id"]),
                                className="kv",
                            ),
                            html.Dl(
                                kv("file_size", f"{rec['file_size_mb']} MB")
                                + kv("n_wells", str(rec["n_wells"])),
                                className="kv",
                            ),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "0 28px"},
                    ),
                    html.Div(
                        html.Dl(
                            kv("cache_key", rec["cache_key"], path=True)
                            + kv("data_path", rec["data_path"], path=True),
                            className="kv",
                            style={"gridTemplateColumns": "90px 1fr"},
                        ),
                        style={
                            "marginTop": "12px",
                            "paddingTop": "12px",
                            "borderTop": "1px solid var(--line-soft)",
                        },
                    ),
                ],
                className="card-body",
            ),
        ],
        className="card",
    )


def _build_wells_table(rec: dict, well_pipeline_status: dict) -> html.Div:
    cache_key = rec["cache_key"]
    n_tasks = len(_TASK_NAMES)
    rows = []
    for compound_well in rec["wells"]:
        pk = f"{cache_key}/{compound_well}"
        task_map = well_pipeline_status.get(pk, {})
        task_cells = [
            html.Td(
                _cell_btn(task_map.get(tn)),
                style={"padding": "4px 4px", "textAlign": "center"},
            )
            for tn in _TASK_NAMES
        ]
        rows.append(
            html.Tr(
                [html.Td(compound_well, className="mono", style={"padding": "6px 12px", "fontSize": "11px"})]
                + task_cells
            )
        )

    empty = [
        html.Tr(
            html.Td(
                "No wells — recording not yet queued onto the pipeline.",
                colSpan=1 + n_tasks,
                style={
                    "padding": "32px 24px",
                    "textAlign": "center",
                    "color": "var(--ink-3)",
                    "fontFamily": "var(--font-mono)",
                    "fontSize": "12px",
                },
            )
        )
    ] if not rows else []

    task_headers = [
        html.Th(
            tn.replace("_", " "),
            style={"padding": "8px 4px", "textAlign": "center", "fontSize": "9px", "width": "36px"},
        )
        for tn in _TASK_NAMES
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"wells · {len(rec['wells'])}", className="h-title"),
                ],
                className="card-head",
            ),
            html.Div(
                html.Div(
                    html.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th(
                                            "compound well_id",
                                            className="well-col",
                                            style={"padding": "8px 12px", "textAlign": "left", "width": "200px"},
                                        ),
                                    ]
                                    + task_headers
                                )
                            ),
                            html.Tbody(rows + empty),
                        ],
                        className="tbl",
                    ),
                    className="tbl-wrap",
                ),
                className="card-body flush",
            ),
        ],
        className="card",
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("recordings-banner-slot", "children"),
    Output("recordings-subtitle", "children"),
    Output("recordings-count", "children"),
    Output("recordings-list", "children"),
    Output("recordings-data-store", "data"),
    Output("recordings-selected-key", "data"),
    Output("recordings-status", "children"),
    Output("recordings-filter-scan-type", "options"),
    Output("recordings-filter-date", "options"),
    Input("recordings-refresh", "n_clicks"),
    Input("recordings-scan", "n_clicks"),
    Input("recordings-filter-scan-type", "value"),
    Input("recordings-filter-date", "value"),
    Input("recordings-filter-queue-status", "value"),
    State("recordings-selected-key", "data"),
    State("recordings-checked-keys", "data"),
)
def _populate(
    _r: int,
    _s: int,
    filter_scan_type: list[str],
    filter_date: list[str],
    filter_queue_status: str,
    current_key: str,
    checked_keys: list[str],
):
    ctx_app = current_app.config["YUXIN_MEA"]
    banner = None if ctx_app.get("config_exists") else no_config_banner()
    analysis_root = ctx_app.get("analysis_root")
    status_msg = ""
    empty_opts: list[dict] = []

    if analysis_root is None:
        return (
            banner, "analysis_root not set", "0", [], {}, "",
            "analysis_root not configured.", empty_opts, empty_opts,
        )

    triggered = dash.ctx.triggered_id
    if isinstance(triggered, str) and triggered == "recordings-scan":
        mgr = load_dataset_mgr()
        if mgr:
            before = len(mgr.recordings)
            mgr.refresh()
            after = len(mgr.recordings)
            status_msg = f"Scanned disk: {after} recording(s) ({after - before:+d})"

    recordings, well_pipeline_status = load_recordings_detail(Path(analysis_root))

    scan_type_options = sorted({r["scan_type"] for r in recordings})
    date_options = sorted({r["date"] for r in recordings})

    filtered = filter_recordings(
        recordings, well_pipeline_status,
        scan_types=filter_scan_type or None,
        dates=filter_date or None,
        queue_status=filter_queue_status or "all",
    )

    subtitle = f"{len(filtered)} of {len(recordings)} recordings · root {analysis_root}"

    valid_keys = {r["cache_key"] for r in filtered}
    selected = current_key if current_key in valid_keys else (
        filtered[0]["cache_key"] if filtered else ""
    )

    store_data = {
        "recordings": recordings,
        "well_pipeline_status": well_pipeline_status,
    }

    list_children = _build_rec_list(
        filtered, well_pipeline_status, selected, checked_keys or [],
    )

    return (
        banner,
        subtitle,
        str(len(filtered)),
        list_children,
        store_data,
        selected,
        status_msg,
        [{"label": s, "value": s} for s in scan_type_options],
        [{"label": d, "value": d} for d in date_options],
    )


@callback(
    Output("recordings-selected-key", "data", allow_duplicate=True),
    Output("recordings-list", "children", allow_duplicate=True),
    Input({"rec-card": ALL}, "n_clicks"),
    State("recordings-data-store", "data"),
    State("recordings-selected-key", "data"),
    State("recordings-checked-keys", "data"),
    State("recordings-filter-scan-type", "value"),
    State("recordings-filter-date", "value"),
    State("recordings-filter-queue-status", "value"),
    prevent_initial_call=True,
)
def _select_recording(
    n_clicks_list,
    store_data,
    current_key,
    checked_keys,
    filter_scan_type,
    filter_date,
    filter_queue_status,
):
    triggered = dash.ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return dash.no_update, dash.no_update

    new_key = triggered.get("rec-card", current_key)
    recordings = store_data.get("recordings", [])
    well_pipeline_status = store_data.get("well_pipeline_status", {})

    filtered = filter_recordings(
        recordings, well_pipeline_status,
        scan_types=filter_scan_type or None,
        dates=filter_date or None,
        queue_status=filter_queue_status or "all",
    )

    list_children = _build_rec_list(
        filtered, well_pipeline_status, new_key, checked_keys or [],
    )
    return new_key, list_children


@callback(
    Output("recordings-checked-keys", "data"),
    Input({"rec-check": ALL}, "value"),
    prevent_initial_call=True,
)
def _update_checked(all_values):
    checked: set[str] = set()
    for vals in all_values:
        if vals:
            checked.update(vals)
    return sorted(checked)


@callback(
    Output("recordings-checked-keys", "data", allow_duplicate=True),
    Output("recordings-list", "children", allow_duplicate=True),
    Input({"rec-group-select": ALL}, "n_clicks"),
    State("recordings-data-store", "data"),
    State("recordings-checked-keys", "data"),
    State("recordings-selected-key", "data"),
    State("recordings-filter-scan-type", "value"),
    State("recordings-filter-date", "value"),
    State("recordings-filter-queue-status", "value"),
    prevent_initial_call=True,
)
def _select_all_in_group(
    n_clicks_list,
    store_data,
    checked_keys,
    selected_key,
    filter_scan_type,
    filter_date,
    filter_queue_status,
):
    triggered = dash.ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return dash.no_update, dash.no_update

    sample_id = triggered.get("rec-group-select")
    if not sample_id:
        return dash.no_update, dash.no_update

    recordings = store_data.get("recordings", [])
    well_pipeline_status = store_data.get("well_pipeline_status", {})

    filtered = filter_recordings(
        recordings, well_pipeline_status,
        scan_types=filter_scan_type or None,
        dates=filter_date or None,
        queue_status=filter_queue_status or "all",
    )

    group_keys = {r["cache_key"] for r in filtered if r["sample_id"] == sample_id}
    checked = set(checked_keys or [])

    if group_keys.issubset(checked):
        checked -= group_keys
    else:
        checked |= group_keys

    new_checked = sorted(checked)
    list_children = _build_rec_list(
        filtered, well_pipeline_status, selected_key or "", new_checked,
    )
    return new_checked, list_children


@callback(
    Output("recordings-detail-meta", "children"),
    Output("recordings-detail-wells", "children"),
    Input("recordings-selected-key", "data"),
    State("recordings-data-store", "data"),
)
def _update_detail(selected_key: str, store_data: dict):
    if not selected_key or not store_data:
        placeholder = html.Div(
            "Select a recording from the list.",
            style={
                "padding": "60px",
                "textAlign": "center",
                "color": "var(--ink-3)",
                "fontFamily": "var(--font-mono)",
                "fontSize": "12px",
            },
        )
        return placeholder, None

    recordings = store_data.get("recordings", [])
    well_pipeline_status = store_data.get("well_pipeline_status", {})
    rec = next((r for r in recordings if r["cache_key"] == selected_key), None)
    if rec is None:
        return html.Div("Recording not found."), None

    return _build_meta_card(rec), _build_wells_table(rec, well_pipeline_status)


@callback(
    Output("recordings-status", "children", allow_duplicate=True),
    Input("recordings-queue-btn", "n_clicks"),
    State("recordings-checked-keys", "data"),
    State("recordings-selected-key", "data"),
    State("recordings-data-store", "data"),
    prevent_initial_call=True,
)
def _queue_recordings(
    _n: int,
    checked_keys: list[str],
    selected_key: str,
    store_data: dict,
):
    keys_to_queue = checked_keys if checked_keys else (
        [selected_key] if selected_key else []
    )
    if not keys_to_queue:
        return "Queue: select a recording first."

    pm = load_pipeline_mgr()
    dataset_mgr = load_dataset_mgr()
    if pm is None or dataset_mgr is None:
        return "Queue: analysis_root or data_root not configured."

    recordings = store_data.get("recordings", [])
    rec_lookup = {r["cache_key"]: r for r in recordings}

    total_wells = 0
    total_recs = 0
    for key in keys_to_queue:
        rec = rec_lookup.get(key)
        if rec is None:
            continue
        n_wells = 0
        for compound_well in rec["wells"]:
            pm.add_well(key, compound_well)
            n_wells += 1
        if n_wells:
            total_recs += 1
            total_wells += n_wells

    return f"Queued {total_wells} well(s) across {total_recs} recording(s)."
