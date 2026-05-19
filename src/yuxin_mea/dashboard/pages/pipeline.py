"""Pipeline page — DAG + status matrix + task inspector.

Layout:
  view-head          — breadcrumb + title + toggle-group filter + actions
  task DAG card      — horizontal dag-nodes with throughput meters
  [matrix | inspector] — html.Table status matrix + task detail panel
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import dash
from dash import ALL, Input, Output, State, callback, clientside_callback, dcc, html
from flask import current_app

from yuxin_mea.dashboard.components import no_config_banner
from yuxin_mea.dashboard.context import load_pipeline_mgr
from yuxin_mea.dashboard.data import load_pipeline_df
from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
from yuxin_mea.tasks import TASK_CLASSES


dash.register_page(__name__, path="/pipeline", name="Pipeline", order=2)

_TASK_NAMES = [tc.task_name for tc in TASK_CLASSES]
_TASK_DEPS: dict[str, list[str]] = {
    tc.task_name: list(getattr(tc, "dependencies", []))
    for tc in TASK_CLASSES
}

_STATUS_OK = "complete"
_STATUS_RUN = "running"
_STATUS_FAIL = "failed"
_STATUS_IDLE = "not_run"


# ---------------------------------------------------------------------------
# Static layout shell
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        dcc.Store(id="pipeline-selected-cell", data={}),
        dcc.Store(id="pipeline-filter-store", data="all"),

        # ── view-head ────────────────────────────────────────────────────
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
                        html.H1("Pipeline"),
                        html.Div(id="pipeline-subtitle", className="subtitle"),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button(
                                    f,
                                    id={"pipeline-filter-btn": f},
                                    n_clicks=0,
                                    className="active" if f == "all" else "",
                                )
                                for f in ("all", "incomplete", "running", "failing")
                            ],
                            className="toggle-group",
                        ),
                        html.Button(
                            [html.Span("↻", className="glyph"), "refresh()"],
                            id="pipeline-refresh",
                            n_clicks=0,
                            className="btn",
                        ),
                        html.Button(
                            [html.Span("▸", className="glyph"), "get_next_task()"],
                            id="pipeline-dispatch",
                            n_clicks=0,
                            className="btn primary",
                        ),
                    ],
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),

        html.Div(id="pipeline-banner-slot"),

        # ── DAG card ─────────────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [html.Span("task DAG · register_task() order", className="h-title")],
                    className="card-head",
                ),
                html.Div(
                    html.Div(id="pipeline-dag", className="dag"),
                    className="card-body flush",
                ),
            ],
            className="card",
        ),

        # ── matrix + inspector ───────────────────────────────────────────
        html.Div(
            [
                # Matrix (left, flex)
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(id="pipeline-matrix-title", className="h-title"),
                                html.Div(
                                    [
                                        html.Span(
                                            [
                                                html.Span(className="swatch", style={"background": "var(--ok)"}),
                                                "complete",
                                            ],
                                            style={"display": "inline-flex", "alignItems": "center"},
                                        ),
                                        html.Span(
                                            [
                                                html.Span(className="swatch", style={"background": "var(--run)"}),
                                                "running",
                                            ],
                                            style={"display": "inline-flex", "alignItems": "center"},
                                        ),
                                        html.Span(
                                            [
                                                html.Span(className="swatch", style={"background": "var(--fail)"}),
                                                "failed",
                                            ],
                                            style={"display": "inline-flex", "alignItems": "center"},
                                        ),
                                        html.Span(
                                            [
                                                html.Span(className="swatch", style={"background": "var(--idle)"}),
                                                "not_run",
                                            ],
                                            style={"display": "inline-flex", "alignItems": "center"},
                                        ),
                                    ],
                                    className="legend h-actions",
                                ),
                            ],
                            className="card-head",
                        ),
                        html.Div(
                            html.Div(
                                html.Table(id="pipeline-matrix", className="matrix"),
                                className="tbl-wrap",
                                style={"maxHeight": "480px"},
                            ),
                            className="card-body flush",
                        ),
                    ],
                    className="card",
                    style={"flex": "1", "minWidth": "0"},
                ),

                # Inspector (right, fixed width)
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(id="pipeline-inspector-title", className="h-title"),
                                html.Div(id="pipeline-inspector-pill", className="h-actions"),
                            ],
                            className="card-head",
                        ),
                        html.Div(id="pipeline-inspector-body", className="card-body"),
                    ],
                    className="card",
                    style={"width": "360px", "flexShrink": "0"},
                ),
            ],
            style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
        ),

        # ── bulk reset card ─────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [
                        html.Span("bulk reset · filter and refresh()", className="h-title"),
                        html.Span(
                            "Cascades dependents. Tells you what would change before it writes.",
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
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Tasks", className="section-label"),
                                        dcc.Dropdown(
                                            id="bulk-reset-tasks",
                                            options=[{"label": tn, "value": tn} for tn in _TASK_NAMES],
                                            value=[],
                                            multi=True,
                                            placeholder="pick one or more tasks",
                                        ),
                                    ],
                                    style={"flex": "1 1 240px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Recordings (empty = all)", className="section-label"),
                                        dcc.Dropdown(
                                            id="bulk-reset-recordings",
                                            options=[],
                                            value=[],
                                            multi=True,
                                            placeholder="any recording",
                                        ),
                                    ],
                                    style={"flex": "1 1 240px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Wells (empty = all in scope)", className="section-label"),
                                        dcc.Dropdown(
                                            id="bulk-reset-wells",
                                            options=[],
                                            value=[],
                                            multi=True,
                                            placeholder="any well",
                                        ),
                                    ],
                                    style={"flex": "1 1 240px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Status (empty = any)", className="section-label"),
                                        dcc.Dropdown(
                                            id="bulk-reset-statuses",
                                            options=[
                                                {"label": s, "value": s}
                                                for s in (_STATUS_OK, _STATUS_RUN, _STATUS_FAIL, _STATUS_IDLE)
                                            ],
                                            value=[],
                                            multi=True,
                                            placeholder="any status",
                                        ),
                                    ],
                                    style={"flex": "1 1 180px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "gap": "16px",
                                "alignItems": "flex-start",
                                "flexWrap": "wrap",
                            },
                        ),
                        html.Div(
                            [
                                html.Button(
                                    [html.Span("…", className="glyph"), "Preview"],
                                    id="bulk-reset-preview",
                                    n_clicks=0,
                                    className="btn",
                                ),
                                html.Button(
                                    [html.Span("⟲", className="glyph"), "Reset selected"],
                                    id="bulk-reset-execute",
                                    n_clicks=0,
                                    className="btn primary",
                                ),
                                html.Span(
                                    id="bulk-reset-status",
                                    style={"fontFamily": "var(--font-mono)",
                                           "fontSize": "11px",
                                           "color": "var(--ink-3)"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center",
                                   "gap": "8px", "marginTop": "14px"},
                        ),
                    ],
                    className="card-body",
                ),
            ],
            className="card",
            style={"marginTop": "16px"},
        ),
    ],
    className="page",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pill(status: str) -> html.Span:
    cls_map = {
        _STATUS_OK: "ok", _STATUS_RUN: "run", _STATUS_FAIL: "fail", _STATUS_IDLE: "idle",
    }
    cls = cls_map.get(status, "idle")
    return html.Span(
        [html.Span(className="swatch"), status],
        className=f"pill {cls}",
    )


def _cell_btn(
    status: str,
    pipeline_key: str,
    task_name: str,
    blocked: bool,
    selected: bool,
) -> html.Button:
    s = status or _STATUS_IDLE
    classes = f"cell-btn {s}"
    if blocked:
        classes += " blocked"
    if selected:
        classes += " selected"
    return html.Button(
        html.Span(className="dot"),
        id={"pipeline-cell": pipeline_key, "task": task_name},
        n_clicks=0,
        className=classes,
        title=f"{task_name}: {s}" + (" · deps not met" if blocked else ""),
    )


def _dag_node(idx: int, task_cls, counts: dict[str, int], selected_task: str | None) -> html.Div:
    tn = task_cls.task_name
    deps = list(getattr(task_cls, "dependencies", []))
    dep_label = f"← {', '.join(deps)}" if deps else "· root"
    c = counts.get(tn, {_STATUS_OK: 0, _STATUS_RUN: 0, _STATUS_FAIL: 0, _STATUS_IDLE: 0})
    total = max(1, c[_STATUS_OK] + c[_STATUS_RUN] + c[_STATUS_FAIL] + c[_STATUS_IDLE])
    ok_pct = f"{100 * c[_STATUS_OK] / total:.1f}%"
    run_pct = f"{100 * c[_STATUS_RUN] / total:.1f}%"
    fail_pct = f"{100 * c[_STATUS_FAIL] / total:.1f}%"
    is_selected = tn == selected_task
    return html.Div(
        [
            html.Div(f"stage {idx} {dep_label}", className="idx"),
            html.Div(tn, className="name"),
            html.Div(
                [
                    html.Span(style={"background": "var(--ok)", "width": ok_pct}),
                    html.Span(style={"background": "var(--run)", "width": run_pct}),
                    html.Span(style={"background": "var(--fail)", "width": fail_pct}),
                ],
                className="meter",
            ),
            html.Div(
                [
                    html.Span([html.B(str(c[_STATUS_OK])), " done"]),
                    html.Span(
                        f"{c[_STATUS_RUN]}▸ · {c[_STATUS_FAIL]}✗ · {c[_STATUS_IDLE]}○",
                        style={"color": "var(--ink-3)"},
                    ),
                ],
                className="stats",
            ),
        ],
        className=f"dag-node {'selected' if is_selected else ''}",
    )


# ---------------------------------------------------------------------------
# Filter toggle: clientside — update button classes + store
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(clicks_all, clicks_incomplete, clicks_running, clicks_failing, current) {
        var triggered = window.dash_clientside.callback_context.triggered;
        if (!triggered || triggered.length === 0) { return [current, 'active', '', '', '']; }
        var prop = triggered[0].prop_id;
        var val = 'all';
        if (prop.indexOf('"all"') >= 0)        val = 'all';
        if (prop.indexOf('"incomplete"') >= 0) val = 'incomplete';
        if (prop.indexOf('"running"') >= 0)    val = 'running';
        if (prop.indexOf('"failing"') >= 0)    val = 'failing';
        return [
            val,
            val === 'all'        ? 'active' : '',
            val === 'incomplete' ? 'active' : '',
            val === 'running'    ? 'active' : '',
            val === 'failing'    ? 'active' : '',
        ];
    }
    """,
    Output("pipeline-filter-store", "data"),
    Output({"pipeline-filter-btn": "all"}, "className"),
    Output({"pipeline-filter-btn": "incomplete"}, "className"),
    Output({"pipeline-filter-btn": "running"}, "className"),
    Output({"pipeline-filter-btn": "failing"}, "className"),
    Input({"pipeline-filter-btn": "all"}, "n_clicks"),
    Input({"pipeline-filter-btn": "incomplete"}, "n_clicks"),
    Input({"pipeline-filter-btn": "running"}, "n_clicks"),
    Input({"pipeline-filter-btn": "failing"}, "n_clicks"),
    State("pipeline-filter-store", "data"),
)


# ---------------------------------------------------------------------------
# Cell selection → store
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function() {
        var triggered = window.dash_clientside.callback_context.triggered;
        if (!triggered || triggered.length === 0) { return window.dash_clientside.no_update; }
        var prop_id = triggered[0].prop_id;
        try {
            var id_part = prop_id.split('.')[0];
            var id_obj = JSON.parse(id_part);
            return {"pipeline_key": id_obj["pipeline-cell"], "task": id_obj["task"]};
        } catch(e) { return window.dash_clientside.no_update; }
    }
    """,
    Output("pipeline-selected-cell", "data"),
    Input({"pipeline-cell": ALL, "task": ALL}, "n_clicks"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Main refresh callback
# ---------------------------------------------------------------------------

@callback(
    Output("pipeline-banner-slot", "children"),
    Output("pipeline-subtitle", "children"),
    Output("pipeline-dag", "children"),
    Output("pipeline-matrix", "children"),
    Output("pipeline-matrix-title", "children"),
    Input("pipeline-refresh", "n_clicks"),
    Input("pipeline-filter-store", "data"),
    Input("pipeline-selected-cell", "data"),
)
def _refresh(_n: int, filter_val: str, selected_cell: dict):
    ctx_app = current_app.config["YUXIN_MEA"]
    banner = None if ctx_app.get("config_exists") else no_config_banner()
    analysis_root = ctx_app.get("analysis_root")

    if analysis_root is None:
        return banner, "analysis_root not set", [], [], "status matrix"

    pipe_df, task_names = load_pipeline_df(Path(analysis_root))
    n_entries = len(pipe_df)
    subtitle = f"{len(task_names)}-task DAG · {n_entries} pipeline entries · cache: pipeline_cache.json"

    # Per-task counts
    counts: dict[str, dict[str, int]] = {
        tn: {_STATUS_OK: 0, _STATUS_RUN: 0, _STATUS_FAIL: 0, _STATUS_IDLE: 0}
        for tn in _TASK_NAMES
    }
    for tn in task_names:
        if tn not in counts:
            counts[tn] = {_STATUS_OK: 0, _STATUS_RUN: 0, _STATUS_FAIL: 0, _STATUS_IDLE: 0}
        for val in pipe_df[tn]:
            if val in counts[tn]:
                counts[tn][val] += 1

    selected_task = (selected_cell or {}).get("task")
    # Insert explicit arrows only on actual dependency edges, not between
    # parallel siblings (e.g. burst_detection and iterative_burst_detection
    # both depend on auto_curation — no arrow between them).
    dag_items: list = []
    for i, tc in enumerate(TASK_CLASSES):
        dag_items.append(_dag_node(i, tc, counts, selected_task))
        if i + 1 < len(TASK_CLASSES):
            next_deps = list(getattr(TASK_CLASSES[i + 1], "dependencies", []))
            if tc.task_name in next_deps:
                dag_items.append(html.Span("→", className="dag-arrow"))

    # Filter rows
    filter_val = filter_val or "all"
    if not pipe_df.empty and task_names:
        if filter_val == "incomplete":
            mask = pipe_df[task_names].apply(lambda r: any(v != _STATUS_OK for v in r), axis=1)
            filtered = pipe_df[mask]
        elif filter_val == "running":
            mask = pipe_df[task_names].apply(lambda r: any(v == _STATUS_RUN for v in r), axis=1)
            filtered = pipe_df[mask]
        elif filter_val == "failing":
            mask = pipe_df[task_names].apply(lambda r: any(v == _STATUS_FAIL for v in r), axis=1)
            filtered = pipe_df[mask]
        else:
            filtered = pipe_df
    else:
        filtered = pipe_df

    # Build matrix
    sel_pk = (selected_cell or {}).get("pipeline_key", "")
    sel_task = (selected_cell or {}).get("task", "")

    task_cols = task_names or _TASK_NAMES
    header = html.Thead(
        html.Tr(
            [html.Th("pipeline entry", className="well-col")]
            + [html.Th(tn.replace("_", " "), style={"whiteSpace": "nowrap"}) for tn in task_cols]
            + [html.Th("")]
        )
    )

    rows = []
    for _, row in filtered.iterrows():
        pk = f"{row['recording_key']}/{row['well_id']}"
        is_selected_row = pk == sel_pk
        cells = []
        for tn in task_cols:
            status = row.get(tn, _STATUS_IDLE)
            deps = _TASK_DEPS.get(tn, [])
            deps_done = all(row.get(d, _STATUS_IDLE) == _STATUS_OK for d in deps if d in pipe_df.columns)
            blocked = not deps_done and status == _STATUS_IDLE
            selected_btn = pk == sel_pk and tn == sel_task
            cells.append(
                html.Td(_cell_btn(status, pk, tn, blocked, selected_btn))
            )
        rows.append(
            html.Tr(
                [
                    html.Td(
                        [
                            html.Div(row["recording_key"], style={"color": "var(--ink)"}),
                            html.Div(
                                row["well_id"],
                                style={"color": "var(--ink-4)", "fontSize": "10px"},
                            ),
                        ],
                        className="well-cell",
                    ),
                ]
                + cells
                + [
                    html.Td(
                        html.Button("⋯", className="btn ghost", style={"height": "22px", "padding": "0 6px"}),
                        style={"textAlign": "right", "paddingRight": "12px"},
                    )
                ],
                className="selected" if is_selected_row else "",
            )
        )

    if not rows:
        rows = [
            html.Tr(
                html.Td(
                    "No entries match the current filter.",
                    colSpan=len(task_cols) + 2,
                    style={
                        "padding": "48px 24px",
                        "textAlign": "center",
                        "color": "var(--ink-3)",
                        "fontFamily": "var(--font-mono)",
                        "fontSize": "12px",
                    },
                )
            )
        ]

    matrix_title = f"status matrix · {len(filtered)} entries × {len(task_cols)} tasks"
    matrix = [header, html.Tbody(rows)]

    return banner, subtitle, dag_items, matrix, matrix_title


# ---------------------------------------------------------------------------
# Inspector callback
# ---------------------------------------------------------------------------

@callback(
    Output("pipeline-inspector-title", "children"),
    Output("pipeline-inspector-pill", "children"),
    Output("pipeline-inspector-body", "children"),
    Input("pipeline-selected-cell", "data"),
    Input("pipeline-refresh", "n_clicks"),
)
def _update_inspector(selected_cell: dict, _n: int):
    empty_title = "select a cell"
    empty_body = html.Div(
        "Click any matrix cell to inspect the TaskRecord, config snapshot, and outputs.",
        style={
            "color": "var(--ink-3)",
            "fontFamily": "var(--font-mono)",
            "fontSize": "12px",
            "padding": "24px 0",
        },
    )

    if not selected_cell:
        return empty_title, None, empty_body

    pipeline_key = selected_cell.get("pipeline_key", "")
    task_name = selected_cell.get("task", "")
    if not pipeline_key or not task_name:
        return empty_title, None, empty_body

    ctx_app = current_app.config["YUXIN_MEA"]
    analysis_root = ctx_app.get("analysis_root")
    if not analysis_root:
        return f"task · {task_name}", None, html.Div("analysis_root not configured.")

    store = JsonPipelineCacheStore(Path(analysis_root))
    entries = store.load()
    entry = entries.get(pipeline_key)
    if entry is None:
        return f"task · {task_name}", None, html.Div("Pipeline entry not found.")

    record = entry.tasks.get(task_name)
    if record is None:
        return f"task · {task_name}", _pill(_STATUS_IDLE), html.Div("Task not yet registered.")

    status = record.status
    last_updated = (
        datetime.fromtimestamp(record.last_updated).strftime("%Y-%m-%d %H:%M:%S")
        if record.last_updated else "—"
    )

    def kv(label: str, value: str, path: bool = False) -> list:
        return [
            html.Dt(label),
            html.Dd(str(value), className="path" if path else ""),
        ]

    body_parts: list = [
        html.Dl(
            kv("pipeline_key", pipeline_key, path=True)
            + kv("task_name", task_name)
            + kv("dependencies", ", ".join(_TASK_DEPS.get(task_name, [])) or "—")
            + kv("status", status)
            + kv("last_updated", last_updated)
            + kv("output_path", str(record.output_path) if record.output_path else "—", path=True),
            className="kv",
            style={"gridTemplateColumns": "110px 1fr"},
        ),
    ]

    if status == _STATUS_FAIL and record.error:
        body_parts += [
            html.Div(
                "error",
                className="section-label",
                style={"marginTop": "14px"},
            ),
            html.Pre(
                record.error,
                className="code",
                style={
                    "background": "var(--fail-soft)",
                    "borderColor": "oklch(0.6 0.16 28 / 0.35)",
                    "color": "oklch(0.4 0.16 28)",
                    "maxHeight": "120px",
                },
            ),
        ]

    if record.config:
        body_parts += [
            html.Div(
                "config snapshot",
                className="section-label",
                style={"marginTop": "14px"},
            ),
            html.Pre(
                json.dumps(record.config, indent=2, default=str),
                className="code",
                style={"maxHeight": "160px"},
            ),
        ]

    body_parts += [
        html.Div(
            [
                html.Button(
                    [html.Span("↻", className="glyph"), "refresh()"],
                    id="pipeline-inspector-refresh",
                    n_clicks=0,
                    className="btn",
                ),
                html.Button(
                    [html.Span("⟲", className="glyph"), "Reset task"],
                    id="pipeline-inspector-reset",
                    n_clicks=0,
                    className="btn primary",
                ),
                html.Span(
                    id="pipeline-inspector-reset-status",
                    style={"fontFamily": "var(--font-mono)", "fontSize": "11px", "color": "var(--ink-3)"},
                ),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "8px", "marginTop": "16px"},
        )
    ]

    return f"task · {task_name}", _pill(status), body_parts


# ---------------------------------------------------------------------------
# Reset task via inspector button
# ---------------------------------------------------------------------------

@callback(
    Output("pipeline-inspector-reset-status", "children"),
    Input("pipeline-inspector-reset", "n_clicks"),
    State("pipeline-selected-cell", "data"),
    prevent_initial_call=True,
)
def _reset_task(_n: int, selected_cell: dict):
    if not selected_cell:
        return "Reset: select a cell first."

    pipeline_key = selected_cell.get("pipeline_key", "")
    task_name = selected_cell.get("task", "")
    if not pipeline_key or not task_name:
        return "Reset: invalid selection."

    # pipeline_key = recording_key/well_id — split at last /
    parts = pipeline_key.rsplit("/", 1)
    if len(parts) != 2:
        return f"Reset: cannot parse pipeline_key {pipeline_key!r}."
    recording_key, well_id = parts

    pm = load_pipeline_mgr()
    if pm is None:
        return "Reset: analysis_root not configured."

    try:
        n = pm.refresh(task_name, recording_key=recording_key, well_ids=[well_id])
    except (ValueError, KeyError) as exc:
        return f"Reset failed: {exc}"

    return f"Reset {task_name} (+ dependents) — {n} record(s). Click refresh() to redraw."


# ---------------------------------------------------------------------------
# Bulk reset card
# ---------------------------------------------------------------------------

@callback(
    Output("bulk-reset-recordings", "options"),
    Input("pipeline-refresh", "n_clicks"),
)
def _populate_bulk_recordings(_n: int):
    """Populate the recordings multi-select from the pipeline cache.

    Re-fires on refresh() so newly-queued recordings appear without a
    full page reload.
    """
    ctx_app = current_app.config["YUXIN_MEA"]
    analysis_root = ctx_app.get("analysis_root")
    if not analysis_root:
        return []
    pipe_df, _ = load_pipeline_df(Path(analysis_root))
    if pipe_df.empty or "recording_key" not in pipe_df.columns:
        return []
    keys = sorted(set(pipe_df["recording_key"].tolist()))
    return [{"label": k, "value": k} for k in keys]


@callback(
    Output("bulk-reset-wells", "options"),
    Input("bulk-reset-recordings", "value"),
)
def _populate_bulk_wells(recording_keys: list[str] | None):
    """Enumerate well_ids for the selected recordings. Empty selection → no
    options (the UI treats empty wells as "all in scope")."""
    if not recording_keys:
        return []
    pm = load_pipeline_mgr()
    if pm is None:
        return []
    wells: set[str] = set()
    for rk in recording_keys:
        for entry in pm.get_entries_for_recording(rk):
            wells.add(entry.well_id)
    return [{"label": w, "value": w} for w in sorted(wells)]


def _filter_entries(pm, recording_keys, well_ids):
    if recording_keys:
        entries = []
        for rk in recording_keys:
            entries.extend(pm.get_entries_for_recording(rk))
    else:
        entries = pm.all_entries()
    if well_ids:
        wanted = set(well_ids)
        entries = [e for e in entries if e.well_id in wanted]
    return entries


def _count_would_reset(pm, tasks, recording_keys, well_ids, statuses):
    """Dry-run count: how many records would each task reset, including cascade?"""
    counts: dict[str, int] = {}
    entries = _filter_entries(pm, recording_keys, well_ids)
    status_set = set(statuses) if statuses else None
    for task in tasks:
        try:
            cascade = pm.cascade_tasks(task)
        except (KeyError, ValueError):
            counts[task] = 0
            continue
        n = 0
        for e in entries:
            for t in cascade:
                rec = e.tasks.get(t)
                if rec is None:
                    continue
                if status_set is not None and rec.status not in status_set:
                    continue
                n += 1
        counts[task] = n
    return counts


@callback(
    Output("bulk-reset-status", "children"),
    Input("bulk-reset-preview", "n_clicks"),
    State("bulk-reset-tasks", "value"),
    State("bulk-reset-recordings", "value"),
    State("bulk-reset-wells", "value"),
    State("bulk-reset-statuses", "value"),
    prevent_initial_call=True,
)
def _bulk_reset_preview(_n, tasks, recordings, wells, statuses):
    if not tasks:
        return "Preview: pick at least one task."
    pm = load_pipeline_mgr()
    if pm is None:
        return "Preview: analysis_root not configured."
    counts = _count_would_reset(pm, tasks, recordings, wells, statuses)
    total = sum(counts.values())
    if total == 0:
        return "Preview: nothing matches the filter."
    breakdown = ", ".join(f"{t}={n}" for t, n in counts.items() if n)
    return f"Preview: would reset {total} record(s) — {breakdown}."


@callback(
    Output("bulk-reset-status", "children", allow_duplicate=True),
    Input("bulk-reset-execute", "n_clicks"),
    State("bulk-reset-tasks", "value"),
    State("bulk-reset-recordings", "value"),
    State("bulk-reset-wells", "value"),
    State("bulk-reset-statuses", "value"),
    prevent_initial_call=True,
)
def _bulk_reset_execute(_n, tasks, recordings, wells, statuses):
    if not tasks:
        return "Reset: pick at least one task."
    pm = load_pipeline_mgr()
    if pm is None:
        return "Reset: analysis_root not configured."
    status_set = set(statuses) if statuses else None
    try:
        counts = pm.bulk_refresh(
            task_names=tasks,
            recording_keys=recordings or None,
            well_ids=wells or None,
            status_filter=status_set,
        )
    except (ValueError, KeyError) as exc:
        return f"Reset failed: {exc}"
    total = sum(counts.values())
    if total == 0:
        return "Reset: nothing matched the filter."
    breakdown = ", ".join(f"{t}={n}" for t, n in counts.items() if n)
    return f"Reset {total} record(s) — {breakdown}. Click refresh() to redraw."
