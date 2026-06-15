"""Home page — Overview matching the mea-chip design.

Layout (top to bottom):
  view-head   — breadcrumb + title + action buttons
  kpi-grid    — 4 tiles: recordings, entries, tasks complete, in-flight/failed
  grid-2      — task throughput table | ready queue table
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import dash
from dash import Input, Output, callback, dcc, html
from flask import current_app

from yuxin_mea.dashboard.components import no_config_banner
from yuxin_mea.dashboard.data import load_pipeline_df, load_recordings_df
from yuxin_mea.tasks import TASK_CLASSES


dash.register_page(__name__, path="/", name="Home", order=0)

_TASK_NAMES = [tc.task_name for tc in TASK_CLASSES]
_TASK_DEPS: dict[str, list[str]] = {
    tc.task_name: list(getattr(tc, "dependencies", []))
    for tc in TASK_CLASSES
}

_STATUS_OK = "complete"
_STATUS_RUN = "running"
_STATUS_FAIL = "failed"
_STATUS_IDLE = "not_run"


def _kpi(label: str, value: object, sub: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="label"),
            html.Div(value if isinstance(value, list) else str(value), className="value"),
            html.Div(sub, className="sub"),
        ],
        className="kpi",
    )


def _throughput_row(task_name: str, counts: dict[str, int]) -> html.Tr:
    c = counts
    total = max(1, c[_STATUS_OK] + c[_STATUS_RUN] + c[_STATUS_FAIL] + c[_STATUS_IDLE])
    ok_pct = f"{100 * c[_STATUS_OK] / total:.1f}%"
    run_pct = f"{100 * c[_STATUS_RUN] / total:.1f}%"
    fail_pct = f"{100 * c[_STATUS_FAIL] / total:.1f}%"
    return html.Tr(
        [
            html.Td(
                task_name,
                style={
                    "fontFamily": "var(--font-mono)",
                    "fontSize": "11px",
                    "color": "var(--ink-2)",
                    "padding": "5px 10px 5px 0",
                    "width": "160px",
                    "whiteSpace": "nowrap",
                },
            ),
            html.Td(
                html.Div(
                    [
                        html.Span(style={"background": "var(--ok)", "width": ok_pct}),
                        html.Span(style={"background": "var(--run)", "width": run_pct}),
                        html.Span(style={"background": "var(--fail)", "width": fail_pct}),
                    ],
                    style={
                        "display": "flex",
                        "gap": "1px",
                        "height": "6px",
                        "background": "var(--bg-deep)",
                        "borderRadius": "3px",
                        "overflow": "hidden",
                    },
                ),
                style={"padding": "5px 0"},
            ),
            html.Td(
                f"{c[_STATUS_OK]}/{total}",
                style={
                    "fontFamily": "var(--font-mono)",
                    "fontSize": "11px",
                    "color": "var(--ink-3)",
                    "textAlign": "right",
                    "width": "72px",
                    "padding": "5px 0 5px 10px",
                    "fontFeatureSettings": "'tnum'",
                    "whiteSpace": "nowrap",
                },
            ),
        ]
    )


def _ready_row(pipeline_key: str, task_name: str) -> html.Tr:
    parts = pipeline_key.rsplit("/", 1)
    rec = parts[0] if len(parts) == 2 else pipeline_key
    well = parts[1] if len(parts) == 2 else ""
    return html.Tr(
        [
            html.Td(
                html.Span(task_name, className="pill idle"),
                style={"padding": "6px 12px"},
            ),
            html.Td(
                [
                    html.Span(rec + "/", className="muted", style={"fontSize": "11px"}),
                    html.Span(well, style={"fontFamily": "var(--font-mono)", "fontSize": "11px"}),
                ],
                className="mono",
                style={"padding": "6px 12px"},
            ),
            html.Td(
                "—",
                className="num muted",
                style={"padding": "6px 12px", "fontSize": "11px"},
            ),
        ]
    )


layout = html.Div(
    [
        dcc.Interval(id="home-auto-refresh", interval=30_000, n_intervals=0),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("workspace"),
                                html.Span("analysis_root"),
                                html.Span("dev"),
                            ],
                            className="breadcrumb",
                        ),
                        html.H1("Analysis operations"),
                        html.Div(id="home-subtitle", className="subtitle"),
                    ]
                ),
                html.Div(
                    [
                        html.Button(
                            [html.Span("↻", className="glyph"), "Refresh"],
                            id="home-refresh",
                            n_clicks=0,
                            className="btn",
                        ),
                    ],
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),
        html.Div(id="home-banner-slot"),
        html.Div(id="home-kpi-grid", className="kpi-grid"),
        html.Div(
            [
                # Task throughput
                html.Div(
                    [
                        html.Div(
                            [html.Span("task throughput", className="h-title")],
                            className="card-head",
                        ),
                        html.Div(
                            html.Table(
                                html.Tbody(id="home-throughput-rows"),
                                style={"width": "100%", "borderCollapse": "collapse"},
                            ),
                            className="card-body",
                        ),
                    ],
                    className="card",
                ),
                # Ready queue
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("ready queue · next dispatch", className="h-title"),
                                html.Div(
                                    html.Button(
                                        [html.Span("▸", className="glyph"), html.Span(id="home-ready-count"), " ready"],
                                        className="btn",
                                    ),
                                    className="h-actions",
                                ),
                            ],
                            className="card-head",
                        ),
                        html.Div(
                            html.Table(
                                [
                                    html.Thead(
                                        html.Tr(
                                            [
                                                html.Th("task", style={"padding": "8px 12px"}),
                                                html.Th("pipeline_key", style={"padding": "8px 12px"}),
                                                html.Th("eta", style={"padding": "8px 12px", "textAlign": "right"}),
                                            ],
                                            style={
                                                "fontFamily": "var(--font-mono)",
                                                "fontSize": "10px",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.06em",
                                                "color": "var(--ink-3)",
                                                "borderBottom": "1px solid var(--line)",
                                                "background": "var(--bg)",
                                            },
                                        )
                                    ),
                                    html.Tbody(id="home-ready-rows"),
                                ],
                                className="tbl",
                            ),
                            className="card-body flush",
                        ),
                    ],
                    className="card",
                ),
            ],
            className="grid-2",
        ),
    ],
    className="page",
)


@callback(
    Output("home-banner-slot", "children"),
    Output("home-subtitle", "children"),
    Output("home-kpi-grid", "children"),
    Output("home-throughput-rows", "children"),
    Output("home-ready-rows", "children"),
    Output("home-ready-count", "children"),
    Input("home-refresh", "n_clicks"),
    Input("home-auto-refresh", "n_intervals"),
)
def _render(_n: int, _intervals: int):
    ctx = current_app.config["YUXIN_MEA"]
    banner = None if ctx.get("config_exists") else no_config_banner()
    analysis_root = ctx.get("analysis_root")

    subtitle = html.Span(
        ["Yuxin_MEA · ", html.Code(str(ctx.get("config_path") or "no config"))],
        style={"fontFamily": "var(--font-mono)", "fontSize": "11px"},
    )

    if analysis_root is None:
        empty_kpis = [_kpi(lbl, "—") for lbl in
                      ("Recordings cached", "Pipeline entries", "Tasks complete", "In flight · failed")]
        return banner, subtitle, empty_kpis, [], [], "0"

    analysis_path = Path(analysis_root)
    rec_df = load_recordings_df(analysis_path)
    pipe_df, task_names = load_pipeline_df(analysis_path)

    n_rec = len(rec_df)
    n_wells = int(rec_df["n_wells"].sum()) if not rec_df.empty else 0
    n_samples = rec_df["sample_id"].nunique() if not rec_df.empty else 0
    n_entries = len(pipe_df)

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

    n_complete_all = sum(c[_STATUS_OK] for c in counts.values())
    n_total_all = n_entries * len(task_names) if task_names else 0
    n_running = sum(c[_STATUS_RUN] for c in counts.values())
    n_failed = sum(c[_STATUS_FAIL] for c in counts.values())
    pct = f"{100 * n_complete_all // n_total_all}%" if n_total_all else "—"

    kpis = [
        _kpi("Recordings cached", n_rec, f"{n_samples} sample(s) · {n_wells} wells"),
        _kpi("Pipeline entries", n_entries, f"{n_entries * len(task_names)} task records · {len(task_names)}-stage DAG"),
        _kpi("Tasks complete", f"{n_complete_all}/{n_total_all}", f"{pct} across all stages"),
        _kpi(
            "In flight · failed",
            [
                html.Span(str(n_running), style={"color": "oklch(0.5 0.13 80)"}),
                html.Span(" · ", style={"color": "var(--ink-4)", "margin": "0 4px"}),
                html.Span(str(n_failed), style={"color": "oklch(0.5 0.16 28)"}),
            ],
            "auto-retry next sweep",
        ),
    ]

    throughput_rows = [
        _throughput_row(tn, counts.get(tn, {_STATUS_OK: 0, _STATUS_RUN: 0, _STATUS_FAIL: 0, _STATUS_IDLE: 0}))
        for tn in (task_names or _TASK_NAMES)
    ]

    # Ready queue: entries where all deps complete but task itself not_run
    ready: list[tuple[str, str]] = []
    if not pipe_df.empty:
        for _, row in pipe_df.iterrows():
            pipeline_key = f"{row['recording_key']}/{row['well_id']}"
            for tn in (task_names or _TASK_NAMES):
                status = row.get(tn, _STATUS_IDLE)
                deps = _TASK_DEPS.get(tn, [])
                deps_done = all(row.get(d, _STATUS_IDLE) == _STATUS_OK for d in deps if d in pipe_df.columns)
                if status == _STATUS_IDLE and deps_done:
                    ready.append((pipeline_key, tn))
                    break
            if len(ready) >= 8:
                break

    if ready:
        ready_rows = [_ready_row(pk, tn) for pk, tn in ready]
    else:
        ready_rows = [
            html.Tr(
                html.Td(
                    "No ready work — all dependencies blocked or complete.",
                    colSpan=3,
                    style={
                        "padding": "40px 24px",
                        "textAlign": "center",
                        "color": "var(--ink-3)",
                        "fontFamily": "var(--font-mono)",
                        "fontSize": "12px",
                    },
                )
            )
        ]

    return banner, subtitle, kpis, throughput_rows, ready_rows, str(len(ready))
