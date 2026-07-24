"""Run page — command builder for `yuxin-mea-run`.

Pick tasks + recordings, see a preview of work items that are eligible right
now, copy the literal CLI invocation. The dashboard does not spawn the worker
itself — task execution stays a terminal concern with the right Unix lifetime
(close the terminal → process dies; reattach by tailing the log).
"""

from __future__ import annotations

import shlex
from pathlib import Path

import dash
from dash import Input, Output, callback, dcc, html
from flask import current_app

from yuxin_mea.dashboard.components import (
    build_filter_bar,
    filter_id,
    filter_kwargs,
    iso_to_yymmdd,
    no_config_banner,
    yymmdd_to_iso,
)
from yuxin_mea.dashboard.context import load_pipeline_mgr
from yuxin_mea.dashboard.data import (
    filter_recordings,
    load_pipeline_df,
    load_recordings_detail,
)
from yuxin_mea.tasks import TASK_CLASSES


dash.register_page(__name__, path="/run", name="Run", order=3)


_TASK_OPTIONS = [{"label": cls.task_name, "value": cls.task_name} for cls in TASK_CLASSES]
_PREVIEW_LIMIT = 25


def _field(label: str, child) -> html.Div:
    return html.Div(
        [html.Label(label, className="section-label"), child],
        style={"flex": "1 1 240px"},
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
                                html.Span("yuxin-mea-run"),
                                html.Span("preview"),
                            ],
                            className="breadcrumb",
                        ),
                        html.H1("Run"),
                        html.Div(
                            "Build the yuxin-mea-run invocation. Preview shows "
                            "work items eligible right now; downstream tasks "
                            "unlock as upstream completes.",
                            className="subtitle",
                        ),
                    ]
                ),
            ],
            className="view-head",
        ),
        html.Div(id="run-banner-slot"),
        # Filters narrow the Recordings dropdown below (keeps the built command
        # sane — you still pick explicit keys from the filtered set).
        # Pinned to the five original facets: this bar picks a recording to run,
        # and an hours-since-media control here would render with no callback
        # behind it. Add "hours" once this page's callback populates and reads it.
        build_filter_bar("run", show=("sample", "scan-type", "date", "group", "status")),
        html.Div(
            [
                html.Div(
                    [html.Span("selection", className="h-title")],
                    className="card-head",
                ),
                html.Div(
                    html.Div(
                        [
                            _field(
                                "Tasks (empty = all)",
                                dcc.Dropdown(
                                    id="run-tasks",
                                    options=_TASK_OPTIONS,
                                    value=[],
                                    multi=True,
                                    placeholder="any task",
                                ),
                            ),
                            _field(
                                "Recordings (empty = all)",
                                dcc.Dropdown(
                                    id="run-recordings",
                                    options=[],
                                    value=[],
                                    multi=True,
                                    placeholder="any recording",
                                ),
                            ),
                            html.Div(
                                [
                                    html.Label("Options", className="section-label"),
                                    dcc.Checklist(
                                        id="run-flags",
                                        options=[{"label": " retry failed", "value": "retry"}],
                                        value=[],
                                        style={"fontFamily": "var(--font-mono)",
                                               "fontSize": "12px",
                                               "color": "var(--ink-2)"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "jobs (concurrent workers)",
                                                style={
                                                    "fontFamily": "var(--font-mono)",
                                                    "fontSize": "11px",
                                                    "color": "var(--ink-3)",
                                                    "marginRight": "6px",
                                                },
                                            ),
                                            dcc.Input(
                                                id="run-jobs",
                                                type="number",
                                                min=1,
                                                step=1,
                                                value=1,
                                                style={"width": "80px"},
                                            ),
                                        ],
                                        style={"marginTop": "8px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "max-tasks (cap, optional)",
                                                style={
                                                    "fontFamily": "var(--font-mono)",
                                                    "fontSize": "11px",
                                                    "color": "var(--ink-3)",
                                                    "marginRight": "6px",
                                                },
                                            ),
                                            dcc.Input(
                                                id="run-max",
                                                type="number",
                                                min=1,
                                                step=1,
                                                placeholder="unlimited",
                                                style={"width": "120px"},
                                            ),
                                        ],
                                        style={"marginTop": "8px"},
                                    ),
                                ],
                                style={"flex": "1 1 220px"},
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
            style={"marginBottom": "16px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Span("preview", className="h-title"),
                        html.Span(
                            id="run-preview-summary",
                            className="h-actions",
                            style={"color": "var(--ink-2)",
                                   "fontFamily": "var(--font-mono)",
                                   "fontSize": "11px",
                                   "textTransform": "none",
                                   "letterSpacing": "0"},
                        ),
                    ],
                    className="card-head",
                ),
                html.Div(
                    html.Pre(id="run-preview-list", className="code"),
                    className="card-body",
                ),
            ],
            className="card",
            style={"marginBottom": "16px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Span("command", className="h-title"),
                        html.Div(
                            dcc.Clipboard(
                                target_id="run-command",
                                title="Copy command",
                                style={"display": "inline-block",
                                       "fontSize": "16px",
                                       "cursor": "pointer",
                                       "color": "var(--ink-3)"},
                            ),
                            className="h-actions",
                        ),
                    ],
                    className="card-head",
                ),
                html.Div(
                    html.Pre(id="run-command", className="code terminal"),
                    className="card-body",
                ),
            ],
            className="card",
        ),
    ],
    className="page",
)


# ---------------------------------------------------------------------------
# Populate the recording dropdown on page entry from the dataset cache.
# ---------------------------------------------------------------------------


@callback(
    Output("run-banner-slot", "children"),
    Output("run-recordings", "options"),
    Output(filter_id("run", "sample"), "options"),
    Output(filter_id("run", "scan-type"), "options"),
    Output(filter_id("run", "date"), "min_date_allowed"),
    Output(filter_id("run", "date"), "max_date_allowed"),
    Output(filter_id("run", "group"), "options"),
    Input(filter_id("run", "sample"), "value"),
    Input(filter_id("run", "scan-type"), "value"),
    Input(filter_id("run", "date"), "start_date"),
    Input(filter_id("run", "date"), "end_date"),
    Input(filter_id("run", "group"), "value"),
    Input(filter_id("run", "status"), "value"),
)
def _populate_recordings(f_sample, f_scan_type, f_date_from, f_date_to, f_group, f_status):
    ctx = current_app.config["YUXIN_MEA"]
    banner = None if ctx.get("config_exists") else no_config_banner()
    analysis_root = ctx.get("analysis_root")
    empty: list[dict] = []
    if analysis_root is None:
        return banner, empty, empty, empty, None, None, empty

    recordings, well_pipeline_status = load_recordings_detail(Path(analysis_root))

    # Filter option universes (all recordings, unaffected by current selection).
    sample_opts = sorted({r["sample_id"] for r in recordings})
    scan_opts = sorted({r["scan_type"] for r in recordings})
    date_opts = sorted({r["date"] for r in recordings})
    date_min = yymmdd_to_iso(date_opts[0]) if date_opts else None
    date_max = yymmdd_to_iso(date_opts[-1]) if date_opts else None
    group_opts = sorted({g for r in recordings for g in r.get("groups", [])})

    kwargs = filter_kwargs(
        {
            "sample": f_sample,
            "scan-type": f_scan_type,
            "date-from": iso_to_yymmdd(f_date_from),
            "date-to": iso_to_yymmdd(f_date_to),
            "group": f_group,
            "status": f_status,
        }
    )
    filtered = filter_recordings(recordings, well_pipeline_status, **kwargs)
    rec_options = [{"label": r["cache_key"], "value": r["cache_key"]} for r in filtered]

    return (
        banner,
        rec_options,
        [{"label": s, "value": s} for s in sample_opts],
        [{"label": s, "value": s} for s in scan_opts],
        date_min,
        date_max,
        [{"label": g, "value": g} for g in group_opts],
    )


# ---------------------------------------------------------------------------
# Build preview + command string off the current selections.
# ---------------------------------------------------------------------------


@callback(
    Output("run-preview-summary", "children"),
    Output("run-preview-list", "children"),
    Output("run-command", "children"),
    Input("run-tasks", "value"),
    Input("run-recordings", "value"),
    Input("run-flags", "value"),
    Input("run-max", "value"),
    Input("run-jobs", "value"),
)
def _build(tasks_selected, recordings_selected, flags, max_tasks, jobs):
    ctx = current_app.config["YUXIN_MEA"]
    config_path = ctx.get("config_path")
    analysis_root = ctx.get("analysis_root")

    cmd = _build_command(
        config_path, tasks_selected, recordings_selected, flags, max_tasks, jobs,
    )

    if analysis_root is None:
        return "analysis_root is not set in the config.", "", cmd

    # The pipeline cache loader is read-only, so previewing is cheap and safe.
    # Build a transient PipelineManager only to call get_next_task with the
    # right eligibility logic — but we don't write through it.
    pm = load_pipeline_mgr()
    if pm is None:
        return "analysis_root not configured.", "", cmd

    retry_failed = "retry" in (flags or [])
    rec_keys = recordings_selected or None
    task_names = tasks_selected or None

    eligible = pm.get_next_task(
        n=_PREVIEW_LIMIT + 1,
        retry_failed=retry_failed,
        recording_keys=rec_keys,
        task_names=task_names,
    )
    # Quick total count: pull a big batch.
    total = len(pm.get_next_task(
        n=100_000,
        retry_failed=retry_failed,
        recording_keys=rec_keys,
        task_names=task_names,
    ))

    cache_df, _task_cols = load_pipeline_df(Path(analysis_root))
    summary = (
        f"{total} eligible work item(s) right now "
        f"({len(cache_df)} (recording, well) entries in cache)."
    )
    if total == 0 and len(cache_df) == 0:
        summary += "  → Queue wells from the Recordings page first."

    preview_text = "\n".join(
        f"{w.task_name:<26s}  {w.recording_key}/{w.well_id}"
        for w in eligible[:_PREVIEW_LIMIT]
    )
    if total > _PREVIEW_LIMIT:
        preview_text += f"\n… and {total - _PREVIEW_LIMIT} more"
    if not preview_text:
        preview_text = "(no eligible work items)"

    return summary, preview_text, cmd


def _build_command(
    config_path: object,
    tasks_selected: list[str] | None,
    recordings_selected: list[str] | None,
    flags: list[str] | None,
    max_tasks: int | None,
    jobs: int | None,
) -> str:
    """Render the literal `yuxin-mea-run ...` invocation, one flag per line."""
    config_str = str(config_path) if config_path else "<set --config>"
    flag_groups: list[str] = [f"--config {shlex.quote(config_str)}"]
    if tasks_selected:
        flag_groups.append(f"--tasks {shlex.quote(','.join(tasks_selected))}")
    if recordings_selected:
        flag_groups.append(f"--recordings {shlex.quote(','.join(recordings_selected))}")
    if flags and "retry" in flags:
        flag_groups.append("--retry-failed")
    if max_tasks:
        flag_groups.append(f"--max-tasks {int(max_tasks)}")
    # jobs=1 is the CLI default — omit to keep the command terse.
    if jobs and int(jobs) > 1:
        flag_groups.append(f"--jobs {int(jobs)}")

    if len(flag_groups) == 1:
        return f"yuxin-mea-run {flag_groups[0]}"
    return "yuxin-mea-run \\\n  " + " \\\n  ".join(flag_groups)
