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

from yuxin_mea.dashboard.components import no_config_banner
from yuxin_mea.dashboard.context import load_pipeline_mgr
from yuxin_mea.dashboard.data import load_pipeline_df, load_recordings_df
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
                                                "max-tasks",
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
    Input("run-tasks", "id"),  # fires once on initial render
)
def _populate_recordings(_id: str):
    ctx = current_app.config["YUXIN_MEA"]
    banner = None if ctx.get("config_exists") else no_config_banner()
    analysis_root = ctx["analysis_root"]
    if analysis_root is None:
        return banner, []
    df = load_recordings_df(Path(analysis_root))
    options = [
        {"label": k, "value": k}
        for k in df.get("cache_key", []).tolist()
    ]
    return banner, options


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
)
def _build(tasks_selected, recordings_selected, flags, max_tasks):
    ctx = current_app.config["YUXIN_MEA"]
    config_path = ctx.get("config_path")
    analysis_root = ctx.get("analysis_root")

    cmd = _build_command(config_path, tasks_selected, recordings_selected, flags, max_tasks)

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

    if len(flag_groups) == 1:
        return f"yuxin-mea-run {flag_groups[0]}"
    return "yuxin-mea-run \\\n  " + " \\\n  ".join(flag_groups)
