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


layout = html.Div(
    [
        html.H2("Run", style={"marginTop": "0"}),
        html.P(
            "Build the `yuxin-mea-run` invocation for the work you want to "
            "drain. The preview shows work items that are eligible RIGHT NOW "
            "given your selections — downstream tasks unlock as upstream "
            "tasks complete in a real run, so the live count there will "
            "usually exceed the preview count."
        ),
        html.Div(id="run-banner-slot"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Tasks (empty = all)", style={"fontSize": "12px", "color": "#555"}),
                        dcc.Dropdown(
                            id="run-tasks",
                            options=_TASK_OPTIONS,
                            value=[],
                            multi=True,
                            placeholder="any task",
                        ),
                    ],
                    style={"flex": "1 1 320px"},
                ),
                html.Div(
                    [
                        html.Label("Recordings (empty = all)", style={"fontSize": "12px", "color": "#555"}),
                        dcc.Dropdown(
                            id="run-recordings",
                            options=[],
                            value=[],
                            multi=True,
                            placeholder="any recording",
                        ),
                    ],
                    style={"flex": "2 1 480px"},
                ),
                html.Div(
                    [
                        html.Label("Options", style={"fontSize": "12px", "color": "#555"}),
                        dcc.Checklist(
                            id="run-flags",
                            options=[
                                {"label": " retry failed", "value": "retry"},
                            ],
                            value=[],
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "max-tasks",
                                    style={"fontSize": "12px", "color": "#555", "marginRight": "6px"},
                                ),
                                dcc.Input(
                                    id="run-max",
                                    type="number",
                                    min=1,
                                    step=1,
                                    placeholder="unlimited",
                                    style={"width": "100px"},
                                ),
                            ],
                            style={"marginTop": "4px"},
                        ),
                    ],
                    style={"flex": "1 1 220px"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "16px",
                "alignItems": "flex-start",
                "marginBottom": "16px",
            },
        ),
        html.H4("Preview"),
        html.Div(
            id="run-preview-summary",
            style={"marginBottom": "6px", "fontSize": "14px"},
        ),
        html.Pre(
            id="run-preview-list",
            style={
                "backgroundColor": "#f6f8fa",
                "border": "1px solid #d0d7de",
                "padding": "10px 12px",
                "borderRadius": "4px",
                "fontSize": "12px",
                "maxHeight": "240px",
                "overflow": "auto",
            },
        ),
        html.H4("Command"),
        html.Pre(
            id="run-command",
            style={
                "backgroundColor": "#0d1117",
                "color": "#c9d1d9",
                "padding": "12px 14px",
                "borderRadius": "4px",
                "fontSize": "13px",
                "whiteSpace": "pre",
                "overflowX": "auto",
            },
        ),
        dcc.Clipboard(
            target_id="run-command",
            title="Copy command",
            style={
                "display": "inline-block",
                "fontSize": "16px",
                "cursor": "pointer",
                "marginTop": "6px",
            },
        ),
        html.Span(
            " ← click the icon to copy the command, then paste in a terminal.",
            style={"fontSize": "12px", "color": "#555"},
        ),
    ]
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

    # Also report cache size so the user knows whether they need to queue wells first.
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
