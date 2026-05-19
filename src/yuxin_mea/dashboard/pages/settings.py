"""Settings page — 3-column config editor matching the mea-chip design.

Layout:
  view-head   — breadcrumb + title + actions
  3-column grid (240px + 1fr + 1fr):
    [left]    section tree (global + per-task nodes)
    [middle]  form · {section}  (form_builder output)
    [right]   JSON preview of current section values

Selection state lives in dcc.Store("settings-section-store").
All form_builder callbacks are preserved; only the layout scaffolding
changes (Tabs → 3-column grid + Store-triggered renders).
"""

from __future__ import annotations

import json
from pathlib import Path

import dash
from dash import ALL, Input, Output, State, callback, ctx, dcc, html
from flask import current_app

from yuxin_mea.config import ConfigManager, GLOBALS_SCHEMA
from yuxin_mea.dashboard.components.form_builder import collect_values, render_form
from yuxin_mea.tasks import TASK_CLASSES


dash.register_page(__name__, path="/settings", name="Settings", order=10)

_TASK_BY_NAME = {tc.task_name: tc for tc in TASK_CLASSES}
_TASK_SCHEMAS = {tc.task_name: tc.params_schema() for tc in TASK_CLASSES}

# Stability badge: tasks without explicit annotation are "exp"
_TASK_STABILITY: dict[str, str] = {
    tc.task_name: getattr(tc, "api_stability", "exp")
    for tc in TASK_CLASSES
}


def _all_form_ids() -> list[str]:
    return ["globals", *(f"task-{name}" for name in _TASK_BY_NAME)]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _tree_node(section_id: str, label: str, badge: str | None, leaf: bool = True) -> html.Div:
    children = [
        html.Span("·" if leaf else "▾", className="leaf" if leaf else "chev"),
        html.Span(label, className="lbl"),
    ]
    if badge:
        children.append(html.Span(badge, className=f"badge {badge}"))
    return html.Div(
        children,
        id={"settings-tree": section_id},
        n_clicks=0,
        className="tree-node",
    )


layout = html.Div(
    [
        dcc.Store(id="settings-section-store", data="globals"),

        # ── view-head ────────────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("workspace"),
                                html.Span("config_manager"),
                                html.Span("pipeline_config.json"),
                            ],
                            className="breadcrumb",
                        ),
                        html.H1("Configuration"),
                        html.Div(id="settings-subtitle", className="subtitle"),
                    ]
                ),
                html.Div(
                    [
                        html.Button(
                            [html.Span("↻", className="glyph"), "load()"],
                            id="settings-load-btn",
                            n_clicks=0,
                            className="btn",
                        ),
                        html.Button(
                            [html.Span("⤓", className="glyph"), "save()"],
                            id="settings-global-save-trigger",
                            n_clicks=0,
                            className="btn primary",
                        ),
                    ],
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),

        html.Div(id="settings-unknown-keys-banner"),

        # ── 3-column grid ────────────────────────────────────────────────
        html.Div(
            [
                # Left: section tree
                html.Div(
                    [
                        html.Div(
                            [html.Span("sections", className="h-title")],
                            className="card-head",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        _tree_node("globals", "global", None, leaf=False),
                                        html.Div(
                                            "tasks",
                                            style={
                                                "padding": "8px 10px 4px",
                                                "fontFamily": "var(--font-mono)",
                                                "fontSize": "10px",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.06em",
                                                "color": "var(--ink-3)",
                                            },
                                        ),
                                        *[
                                            _tree_node(
                                                name,
                                                name,
                                                _TASK_STABILITY.get(name, "exp"),
                                                leaf=True,
                                            )
                                            for name in _TASK_BY_NAME
                                        ],
                                    ],
                                    className="tree",
                                    id="settings-tree-container",
                                    style={"padding": "6px"},
                                ),
                                html.Div(
                                    [
                                        html.Code("register_task()"),
                                        " seeds new sections. Task ",
                                        html.Code("default_params()"),
                                        " applied via ",
                                        html.Code("resolve_params()"),
                                        ".",
                                    ],
                                    className="rail-footer",
                                    style={"margin": "0", "borderTop": "1px solid var(--line)", "padding": "10px 12px"},
                                ),
                            ],
                            className="card-body flush",
                        ),
                    ],
                    className="card",
                    style={"width": "240px", "flexShrink": "0"},
                ),

                # Middle: form
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(id="settings-form-title", className="h-title"),
                                html.Div(
                                    html.Span("schema", className="badge"),
                                    className="h-actions",
                                ),
                            ],
                            className="card-head",
                        ),
                        html.Div(
                            html.Div(id="settings-form-panel", style={"maxHeight": "540px", "overflow": "auto"}),
                            className="card-body flush",
                        ),
                    ],
                    className="card",
                    style={"flex": "1", "minWidth": "0"},
                ),

                # Right: JSON preview
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(id="settings-json-title", className="h-title"),
                                html.Div(
                                    [
                                        html.Span(id="settings-json-pill"),
                                        html.Button(
                                            "⎘",
                                            className="btn ghost",
                                            style={"height": "22px", "padding": "0 8px"},
                                        ),
                                    ],
                                    className="h-actions",
                                ),
                            ],
                            className="card-head",
                        ),
                        html.Div(
                            html.Pre(id="settings-json-preview", className="code"),
                            className="card-body",
                        ),
                    ],
                    className="card",
                    style={"flex": "1", "minWidth": "0"},
                ),
            ],
            style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
        ),
    ],
    className="page",
)


# ---------------------------------------------------------------------------
# 1) Tree node click → update section store + active styling
# ---------------------------------------------------------------------------

@callback(
    Output("settings-section-store", "data"),
    Output("settings-tree-container", "children"),
    Input({"settings-tree": ALL}, "n_clicks"),
    State("settings-section-store", "data"),
    prevent_initial_call=True,
)
def _select_section(_clicks, current_section: str):
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return dash.no_update, dash.no_update

    new_section = triggered.get("settings-tree", current_section)

    def _node(section_id: str, label: str, badge: str | None, leaf: bool) -> html.Div:
        is_active = section_id == new_section
        children = [
            html.Span("·" if leaf else "▾", className="leaf" if leaf else "chev"),
            html.Span(label, className="lbl"),
        ]
        if badge:
            children.append(html.Span(badge, className=f"badge {badge}"))
        return html.Div(
            children,
            id={"settings-tree": section_id},
            n_clicks=0,
            className=f"tree-node {'active' if is_active else ''}",
        )

    tree_children = [
        _node("globals", "global", None, leaf=False),
        html.Div(
            "tasks",
            style={
                "padding": "8px 10px 4px",
                "fontFamily": "var(--font-mono)",
                "fontSize": "10px",
                "textTransform": "uppercase",
                "letterSpacing": "0.06em",
                "color": "var(--ink-3)",
            },
        ),
        *[
            _node(name, name, _TASK_STABILITY.get(name, "exp"), leaf=True)
            for name in _TASK_BY_NAME
        ],
    ]
    return new_section, tree_children


# ---------------------------------------------------------------------------
# 2) Section change → populate form + JSON preview + titles
# ---------------------------------------------------------------------------

@callback(
    Output("settings-subtitle", "children"),
    Output("settings-unknown-keys-banner", "children"),
    Output("settings-form-title", "children"),
    Output("settings-form-panel", "children"),
    Output("settings-json-title", "children"),
    Output("settings-json-preview", "children"),
    Output("settings-json-pill", "children"),
    Input("settings-section-store", "data"),
    Input("settings-load-btn", "n_clicks"),
)
def _populate_section(section: str, _load: int):
    section = section or "globals"
    yuxin_ctx = current_app.config.get("YUXIN_MEA", {})
    config_path = yuxin_ctx.get("config_path")

    cm = ConfigManager()
    unknown: dict[str, list[str]] = {}
    if config_path and Path(config_path).exists():
        cm.load(config_path)
        unknown = cm.validate_loaded(_TASK_SCHEMAS)

    subtitle = html.Span(
        [
            "ConfigManager · loaded from ",
            html.Code(str(config_path) if config_path else "(none)"),
        ],
        style={"fontFamily": "var(--font-mono)", "fontSize": "11px"},
    )

    banner = _build_unknown_keys_banner(unknown)

    if section == "globals":
        schema = GLOBALS_SCHEMA
        values = {key: cm.get_global(key, "") for key in schema}
        form_id = "globals"
        form_title = "form · global"
        json_title = "pipeline_config.json · global"
    else:
        task_name = section
        schema = _TASK_SCHEMAS.get(task_name, {})
        values = cm.get_task_params(task_name) if schema else {}
        form_id = f"task-{task_name}"
        form_title = f"form · {task_name}"
        json_title = f"pipeline_config.json · {task_name}"

    if schema:
        form = render_form(form_id, schema, values, title=None)
    else:
        form = html.P(
            f"{section} has no params_schema(); edit the JSON file directly.",
            style={"color": "var(--ink-3)", "fontFamily": "var(--font-mono)", "fontSize": "12px", "padding": "16px"},
        )

    json_text = json.dumps(values, indent=2, default=str) if values else "{}"
    json_pill = html.Span("unsaved", className="pill idle")

    return subtitle, banner, form_title, form, json_title, json_text, json_pill


def _build_unknown_keys_banner(unknown: dict[str, list[str]]):
    if not unknown:
        return None
    items = [
        html.Li(
            f"{task}: {', '.join(keys)}",
            style={"fontFamily": "var(--font-mono)", "fontSize": "11px"},
        )
        for task, keys in unknown.items()
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Strong("Stale config keys detected. "),
                    html.Span(
                        "These keys are in the loaded config but not in any "
                        "task schema. They will be dropped on the next Save."
                    ),
                ]
            ),
            html.Ul(items, style={"margin": "8px 0 0 0", "paddingLeft": "18px"}),
        ],
        className="banner warn",
    )


# ---------------------------------------------------------------------------
# 3) Dirty store: any field change → mark dirty
# ---------------------------------------------------------------------------

@callback(
    Output({"form": ALL, "key": "dirty"}, "data"),
    Input({"form": ALL, "field": ALL}, "value"),
    State({"form": ALL, "key": "dirty"}, "id"),
    prevent_initial_call=True,
)
def _mark_forms_dirty(_field_values, dirty_ids):
    return [True] * len(dirty_ids)


# ---------------------------------------------------------------------------
# 4) Save-button disabled toggle
# ---------------------------------------------------------------------------

@callback(
    Output({"form": ALL, "key": "save"}, "disabled"),
    Input({"form": ALL, "key": "dirty"}, "data"),
)
def _toggle_save_buttons(dirty_flags):
    return [not flag for flag in dirty_flags]


# ---------------------------------------------------------------------------
# 5) Save: pattern-matched, handles all forms
# ---------------------------------------------------------------------------

@callback(
    Output({"form": ALL, "key": "status"}, "children"),
    Output({"form": ALL, "key": "dirty"}, "data", allow_duplicate=True),
    Output({"form": ALL, "field-error": ALL}, "children"),
    Output("settings-json-preview", "children", allow_duplicate=True),
    Output("settings-json-pill", "children", allow_duplicate=True),
    Input({"form": ALL, "key": "save"}, "n_clicks"),
    State({"form": ALL, "field": ALL}, "value"),
    State({"form": ALL, "field": ALL}, "id"),
    State("settings-section-store", "data"),
    prevent_initial_call=True,
)
def _save_any_form(_save_clicks, field_values, field_ids, active_section):
    triggered = ctx.triggered_id
    if not triggered or triggered.get("key") != "save":
        return (
            [dash.no_update] * len(_save_clicks),
            [dash.no_update] * len(_save_clicks),
            [dash.no_update] * len(field_ids),
            dash.no_update,
            dash.no_update,
        )
    target_form_id = triggered["form"]

    fields_by_form: dict[str, dict[str, object]] = {}
    for value, id_ in zip(field_values, field_ids):
        fields_by_form.setdefault(id_["form"], {})[id_["field"]] = value

    raw = fields_by_form.get(target_form_id, {})

    if target_form_id == "globals":
        schema = GLOBALS_SCHEMA
    elif target_form_id.startswith("task-"):
        schema = _TASK_SCHEMAS.get(target_form_id[len("task-"):], {})
    else:
        schema = {}

    parsed, errors = collect_values(schema, raw)

    errors_out = []
    for id_ in field_ids:
        if id_["form"] == target_form_id and id_["field"] in errors:
            errors_out.append(errors[id_["field"]])
        elif id_["form"] == target_form_id:
            errors_out.append("")
        else:
            errors_out.append(dash.no_update)

    status_out = [dash.no_update] * len(_save_clicks)
    dirty_out = [dash.no_update] * len(_save_clicks)

    target_index = next(
        (i for i, n in enumerate(_save_clicks)
         if _id_match_for_save(ctx.outputs_list[0][i], target_form_id)),
        None,
    )

    if errors:
        if target_index is not None:
            status_out[target_index] = f"❌ {len(errors)} field(s) need fixing"
        return status_out, dirty_out, errors_out, dash.no_update, dash.no_update

    yuxin_ctx = current_app.config.get("YUXIN_MEA", {})
    config_path: Path = yuxin_ctx.get("config_path")
    if config_path is None:
        if target_index is not None:
            status_out[target_index] = "❌ No config path configured for this dashboard."
        return status_out, dirty_out, errors_out, dash.no_update, dash.no_update

    cm = ConfigManager()
    if Path(config_path).exists():
        cm.load(config_path)
    if target_form_id == "globals":
        cm.set_globals(parsed)
        suffix = (
            " Restart the dashboard for data_root/analysis_root/figure_root changes."
        )
    else:
        task_name = target_form_id[len("task-"):]
        cm.set_task_params(task_name, parsed)
        suffix = ""
    cm.save(config_path)

    current_app.config["YUXIN_MEA"]["config_exists"] = True

    if target_index is not None:
        status_out[target_index] = f"✓ Saved to {config_path}.{suffix}"
        dirty_out[target_index] = False

    # Refresh JSON preview with saved values
    json_text = json.dumps(parsed, indent=2, default=str)
    json_pill = html.Span(
        [html.Span(className="swatch"), "saved"],
        className="pill ok",
    )

    return status_out, dirty_out, errors_out, json_text, json_pill


def _id_match_for_save(output_spec, target_form_id: str) -> bool:
    id_ = output_spec.get("id", {})
    return isinstance(id_, dict) and id_.get("form") == target_form_id
