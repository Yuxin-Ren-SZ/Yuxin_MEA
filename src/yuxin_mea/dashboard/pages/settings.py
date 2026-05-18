"""Settings page — schema-driven config editor.

Layout: top-level `dcc.Tabs` with a "Globals" tab followed by one tab per
task that has a non-empty `params_schema()`. Each tab renders a form via
`form_builder.render_form()`. Per-form Save buttons validate field values,
call `ConfigManager.set_task_params()` / `set_globals()`, and persist via
`ConfigManager.save()`.

Pattern-matched IDs (`{"form": <form_id>, "field": <name>}` and
`{"form": <form_id>, "key": ...}`) let one set of callbacks handle every
form uniformly — no per-form code duplication.

If the loaded config has keys that aren't in any task's schema (e.g. a
renamed param after a code update), the page surfaces a yellow banner at
the top so the user sees what will be dropped on the next Save.
"""

from __future__ import annotations

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


def _all_form_ids() -> list[str]:
    return ["globals", *(f"task-{name}" for name in _TASK_BY_NAME)]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


layout = html.Div(
    [
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
                        html.H1("Settings"),
                        html.Div(
                            "Per-tab Save. data_root / analysis_root / "
                            "figure_root changes require a dashboard restart.",
                            className="subtitle",
                        ),
                    ]
                ),
            ],
            className="view-head",
        ),
        html.Div(id="settings-unknown-keys-banner"),
        dcc.Tabs(
            id="settings-tabs",
            value="globals",
            parent_className="tab-strip",
            className="dash-tabs-container",
            children=[
                dcc.Tab(
                    label="Globals",
                    value="globals",
                    className="tab--regular",
                    selected_className="tab--selected",
                    children=[html.Div(id="settings-tab-globals",
                                       style={"marginTop": "16px"})],
                ),
                *[
                    dcc.Tab(
                        label=name,
                        value=f"task-{name}",
                        className="tab--regular",
                        selected_className="tab--selected",
                        children=[
                            html.Div(id={"settings-tab": name},
                                     style={"marginTop": "16px"})
                        ],
                    )
                    for name in _TASK_BY_NAME
                ],
            ],
        ),
    ],
    className="page",
)


# ---------------------------------------------------------------------------
# 1) Populate the Globals tab on page entry
# ---------------------------------------------------------------------------


@callback(
    Output("settings-tab-globals", "children"),
    Output("settings-unknown-keys-banner", "children"),
    Input("settings-tabs", "id"),  # fires once on initial render
)
def _populate_globals(_id: str):
    yuxin_ctx = current_app.config.get("YUXIN_MEA", {})
    config_path = yuxin_ctx.get("config_path")

    values: dict = {}
    unknown: dict[str, list[str]] = {}
    if config_path is not None and Path(config_path).exists():
        cm = ConfigManager()
        cm.load(config_path)
        values = {key: cm.get_global(key, "") for key in GLOBALS_SCHEMA}
        unknown = cm.validate_loaded(_TASK_SCHEMAS)

    banner = _build_unknown_keys_banner(unknown)
    form = render_form("globals", GLOBALS_SCHEMA, values, title="Global settings")
    return form, banner


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
# 2) Populate each task tab on first view (lazy — heavy schemas defer until then)
# ---------------------------------------------------------------------------


@callback(
    Output({"settings-tab": ALL}, "children"),
    Input("settings-tabs", "id"),
)
def _populate_task_tabs(_id: str):
    yuxin_ctx = current_app.config.get("YUXIN_MEA", {})
    config_path = yuxin_ctx.get("config_path")

    cm = ConfigManager()
    if config_path is not None and Path(config_path).exists():
        cm.load(config_path)

    outputs = []
    for task_name, schema in _TASK_SCHEMAS.items():
        if not schema:
            outputs.append(html.P(
                f"{task_name} has no params_schema(); edit the JSON file directly.",
                style={"color": "#888"},
            ))
            continue
        values = cm.get_task_params(task_name)
        outputs.append(render_form(f"task-{task_name}", schema, values,
                                   title=f"{task_name} parameters"))
    return outputs


# ---------------------------------------------------------------------------
# 3) Dirty store: any field change → flip dirty=True per form
# ---------------------------------------------------------------------------


@callback(
    Output({"form": ALL, "key": "dirty"}, "data"),
    Input({"form": ALL, "field": ALL}, "value"),
    State({"form": ALL, "key": "dirty"}, "id"),
    prevent_initial_call=True,
)
def _mark_forms_dirty(_field_values, dirty_ids):
    """Mark every form whose fields were just edited as dirty.

    Granular per-form tracking would require ALL_MATCH over the form key,
    which Dash doesn't expose cleanly. We mark all forms dirty whenever
    ANY field changes — slightly conservative but safe (Save still
    validates per-form, so no spurious writes occur).
    """
    return [True] * len(dirty_ids)


# ---------------------------------------------------------------------------
# 4) Save-button-disabled toggle
# ---------------------------------------------------------------------------


@callback(
    Output({"form": ALL, "key": "save"}, "disabled"),
    Input({"form": ALL, "key": "dirty"}, "data"),
)
def _toggle_save_buttons(dirty_flags):
    return [not flag for flag in dirty_flags]


# ---------------------------------------------------------------------------
# 5) Save: one callback handles every form via pattern-matching
# ---------------------------------------------------------------------------


@callback(
    Output({"form": ALL, "key": "status"}, "children"),
    Output({"form": ALL, "key": "dirty"}, "data", allow_duplicate=True),
    Output({"form": ALL, "field-error": ALL}, "children"),
    Input({"form": ALL, "key": "save"}, "n_clicks"),
    State({"form": ALL, "field": ALL}, "value"),
    State({"form": ALL, "field": ALL}, "id"),
    prevent_initial_call=True,
)
def _save_any_form(_save_clicks, field_values, field_ids):
    triggered = ctx.triggered_id
    if not triggered or triggered.get("key") != "save":
        return [dash.no_update] * len(_save_clicks), \
               [dash.no_update] * len(_save_clicks), \
               [dash.no_update] * len(field_ids)
    target_form_id = triggered["form"]

    # Bucket fields by form_id.
    fields_by_form: dict[str, dict[str, object]] = {}
    for value, id_ in zip(field_values, field_ids):
        fields_by_form.setdefault(id_["form"], {})[id_["field"]] = value

    raw = fields_by_form.get(target_form_id, {})

    if target_form_id == "globals":
        schema = GLOBALS_SCHEMA
    elif target_form_id.startswith("task-"):
        task_name = target_form_id[len("task-"):]
        schema = _TASK_SCHEMAS.get(task_name, {})
    else:
        schema = {}

    parsed, errors = collect_values(schema, raw)

    # Build the per-field-error output list in the same order as field_ids.
    errors_out = []
    for id_ in field_ids:
        if id_["form"] == target_form_id and id_["field"] in errors:
            errors_out.append(errors[id_["field"]])
        elif id_["form"] == target_form_id:
            errors_out.append("")
        else:
            errors_out.append(dash.no_update)

    # Build status output list (one per form, ordered by _save_clicks).
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
        return status_out, dirty_out, errors_out

    # No errors — persist.
    yuxin_ctx = current_app.config.get("YUXIN_MEA", {})
    config_path: Path = yuxin_ctx.get("config_path")
    if config_path is None:
        if target_index is not None:
            status_out[target_index] = "❌ No config path configured for this dashboard."
        return status_out, dirty_out, errors_out

    cm = ConfigManager()
    if Path(config_path).exists():
        cm.load(config_path)
    if target_form_id == "globals":
        cm.set_globals(parsed)
        suffix = (
            " Restart the dashboard for `data_root` / `analysis_root` / "
            "`figure_root` changes to take effect on other pages."
        )
    else:
        task_name = target_form_id[len("task-"):]
        cm.set_task_params(task_name, parsed)
        suffix = ""
    cm.save(config_path)

    # Update the dashboard's stash so the file-existence check flips to True
    # (in case this was the first-ever save).
    current_app.config["YUXIN_MEA"]["config_exists"] = True

    if target_index is not None:
        status_out[target_index] = f"✓ Saved to {config_path}.{suffix}"
        dirty_out[target_index] = False
    return status_out, dirty_out, errors_out


def _id_match_for_save(output_spec, target_form_id: str) -> bool:
    """Whether the Output at `output_spec` belongs to the form we just saved."""
    id_ = output_spec.get("id", {})
    return isinstance(id_, dict) and id_.get("form") == target_form_id
