"""Schema-driven form builder for the Settings page.

Given a `{param_name: ParamSpec}` dict and the current loaded values, this
module emits Dash components and a callback-friendly value-collection
helper. The Settings page composes one `render_form()` per tab (Globals +
one per task) and wires per-form callbacks via pattern-matched IDs:

    {"form": "<form_id>", "field": "<param_name>"}

Validation is reused from `yuxin_mea.config.schema.validate_value`. Errors
are returned per-field and rendered next to each input via a sibling div.

The form-builder is intentionally **dependency-free** beyond Dash and the
schema module — no `dash-daq` or other component packs. `bool` renders as
a single-option `dcc.Checklist` for predictability.
"""

from __future__ import annotations

from typing import Any

from dash import dcc, html

from yuxin_mea.config.schema import ParamSpec, ValidationError, validate_value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_form(
    form_id: str,
    schema: dict[str, ParamSpec],
    values: dict[str, Any],
    title: str | None = None,
) -> html.Div:
    """Build a form Div: header + Save button + N rendered fields + status."""
    header = []
    if title:
        header.append(html.H3(title, style={"marginTop": "0"}))
    save_btn = html.Button(
        "Save",
        id={"form": form_id, "key": "save"},
        n_clicks=0,
        disabled=True,
        style={"marginRight": "12px"},
    )
    status = html.Span(
        id={"form": form_id, "key": "status"},
        style={"color": "#555"},
    )
    return html.Div(
        [
            *header,
            html.Div([save_btn, status], style={"marginBottom": "16px"}),
            dcc.Store(id={"form": form_id, "key": "dirty"}, data=False),
            html.Div(
                [render_field(form_id, name, spec, values.get(name, spec.default))
                 for name, spec in schema.items()],
                style={"display": "flex", "flexDirection": "column", "gap": "12px"},
            ),
        ],
        style={"padding": "8px 0"},
    )


def render_field(
    form_id: str,
    name: str,
    spec: ParamSpec,
    value: Any,
) -> html.Div:
    """One label + widget + per-field error slot."""
    widget = _build_widget(form_id, name, spec, value)
    label = html.Label(
        [
            html.Span(name, style={"fontWeight": "600"}),
            _description_node(spec),
        ],
        style={"display": "block", "marginBottom": "4px"},
    )
    error = html.Div(
        id={"form": form_id, "field-error": name},
        style={"color": "#c62828", "fontSize": "12px", "marginTop": "4px",
               "minHeight": "14px"},
    )
    return html.Div([label, widget, error])


def collect_values(
    schema: dict[str, ParamSpec],
    raw_by_name: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate every schema field; return (parsed, errors).

    `raw_by_name` is `{param_name: widget_value}` (caller pulls these out of
    `State` lists by zipping ids and values). Nested-dict fields are
    rendered with dotted IDs (`"parent.child"`); this function reconstructs
    them into `{"parent": {"child": value}}` before validation so
    ``validate_value`` sees the proper nested dict.

    The returned `parsed` dict contains coerced values for every schema key
    that validated; the `errors` dict contains a user-facing message for
    every key that didn't. The two dicts have disjoint key sets — a caller
    can short-circuit on `errors` being non-empty.
    """
    raw_by_name = _reconstruct_nested(raw_by_name)
    parsed: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, spec in schema.items():
        raw = raw_by_name.get(name, spec.default)
        # `bool` widgets emit a list (checked = [True]); unwrap before validation.
        if spec.type == "bool" and isinstance(raw, list):
            raw = bool(raw)
        try:
            parsed[name] = validate_value(spec, raw)
        except ValidationError as exc:
            errors[name] = str(exc)
    return parsed, errors


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _reconstruct_nested(raw_by_name: dict[str, Any]) -> dict[str, Any]:
    """Fold dotted field IDs (`"parent.child"`) back into nested dicts.

    Nested-dict ParamSpecs render sub-fields with IDs like
    `"high_vram_sorter_kwargs.batch_size_seconds"` — but the Save callback
    builds a flat `{field: value}` dict that strips that hierarchy. Without
    this step, ``collect_values`` looks up the parent name, fails, and
    falls back to ``spec.default`` — silently discarding every user edit
    to nested fields.

    Inputs that contain no dotted keys pass through unchanged.
    """
    if not any("." in k for k in raw_by_name):
        return dict(raw_by_name)
    out: dict[str, Any] = {}
    for key, value in raw_by_name.items():
        if "." in key:
            parent, _, child = key.partition(".")
            out.setdefault(parent, {})[child] = value
        else:
            out.setdefault(key, value)
    return out


def _field_id(form_id: str, name: str) -> dict[str, str]:
    return {"form": form_id, "field": name}


def _description_node(spec: ParamSpec) -> Any:
    if not spec.description:
        return ""
    return html.Span(
        f"  {spec.description}",
        style={"fontWeight": "400", "color": "#666", "fontSize": "12px"},
    )


def _build_widget(form_id: str, name: str, spec: ParamSpec, value: Any) -> Any:
    common = {"id": _field_id(form_id, name)}

    if spec.choices is not None and spec.type in {"str", "int", "float"}:
        return dcc.Dropdown(
            options=[{"label": str(c), "value": c} for c in spec.choices],
            value=value if value is not None else spec.default,
            clearable=False,
            style={"maxWidth": "320px"},
            **common,
        )

    if spec.type == "bool":
        return dcc.Checklist(
            options=[{"label": "", "value": True}],
            value=[True] if value else [],
            **common,
        )

    if spec.type in {"int", "float"}:
        step = 1 if spec.type == "int" else "any"
        return dcc.Input(
            type="number",
            value=value if value is not None else spec.default,
            step=step,
            min=spec.min, max=spec.max,
            style={"width": "160px"},
            **common,
        )

    if spec.type == "list_str" and spec.choices is not None:
        return dcc.Dropdown(
            options=[{"label": str(c), "value": c} for c in spec.choices],
            value=list(value) if value else list(spec.default or []),
            multi=spec.multiselect,
            style={"maxWidth": "480px"},
            **common,
        )

    if spec.type in {"list_int", "list_float", "list_str"}:
        # Render as comma-separated text input. collect_values parses.
        display = ", ".join(str(x) for x in (value or spec.default or []))
        return dcc.Input(
            type="text",
            value=display,
            placeholder="comma-separated",
            style={"width": "320px"},
            **common,
        )

    if spec.type == "dict":
        # Recurse into nested_schema if provided; otherwise render as JSON
        # textarea. Keep nested forms simple (no Save button — outer form
        # is responsible). For Phase 3 simplicity, dicts without a
        # nested_schema fall back to a JSON textarea.
        if spec.nested_schema is not None:
            children = [
                render_field(form_id, f"{name}.{sub_name}", sub_spec,
                             (value or {}).get(sub_name, sub_spec.default))
                for sub_name, sub_spec in spec.nested_schema.items()
            ]
            return html.Details(
                [
                    html.Summary(f"{len(children)} nested fields",
                                 style={"cursor": "pointer", "color": "#1f5aa6"}),
                    html.Div(
                        children,
                        style={"borderLeft": "2px solid #ddd",
                               "paddingLeft": "12px", "marginTop": "8px",
                               "display": "flex", "flexDirection": "column",
                               "gap": "10px"},
                    ),
                ],
                open=False,
            )
        # Free-form dict — JSON textarea
        import json as _json
        return dcc.Textarea(
            value=_json.dumps(value or {}, indent=2),
            style={"width": "480px", "minHeight": "120px", "fontFamily": "monospace"},
            **common,
        )

    # path | str (no choices) → text input
    return dcc.Input(
        type="text",
        value="" if value is None else str(value),
        style={"width": "480px", "fontFamily": "monospace" if spec.type == "path" else "inherit"},
        **common,
    )
