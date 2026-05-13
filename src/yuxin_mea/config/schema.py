"""Schema metadata for ConfigManager-driven UIs.

A `ParamSpec` describes one editable field: its Python type, default value,
human-readable description, and optional choices / bounds / nested schema.
Tasks declare a `params_schema()` classmethod returning
`{param_name: ParamSpec}`; the dashboard config builder consumes it to
render forms and validate edits.

`validate_value(spec, raw)` is a small coercion + validation layer:
- coerces strings (which is what HTML form widgets emit) into the right
  Python type,
- enforces `choices`, `min`, `max`,
- recurses into `nested_schema` for `type="dict"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ParamType = Literal[
    "str", "int", "float", "bool",
    "list_int", "list_float", "list_str",
    "path", "dict",
]


@dataclass(frozen=True)
class ParamSpec:
    """One editable parameter description.

    `nullable=True` permits `None` as a valid value (e.g. some SI sorters
    treat `None` as "no limit" while a normal `int` like 50000 is also
    valid). The form widget still renders the underlying type's widget;
    the user clears the input to emit `None`.
    """

    type: ParamType
    default: Any
    description: str = ""
    choices: list[Any] | None = None
    multiselect: bool = False
    min: float | None = None
    max: float | None = None
    nested_schema: dict[str, "ParamSpec"] | None = None
    nullable: bool = False


class ValidationError(ValueError):
    """Raised by validate_value when a field fails its schema."""


def validate_value(spec: ParamSpec, value: Any) -> Any:
    """Coerce a raw form value to its declared type and check constraints.

    Returns the coerced value. Raises :class:`ValidationError` with a
    user-facing message on failure.

    `None` is allowed for: `path` (treated as the empty string), any spec
    with `nullable=True`. Other types reject `None` with "required".
    """
    if value is None and spec.nullable and spec.type != "path":
        return None
    if spec.type == "bool":
        return _coerce_bool(value)
    if spec.type == "int":
        return _coerce_numeric(value, spec, int, "integer")
    if spec.type == "float":
        return _coerce_numeric(value, spec, float, "number")
    if spec.type == "str":
        return _coerce_str(value, spec)
    if spec.type == "path":
        return "" if value is None else str(value)
    if spec.type == "list_int":
        return _coerce_list(value, spec, int, "integer")
    if spec.type == "list_float":
        return _coerce_list(value, spec, float, "number")
    if spec.type == "list_str":
        return _coerce_list_str(value, spec)
    if spec.type == "dict":
        return _coerce_dict(value, spec)
    raise ValidationError(f"Unknown ParamSpec.type {spec.type!r}")


# ---------------------------------------------------------------------------
# Type-specific coercion helpers
# ---------------------------------------------------------------------------


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1", "on"}:
            return True
        if v in {"false", "no", "0", "off", ""}:
            return False
    if isinstance(value, list):
        return bool(value)
    raise ValidationError(f"Expected boolean, got {value!r}")


def _coerce_numeric(value: Any, spec: ParamSpec, cast, label: str):
    if value is None or value == "":
        raise ValidationError(f"Required ({label}).")
    try:
        v = cast(value) if not isinstance(value, str) else cast(float(value))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Not a valid {label}: {value!r}") from exc
    if spec.min is not None and v < spec.min:
        raise ValidationError(f"Must be ≥ {spec.min}.")
    if spec.max is not None and v > spec.max:
        raise ValidationError(f"Must be ≤ {spec.max}.")
    if spec.choices is not None and v not in spec.choices:
        raise ValidationError(f"Must be one of {spec.choices}.")
    return v


def _coerce_str(value: Any, spec: ParamSpec) -> str:
    if value is None:
        if spec.choices and "" not in spec.choices:
            raise ValidationError("Required.")
        return ""
    v = str(value)
    if spec.choices is not None and v not in spec.choices:
        raise ValidationError(f"Must be one of {spec.choices}.")
    return v


def _coerce_list(value: Any, spec: ParamSpec, cast, label: str) -> list:
    items = _split_list_input(value)
    out: list = []
    for raw in items:
        try:
            out.append(cast(raw) if not isinstance(raw, str) else cast(float(raw)))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Element {raw!r} is not a valid {label}."
            ) from exc
    if spec.min is not None and len(out) < spec.min:
        raise ValidationError(f"Need at least {int(spec.min)} element(s).")
    if spec.max is not None and len(out) > spec.max:
        raise ValidationError(f"Max {int(spec.max)} element(s).")
    return out


def _coerce_list_str(value: Any, spec: ParamSpec) -> list[str]:
    items = _split_list_input(value)
    out = [str(x) for x in items]
    if spec.choices is not None:
        bad = [x for x in out if x not in spec.choices]
        if bad:
            raise ValidationError(
                f"Unknown choice(s): {bad}. Allowed: {spec.choices}."
            )
    return out


def _coerce_dict(value: Any, spec: ParamSpec) -> dict:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValidationError(f"Expected dict, got {type(value).__name__}.")
    if spec.nested_schema is None:
        return dict(value)
    out: dict[str, Any] = {}
    for key, sub_spec in spec.nested_schema.items():
        raw = value.get(key, sub_spec.default)
        out[key] = validate_value(sub_spec, raw)
    return out


def _split_list_input(value: Any) -> list:
    """Split a list-input either from a Python list or comma-separated string."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return [value]
