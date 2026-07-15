"""Shared filter bar used by the Recordings, Pipeline, and Run pages.

Renders a single ``.card`` with a flex row of labelled dropdowns in the
canonical order **Sample · Scan Type · Date (from/to) · Group · Status**.
Control ids follow ``filter_id(prefix, field)`` → ``f"{prefix}-filter-{field}"``
so each page wires its own callbacks by explicit id. Option lists for the
data-driven fields (everything except status, whose values are fixed) are
populated by the page's render callback.

The status options are the real ``TaskStatus`` values only — there is no
"skipped"/"queued" state in the pipeline.

`ALL_FIELDS` is the canonical order; `filter_kwargs()` maps a field→value dict
straight onto the keyword args of `data.filter_recordings` /
`data.filter_pipeline_df`.
"""

from __future__ import annotations

from dash import dcc, html


# Canonical field order. Date is a single calendar range picker.
ALL_FIELDS = ("sample", "scan-type", "date", "group", "status")

# Data-driven fields whose option lists are filled by the page callback
# (status options are fixed; date bounds are set via min/max_date_allowed).
POPULATED_FIELDS = ("sample", "scan-type", "group")

# Fixed status options (real TaskStatus values; see pipeline/task_record.py).
STATUS_OPTIONS = [
    {"label": "complete", "value": "complete"},
    {"label": "running", "value": "running"},
    {"label": "failed", "value": "failed"},
    {"label": "not run", "value": "not_run"},
]

# field → keyword arg name on filter_recordings / filter_pipeline_df.
# The date range picker is one control, but yields two kwargs — pages split its
# start_date/end_date into the "date-from"/"date-to" keys (as "YYMMDD") before
# calling filter_kwargs.
_KWARG = {
    "sample": "sample_ids",
    "scan-type": "scan_types",
    "date-from": "date_from",
    "date-to": "date_to",
    "group": "groups",
    "status": "statuses",
}

_LABELS = {
    "sample": "Sample(s)",
    "scan-type": "Scan type(s)",
    "date": "Date range",
    "group": "Group(s)",
    "status": "Status",
}

_LABEL_STYLE = {
    "fontFamily": "var(--font-mono)",
    "fontSize": "10px",
    "color": "var(--ink-3)",
    "marginBottom": "4px",
    "display": "block",
}


def filter_id(prefix: str, field: str) -> str:
    """Stable string id for a filter control, e.g. ``recordings-filter-sample``."""
    return f"{prefix}-filter-{field}"


def iso_to_yymmdd(iso: str | None) -> str | None:
    """``"2026-02-03"`` → ``"260203"`` (the cache's 6-digit date form).

    Accepts the ISO date a `dcc.DatePickerRange` yields (optionally with a time
    suffix); returns ``None`` for an empty/None input.
    """
    if not iso:
        return None
    y, m, d = iso[:10].split("-")
    return f"{y[2:]}{m}{d}"


def yymmdd_to_iso(yymmdd: str | None) -> str | None:
    """``"260203"`` → ``"2026-02-03"`` for the picker's min/max bounds.

    Cache dates are always ``20YY``. Returns ``None`` for empty/None input.
    """
    if not yymmdd:
        return None
    return f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"


def filter_kwargs(values: dict[str, object]) -> dict[str, object]:
    """Map a ``{field: value}`` dict onto filter-function keyword args.

    Empty selections (``[]``/``None``/``""``) become ``None`` so the pure
    filter functions treat them as "no constraint".
    """
    out: dict[str, object] = {}
    for field, kwarg in _KWARG.items():
        v = values.get(field)
        out[kwarg] = v if v else None
    return out


def _field(prefix: str, field: str) -> html.Div:
    if field == "date":
        control = dcc.DatePickerRange(
            id=filter_id(prefix, field),
            start_date_placeholder_text="from",
            end_date_placeholder_text="to",
            display_format="YY-MM-DD",
            clearable=True,
            first_day_of_week=1,
            # min/max_date_allowed set per render by the page callback.
        )
        flex = "1 1 240px"
    else:
        is_status = field == "status"
        control = dcc.Dropdown(
            id=filter_id(prefix, field),
            options=STATUS_OPTIONS if is_status else [],
            value=[],
            multi=True,
            placeholder="any",
            clearable=True,
        )
        flex = "1 1 170px"
    return html.Div(
        [html.Label(_LABELS[field], style=_LABEL_STYLE), control],
        style={"flex": flex},
    )


def build_filter_bar(prefix: str, show: tuple[str, ...] = ALL_FIELDS) -> html.Div:
    """Return the filter card for a page.

    Args:
        prefix: page namespace for control ids (e.g. "recordings").
        show:   which fields to render, in canonical order. Defaults to all.
    """
    fields = [f for f in ALL_FIELDS if f in show]
    return html.Div(
        [
            html.Div(
                [html.Span("filters", className="h-title")],
                className="card-head",
            ),
            html.Div(
                html.Div(
                    [_field(prefix, f) for f in fields],
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
    )
