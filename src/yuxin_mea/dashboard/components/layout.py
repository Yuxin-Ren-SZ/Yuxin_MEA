"""App shell — topbar, left rail (grouped nav), main viewport.

Uses the `mea-chip` design's visual vocabulary; CSS lives in
`dashboard/assets/styles.css` (auto-loaded by Dash).

Also exports `no_config_banner()` for data pages to render when the
dashboard is running in config-only mode (no `pipeline_config.json` yet).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import dash
from dash import Input, Output, callback, clientside_callback, dcc, html
from flask import current_app


# Per-page presentation metadata: section and glyph.
# Pages not listed here fall back to the "operations" section with a dot
# glyph — keeps the rail resilient if someone adds a page without
# updating this dict.
_NAV_META: dict[str, tuple[str, str]] = {
    "Home": ("operations", "◐"),
    "Recordings": ("operations", "▦"),
    "Pipeline": ("operations", "≡"),
    "Run": ("operations", "▸"),
    "Burst diagnostic": ("analysis", "∿"),
    "Plate viewer": ("analysis", "▤"),
    "Settings": ("system", "{}"),
}

_SECTION_ORDER = ("operations", "analysis", "system")


def _git_rev() -> str:
    """Return short git SHA for the topbar chip; empty if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
            timeout=1,
        )
        return out.decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""


_GIT_REV = _git_rev()


def no_config_banner() -> html.Div:
    """Banner shown on data pages when the config file doesn't exist yet."""
    return html.Div(
        [
            html.Strong("No config loaded. "),
            html.Span("Go to "),
            dcc.Link("Settings", href="/settings"),
            html.Span(
                " to fill in data_root, analysis_root, and per-task "
                "parameters, then Save."
            ),
        ],
        className="banner info",
    )


def build_layout() -> html.Div:
    """Return the static app shell. Pages mount inside `.viewport`."""
    return html.Div(
        [
            dcc.Location(id="dashboard-url"),
            _topbar(),
            _rail(),
            html.Main(
                html.Div(dash.page_container, className="viewport"),
                className="main",
            ),
        ],
        className="app",
    )


# ---------------------------------------------------------------------------
# Topbar
# ---------------------------------------------------------------------------


def _topbar() -> html.Div:
    branch_chip_label = f"dev · {_GIT_REV}" if _GIT_REV else "dev"
    return html.Div(
        [
            html.Div(
                [
                    html.Span(className="brand-mark"),
                    html.Span("MEA Chip"),
                    html.Span("/", className="brand-sep"),
                    html.Span("analysis ops", style={"color": "var(--ink-3)"}),
                ],
                className="brand",
            ),
            html.Span(branch_chip_label, className="branch-chip"),
            html.Div(className="topbar-spacer"),
            html.Div(
                [
                    html.Span(
                        [html.Span(className="dot"), "scheduler · online"],
                    ),
                ],
                className="topbar-meta",
            ),
            html.Div("YR", className="avatar"),
        ],
        className="topbar",
    )


# ---------------------------------------------------------------------------
# Left rail
# ---------------------------------------------------------------------------


def _rail() -> html.Aside:
    pages = sorted(
        dash.page_registry.values(),
        key=lambda p: (p.get("order", 100), p["name"]),
    )
    by_section: dict[str, list] = {s: [] for s in _SECTION_ORDER}
    for page in pages:
        section, _glyph = _NAV_META.get(page["name"], ("operations", "·"))
        by_section.setdefault(section, []).append(page)

    children: list = []
    for section in _SECTION_ORDER:
        items = by_section.get(section, [])
        if not items:
            continue
        children.append(html.Div(section, className="rail-section"))
        for page in items:
            children.append(_rail_item(page))

    children.append(html.Div(id="rail-footer-slot"))
    return html.Aside(children, className="rail")


def _rail_item(page: dict) -> dcc.Link:
    name = page["name"]
    _section, glyph = _NAV_META.get(name, ("operations", "·"))
    return dcc.Link(
        [
            html.Span(glyph, className="glyph"),
            html.Span(name),
        ],
        href=page["relative_path"],
        className="rail-item",
        # `id` lets the clientside active-toggle callback target each item.
        id={"rail-link": page["relative_path"]},
        refresh=False,
    )


# Populate the rail-footer once the app context is up. We can't read
# `current_app.config` at layout-build time reliably; doing it in a
# callback (which always runs inside a request) is the safe path.
@callback(
    Output("rail-footer-slot", "children"),
    Input("dashboard-url", "pathname"),
)
def _render_rail_footer(_pathname: str):
    try:
        ctx = current_app.config.get("YUXIN_MEA", {})
        analysis_root = ctx.get("analysis_root")
    except RuntimeError:
        analysis_root = None

    root_str = str(analysis_root) if analysis_root else "(not set)"
    return html.Div(
        [
            html.Div(
                "analysis_root",
                style={"color": "var(--ink-3)", "marginBottom": "4px"},
            ),
            html.Code(root_str),
            html.Div(
                "caches",
                style={"color": "var(--ink-3)", "marginTop": "8px"},
            ),
            html.Code("experiment_cache.json"),
            html.Br(),
            html.Code("pipeline_cache.json"),
        ],
        className="rail-footer",
    )


# Active-link toggle: add `.active` to the rail item whose href matches
# the current pathname. Runs entirely in the browser — no server hop on
# nav clicks.
clientside_callback(
    """
    function(pathname, ids) {
        if (!ids) { return []; }
        return ids.map(function(id) {
            var href = id["rail-link"];
            var isActive = (pathname === href) ||
                           (href !== "/" && pathname && pathname.indexOf(href) === 0);
            return isActive ? "rail-item active" : "rail-item";
        });
    }
    """,
    Output({"rail-link": dash.ALL}, "className"),
    Input("dashboard-url", "pathname"),
    Input({"rail-link": dash.ALL}, "id"),
)
