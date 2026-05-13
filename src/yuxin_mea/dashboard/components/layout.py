"""Top-level app layout: navbar + active page container.

Also exports `no_config_banner()` for data pages to render when the
dashboard is running in config-only mode (no `pipeline_config.json` yet).
"""

from __future__ import annotations

import dash
from dash import dcc, html


_NAVBAR_STYLE = {
    "display": "flex",
    "gap": "16px",
    "padding": "12px 24px",
    "borderBottom": "1px solid #e0e0e0",
    "marginBottom": "16px",
    "fontFamily": "sans-serif",
}

_LINK_STYLE = {"textDecoration": "none", "color": "#1f5aa6", "fontWeight": "500"}
_TITLE_STYLE = {"marginRight": "32px", "fontWeight": "700", "color": "#222"}


def no_config_banner() -> html.Div:
    """Banner shown on data pages when the config file doesn't exist yet."""
    return html.Div(
        [
            html.Strong("No config loaded. "),
            html.Span("Go to "),
            dcc.Link("Settings", href="/settings",
                     style={"color": "#1f5aa6", "fontWeight": "600"}),
            html.Span(" to fill in `data_root`, `analysis_root`, and per-task "
                      "parameters, then Save."),
        ],
        style={
            "backgroundColor": "#e3f2fd",
            "border": "1px solid #90caf9",
            "padding": "12px 16px",
            "borderRadius": "4px",
            "marginBottom": "16px",
        },
    )


def build_layout() -> html.Div:
    """Return the static app shell. Pages plug in via `dash.page_container`."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span("yuxin_mea dashboard", style=_TITLE_STYLE),
                    *_nav_links(),
                ],
                style=_NAVBAR_STYLE,
            ),
            html.Div(dash.page_container, style={"padding": "0 24px"}),
        ]
    )


def _nav_links() -> list[dcc.Link]:
    """Build one nav link per registered page, ordered by `order` then name."""
    pages = sorted(
        dash.page_registry.values(),
        key=lambda p: (p.get("order", 100), p["name"]),
    )
    return [
        dcc.Link(p["name"], href=p["relative_path"], style=_LINK_STYLE) for p in pages
    ]
