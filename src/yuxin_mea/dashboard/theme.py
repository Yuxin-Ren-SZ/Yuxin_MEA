"""Plotly template matching the dashboard's `mea-chip` warm-paper design.

Hex values mirror the CSS custom properties in `assets/styles.css`:
- `--bg`     → `#f4f1ea`  (paper background)
- `--bg-elev`→ `#fbf9f3`  (plot area background)
- `--ink`    → `#1c1a15`  (text)
- `--ink-3`  → `#84807a`  (axis labels)
- `--line`   → `#d9d3c5`  (gridlines)
- accents (oklch in CSS, hex here for Plotly compatibility):
  sage `#5d9e7e` · amber `#c89a3a` · fail `#c0623a` · info `#5b7fac`
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio


TEMPLATE_NAME = "mea_paper"

_PAPER = "#f4f1ea"
_PLOT = "#fbf9f3"
_INK = "#1c1a15"
_INK3 = "#84807a"
_LINE = "#d9d3c5"
_LINE_SOFT = "#e6e1d2"

_STATUS_PALETTE = ["#5d9e7e", "#c89a3a", "#c0623a", "#5b7fac", "#84807a"]


def _build_template() -> go.layout.Template:
    return go.layout.Template(
        layout=dict(
            paper_bgcolor=_PAPER,
            plot_bgcolor=_PLOT,
            font=dict(
                family='"Geist", ui-sans-serif, system-ui, sans-serif',
                size=12,
                color=_INK,
            ),
            colorway=_STATUS_PALETTE,
            xaxis=dict(
                gridcolor=_LINE_SOFT,
                linecolor=_LINE,
                tickcolor=_LINE,
                tickfont=dict(color=_INK3, family='"Geist Mono", monospace', size=10),
                zerolinecolor=_LINE,
            ),
            yaxis=dict(
                gridcolor=_LINE_SOFT,
                linecolor=_LINE,
                tickcolor=_LINE,
                tickfont=dict(color=_INK3, family='"Geist Mono", monospace', size=10),
                zerolinecolor=_LINE,
            ),
            legend=dict(
                bgcolor=_PAPER,
                bordercolor=_LINE,
                borderwidth=1,
                font=dict(size=11, color=_INK),
            ),
            margin=dict(l=48, r=24, t=32, b=40),
            hoverlabel=dict(
                bgcolor=_PAPER,
                bordercolor=_LINE,
                font=dict(color=_INK, family='"Geist Mono", monospace', size=11),
            ),
        )
    )


def apply_default_theme() -> None:
    """Register `mea_paper` and set it as the Plotly default.

    Safe to call more than once: re-registering overwrites in place.
    """
    pio.templates[TEMPLATE_NAME] = _build_template()
    pio.templates.default = TEMPLATE_NAME
