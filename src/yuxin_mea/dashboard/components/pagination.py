"""Tiny pagination helper shared by the Pipeline and Recordings pages.

`page_bounds` is a pure function (index math only) so callers can slice either
a list (``items[start:end]``) or a DataFrame (``df.iloc[start:end]``).
`pager_bar` renders the static prev/next buttons + "page X / Y" label (always
present in the layout); `pager_state` computes the label text + disabled flags
each render. The page index lives in a per-page ``dcc.Store`` the caller owns.
"""

from __future__ import annotations

from dash import html


PAGE_SIZE = 50


def page_bounds(n_items: int, page: int, size: int = PAGE_SIZE) -> tuple[int, int, int, int]:
    """Return ``(start, end, n_pages, clamped_page)`` for the requested page.

    `page` is clamped into ``[0, n_pages - 1]`` so a stale index (e.g. after a
    filter shrinks the result set) never yields an empty slice.
    """
    n_pages = max(1, (n_items + size - 1) // size)
    clamped = max(0, min(page or 0, n_pages - 1))
    start = clamped * size
    end = min(start + size, n_items)
    return start, end, n_pages, clamped


def pager_bar(prefix: str) -> html.Div:
    """Static prev/next pager for a page namespace — rendered once in the layout.

    The buttons and label live in the static layout (ids
    ``f"{prefix}-page-prev"`` / ``f"{prefix}-page-label"`` /
    ``f"{prefix}-page-next"``) so they always exist. The page's render callback
    updates the label text and the buttons' ``disabled`` state via
    :func:`pager_state`; it does **not** mint the buttons (which would create a
    chicken-and-egg: the callback's own prev/next Inputs wouldn't exist at load).
    """
    return html.Div(
        [
            html.Button(
                "‹ prev",
                id=f"{prefix}-page-prev",
                n_clicks=0,
                className="btn",
                disabled=True,
            ),
            html.Span(
                "",
                id=f"{prefix}-page-label",
                style={
                    "fontFamily": "var(--font-mono)",
                    "fontSize": "11px",
                    "color": "var(--ink-3)",
                    "padding": "0 4px",
                },
            ),
            html.Button(
                "next ›",
                id=f"{prefix}-page-next",
                n_clicks=0,
                className="btn",
                disabled=True,
            ),
        ],
        style={
            "display": "flex",
            "gap": "10px",
            "alignItems": "center",
            "justifyContent": "flex-end",
            "padding": "8px 4px",
        },
    )


def pager_state(page: int, n_pages: int, n_items: int) -> tuple[str, bool, bool]:
    """Return ``(label_text, prev_disabled, next_disabled)`` for a pager bar."""
    label = f"page {page + 1} / {n_pages} · {n_items} total"
    return label, page <= 0, page >= n_pages - 1
