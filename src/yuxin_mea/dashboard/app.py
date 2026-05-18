"""Dash app factory: `build_app(config_path)` returns a configured Dash."""

from __future__ import annotations

from pathlib import Path

from dash import Dash

from yuxin_mea.config import ConfigManager

from .components.layout import build_layout
from .theme import apply_default_theme


_INDEX_STRING = """\
<!DOCTYPE html>
<html data-theme="warm">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def build_app(config_path: Path) -> Dash:
    """Build a multipage Dash app pinned to the given config file.

    The resolved global paths (`data_root`, `analysis_root`, `figure_root`)
    are stashed on the underlying Flask app at `server.config["YUXIN_MEA"]`
    so page modules can read them from `flask.current_app` inside callbacks.

    Tolerates a nonexistent `config_path`: launches in "config-only" mode
    with empty globals so the Settings page can create the file.
    `config_exists` on the stash distinguishes the two states so data
    pages can render a "no config yet" banner.
    """
    config_path = Path(config_path)
    cm = ConfigManager()
    config_exists = config_path.exists()
    if config_exists:
        cm.load(config_path)
    analysis_root = _resolve_optional_path(cm.get_global("analysis_root"))
    data_root = _resolve_optional_path(cm.get_global("data_root"))
    figure_root = _resolve_optional_path(cm.get_global("figure_root"))

    apply_default_theme()

    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="pages",
        title="yuxin_mea dashboard",
        suppress_callback_exceptions=True,
        index_string=_INDEX_STRING,
    )
    app.server.config["YUXIN_MEA"] = {
        "config_path": config_path,
        "config_exists": config_exists,
        "analysis_root": analysis_root,
        "data_root": data_root,
        "figure_root": figure_root,
    }
    app.layout = build_layout()
    return app


def _resolve_optional_path(value: object) -> Path | None:
    """Return a `Path` if `value` is a non-empty string, else `None`.

    A missing global is not fatal at app-build time — the pages render an
    empty state. This keeps the dashboard usable for a freshly-initialized
    config that hasn't been filled in yet.
    """
    if not value:
        return None
    return Path(str(value))
