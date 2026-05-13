"""Dash app factory: `build_app(config_path)` returns a configured Dash."""

from __future__ import annotations

from pathlib import Path

from dash import Dash

from yuxin_mea.config import ConfigManager

from .components.layout import build_layout


def build_app(config_path: Path) -> Dash:
    """Build a multipage Dash app pinned to the given config file.

    The resolved global paths (`data_root`, `analysis_root`) are stashed on
    the underlying Flask app at `server.config["YUXIN_MEA"]` so page modules
    can read them from `flask.current_app` inside callbacks.
    """
    cm = ConfigManager()
    cm.load(config_path)
    analysis_root = _resolve_optional_path(cm.get_global("analysis_root"))
    data_root = _resolve_optional_path(cm.get_global("data_root"))

    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="pages",
        title="yuxin_mea dashboard",
        suppress_callback_exceptions=True,
    )
    app.server.config["YUXIN_MEA"] = {
        "config_path": Path(config_path),
        "analysis_root": analysis_root,
        "data_root": data_root,
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
