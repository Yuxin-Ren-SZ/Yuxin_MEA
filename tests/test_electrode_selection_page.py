"""Smoke tests for the Electrode selection Dash page.

Read-only page at ``/electrode-selection``; callback behaviour is exercised in
the browser. These guard import + registration so a typo in a callback signature
cannot slip past CI. ``build_app`` runs *before* the page import because
``dash.register_page`` requires an instantiated app.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import dash

from yuxin_mea.dashboard import build_app


def _write_config(path: Path, analysis_root: Path) -> None:
    path.write_text(json.dumps({
        "global": {"data_root": str(analysis_root / "raw"),
                   "analysis_root": str(analysis_root),
                   "figure_root": str(analysis_root / "figures")},
        "tasks": {},
    }))


def test_page_registered_at_expected_path():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = tmp_path / "pipeline_config.json"
        _write_config(cfg, tmp_path)

        build_app(cfg)                       # must precede the page import
        from yuxin_mea.dashboard.pages import electrode_selection  # noqa: F401

        by_path = {p["path"]: p for p in dash.page_registry.values()}
        assert "/electrode-selection" in by_path
        assert by_path["/electrode-selection"]["name"] == "Electrode selection"


def test_page_has_a_nav_entry():
    """Without this the rail renders the page with no section and no glyph."""
    from yuxin_mea.dashboard.components.layout import _NAV_META, _SECTION_ORDER

    section, glyph = _NAV_META["Electrode selection"]
    assert section in _SECTION_ORDER
    assert glyph


def test_grid_callback_is_graceful_without_a_selection():
    """The empty state must render, not raise, before a chip is picked."""
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = tmp_path / "pipeline_config.json"
        _write_config(cfg, tmp_path)
        app = build_app(cfg)
        from yuxin_mea.dashboard.pages import electrode_selection as page

        with app.server.test_request_context():
            grid, legend, context = page._render_grid(None, None, ["on"], "")
        assert context is None
        assert legend == ""


def test_unified_toggle_is_wired_into_the_grid_callback():
    """The checkbox must reach _render_grid, or flipping it would do nothing."""
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = tmp_path / "pipeline_config.json"
        _write_config(cfg, tmp_path)
        build_app(cfg)
        from yuxin_mea.dashboard.pages import electrode_selection as page

        import inspect
        assert "unified" in inspect.signature(page._render_grid).parameters
