"""Smoke test for the Activity scan Dash page.

Read-only page registered at ``/activity-scan``; real callback behaviour is
exercised in the browser. This guards import + registration so a typo in a
callback signature can't slip past CI. ``build_app`` is invoked *before* the
page module is imported, because ``dash.register_page`` requires an instantiated
app (importing the module first raises PageError).
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


def test_activity_scan_page_registered_at_expected_path():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = tmp_path / "pipeline_config.json"
        _write_config(cfg, tmp_path)

        build_app(cfg)                       # must precede the page import
        from yuxin_mea.dashboard.pages import activity_scan  # noqa: F401

        paths = {p["path"] for p in dash.page_registry.values()}
        assert "/activity-scan" in paths
        names = {p["name"] for p in dash.page_registry.values()}
        assert "Activity scan" in names
