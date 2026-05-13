"""Smoke test for the Phase 5 plate viewer Dash page.

The page is read-only with respect to caches and registers at
`/plate-viewer`. Real callback behavior is exercised manually in the
browser; this file just guards import correctness + registration so a
typo in a callback signature can't slip past CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import dash

from yuxin_mea.dashboard import build_app


def _write_minimal_config(path: Path, analysis_root: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "global": {
                    "data_root": str(analysis_root / "raw"),
                    "analysis_root": str(analysis_root),
                    "figure_root": str(analysis_root / "figures"),
                },
                "tasks": {},
            }
        )
    )


def test_plate_viewer_page_imports():
    """A typo in the page module fails this test, not a user opening the browser."""
    from yuxin_mea.dashboard.pages import plate_viewer  # noqa: F401


def test_plate_viewer_page_registered_at_expected_path():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = tmp_path / "pipeline_config.json"
        _write_minimal_config(cfg, tmp_path)

        build_app(cfg)
        paths = {p["path"] for p in dash.page_registry.values()}
        assert "/plate-viewer" in paths


def test_plate_viewer_page_orders_before_burst_diagnostic():
    """Plate viewer is more general (read existing outputs) → comes before burst-diagnostic."""
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = tmp_path / "pipeline_config.json"
        _write_minimal_config(cfg, tmp_path)

        build_app(cfg)
        registry = dash.page_registry
        plate_order = next(
            p["order"] for p in registry.values() if p["path"] == "/plate-viewer"
        )
        burst_order = next(
            p["order"] for p in registry.values() if p["path"] == "/burst-diagnostic"
        )
        assert plate_order < burst_order
