"""Smoke tests for the Activity scan Dash page.

Read-only page registered at ``/activity-scan``; most callback behaviour is
exercised in the browser. These guard import + registration so a typo in a
callback signature can't slip past CI, plus one direct invocation of the
recording-populate callback (its filter wiring broke silently once — a duplicate
``scan_types`` keyword raised on every request, leaving the page empty).
``build_app`` is invoked *before* the page module is imported, because
``dash.register_page`` requires an instantiated app (importing the module first
raises PageError).
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


def _recording(cache_key: str, scan_type: str) -> dict:
    sample, date, plate, _scan, run = cache_key.split("/")
    return {
        "cache_key": cache_key, "sample_id": sample, "date": date,
        "plate_id": plate, "scan_type": scan_type, "run_id": run,
        "groups": ["ctrl"], "wells": [], "n_wells": 0,
    }


def test_populate_recordings_lists_only_activity_scans(monkeypatch):
    """The populate callback runs and restricts the dropdown to ActivityScans.

    Calls the undecorated function (``@callback`` returns it unchanged) inside a
    Flask app context, with the two data sources stubbed so no cache or output
    tree is needed.
    """
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = tmp_path / "pipeline_config.json"
        _write_config(cfg, tmp_path)

        app = build_app(cfg)
        from yuxin_mea.analysis import activity_scan_inspector as vi
        from yuxin_mea.dashboard import data as dash_data
        from yuxin_mea.dashboard.pages import activity_scan

        act_key = "S1/260101/P1/ActivityScan/000"
        net_key = "S1/260101/P1/Network/000"
        recordings = [_recording(act_key, "ActivityScan"),
                      _recording(net_key, "Network")]
        # Both recordings have output on disk; only the ActivityScan may show.
        monkeypatch.setattr(vi, "recordings_with_activity_scan",
                            lambda root, override=None: [act_key, net_key])
        monkeypatch.setattr(dash_data, "load_recordings_detail",
                            lambda root: (recordings, {}))

        with app.server.app_context():
            out = activity_scan._populate_recordings(0, None, None, None, None,
                                                     "", None)

        _banner, options, value, status, sample_opts, _dmin, _dmax, group_opts = out
        assert [o["value"] for o in options] == [act_key]
        assert value == act_key
        assert "1 of 2 computed recording(s) match." == status
        assert [o["value"] for o in sample_opts] == ["S1"]
        assert [o["value"] for o in group_opts] == ["ctrl"]


def _layout_ids(component) -> set[str]:
    """Every string ``id`` in a Dash component tree."""
    ids: set[str] = set()
    cid = getattr(component, "id", None)
    if isinstance(cid, str):
        ids.add(cid)
    children = getattr(component, "children", None)
    if children is None:
        return ids
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            ids |= _layout_ids(child)
    return ids


def _page_module():
    """Build the app (required before page import) and return the page module."""
    tmp = TemporaryDirectory()
    tmp_path = Path(tmp.name)
    cfg = tmp_path / "pipeline_config.json"
    _write_config(cfg, tmp_path)
    build_app(cfg)
    from yuxin_mea.dashboard.pages import activity_scan

    return activity_scan, tmp


def test_layout_exposes_plate_and_well_nav_ids():
    """The keyboard shortcut clicks these ids, so they must exist in the layout."""
    activity_scan, tmp = _page_module()
    try:
        ids = _layout_ids(activity_scan.layout)
        assert {"as-prev-plate", "as-next-plate", "as-prev-well", "as-next-well",
                "as-keydummy", "as-modal"} <= ids
    finally:
        tmp.cleanup()


def test_keydown_js_clicks_real_ids_and_guards_arrow_key_controls():
    """Every id the listener clicks exists, and the arrow-owning controls are skipped.

    The JS itself is not executed here — this pins the contract between the
    listener and the layout (a renamed button id, or a dropped guard that would
    let arrow keys hijack the threshold slider, fails the test).
    """
    activity_scan, tmp = _page_module()
    try:
        js = activity_scan._KEYDOWN_JS
        ids = _layout_ids(activity_scan.layout)
        for clicked in ("as-prev-plate", "as-next-plate",
                        "as-prev-well", "as-next-well", "as-modal"):
            assert f"'{clicked}'" in js
            assert clicked in ids
        # dcc.Slider handle is role="slider"; dcc.Input/Dropdown the other two.
        for guard in ("role=slider", "role=combobox", "TEXTAREA", "shiftKey"):
            assert guard in js
    finally:
        tmp.cleanup()


class _FakeCtx:
    """Minimal stand-in for ``dash.ctx`` (only what `_on_plate_nav` reads)."""

    def __init__(self, trigger_id: str):
        self.triggered = [{"prop_id": f"{trigger_id}.n_clicks", "value": 1}]
        self.triggered_id = trigger_id


def test_plate_nav_steps_recording_and_closes_modal(monkeypatch):
    activity_scan, tmp = _page_module()
    try:
        options = [{"label": "A", "value": "A"}, {"label": "B", "value": "B"}]
        monkeypatch.setattr(activity_scan, "ctx", _FakeCtx("as-next-plate"))
        value, style, active = activity_scan._on_plate_nav(0, 1, "A", options)
        assert value == "B"
        assert style == activity_scan._MODAL_HIDDEN
        assert active is None

        monkeypatch.setattr(activity_scan, "ctx", _FakeCtx("as-prev-plate"))
        value, _style, _active = activity_scan._on_plate_nav(1, 0, "B", options)
        assert value == "A"
    finally:
        tmp.cleanup()


def test_plate_nav_is_a_no_op_at_the_ends(monkeypatch):
    activity_scan, tmp = _page_module()
    try:
        options = [{"label": "A", "value": "A"}, {"label": "B", "value": "B"}]
        nu = dash.no_update
        monkeypatch.setattr(activity_scan, "ctx", _FakeCtx("as-next-plate"))
        assert activity_scan._on_plate_nav(0, 1, "B", options) == (nu, nu, nu)
        monkeypatch.setattr(activity_scan, "ctx", _FakeCtx("as-prev-plate"))
        assert activity_scan._on_plate_nav(1, 0, "A", options) == (nu, nu, nu)
        # No options at all (nothing computed yet) — must not raise.
        assert activity_scan._on_plate_nav(1, 0, None, []) == (nu, nu, nu)
    finally:
        tmp.cleanup()
