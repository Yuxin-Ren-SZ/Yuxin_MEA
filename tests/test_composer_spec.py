"""Spec round-trips, validation, and the split/merge layout operations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report.composer import spec as SP  # noqa: E402


def _comp() -> SP.Composition:
    return SP.Composition(name="t", figures=[
        SP.FigureSpec(id="F1", title="one", rows=8, panels=[
            SP.PanelPlacement(uid="a", panel="forest.burst", x=0, y=0, w=6, h=4),
            SP.PanelPlacement(uid="b", panel="forest.node", x=6, y=0, w=6, h=4),
            SP.PanelPlacement(uid="c", panel="ml.umap", x=0, y=4, w=12, h=4),
        ]),
        SP.FigureSpec(id="F2", title="two", rows=4, panels=[
            SP.PanelPlacement(uid="d", panel="method.compare", x=0, y=0, w=12, h=4),
        ]),
    ])


def test_json_round_trip_is_lossless(tmp_path):
    c = _comp()
    p = c.save(tmp_path / "s.json")
    back = SP.Composition.load(p)
    assert back.to_json() == c.to_json()


def test_load_fills_in_missing_globals(tmp_path):
    """A hand-edited spec that omits optional globals still loads."""
    d = _comp().to_json()
    d["globals"] = {"groups": ["Control", "IVH_Late"]}
    p = tmp_path / "sparse.json"
    p.write_text(json.dumps(d))
    back = SP.Composition.load(p)
    assert back.groups == ["Control", "IVH_Late"]
    assert back.grid_cols == SP.DEFAULT_GRID_COLS
    assert back.gutter_x_in > 0 and back.gutter_y_in > 0
    assert back.dpi == 300


def test_future_version_is_refused(tmp_path):
    d = _comp().to_json()
    d["version"] = SP.SPEC_VERSION + 1
    p = tmp_path / "future.json"
    p.write_text(json.dumps(d))
    with pytest.raises(SP.SpecError):
        SP.Composition.load(p)


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: setattr(c.figures[0].panels[0], "w", 20), "past the grid"),
    (lambda c: setattr(c.figures[0].panels[0], "h", 99), "past the figure"),
    (lambda c: setattr(c.figures[0].panels[0], "x", -1), "must be >= 0"),
    (lambda c: setattr(c.figures[0].panels[1], "uid", "a"), "duplicate panel uid"),
    (lambda c: setattr(c.figures[1], "id", "F1"), "duplicate figure id"),
])
def test_validate_rejects_structural_errors(mutate, msg):
    c = _comp()
    mutate(c)
    with pytest.raises(SP.SpecError, match=msg):
        c.validate()


def test_validate_warns_but_does_not_raise_on_overlap_and_unknown_panel():
    c = _comp()
    c.figures[0].panels[1].x = 0            # now overlaps panel "a"
    problems = c.validate(known_panels={"forest.burst", "ml.umap"})
    assert any("overlap" in m for m in problems)
    assert any("unknown panel" in m for m in problems)


def test_auto_labels_follow_reading_order():
    f = _comp().figures[0]
    labels = f.auto_labels()
    assert [labels[p.uid] for p in f.ordered_panels()] == ["A", "B", "C"]


def test_explicit_label_overrides_auto():
    f = _comp().figures[0]
    f.panels[0].label = "Z"
    assert f.auto_labels()["a"] == "Z"


def test_split_moves_panels_into_a_new_figure_placed_after_the_source():
    c = _comp()
    dst = SP.split_figure(c, "F1", ["c"])
    src = c.figure("F1")
    assert [p.uid for p in src.panels] == ["a", "b"]
    assert [p.uid for p in dst.panels] == ["c"]
    assert c.figures.index(dst) == c.figures.index(src) + 1
    assert c.validate() == []                 # both figures still structurally sound


def test_split_refuses_to_empty_the_source():
    c = _comp()
    with pytest.raises(SP.SpecError, match="empty the source"):
        SP.split_figure(c, "F1", ["a", "b", "c"])


def test_merge_concatenates_and_reflows_without_collisions():
    c = _comp()
    head = SP.merge_figures(c, ["F1", "F2"], new_title="merged")
    assert len(c.figures) == 1
    assert head.title == "merged"
    assert {p.uid for p in head.panels} == {"a", "b", "c", "d"}
    assert c.validate() == []                 # reflow removed every overlap


def test_merge_keeps_the_source_figures_panel_order():
    """Reflowing by position would interleave the two figures and scramble labels."""
    c = _comp()
    head = SP.merge_figures(c, ["F1", "F2"])
    labels = head.auto_labels()
    assert [labels[u] for u in ("a", "b", "c", "d")] == ["A", "B", "C", "D"]


def test_reflow_by_list_order_ignores_current_positions():
    f = SP.FigureSpec(id="X", rows=1, panels=[
        SP.PanelPlacement(uid="second", panel="p", x=6, y=0, w=6, h=2),
        SP.PanelPlacement(uid="first", panel="p", x=0, y=0, w=6, h=2)])
    SP.reflow(f, order="list")
    assert [(p.uid, p.x) for p in f.panels] == [("second", 0), ("first", 6)]


def test_reflow_wraps_at_the_grid_width():
    f = SP.FigureSpec(id="X", rows=1, panels=[
        SP.PanelPlacement(uid=str(i), panel="p", x=0, y=0, w=5, h=2)
        for i in range(3)])
    SP.reflow(f, grid_cols=12)
    assert [(p.x, p.y) for p in f.ordered_panels()] == [(0, 0), (5, 0), (0, 2)]
    assert f.rows == 4


def test_fingerprint_delta_reports_only_real_changes():
    base = {"tidy_sha256": "aa", "tidy_rows": 10, "tidy_cols": 3,
            "analysis_root": "/a"}
    assert SP.fingerprint_delta(base, dict(base)) == []
    changed = {**base, "tidy_rows": 11, "tidy_sha256": "bb"}
    msgs = SP.fingerprint_delta(base, changed)
    assert any("content changed" in m for m in msgs)
    assert any("10 -> 11" in m for m in msgs)


def test_safe_name_strips_path_separators():
    assert SP.safe_name("../../etc/passwd") == "etc_passwd"
    assert SP.safe_name("  ") == "composition"


def test_default_composition_is_valid_and_covers_the_report():
    from scripts.report import panels as P

    c = SP.default_composition()
    assert c.validate(known_panels=set(P.load_registry())) == []
    assert {f.id for f in c.figures} >= {"F4", "F5", "F6", "F7", "F7b", "F8"}
