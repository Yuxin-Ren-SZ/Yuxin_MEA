"""The layout contract: a panel's grid cell and its position in the PDF agree.

The composer's whole premise is that dragging a panel on a 12-column canvas moves
it to the corresponding place in the exported figure. That holds only because
each panel is pinned to a rectangle computed from ``(x, y, w, h)`` alone. These
tests pin that arithmetic, so a future "just use tight_layout" or a shared
GridSpec with wspace cannot quietly break it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report import panels as P  # noqa: E402
from scripts.report.composer import compose as C  # noqa: E402
from scripts.report.composer import spec as SP  # noqa: E402

TOL = 1e-9


def _tidy(n_wells: int = 12) -> pd.DataFrame:
    rows = []
    for w in range(n_wells):
        for tau in (-3, 3):
            rows.append({
                "well_uid": f"CHIP|w{w}", "sample_id": f"CHIP{w % 3}",
                "well_name": f"W{w}", "canonical_group":
                    "Control" if w % 2 else "IVH_Late",
                "tau": tau, "DIV": 14 + tau, "recording_key": "K", "rec_name": "r",
                "well_id": f"well{w:03d}",
                "nb_rate": 0.5 + 0.1 * w, "median_firing_rate": 1.0 + 0.05 * w,
                "n_curated": 10 + w, "nb_duration_mean": 0.3,
                "nb_spikes_per_burst_mean": 20.0, "nb_ibi_mean": 5.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def ctx(tmp_path):
    return P.RenderContext(_tidy(), tmp_path, tmp_path,
                           groups=["Control", "IVH_Late"])


def _comp(panels, rows=8, cols=12, gutter=0.0):
    c = SP.Composition(name="t", grid_cols=cols, gutter_x=gutter, gutter_y=gutter)
    c.figures = [SP.FigureSpec(id="F", title="", show_suptitle=False, rows=rows,
                               panels=panels)]
    return c


def _placement(uid, x, y, w, h, panel="forest.burst"):
    return SP.PanelPlacement(uid=uid, panel=panel, x=x, y=y, w=w, h=h)


# --------------------------------------------------------------------------- #
# Cell arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("x,y,w,h", [
    (0, 0, 6, 4), (6, 0, 6, 4), (0, 4, 4, 4), (4, 4, 8, 4), (0, 0, 12, 8),
])
def test_cell_rect_is_the_exact_grid_fraction(x, y, w, h):
    """With no gutter and no suptitle margin, cells tile the drawing area exactly."""
    comp = _comp([_placement("p", x, y, w, h)])
    f = comp.figures[0]
    m = C.MARGIN_NO_TITLE
    left, right, bottom, top = C.cell_rect(f.panels[0], f, comp, m)

    ax0, ax1 = m["left"], 1 - m["right"]
    ay0, ay1 = m["bottom"], 1 - m["top"]
    span_x, span_y = ax1 - ax0, ay1 - ay0
    assert left == pytest.approx(ax0 + x / 12 * span_x, abs=TOL)
    assert right == pytest.approx(ax0 + (x + w) / 12 * span_x, abs=TOL)
    assert top == pytest.approx(ay1 - y / 8 * span_y, abs=TOL)
    assert bottom == pytest.approx(ay1 - (y + h) / 8 * span_y, abs=TOL)


def test_grid_y_counts_downward_from_the_top():
    """Screen convention: y=0 is the top row, which is the *highest* in figure coords."""
    comp = _comp([_placement("top", 0, 0, 12, 4), _placement("bot", 0, 4, 12, 4)])
    f = comp.figures[0]
    top_rect = C.cell_rect(f.panels[0], f, comp, C.MARGIN_NO_TITLE)
    bot_rect = C.cell_rect(f.panels[1], f, comp, C.MARGIN_NO_TITLE)
    assert top_rect[2] > bot_rect[3] - TOL          # top.bottom >= bot.top


def test_adjacent_cells_share_an_edge_when_the_gutter_is_zero():
    comp = _comp([_placement("l", 0, 0, 6, 8), _placement("r", 6, 0, 6, 8)])
    f = comp.figures[0]
    lr = C.cell_rect(f.panels[0], f, comp, C.MARGIN_NO_TITLE)
    rr = C.cell_rect(f.panels[1], f, comp, C.MARGIN_NO_TITLE)
    assert lr[1] == pytest.approx(rr[0], abs=TOL)


def test_gutter_insets_symmetrically():
    plain = _comp([_placement("p", 0, 0, 6, 4)], gutter=0.0)
    inset = _comp([_placement("p", 0, 0, 6, 4)], gutter=0.02)
    a = C.cell_rect(plain.figures[0].panels[0], plain.figures[0], plain, C.MARGIN_NO_TITLE)
    b = C.cell_rect(inset.figures[0].panels[0], inset.figures[0], inset, C.MARGIN_NO_TITLE)
    assert b[0] - a[0] == pytest.approx(0.02, abs=TOL)
    assert a[1] - b[1] == pytest.approx(0.02, abs=TOL)


def test_an_oversized_gutter_cannot_invert_a_cell():
    comp = _comp([_placement("p", 0, 0, 1, 1)], rows=8, gutter=0.9)
    left, right, bottom, top = C.cell_rect(
        comp.figures[0].panels[0], comp.figures[0], comp, C.MARGIN_NO_TITLE)
    assert right > left and top > bottom


# --------------------------------------------------------------------------- #
# What actually lands on the figure
# --------------------------------------------------------------------------- #
def test_rendered_axes_land_on_their_cells(ctx):
    cells = [("a", 0, 0, 6, 4), ("b", 6, 0, 6, 4), ("c", 0, 4, 12, 4)]
    comp = _comp([_placement(*c) for c in cells])
    f = comp.figures[0]
    fig, records = C.build_figure(f, comp, ctx, draw_labels=False)

    assert [r["status"] for r in records] == ["ok"] * 3
    by_uid = {p.uid: p for p in f.panels}
    for ax, rec in zip(fig.axes, records):
        want = C.cell_rect(by_uid[rec["uid"]], f, comp, C.MARGIN_NO_TITLE)
        pos = ax.get_position()
        assert (pos.x0, pos.x1, pos.y0, pos.y1) == pytest.approx(want, abs=1e-6)


def test_figure_size_follows_rows_and_width(ctx):
    comp = _comp([_placement("a", 0, 0, 12, 6)], rows=6)
    comp.row_height_in = 0.5
    comp.figures[0].width_in = 7.5
    fig, _ = C.build_figure(comp.figures[0], comp, ctx)
    assert tuple(fig.get_size_inches()) == pytest.approx((7.5, 3.0))


def test_a_failing_panel_becomes_a_note_not_an_exception(ctx):
    comp = _comp([_placement("a", 0, 0, 12, 8, panel="does.not.exist")])
    fig, records = C.build_figure(comp.figures[0], comp, ctx)
    assert records[0]["status"] == "unknown"
    assert len(fig.axes) == 1                    # the note still occupies the cell


def test_missing_data_is_reported_rather_than_raised(ctx):
    """A criticality panel against a tidy table with no criticality columns."""
    comp = _comp([_placement("a", 0, 0, 12, 8, panel="forest.criticality")])
    _, records = C.build_figure(comp.figures[0], comp, ctx)
    assert records[0]["status"] == "no-data"
    assert records[0]["warning"]


def test_labels_are_drawn_once_per_panel(ctx):
    comp = _comp([_placement("a", 0, 0, 6, 8), _placement("b", 6, 0, 6, 8)])
    fig, _ = C.build_figure(comp.figures[0], comp, ctx, draw_labels=True)
    texts = [t.get_text() for t in fig.texts]
    assert texts == ["A", "B"]


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def test_export_writes_pdf_pages_spec_and_manifest(ctx, tmp_path):
    comp = _comp([_placement("a", 0, 0, 12, 8)])
    comp.figures.append(SP.FigureSpec(
        id="G", rows=4, panels=[_placement("b", 0, 0, 12, 4)]))
    manifest = C.export(comp, ctx, tmp_path, formats=("pdf", "png"))

    out = Path(manifest["out_dir"])
    assert (out / "figures.pdf").exists()
    assert (out / "F.pdf").exists() and (out / "G.png").exists()
    assert (out / "figure_spec.json").exists()
    assert len(manifest["figures"]) == 2

    # page count, when a PDF reader happens to be installed
    pypdf = pytest.importorskip("pypdf", reason="pypdf not installed")
    assert len(pypdf.PdfReader(str(out / "figures.pdf")).pages) == 2


def test_exported_spec_reloads_to_an_identical_composition(ctx, tmp_path):
    comp = _comp([_placement("a", 0, 0, 6, 4), _placement("b", 6, 0, 6, 4)])
    comp.name = "resume_me"
    manifest = C.export(comp, ctx, tmp_path, formats=("png",))
    back = SP.Composition.load(Path(manifest["spec"]))
    assert back.to_json() == comp.to_json()


def test_export_is_deterministic_for_the_same_spec(ctx, tmp_path):
    comp = _comp([_placement("a", 0, 0, 12, 8)])
    a = C.export(comp, ctx, tmp_path / "one", formats=("png",))
    b = C.export(comp, ctx, tmp_path / "two", formats=("png",))
    strip = lambda m: [f["panels"] for f in m["figures"]]  # noqa: E731
    assert strip(a) == strip(b)
    assert (Path(a["out_dir"]) / "F.png").read_bytes() == \
           (Path(b["out_dir"]) / "F.png").read_bytes()
