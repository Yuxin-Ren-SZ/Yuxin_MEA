"""Lay the rendered panels out as a portrait US-Letter PowerPoint deck.

    python -m scripts.report.panel_pptx
    python -m scripts.report.panel_pptx --figures f4 f7 --out my_deck.pptx

One slide per figure, on an 8.5 x 11 in portrait canvas so the page matches the
manuscript's. The panels are placed as images; **everything else is a native,
editable PowerPoint text box** — the figure number, each panel's bold letter, and
the caption. Nothing is baked into the images, which is the point of the panel
split: the deck can be re-typed and re-laid-out without re-rendering anything.

Captions are written the way a journal expects them::

    Figure 4. HD-MEA recording schema and network maturation.
    (A) Spike raster of one Control well ... (B) ... (C) ...

so a reader can map every sentence to the panel it describes by its letter. Text
repeated across panels — the difference-in-differences method paragraph, say — is
stated once at the end as "In (B)-(F): …" instead of five times.

A dense figure whose caption would squeeze the panels below print legibility gets
the whole page, with its caption on the facing slide, the way a journal handles a
full-page figure.

Panel order, and therefore lettering, comes from ``assemble_figure.LAYOUTS``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

from . import panel_data as D
from .assemble_figure import FIGURE_TITLES, LAYOUTS, _letters
from .panel_base import captions_dir, panels_dir
from .report_style import report_dir

logger = logging.getLogger("report.pptx")

# --- page geometry (inches), US Letter portrait ------------------------------ #
PAGE_W, PAGE_H = 8.5, 11.0
MARGIN = 0.55
CONTENT_W = PAGE_W - 2 * MARGIN

TITLE_TOP, TITLE_H, TITLE_FS = 0.42, 0.30, 12.0
#: Gutter above each panel row for its letter label.
LETTER_GUTTER, LETTER_FS = 0.22, 11.0
PANEL_GAP = 0.10                  # between panels in a row, and between rows
CAP_GAP = 0.16                    # between the last row and the caption

#: Caption type is shrunk to fit before it is allowed to overflow to a second
#: slide; below the floor it stops being readable in print.
CAP_FS_MAX, CAP_FS_MIN, CAP_FS_STEP = 9.0, 7.0, 0.5
#: Below this scale the panels stop being legible at print size. Rather than
#: shrink past it to make room for the caption, the figure takes the whole page
#: and its caption moves to the facing page — what a journal does with a large
#: figure, and the reason the caption is a separate text box in the first place.
MIN_SCALE = 0.82


def _aspect(path: Path) -> float:
    from PIL import Image

    with Image.open(path) as im:
        return im.width / im.height


#: Fraction of the content width a row may occupy, by how many panels it holds.
#: A lone panel stretched to the full page is tall, and on a portrait page height
#: is the scarce resource — every row that overreaches shrinks *all* the others,
#: because one uniform scale keeps the panels comparable.
_ROW_WIDTH_FRAC = {1: 0.72, 2: 0.94}


def _row_layout(rows: list[list[Path]], width: float):
    """Scale each row; return ``[(row, height)]`` and the stacked height.

    Panels within a row share a height, and the row fills its allotted width, so
    rows align on both edges — the arrangement a reader expects from a journal
    figure.
    """
    out, total = [], 0.0
    for row in rows:
        aspects = [_aspect(p) for p in row]
        avail = width * _ROW_WIDTH_FRAC.get(len(row), 1.0)
        avail -= PANEL_GAP * (len(row) - 1)
        h = avail / sum(aspects)
        out.append((list(zip(row, aspects)), h))
        total += LETTER_GUTTER + h + PANEL_GAP
    return out, total


#: Average glyph advance as a fraction of the point size, calibrated against a
#: rendered slide (Calibri body text). Guessing low here is what pushed the last
#: caption line off the bottom of the page.
_CHAR_W = 0.68


def _wrapped_lines(text: str, width_in: float, fs: float) -> int:
    """Rough line count for a text box — enough to choose a font size."""
    chars = max(20, int(width_in * 72 / (fs * _CHAR_W)))
    n = 0
    for para in text.split("\n"):
        n += max(1, -(-len(para) // chars))
    return n


def _caption_height(text: str, width_in: float, fs: float) -> float:
    """Height the caption needs, with headroom — a clipped last line is silent
    data loss, so err on the side of asking for too much."""
    return _wrapped_lines(text, width_in, fs) * fs * 1.22 / 72.0 * 1.08


def _textbox(slide, left, top, width, height, text, *, size, bold=False,
             anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width),
                                   Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    para = tf.paragraphs[0]
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    return box


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


#: A shared block has to be worth factoring out — below this it reads better
#: inline than as a cross-reference.
_MIN_SHARED = 150


def _common_suffix(a: str, b: str) -> str:
    """Longest shared tail of two captions, trimmed to a sentence boundary."""
    i = 0
    while i < min(len(a), len(b)) and a[-1 - i] == b[-1 - i]:
        i += 1
    tail = a[len(a) - i:]
    cut = tail.find(". ")            # start the shared block at a full sentence
    return tail[cut + 2:] if cut >= 0 else tail


def _factor_shared(caps: dict) -> tuple[dict, list[tuple[list[str], str]]]:
    """Pull text repeated across panels out into once-stated shared blocks.

    Most DiD panels end with the same paragraph describing the estimator and the
    significance rule. Repeating it five times is how the caption came to be
    longer than the figure; a journal states it once and says which panels it
    applies to. Blocks are found by shared suffix rather than by name, so this
    also catches the three cross-correlogram panels, whose captions differ only
    in the group named near the start.
    """
    caps = dict(caps)
    shared: list[tuple[list[str], str]] = []
    for _round in range(3):
        best = None                                   # (saving, suffix, letters)
        letters = list(caps)
        for i, a in enumerate(letters):
            for b in letters[i + 1:]:
                suf = _common_suffix(caps[a], caps[b])
                if len(suf) < _MIN_SHARED:
                    continue
                group = [c for c in letters if caps[c].endswith(suf)]
                saving = (len(group) - 1) * len(suf)
                if best is None or saving > best[0]:
                    best = (saving, suf, group)
        if best is None:
            break
        _saving, suf, group = best
        for c in group:
            caps[c] = caps[c][: -len(suf)].strip()
        shared.append((group, suf))
    return caps, shared


def _range_label(letters: list[str]) -> str:
    """``(B)-(F)`` when the panels are contiguous, else ``(B), (D), (F)``."""
    idx = [ord(c) for c in letters if len(c) == 1]
    if len(idx) == len(letters) and idx == list(range(idx[0], idx[0] + len(idx))):
        return f"({letters[0]})-({letters[-1]})" if len(letters) > 2 else \
            ", ".join(f"({c})" for c in letters)
    return ", ".join(f"({c})" for c in letters)


def figure_caption(fig_id: str, names: list[str], cap_dir: Path) -> str:
    """``Figure N. Title. (A) … (B) …`` — one string, panel letters inline."""
    num = fig_id.upper().lstrip("F")
    head = f"Figure {num}. {FIGURE_TITLES.get(fig_id, '')}".rstrip(". ") + "."
    letters = _letters(len(names))
    caps = {}
    for letter, name in zip(letters, names):
        p = cap_dir / f"{name}.txt"
        if p.exists():
            caps[letter] = p.read_text().strip()
    caps, shared = _factor_shared(caps)
    body = " ".join(f"({c}) {caps[c]}" for c in letters if c in caps)
    tail = " ".join(f"In {_range_label(g)}: {txt}" for g, txt in shared)
    return " ".join(x for x in (head, body, tail) if x)


def build_slide(prs, fig_id: str, src: Path, cap_dir: Path) -> int:
    """Add the slides for one figure; returns how many were used."""
    rows_spec = LAYOUTS.get(fig_id)
    if not rows_spec:
        return 0
    rows, names = [], []
    for row in rows_spec:
        present = [(n, src / f"{n}.png") for n in row]
        present = [(n, p) for n, p in present if p.exists()]
        if present:
            rows.append([p for _n, p in present])
            names.extend(n for n, _p in present)
    if not rows:
        logger.warning("[%s] no rendered panels in %s", fig_id, src)
        return 0

    caption = figure_caption(fig_id, names, cap_dir)
    panels_top = TITLE_TOP + TITLE_H + 0.06
    page_bottom = PAGE_H - MARGIN
    max_panel_h = page_bottom - panels_top

    laid, natural_h = _row_layout(rows, CONTENT_W)

    # Largest readable caption size that still leaves the panels a legible scale.
    fs, cap_h, scale = CAP_FS_MAX, 0.0, 0.0
    while True:
        cap_h = _caption_height(caption, CONTENT_W, fs)
        scale = (max_panel_h - CAP_GAP - cap_h) / natural_h if natural_h else 1.0
        if scale >= MIN_SCALE or fs - CAP_FS_STEP < CAP_FS_MIN:
            break
        fs -= CAP_FS_STEP
    # Still too cramped: give the panels the page and the caption its own.
    caption_on_own_page = scale < MIN_SCALE
    if caption_on_own_page:
        scale = max_panel_h / natural_h if natural_h else 1.0
    scale = min(scale, 1.0)

    slide = _blank(prs)
    _textbox(slide, MARGIN, TITLE_TOP, CONTENT_W, TITLE_H,
             f"Figure {fig_id.upper().lstrip('F')}", size=TITLE_FS, bold=True)

    letters = _letters(len(names))
    k, y = 0, panels_top
    for row, h in laid:
        h *= scale
        y += LETTER_GUTTER * scale
        row_w = sum(h * a for _p, a in row) + PANEL_GAP * scale * (len(row) - 1)
        x = MARGIN + (CONTENT_W - row_w) / 2      # centre short rows on the page
        for path, aspect in row:
            w = h * aspect
            _textbox(slide, x, y - LETTER_GUTTER * scale, 0.3,
                     LETTER_GUTTER * scale, letters[k],
                     size=LETTER_FS * min(1.0, max(scale, 0.75)), bold=True,
                     anchor=MSO_ANCHOR.BOTTOM)
            slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                     width=Inches(w), height=Inches(h))
            x += w + PANEL_GAP * scale
            k += 1
        y += h + PANEL_GAP * scale

    if not caption_on_own_page:
        _textbox(slide, MARGIN, y + CAP_GAP - PANEL_GAP * scale, CONTENT_W,
                 page_bottom - y, caption, size=fs)
        return 1

    cap_slide = _blank(prs)
    _textbox(cap_slide, MARGIN, TITLE_TOP, CONTENT_W, TITLE_H,
             f"Figure {fig_id.upper().lstrip('F')} — caption",
             size=TITLE_FS, bold=True)
    _textbox(cap_slide, MARGIN, TITLE_TOP + TITLE_H + 0.12, CONTENT_W,
             page_bottom - TITLE_TOP - TITLE_H - 0.12, caption, size=CAP_FS_MAX)
    return 2


def build(config=None, figures=None, out: Path | None = None) -> Path:
    ctx = D.context(config)
    src = panels_dir(ctx.figure_root)
    caps = captions_dir(ctx.figure_root)

    prs = Presentation()
    prs.slide_width = Inches(PAGE_W)
    prs.slide_height = Inches(PAGE_H)
    n = 0
    for fid in (figures or sorted(LAYOUTS)):
        used = build_slide(prs, fid.lower(), src, caps)
        n += used
        logger.info("[%s] %d slide(s)", fid, used)

    path = Path(out) if out else report_dir(ctx.figure_root, "pptx") / "report_panels.pptx"
    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(path))
    logger.info("wrote %s (%d slides)", path, n)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--figures", nargs="*", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    print(build(args.config, args.figures, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
