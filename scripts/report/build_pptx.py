"""Assemble the **supplementary** figures into an editable PowerPoint deck.

Each figure becomes one slide: the plot as a high-resolution image (>=300 dpi as
displayed, >=600 for detail-heavy figures) plus the figure **title** and
**caption** as native, editable PowerPoint text boxes — not baked into the image,
so they can be re-typed and re-laid-out later.

    <figure_root>/report/pptx/report_supp.pptx   (S1-S5)

The main figures are panels now and their deck is built by ``panel_pptx.py``.
This module used to advertise a ``report_main.pptx`` as well; that branch had
been unreachable since the main list emptied, so it is gone rather than left
promising a file it could not produce.

The run has two passes:

  * **render** — re-runs the figure builders once with caption drawing
    suppressed (``report_style._CAPTION_DRAW``), grabs each live figure via
    ``report_style._COLLECT``, and caches a caption-less PNG + metadata under
    ``pptx/_assets/`` (``manifest.json``). The standalone captioned PNG/PDF/SVG
    files are left untouched.
  * **assemble** — reads the cached assets and lays out the two decks. Layout-only
    tweaks can rerun just this pass with ``--assemble-only`` (fast; no builders).

S5 needs ``--qpcr-dir`` and skips without it.

Usage::

    python -m scripts.report.build_pptx --config pipeline_config_local.json \
        --qpcr-dir /mnt/Vol20tb2/SadeghLab/yuxin/analysis/qPCR
    # after editing layout constants, reassemble without re-rendering:
    python -m scripts.report.build_pptx --config pipeline_config_local.json \
        --assemble-only
"""
from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

from . import make_report, report_style
from .report_style import report_dir, resolve_roots

logger = logging.getLogger("report.pptx")

# --- slide layout (inches), 16:9 canvas -------------------------------------- #
SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.4
TITLE_TOP, TITLE_H, TITLE_FS = 0.12, 0.85, 15   # room for a 2-line title
CAP_H, CAP_FS = 1.5, 9.0
CAP_TOP = SLIDE_H - 0.08 - CAP_H
IMG_TOP = TITLE_TOP + TITLE_H + 0.06
IMG_H = CAP_TOP - IMG_TOP - 0.06
IMG_W = SLIDE_W - 2 * MARGIN

#: Figures with many small features that get 600 dpi (as displayed).
_DETAIL_PREFIXES = ("F6_", "F6b", "F6c", "F7", "F8", "S2", "S3", "S5")


def _dpi_for(name: str) -> int:
    return 600 if name.startswith(_DETAIL_PREFIXES) else 300


def _display_size(aspect: float) -> tuple[float, float]:
    """Size (inches) the image is shown at once fit into the content box."""
    if IMG_W / IMG_H > aspect:          # box wider than image -> fit by height
        return IMG_H * aspect, IMG_H
    return IMG_W, IMG_W / aspect        # fit by width


def _render_png(fig, target_dpi: int, path: Path) -> tuple[float, int]:
    """Render a figure to ``path`` (title suppressed) so its *displayed*
    resolution is at least ``target_dpi``; return (aspect, rendered_dpi).

    Figures are scaled to fit the content box, so a wide/short figure gets
    up-scaled and its effective dpi would fall below the source dpi. A cheap
    150-dpi probe measures the tight-bbox size; the final dpi is raised by the
    up-scale factor (capped at 900 to bound file size).
    """
    if fig._suptitle is not None:
        fig.suptitle("")
    try:
        from PIL import Image

        probe = io.BytesIO()
        fig.savefig(probe, format="png", dpi=150, bbox_inches="tight",
                    pad_inches=0.1)
        probe.seek(0)
        iw, ih = Image.open(probe).size
        aspect = iw / ih
        disp_w, disp_h = _display_size(aspect)
        scale = max(disp_w / (iw / 150), disp_h / (ih / 150))   # >1 if up-scaled
        dpi = int(min(max(target_dpi * scale, target_dpi), 900))
    except Exception:                    # Pillow missing -> heuristic headroom
        w, h = fig.get_size_inches()
        aspect = float(w / h)
        dpi = int(min(target_dpi * 1.6, 900))
    fig.savefig(path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    return aspect, dpi


def _render_pass(cli, figure_root: Path) -> list[dict]:
    """Re-run the builders (captions off), caching a caption-less PNG + metadata
    per figure. Returns the manifest and writes it to ``pptx/_assets``."""
    asset_dir = report_dir(figure_root) / "pptx" / "_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    args = argparse.Namespace(config=cli.config, figures=["all"],
                              qpcr_dir=cli.qpcr_dir, verbose=False)

    prev_draw = report_style._CAPTION_DRAW
    report_style._CAPTION_DRAW = False
    collected: list = []
    report_style._COLLECT = collected
    try:
        make_report.build(args)
    finally:
        report_style._COLLECT = None
        report_style._CAPTION_DRAW = prev_draw

    manifest = []
    for name, subdir, fig in collected:
        title = fig._suptitle.get_text() if fig._suptitle is not None else name
        cap = getattr(fig, "_report_caption", "")
        png = asset_dir / f"{name}.png"
        aspect, dpi = _render_png(fig, _dpi_for(name), png)
        manifest.append(dict(name=name, subdir=subdir, title=title, caption=cap,
                             png=png.name, aspect=aspect, dpi=dpi))
        logger.info("[pptx] cached %s (target %d, rendered %d dpi)",
                    name, _dpi_for(name), dpi)
    (asset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _add_slide(prs, entry: dict, asset_dir: Path) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    tb = slide.shapes.add_textbox(Inches(MARGIN), Inches(TITLE_TOP),
                                  Inches(IMG_W), Inches(TITLE_H))
    tf = tb.text_frame
    tf.word_wrap = True
    run = tf.paragraphs[0].add_run()
    run.text = entry["title"]
    run.font.size = Pt(TITLE_FS)
    run.font.bold = True

    aspect = entry["aspect"]
    pw, ph = _display_size(aspect)
    left = MARGIN + (IMG_W - pw) / 2
    top = IMG_TOP + (IMG_H - ph) / 2
    slide.shapes.add_picture(str(asset_dir / entry["png"]), Inches(left),
                             Inches(top), width=Inches(pw), height=Inches(ph))

    cb = slide.shapes.add_textbox(Inches(MARGIN), Inches(CAP_TOP),
                                  Inches(IMG_W), Inches(CAP_H))
    cf = cb.text_frame
    cf.word_wrap = True
    crun = cf.paragraphs[0].add_run()
    crun.text = entry["caption"]
    crun.font.size = Pt(CAP_FS)


def _new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    return prs


def _assemble(figure_root: Path, manifest: list[dict] | None = None) -> list[Path]:
    asset_dir = report_dir(figure_root) / "pptx" / "_assets"
    if manifest is None:
        manifest = json.loads((asset_dir / "manifest.json").read_text())
    # One deck. There used to be a "main" deck here too, but the main figures
    # became panels and ``panel_pptx`` builds their deck; this branch had been
    # unreachable ever since, quietly promising a file it could not produce.
    deck, n = _new_deck(), 0
    for entry in manifest:
        _add_slide(deck, entry, asset_dir)
        n += 1

    out_dir = report_dir(figure_root) / "pptx"
    if not n:
        logger.warning("[pptx] no supplement figures collected; no deck written")
        return []
    p = out_dir / "report_supp.pptx"
    deck.save(str(p))
    logger.info("[pptx] wrote %s (%d slides)", p, n)
    return [p]


def build(cli) -> list[Path]:
    report_style.apply_style()
    _, figure_root = resolve_roots(cli.config)
    manifest = None
    if not cli.assemble_only:
        manifest = _render_pass(cli, figure_root)
    return _assemble(figure_root, manifest)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--qpcr-dir", default=None,
                   help="qPCR plate root — S5 skips without it")
    p.add_argument("--assemble-only", action="store_true",
                   help="skip the render pass; reassemble decks from cached "
                        "pptx/_assets (fast, for layout tweaks)")
    cli = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    written = build(cli)
    print(f"\nPPTX export complete: {len(written)} deck(s).")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
