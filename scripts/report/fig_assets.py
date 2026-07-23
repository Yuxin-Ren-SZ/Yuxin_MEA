"""F1 (graphical abstract) & F2 (ICC) — composition of user-supplied images.

These figures are *not* data-generated: they lay out microscopy / schematic
image files the user provides into a labelled, journal-styled panel grid with
optional scale bars. No pixel data is synthesised. If no image paths are given,
``make_report`` skips these and prints the expected input contract.

Input contract
--------------
* **F1 graphical abstract**: one or more schematic asset files (the
  culture-method → MEA → pipeline diagram, e.g. exported from BioRender), plus
  optionally an ICC inset. Passed as an ordered list of image paths.
* **F2 ICC**: a list of micrograph files + matching channel/marker labels, an
  optional scale-bar length (µm) and image scale (pixels per µm).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .report_style import caption, save_fig


def _imread(path):
    import matplotlib.image as mpimg

    return mpimg.imread(str(path))


def compose_grid(
    image_paths: Sequence[str | Path],
    ncols: int = 2,
    labels: Sequence[str] | None = None,
    panel_letters: bool = True,
    scalebar_um: float | None = None,
    px_per_um: float | None = None,
    title: str | None = None,
    cap: str | None = None,
    figsize=None,
):
    """Lay images into an ncols grid. Missing files render an explicit stub."""
    import matplotlib.pyplot as plt

    n = len(image_paths)
    nrows = (n + ncols - 1) // ncols
    figsize = figsize or (4.5 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if i >= n:
            ax.set_visible(False)
            continue
        p = Path(image_paths[i])
        if not p.exists():
            ax.text(0.5, 0.5, f"missing:\n{p.name}", ha="center", va="center",
                    fontsize=7, color="0.4", transform=ax.transAxes)
            continue
        img = _imread(p)
        ax.imshow(img)
        if labels and i < len(labels) and labels[i]:
            ax.text(0.03, 0.95, labels[i], transform=ax.transAxes, fontsize=8,
                    color="white", va="top", ha="left",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
        if panel_letters:
            ax.text(-0.02, 1.02, chr(ord("A") + i), transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="bottom", ha="right")
        # scale bar on the last panel
        if scalebar_um and px_per_um and i == n - 1:
            h, w = img.shape[:2]
            bar_px = scalebar_um * px_per_um
            x0, y0 = w * 0.95 - bar_px, h * 0.93
            ax.plot([x0, x0 + bar_px], [y0, y0], color="white", lw=3)
            ax.text(x0 + bar_px / 2, y0 - h * 0.02, f"{scalebar_um:g} µm",
                    color="white", ha="center", va="bottom", fontsize=6)
    if title:
        fig.suptitle(title, fontsize=9, y=1.0)
    fig.tight_layout()
    if cap:
        caption(fig, cap)
    return fig


def render_f1(figure_root, asset_paths: Sequence[str | Path], ncols: int = 2):
    fig = compose_grid(
        asset_paths, ncols=ncols, panel_letters=False,
        title="Figure 1 — Graphical abstract (culture · MEA · pipeline)",
        cap="Graphical abstract: study schematic — iPSC-derived neuronal culture "
            "on the HD-MEA (MaxWell MaxTwo), the within-well pre/post treatment "
            "design (IVH CSF / oxidative stress applied on the treatment day), "
            "and the analysis pipeline from spike sorting through burst, "
            "connectivity and criticality read-outs. Composed from user-supplied "
            "schematic assets.")
    return save_fig(fig, "F1_graphical_abstract", figure_root, subdir="main")


def render_f2(figure_root, image_paths: Sequence[str | Path],
              labels: Sequence[str] | None = None, ncols: int = 3,
              scalebar_um: float | None = None, px_per_um: float | None = None):
    fig = compose_grid(
        image_paths, ncols=ncols, labels=labels, scalebar_um=scalebar_um,
        px_per_um=px_per_um,
        title="Figure 2 — Immunocytochemical characterisation",
        cap="Immunocytochemistry of the culture: each panel is a fluorescence "
            "channel for a cell-type / structural marker (channel labelled "
            "top-left; scale bar bottom-right of the last panel). Confirms "
            "neuronal and glial identity of the network recorded on the MEA. "
            "Representative micrographs supplied by the user.")
    return save_fig(fig, "F2_icc", figure_root, subdir="main")
