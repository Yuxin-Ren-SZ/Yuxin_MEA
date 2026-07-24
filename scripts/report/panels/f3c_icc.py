"""F3c — Immunofluorescence micrographs (blank placeholders until supplied)."""
from __future__ import annotations

import numpy as np

from ..panel_base import build_parser, panel_main
from ..report_style import GRID, MUTED

NAME = "F3c_icc"
#: Markers down the rows, sampling days across the columns — the same days the
#: qPCR panels use, so the two read together.
MARKERS = ["MAP2", "GFAP", "DAPI / merge"]
DAYS = [0, 7, 14, 21]
FIGSIZE = (1.55 * len(DAYS) + 0.9, 1.55 * len(MARKERS) + 0.4)
CAPTION = (
    "Immunofluorescence characterisation of the cultures at the same time points "
    "as the qPCR panels: the treatment day (day 0, pre-treatment) and days 7, 14 "
    "and 21 after. Rows are markers — MAP2 for neurons, GFAP for astrocytes, and "
    "a DAPI/merge view — and columns are sampling days, so a row reads as one "
    "marker's time course and a column as the culture's composition on that day. "
    "Scale bar 50 µm. Images pending: the frames are reserved placeholders and "
    "carry no data."
)


def _blank(ax, text: str) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linestyle((0, (4, 4)))
        sp.set_color(GRID)
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            fontsize=6, color=MUTED)


def _show(ax, path, scalebar_um=None, px_per_um=None) -> None:
    import matplotlib.pyplot as plt

    img = plt.imread(str(path))
    ax.imshow(img, cmap=(None if np.ndim(img) == 3 else "gray"))
    ax.set_xticks([]); ax.set_yticks([])
    if scalebar_um and px_per_um:
        w = scalebar_um * px_per_um
        x0 = img.shape[1] * 0.94 - w
        y0 = img.shape[0] * 0.93
        ax.plot([x0, x0 + w], [y0, y0], color="white", lw=2.0,
                solid_capstyle="butt")
        ax.text(x0 + w / 2, y0 * 0.985, f"{scalebar_um:g} µm", color="white",
                ha="center", va="bottom", fontsize=5.5)


def build_fig(fig, ctx):
    """Grid of markers × days; real micrographs when supplied, else reserved frames.

    Images are passed on the command line rather than discovered, because there is
    no ICC output in the pipeline tree to discover — the microscope files live
    outside it. Order is row-major (marker by marker, days left to right).
    """
    images = list(getattr(ctx, "icc_images", None) or [])
    gs = fig.add_gridspec(len(MARKERS), len(DAYS), hspace=0.12, wspace=0.06)
    for r, marker in enumerate(MARKERS):
        for c, day in enumerate(DAYS):
            ax = fig.add_subplot(gs[r, c])
            k = r * len(DAYS) + c
            if k < len(images):
                _show(ax, images[k], getattr(ctx, "icc_scalebar_um", None),
                      getattr(ctx, "icc_px_per_um", None))
            else:
                _blank(ax, f"{marker}\nday {day}")
            if r == 0:
                ax.set_title(f"day {day}", fontsize=7, fontweight="bold")
            if c == 0:
                ax.set_ylabel(marker, fontsize=7, fontweight="bold")
    return None


def main(argv=None) -> int:
    """Own CLI: this panel takes image files, which no other panel does."""
    import sys

    from .. import panel_data as D
    from ..panel_base import render_panel

    p = build_parser(__doc__)
    p.add_argument("--images", nargs="*", default=None,
                   help="micrograph files, row-major (marker by marker, days L-R)")
    p.add_argument("--scalebar-um", type=float, default=50.0)
    p.add_argument("--px-per-um", type=float, default=None)
    args = p.parse_args(argv)

    ctx = D.context(args.config, refresh=args.refresh_cache, qpcr_dir=args.qpcr_dir)
    ctx.icc_images = args.images
    ctx.icc_scalebar_um = args.scalebar_um
    ctx.icc_px_per_um = args.px_per_um
    rec = render_panel(sys.modules[__name__], ctx, formats=tuple(args.formats),
                       dpi=args.dpi, write_caption=not args.no_caption_file)
    for f in rec["files"]:
        print(f)
    return 0 if rec["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
