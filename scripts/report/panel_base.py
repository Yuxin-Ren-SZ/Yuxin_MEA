"""Runner shared by every standalone panel script.

Each subplot of the report lives in its own file under ``scripts/report/panels/``
and is runnable on its own::

    python -m scripts.report.panels.f4c_firing_rate
    python -m scripts.report.panels.f4c_firing_rate --formats png --dpi 600

A panel module declares four things and nothing else:

``NAME``     output basename (``F4c_firing_rate``)
``CAPTION``  the explanatory text — written to ``captions/<NAME>.txt``, **never**
             drawn on the image. The PNG must stay clean enough to drop into a
             slide, a poster or a different journal layout unmodified.
``FIGSIZE``  ``(w, h)`` inches
``build``    ``build(ax, ctx)`` for a one-axes panel, or ``build_fig(fig, ctx)``
             when the panel is a composite (heatmap + trace + colourbar).

Either builder may return a :class:`pandas.DataFrame` of the statistics behind
the panel; it is written next to the image as ``<NAME>_stats.csv`` so a reader
can check that every ``*`` really is q<0.05 and every ``△`` really is p<0.05.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("report.panel")

DEFAULT_FORMATS = ("png", "pdf", "svg")


# --------------------------------------------------------------------------- #
# Output locations
# --------------------------------------------------------------------------- #
def panels_dir(figure_root) -> Path:
    from .report_style import report_dir
    return report_dir(figure_root, "panels")


def captions_dir(figure_root) -> Path:
    from .report_style import report_dir
    return report_dir(figure_root, "captions")


# --------------------------------------------------------------------------- #
# Rendering one panel
# --------------------------------------------------------------------------- #
def render_panel(
    module,
    ctx,
    *,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int | None = None,
    figsize: tuple[float, float] | None = None,
    out_dir: Path | None = None,
    write_caption: bool = True,
) -> dict[str, Any]:
    """Render one panel module against ``ctx``; return a small manifest record.

    Never draws the caption — that is the whole point of the split. The figure is
    saved with a tight bbox so the PNG has no dead margin when it is placed.
    """
    import matplotlib.pyplot as plt

    from .report_style import apply_style

    name = getattr(module, "NAME", None) or module.__name__.rsplit(".", 1)[-1]
    size = figsize or getattr(module, "FIGSIZE", (3.4, 2.6))
    apply_style()

    fig = None
    stats = None
    status = "ok"
    warning = ""
    try:
        if hasattr(module, "make_fig"):
            # Panels whose grid size depends on the data build their own figure.
            fig, stats = module.make_fig(ctx)
        elif hasattr(module, "build_fig"):
            fig = plt.figure(figsize=size)
            stats = module.build_fig(fig, ctx)
        elif hasattr(module, "build"):
            fig = plt.figure(figsize=size)
            stats = module.build(fig.add_subplot(111), ctx)
        else:
            raise AttributeError(
                f"{name}: needs build(ax, ctx), build_fig(fig, ctx) or make_fig(ctx)")
    except Exception as exc:  # noqa: BLE001 — one broken panel must not stop a batch
        status, warning = "error", f"{type(exc).__name__}: {exc}"
        logger.error("[%s] render failed: %s", name, exc, exc_info=True)
        if fig is not None:
            plt.close(fig)
        return {"panel": name, "status": status, "warning": warning, "files": []}

    out = Path(out_dir) if out_dir else panels_dir(ctx.figure_root)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for ext in formats:
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02,
                    **({"dpi": dpi} if dpi else {}))
        written.append(str(p))
    plt.close(fig)

    if write_caption:
        text = " ".join((getattr(module, "CAPTION", "") or "").split())
        if text:
            cdir = captions_dir(ctx.figure_root)
            cdir.mkdir(parents=True, exist_ok=True)
            cpath = cdir / f"{name}.txt"
            cpath.write_text(text + "\n")
            written.append(str(cpath))
        else:
            warning = "no CAPTION defined"

    if stats is not None and getattr(stats, "empty", True) is False:
        spath = out / f"{name}_stats.csv"
        stats.to_csv(spath, index=False)
        written.append(str(spath))

    return {"panel": name, "status": status, "warning": warning, "files": written}


# --------------------------------------------------------------------------- #
# CLI entry point used by every panel module's __main__ block
# --------------------------------------------------------------------------- #
def build_parser(description: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None,
                   help="pipeline config JSON (default pipeline_config_local.json)")
    p.add_argument("--out", default=None, help="output directory override")
    p.add_argument("--formats", nargs="*", default=list(DEFAULT_FORMATS))
    p.add_argument("--dpi", type=int, default=None)
    p.add_argument("--size", nargs=2, type=float, default=None,
                   metavar=("W", "H"), help="figure size in inches")
    p.add_argument("--no-caption-file", action="store_true")
    p.add_argument("--qpcr-dir", default=None,
                   help="qPCR plate export root (default <analysis_root>/qPCR)")
    p.add_argument("--refresh-cache", action="store_true",
                   help="recompute the shared derived datasets")
    p.add_argument("--verbose", action="store_true")
    return p


def panel_main(module_name: str, argv=None) -> int:
    """``if __name__ == "__main__": panel_main(__name__)`` in every panel script."""
    from . import panel_data as D

    module = sys.modules.get(module_name) or importlib.import_module(module_name)
    args = build_parser(module.__doc__ or "").parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    ctx = D.context(args.config, refresh=args.refresh_cache,
                    qpcr_dir=args.qpcr_dir)
    rec = render_panel(
        module, ctx,
        formats=tuple(args.formats),
        dpi=args.dpi,
        figsize=tuple(args.size) if args.size else None,
        out_dir=Path(args.out) if args.out else None,
        write_caption=not args.no_caption_file,
    )
    for f in rec["files"]:
        print(f)
    if rec["warning"]:
        print(f"[{rec['panel']}] {rec['warning']}", file=sys.stderr)
    return 0 if rec["status"] == "ok" else 1
