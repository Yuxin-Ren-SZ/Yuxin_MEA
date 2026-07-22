"""Spec -> matplotlib figures -> PDF. The half of the composer that has no UI.

This module is deliberately usable on its own::

    python -m scripts.report.composer.compose my_spec.json --out /tmp/figs

so the layout contract can be exercised, tested and scripted without a browser.

Two rules keep the canvas and the PDF in agreement:

1. **Each panel owns a pinned rectangle.** Rather than one shared ``GridSpec``
   with ``wspace``/``hspace`` — whose cell edges depend on the surrounding cells
   — every panel gets its own 1x1 ``GridSpec`` pinned to the exact figure
   fraction its grid cell occupies. Placement is then an arithmetic function of
   ``(x, y, w, h)`` alone, which is what makes it testable and what lets a
   composite panel call ``subspec.subgridspec(...)`` without escaping its cell.
2. **Never call ``tight_layout``.** Several ``build_fX`` functions do; here it
   would reflow panels off their assigned cells and break rule 1.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .. import panels as P
from ..report_style import apply_style, figures_base
from .spec import Composition, FigureSpec, PanelPlacement

# Outer margins as figure fractions. Generous top leaves room for the suptitle.
MARGIN = {"left": 0.055, "right": 0.02, "bottom": 0.055, "top": 0.085}
MARGIN_NO_TITLE = {**MARGIN, "top": 0.03}


def cell_rect(p: PanelPlacement, fig_spec: FigureSpec, comp: Composition,
              margin: dict | None = None) -> tuple[float, float, float, float]:
    """``(left, right, bottom, top)`` of a grid cell, in figure fractions.

    Grid y counts downward from the top (screen convention); matplotlib measures
    from the bottom, hence the flip.
    """
    m = margin or MARGIN
    ax0, ax1 = m["left"], 1.0 - m["right"]
    ay0, ay1 = m["bottom"], 1.0 - m["top"]
    span_x, span_y = ax1 - ax0, ay1 - ay0
    gx, gy = comp.gutter_x, comp.gutter_y
    left = ax0 + (p.x / comp.grid_cols) * span_x + gx
    right = ax0 + ((p.x + p.w) / comp.grid_cols) * span_x - gx
    top = ay1 - (p.y / fig_spec.rows) * span_y - gy
    bottom = ay1 - ((p.y + p.h) / fig_spec.rows) * span_y + gy
    # a gutter wider than the cell would invert it
    if right <= left:
        left, right = (left + right) / 2 - 1e-3, (left + right) / 2 + 1e-3
    if top <= bottom:
        bottom, top = (bottom + top) / 2 - 1e-3, (bottom + top) / 2 + 1e-3
    return left, right, bottom, top


def build_figure(fig_spec: FigureSpec, comp: Composition, ctx: P.RenderContext,
                 draw_labels: bool = True) -> tuple[Figure, list[dict]]:
    """Render one figure. Returns the figure and one record per panel."""
    P.load_registry()
    height_in = max(1.0, fig_spec.rows * comp.row_height_in)
    fig = Figure(figsize=(fig_spec.width_in, height_in), dpi=comp.dpi)
    FigureCanvasAgg(fig)                       # OO API: no pyplot, thread-safe

    margin = MARGIN if (fig_spec.show_suptitle and fig_spec.title) else MARGIN_NO_TITLE
    labels = fig_spec.auto_labels()
    records: list[dict] = []

    for p in fig_spec.ordered_panels():
        left, right, bottom, top = cell_rect(p, fig_spec, comp, margin)
        gs = fig.add_gridspec(1, 1, left=left, right=right, bottom=bottom, top=top)
        rec = P.draw_panel(fig, gs[0, 0], p.panel, ctx, p.params)
        rec.update(uid=p.uid, label=labels[p.uid],
                   cell={"x": p.x, "y": p.y, "w": p.w, "h": p.h})
        records.append(rec)
        if draw_labels and labels[p.uid]:
            fig.text(max(left - 0.035, 0.002), min(top + 0.012, 0.998),
                     labels[p.uid], fontsize=10, fontweight="bold",
                     ha="left", va="bottom")

    if fig_spec.show_suptitle and fig_spec.title:
        fig.suptitle(fig_spec.title, fontsize=9, y=0.995)
    return fig, records


def build_all(comp: Composition, ctx: P.RenderContext) -> list[tuple[FigureSpec, Figure, list[dict]]]:
    out = []
    for f in comp.figures:
        fig, recs = build_figure(f, comp, ctx)
        out.append((f, fig, recs))
    return out


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export(comp: Composition, ctx: P.RenderContext, out_dir: Path | None = None,
           formats: tuple[str, ...] = ("pdf", "png", "svg"),
           combined_pdf: str = "figures.pdf") -> dict:
    """Write every figure, the combined PDF, the spec, and a manifest.

    The spec written here is the resume file: loading it back restores the
    groups, the figures, every panel's placement and every panel's parameters.
    """
    apply_style()
    out_dir = Path(out_dir or (figures_base(ctx.figure_root) / "report" / "composed"
                               / comp.name))
    out_dir.mkdir(parents=True, exist_ok=True)

    built = build_all(comp, ctx)
    written: dict[str, list[str]] = {}
    warnings: list[dict] = []

    for fig_spec, fig, recs in built:
        paths = []
        for ext in formats:
            p = out_dir / f"{fig_spec.id}.{ext}"
            fig.savefig(p, dpi=comp.dpi)       # no bbox='tight': keeps the grid
            paths.append(str(p))
        written[fig_spec.id] = paths
        for r in recs:
            if r.get("status") != "ok" or r.get("warning"):
                warnings.append({"figure": fig_spec.id, **r})

    combined = out_dir / combined_pdf
    with PdfPages(combined) as pdf:
        for _, fig, _ in built:
            pdf.savefig(fig)
    for _, fig, _ in built:
        fig.clear()

    comp.created = comp.created or datetime.now().astimezone().isoformat(timespec="seconds")
    spec_path = comp.save(out_dir / "figure_spec.json")

    manifest = {
        "name": comp.name,
        "exported": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": comp.to_json()["generator"],
        "groups": list(comp.groups),
        "dpi": comp.dpi,
        "data": dict(comp.data),
        "combined_pdf": str(combined),
        "spec": str(spec_path),
        "figures": [
            {"id": f.id, "title": f.title,
             "size_in": [f.width_in, round(f.rows * comp.row_height_in, 4)],
             "grid": {"cols": comp.grid_cols, "rows": f.rows},
             "files": written[f.id],
             "panels": [{k: v for k, v in r.items() if k != "warning"}
                        for r in recs]}
            for f, _, recs in built
        ],
        "warnings": warnings,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["manifest"] = str(manifest_path)
    manifest["out_dir"] = str(out_dir)
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _context_for(comp: Composition, config: str | None) -> P.RenderContext:
    from .. import load as L
    from ..report_style import resolve_roots
    from .spec import data_fingerprint

    analysis_root, figure_root = resolve_roots(config)
    tidy = L.load_tidy(figure_root)
    comp.data = comp.data or data_fingerprint(tidy, analysis_root, config)
    return P.RenderContext(tidy, analysis_root, figure_root, groups=comp.groups,
                           seed=comp.seed)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec", nargs="?", help="figure_spec.json (default: built-in report)")
    ap.add_argument("--config", default=None, help="pipeline config JSON")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--name", default=None, help="override the composition name")
    ap.add_argument("--formats", nargs="*", default=["pdf", "png", "svg"])
    args = ap.parse_args(argv)

    if args.spec:
        comp = Composition.load(Path(args.spec))
    else:
        from .spec import default_composition
        comp = default_composition()
    if args.name:
        comp.name = args.name

    P.load_registry()
    problems = comp.validate(known_panels=set(P.PANELS))
    for m in problems:
        print(f"  warning: {m}")

    ctx = _context_for(comp, args.config)
    manifest = export(comp, ctx, args.out, formats=tuple(args.formats))

    print(f"\nwrote {len(manifest['figures'])} figures -> {manifest['out_dir']}")
    print(f"  combined PDF : {manifest['combined_pdf']}")
    print(f"  resume spec  : {manifest['spec']}")
    if manifest["warnings"]:
        print(f"  {len(manifest['warnings'])} panel warning(s):")
        for w in manifest["warnings"]:
            print(f"    {w['figure']}/{w.get('label', '?')} {w['panel']}: "
                  f"{w.get('warning', w.get('status'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
