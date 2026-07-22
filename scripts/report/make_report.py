"""Driver — build the journal report figures.

Usage::

    python -m scripts.report.make_report --config pipeline_config_local.json --all
    python -m scripts.report.make_report --figures f5 f6 s4
    python -m scripts.report.make_report --figures f3 --qpcr-table qpcr.csv
    python -m scripts.report.make_report --figures f2 \
        --f2-images a.tif b.tif c.tif --f2-labels GFAP MAP2 DAPI \
        --f2-scalebar-um 50 --f2-px-per-um 1.5

Data-generated MEA figures (F4-F6, S1-S4) need no extra input. Wet-lab / asset
figures skip with an explicit message unless their inputs are supplied:
  * F1 needs ``--f1-assets`` (schematic image files),
  * F2 needs ``--f2-images`` (+ optional labels / scale bar),
  * F3 needs ``--qpcr-table`` (else a clearly-named SYNTHETIC placeholder).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import (
    fig3_qpcr, fig4_activity_dev, fig5_burst_phenotype, fig6_ml,
    fig6c_bursttypes, fig7_spatial_connectivity, fig_umap_migration, fig_assets,
    fig_s1_method_compare, fig_s2_units, fig_s3_rasters, fig_s4_crossgroup, load,
)
from .report_style import apply_style, report_dir, resolve_roots

logger = logging.getLogger("report")

MAIN = ["f1", "f2", "f3", "f4", "f5", "f6", "f6b", "f6c", "f7"]
SUPP = ["s1", "s2", "s3", "s4"]


def _paths(result):
    """Normalise a render() return (paths | (paths, data)) to a path list."""
    if isinstance(result, tuple):
        result = result[0]
    return [str(p) for p in (result or [])]


def build(args) -> dict:
    apply_style()
    analysis_root, figure_root = resolve_roots(args.config)
    logger.info("analysis_root=%s figure_root=%s", analysis_root, figure_root)
    tidy = load.load_tidy(figure_root)

    which = args.figures
    if not which or "all" in which:
        which = MAIN + SUPP
    which = [w.lower() for w in which]

    manifest: dict[str, object] = {}

    def run(fid, fn):
        if fid not in which:
            return
        try:
            manifest[fid] = _paths(fn())
            logger.info("[%s] wrote %d files", fid, len(manifest[fid]))
        except Exception as exc:  # keep building the rest
            manifest[fid] = {"error": str(exc)}
            logger.error("[%s] FAILED: %s", fid, exc, exc_info=args.verbose)

    # --- wet-lab / asset figures (conditional inputs) ---
    if "f1" in which:
        if args.f1_assets:
            run("f1", lambda: fig_assets.render_f1(figure_root, args.f1_assets))
        else:
            manifest["f1"] = {"skipped": "needs --f1-assets (schematic image files)"}
            logger.warning("[f1] skipped: provide --f1-assets")
    if "f2" in which:
        if args.f2_images:
            run("f2", lambda: fig_assets.render_f2(
                figure_root, args.f2_images, labels=args.f2_labels,
                scalebar_um=args.f2_scalebar_um, px_per_um=args.f2_px_per_um))
        else:
            manifest["f2"] = {"skipped": "needs --f2-images (ICC micrographs)"}
            logger.warning("[f2] skipped: provide --f2-images")
    if "f3" in which:
        if not args.qpcr_table:
            logger.warning("[f3] no --qpcr-table; writing SYNTHETIC placeholder")
        run("f3", lambda: fig3_qpcr.render(figure_root, args.qpcr_table))

    # --- data-generated MEA figures ---
    run("f4", lambda: fig4_activity_dev.render(tidy, analysis_root, figure_root))
    run("f5", lambda: fig5_burst_phenotype.render(tidy, figure_root, args.rosglo_table))
    run("f6", lambda: fig6_ml.render(tidy, analysis_root, figure_root))
    run("f6b", lambda: fig_umap_migration.render(tidy, analysis_root, figure_root))
    run("f6c", lambda: fig6c_bursttypes.render(tidy, analysis_root, figure_root))
    run("f7", lambda: fig7_spatial_connectivity.render(tidy, analysis_root, figure_root))
    run("s1", lambda: fig_s1_method_compare.render(figure_root))
    run("s2", lambda: fig_s2_units.render(tidy, analysis_root, figure_root))
    run("s3", lambda: fig_s3_rasters.render(tidy, analysis_root, figure_root))
    run("s4", lambda: fig_s4_crossgroup.render(tidy, figure_root))

    out = report_dir(figure_root)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("manifest -> %s", out / "manifest.json")
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="pipeline config JSON (default: pipeline_config_local.json)")
    p.add_argument("--figures", nargs="*", default=["all"],
                   help="figure ids to build: f1..f6 s1..s4, or 'all'")
    p.add_argument("--qpcr-table", default=None, help="qPCR CSV/xlsx for F3")
    p.add_argument("--rosglo-table", default=None,
                   help="ROS-Glo CSV/xlsx (group,value) for F5 panel B")
    p.add_argument("--f1-assets", nargs="*", default=None,
                   help="schematic image files for F1 graphical abstract")
    p.add_argument("--f2-images", nargs="*", default=None,
                   help="ICC micrograph files for F2")
    p.add_argument("--f2-labels", nargs="*", default=None, help="F2 channel labels")
    p.add_argument("--f2-scalebar-um", type=float, default=None)
    p.add_argument("--f2-px-per-um", type=float, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    manifest = build(args)
    ok = sum(1 for v in manifest.values() if isinstance(v, list))
    print(f"\nReport build complete: {ok}/{len(manifest)} figures written.")
    for fid, v in manifest.items():
        tag = (f"{len(v)} files" if isinstance(v, list)
               else next(iter(v.values())))
        print(f"  {fid}: {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
