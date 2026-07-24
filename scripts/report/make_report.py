"""Driver — build the supplementary figures S1-S5.

The **main** figures are not built here. Every F-panel is its own script under
``panels/``, rendered by ``render_panels.py`` and arranged by
``assemble_figure.py`` / ``panel_pptx.py``; see ``README.md``. This driver
survives only because S1-S5 are still composed figures with no panel equivalent.

Usage::

    python -m scripts.report.make_report --config pipeline_config_local.json
    python -m scripts.report.make_report --figures s2 s3
    python -m scripts.report.make_report --figures s5 --qpcr-dir /path/to/qPCR

S1-S4 need no extra input. S5 (reference-gene screen) needs ``--qpcr-dir`` and
skips with an explicit message without it.

The cohort comes from :class:`panel_data.PanelContext`, the same object every
panel uses, so the supplement and the main figures describe one dataset. Reading
``tidy_long.csv`` directly here — as this did until the media-change window
landed — silently gave the supplement 1843 recordings while the panels used 1367.
"""
from __future__ import annotations

import argparse
import json
import logging

from . import (
    fig_s1_method_compare, fig_s2_units, fig_s3_rasters, fig_s4_crossgroup,
    fig_s5_refgene,
)
from .report_style import apply_style, report_dir

logger = logging.getLogger("report")

#: The supplement is the whole of this driver's job. S6 used to live here too,
#: but it is the same figure as the F3b panel — one ``build_s6``, two output
#: names — so it is drawn once, as a panel.
SUPP = ["s1", "s2", "s3", "s4", "s5"]


def _paths(result):
    """Normalise a render() return (paths | (paths, data)) to a path list."""
    if isinstance(result, tuple):
        result = result[0]
    return [str(p) for p in (result or [])]


def build(args) -> dict:
    from . import panel_data as D

    apply_style()
    ctx = D.context(args.config)
    analysis_root, figure_root = ctx.analysis_root, ctx.figure_root
    logger.info("analysis_root=%s figure_root=%s", analysis_root, figure_root)
    logger.info("cohort: %d well-recordings (same as the panels)", len(ctx.tidy))
    tidy = ctx.tidy

    which = args.figures
    if not which or "all" in which:
        which = SUPP
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

    run("s1", lambda: fig_s1_method_compare.render(figure_root))
    run("s2", lambda: fig_s2_units.render(tidy, analysis_root, figure_root))
    run("s3", lambda: fig_s3_rasters.render(tidy, analysis_root, figure_root))
    run("s4", lambda: fig_s4_crossgroup.render(tidy, figure_root))
    if "s5" in which:
        if args.qpcr_dir:
            run("s5", lambda: fig_s5_refgene.render(figure_root, args.qpcr_dir))
        else:
            manifest["s5"] = {"skipped": "needs --qpcr-dir (qPCR plate directories)"}
            logger.warning("[s5] skipped: provide --qpcr-dir")

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
                   help="supplement figure ids to build (s1..s5). 'all' means all "
                        "of them. The main figures F2-F8 are built as individual "
                        "panels by scripts.report.render_panels.")
    p.add_argument("--qpcr-dir", default=None,
                   help="qPCR analysis root holding one directory per plate "
                        "(S5 reference-gene screen)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    manifest = build(args)
    ok = sum(1 for v in manifest.values() if isinstance(v, list))
    print(f"\nSupplement build complete: {ok}/{len(manifest)} figures written.")
    for fid, v in manifest.items():
        tag = (f"{len(v)} files" if isinstance(v, list)
               else next(iter(v.values())))
        print(f"  {fid}: {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
