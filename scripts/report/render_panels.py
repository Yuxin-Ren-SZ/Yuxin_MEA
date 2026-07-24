"""Render many panels in one process (so the shared cache is paid once).

    python -m scripts.report.render_panels --all
    python -m scripts.report.render_panels --figures f4 f7
    python -m scripts.report.render_panels --panels f4c_firing_rate f4d_nb_rate

Each panel is still a standalone script; this only saves the repeated process
start-up and, more importantly, the repeated expensive fits. A panel that fails
is reported and the batch continues — one broken input must not cost the rest.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import pkgutil
from pathlib import Path

from . import panel_data as D
from . import panels as panels_pkg
from .panel_base import DEFAULT_FORMATS, render_panel, panels_dir

logger = logging.getLogger("report.render")


def discover() -> list[str]:
    """Panel module names, in file order (``f4a`` before ``f4b`` before ``f5a``)."""
    return sorted(
        m.name for m in pkgutil.iter_modules(panels_pkg.__path__)
        if not m.name.startswith("_")
    )


def _figure_of(mod_name: str) -> str:
    """``f7b_ccg_ivh_early`` -> ``f7``; the panel letter is not the figure."""
    stem = mod_name.split("_", 1)[0]        # "f7b"
    i = 1
    while i < len(stem) and stem[i].isdigit():
        i += 1
    return stem[:i]


def select(all_mods: list[str], figures=None, panels=None) -> list[str]:
    if panels:
        want = {p.removesuffix(".py") for p in panels}
        return [m for m in all_mods if m in want]
    if figures:
        figs = {f.lower() for f in figures}
        return [m for m in all_mods if _figure_of(m) in figs]
    return all_mods


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--figures", nargs="*", default=None,
                   help="figure ids, e.g. f4 f5 f7")
    p.add_argument("--panels", nargs="*", default=None, help="panel module names")
    p.add_argument("--formats", nargs="*", default=list(DEFAULT_FORMATS))
    p.add_argument("--dpi", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--qpcr-dir", default=None)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    mods = select(discover(), args.figures, args.panels)
    if not mods and not args.all:
        p.error("nothing selected: pass --all, --figures or --panels")

    ctx = D.context(args.config, refresh=args.refresh_cache,
                    qpcr_dir=args.qpcr_dir)
    out_dir = Path(args.out) if args.out else panels_dir(ctx.figure_root)
    manifest = []
    for name in mods:
        module = importlib.import_module(f"{panels_pkg.__name__}.{name}")
        rec = render_panel(module, ctx, formats=tuple(args.formats), dpi=args.dpi,
                           out_dir=out_dir)
        manifest.append(rec)
        flag = "ok " if rec["status"] == "ok" else "FAIL"
        logger.info("[%s] %s %s", flag, rec["panel"], rec["warning"])

    (out_dir / "panel_manifest.json").write_text(json.dumps(manifest, indent=2))
    ok = sum(1 for r in manifest if r["status"] == "ok")
    print(f"\n{ok}/{len(manifest)} panels rendered -> {out_dir}")
    for r in manifest:
        if r["status"] != "ok":
            print(f"  FAILED {r['panel']}: {r['warning']}")
    return 0 if ok == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
