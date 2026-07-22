"""Flask app for the composer — HTTP in, calls on :mod:`spec` and :mod:`compose` out.

Deliberately thin: every operation the UI performs (validate, split, merge,
export) is a function in the headless modules, so the same things can be done
from a script and are covered by tests that never start a server.

State lives in the browser. The server keeps only the loaded dataset, the
preview cache and the specs directory on disk; the working composition is posted
with each request. That keeps the page reloadable and makes "resume from a spec
file" the same code path as "the user pressed Load".
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from .. import load as L
from .. import panels as P
from ..report_style import apply_style, figures_base, resolve_roots
from . import compose as CM
from .cache import PreviewCache
from .spec import (
    Composition, SpecError, data_fingerprint, default_composition,
    fingerprint_delta, merge_figures, safe_name, split_figure,
)

logger = logging.getLogger("composer.server")


def _json_error(exc: Exception, status: int = 400):
    logger.warning("%s: %s", type(exc).__name__, exc)
    return jsonify({"error": str(exc), "type": type(exc).__name__}), status


def create_app(config_path: str | None = None, cache_dir: Path | None = None) -> Flask:
    apply_style()
    P.load_registry()

    analysis_root, figure_root = resolve_roots(config_path)
    tidy = L.load_tidy(figure_root)
    fingerprint = data_fingerprint(tidy, analysis_root, config_path)
    specs_dir = figures_base(figure_root) / "report" / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    cache = PreviewCache(
        cache_dir or (Path.home() / ".cache" / "yuxin_mea" / "composer"),
        config_path, fingerprint["tidy_sha256"], list(default_composition().groups))

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(JSON_SORT_KEYS=False, TEMPLATES_AUTO_RELOAD=True)
    app.extensions["composer"] = {
        "tidy": tidy, "analysis_root": analysis_root, "figure_root": figure_root,
        "fingerprint": fingerprint, "specs_dir": specs_dir, "cache": cache,
        "config_path": config_path,
    }

    def ctx_for(groups, seed=0) -> P.RenderContext:
        return P.RenderContext(tidy, analysis_root, figure_root, groups=groups,
                               seed=seed)

    # -- page ----------------------------------------------------------------
    @app.get("/")
    def index():
        from flask import render_template
        return render_template("composer.html")

    # -- bootstrap -----------------------------------------------------------
    @app.get("/api/bootstrap")
    def bootstrap():
        cov = P.coverage(tidy)
        return jsonify({
            "panels": [s.to_json() for s in P.PANELS.values()],
            "sections": {sec: [s.id for s in specs]
                         for sec, specs in P.sections().items()},
            "groups": P.ordered_groups(tidy.canonical_group.dropna().unique()),
            "coverage": cov,
            "families": P.METRIC_FAMILIES,
            "metric_labels": P.METRIC_LABELS,
            "data": fingerprint,
            "roots": {"analysis": str(analysis_root), "figure": str(figure_root)},
            "specs": sorted(p.stem for p in specs_dir.glob("*.json")),
            "default": default_composition().to_json(),
        })

    # -- previews ------------------------------------------------------------
    @app.post("/api/preview")
    def preview():
        try:
            body = request.get_json(force=True) or {}
            rec = cache.render(
                panel_id=str(body["panel"]),
                params=dict(body.get("params") or {}),
                groups=list(body.get("groups") or []),
                cell_w=int(body.get("w", 6)), cell_h=int(body.get("h", 4)))
        except Exception as exc:  # noqa: BLE001
            return _json_error(exc)
        rec["url"] = f"/api/panel/{rec['key']}.png"
        return jsonify(rec)

    @app.get("/api/panel/<key>.png")
    def panel_png(key: str):
        p = cache.path_for(safe_name(key))
        if not p.exists():
            return jsonify({"error": "not rendered"}), 404
        return send_file(p, mimetype="image/png", max_age=0)

    @app.post("/api/cache/clear")
    def cache_clear():
        return jsonify({"cleared": cache.clear()})

    # -- whole-figure preview (true WYSIWYG check) ---------------------------
    @app.post("/api/figure_preview")
    def figure_preview():
        try:
            body = request.get_json(force=True) or {}
            comp = Composition.from_json(body["composition"])
            fig_spec = comp.figure(str(body["figure"]))
            comp.validate(known_panels=set(P.PANELS))
            fig, _ = CM.build_figure(fig_spec, comp, ctx_for(comp.groups, comp.seed))
            out = cache.dir / f"figure_{safe_name(comp.name)}_{safe_name(fig_spec.id)}.png"
            fig.savefig(out, dpi=110, facecolor="white")
            fig.clear()
        except Exception as exc:  # noqa: BLE001
            return _json_error(exc)
        return jsonify({"url": f"/api/figure_png/{out.name}"})

    @app.get("/api/figure_png/<name>")
    def figure_png(name: str):
        p = cache.dir / safe_name(name)
        if not p.exists():
            return jsonify({"error": "not rendered"}), 404
        return send_file(p, mimetype="image/png", max_age=0)

    # -- spec validate / save / load -----------------------------------------
    @app.post("/api/validate")
    def validate():
        try:
            comp = Composition.from_json((request.get_json(force=True) or {}))
            problems = comp.validate(known_panels=set(P.PANELS))
        except SpecError as exc:
            return _json_error(exc)
        return jsonify({"ok": True, "warnings": problems,
                        "data_delta": fingerprint_delta(comp.data, fingerprint)})

    @app.post("/api/spec")
    def save_spec():
        try:
            comp = Composition.from_json(request.get_json(force=True) or {})
            comp.validate(known_panels=set(P.PANELS))
            comp.data = comp.data or fingerprint
            path = comp.save(specs_dir / f"{safe_name(comp.name)}.json")
        except SpecError as exc:
            return _json_error(exc)
        return jsonify({"saved": str(path), "name": safe_name(comp.name),
                        "specs": sorted(p.stem for p in specs_dir.glob("*.json"))})

    def _loaded(comp: Composition):
        return jsonify({"composition": comp.to_json(),
                        "data_delta": fingerprint_delta(comp.data, fingerprint),
                        "warnings": comp.validate(known_panels=set(P.PANELS))})

    @app.get("/api/spec/<name>")
    def load_spec(name: str):
        path = specs_dir / f"{safe_name(name)}.json"
        if not path.exists():
            return jsonify({"error": f"no spec {name!r}"}), 404
        try:
            return _loaded(Composition.load(path))
        except SpecError as exc:
            return _json_error(exc)

    @app.post("/api/spec_from_path")
    def load_spec_from_path():
        """Load any ``figure_spec.json`` by path.

        Export writes its spec next to the PDF; this is what lets that file be
        fed straight back in, including one carried over from another machine.
        """
        raw = str((request.get_json(force=True) or {}).get("path", "")).strip()
        if not raw:
            return jsonify({"error": "no path given"}), 400
        path = Path(raw).expanduser()
        if path.is_dir():
            path = path / "figure_spec.json"
        if not path.exists():
            return jsonify({"error": f"no such file: {path}"}), 404
        try:
            return _loaded(Composition.load(path))
        except (SpecError, ValueError) as exc:
            return _json_error(exc)

    # -- layout operations ----------------------------------------------------
    @app.post("/api/split")
    def split():
        try:
            body = request.get_json(force=True) or {}
            comp = Composition.from_json(body["composition"])
            split_figure(comp, str(body["figure"]), list(body["uids"]),
                         new_title=str(body.get("title", "")))
        except (SpecError, KeyError) as exc:
            return _json_error(exc)
        return jsonify({"composition": comp.to_json()})

    @app.post("/api/merge")
    def merge():
        try:
            body = request.get_json(force=True) or {}
            comp = Composition.from_json(body["composition"])
            merge_figures(comp, list(body["figures"]),
                          new_title=str(body.get("title", "")))
        except (SpecError, KeyError) as exc:
            return _json_error(exc)
        return jsonify({"composition": comp.to_json()})

    # -- export ---------------------------------------------------------------
    @app.post("/api/export")
    def export():
        try:
            body = request.get_json(force=True) or {}
            comp = Composition.from_json(body["composition"])
            comp.validate(known_panels=set(P.PANELS))
            comp.data = fingerprint
            formats = tuple(body.get("formats") or ("pdf", "png", "svg"))
            manifest = CM.export(comp, ctx_for(comp.groups, comp.seed),
                                 out_dir=body.get("out_dir"), formats=formats)
            # Also register the spec under its name, so the exported layout is
            # in the Load list next session. Exporting is the action users
            # actually take at the end of a session; requiring a separate Save
            # to be able to resume would lose the work they just committed to.
            comp.save(specs_dir / f"{safe_name(comp.name)}.json")
            manifest["specs"] = sorted(p.stem for p in specs_dir.glob("*.json"))
        except SpecError as exc:
            return _json_error(exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("export failed\n%s", traceback.format_exc())
            return _json_error(exc, 500)
        return jsonify(manifest)

    return app
