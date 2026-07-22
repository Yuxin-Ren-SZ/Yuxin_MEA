"""Panel-preview rendering and cache.

Previews are the same matplotlib code the PDF uses, rasterised small. That is
the point — a preview drawn by a different renderer would look right and export
wrong. It also means previews are not cheap (the ML feature UMAP takes ~14 s), so
they are cached on disk and rendered in a process pool.

Cache key
---------
``(panel, params, groups, cell_w, cell_h, data fingerprint, style version)``.
The **cell size belongs in the key**: resizing a panel changes its aspect ratio
in the exported figure, and a key without it would serve a stale thumbnail —
diverging preview from output on precisely the dimension being edited.

Why processes
-------------
A worker holds one :class:`RenderContext` for its lifetime, so the memoised
bootstraps and pooled unit tables are reused across the previews it renders,
while a slow panel cannot block the rest of the palette. Matplotlib is driven
through the object-oriented API (``Figure`` + ``FigureCanvasAgg``) rather than
pyplot, whose global figure registry is not safe to share.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger("composer.cache")

STYLE_VERSION = "1"          # bump to invalidate every cached preview
PREVIEW_DPI = 96


def preview_key(panel_id: str, params: dict, groups: list[str], cell_w: int,
                cell_h: int, fingerprint: str) -> str:
    h = hashlib.sha256()
    h.update(panel_id.encode())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    h.update(json.dumps(sorted(groups)).encode())
    h.update(f"{cell_w}x{cell_h}".encode())
    h.update(fingerprint.encode())
    h.update(STYLE_VERSION.encode())
    return h.hexdigest()[:24]


# --------------------------------------------------------------------------- #
# Worker side
# --------------------------------------------------------------------------- #
_WORKER: dict[str, Any] = {}


def _worker_init(config_path: str | None, groups: list[str], seed: int):
    import matplotlib
    matplotlib.use("Agg")

    from .. import load as L
    from .. import panels as P
    from ..report_style import apply_style, resolve_roots

    apply_style()
    P.load_registry()
    analysis_root, figure_root = resolve_roots(config_path)
    tidy = L.load_tidy(figure_root)
    _WORKER["tidy"] = tidy
    _WORKER["roots"] = (analysis_root, figure_root)
    _WORKER["ctx"] = P.RenderContext(tidy, analysis_root, figure_root,
                                     groups=groups, seed=seed)
    _WORKER["groups"] = list(groups)
    _WORKER["seed"] = seed


def _worker_ctx(groups: list[str], seed: int):
    """Reuse the worker's context unless the group selection actually changed."""
    from .. import panels as P

    if _WORKER.get("groups") != list(groups) or _WORKER.get("seed") != seed:
        analysis_root, figure_root = _WORKER["roots"]
        _WORKER["ctx"] = P.RenderContext(_WORKER["tidy"], analysis_root,
                                         figure_root, groups=groups, seed=seed)
        _WORKER["groups"] = list(groups)
        _WORKER["seed"] = seed
    return _WORKER["ctx"]


def _render_preview_task(panel_id: str, params: dict, groups: list[str],
                         cell_w: int, cell_h: int, out_path: str,
                         px_per_col: int, seed: int) -> dict:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from .. import panels as P

    ctx = _worker_ctx(groups, seed)
    # Preview aspect mirrors the cell so what is previewed is what is exported.
    w_in = max(1.6, cell_w * px_per_col / PREVIEW_DPI)
    h_in = max(1.2, cell_h * px_per_col * 0.62 / PREVIEW_DPI)
    fig = Figure(figsize=(w_in, h_in), dpi=PREVIEW_DPI)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 1, left=0.14, right=0.97, bottom=0.16, top=0.90)
    rec = P.draw_panel(fig, gs[0, 0], panel_id, ctx, params)
    tmp = Path(f"{out_path}.{os.getpid()}.tmp")
    # format is explicit: matplotlib would otherwise infer it from the .tmp suffix
    fig.savefig(tmp, format="png", dpi=PREVIEW_DPI, facecolor="white")
    tmp.replace(out_path)          # atomic: a half-written PNG is never served
    fig.clear()
    rec["path"] = out_path
    return rec


# --------------------------------------------------------------------------- #
# Server side
# --------------------------------------------------------------------------- #
class PreviewCache:
    """Disk-cached panel previews rendered in a process pool."""

    def __init__(self, cache_dir: Path, config_path: str | None,
                 fingerprint: str, groups: list[str], seed: int = 0,
                 max_workers: int | None = None, px_per_col: int = 78):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config_path = config_path
        self.fingerprint = fingerprint
        self.px_per_col = px_per_col
        self.seed = seed
        self._pool: ProcessPoolExecutor | None = None
        self._pool_args = (config_path, list(groups), seed)
        self._max_workers = max_workers or max(2, min(8, (os.cpu_count() or 4) - 2))
        self._records: dict[str, dict] = {}

    # -- pool lifecycle ----------------------------------------------------- #
    @property
    def pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=_worker_init, initargs=self._pool_args)
        return self._pool

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    # -- lookup / render ---------------------------------------------------- #
    def path_for(self, key: str) -> Path:
        return self.dir / f"{key}.png"

    def get(self, panel_id: str, params: dict, groups: list[str],
            cell_w: int, cell_h: int) -> tuple[str, Path, bool]:
        key = preview_key(panel_id, params, groups, cell_w, cell_h, self.fingerprint)
        p = self.path_for(key)
        return key, p, p.exists()

    def render(self, panel_id: str, params: dict, groups: list[str],
               cell_w: int, cell_h: int, timeout: float = 300.0) -> dict:
        key, path, cached = self.get(panel_id, params, groups, cell_w, cell_h)
        if cached:
            rec = self._records.get(key, {"panel": panel_id, "status": "ok"})
            return {**rec, "key": key, "cached": True}
        fut = self.pool.submit(_render_preview_task, panel_id, params,
                               list(groups), int(cell_w), int(cell_h), str(path),
                               self.px_per_col, self.seed)
        try:
            rec = fut.result(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - a dead worker must not 500 the page
            logger.warning("preview failed for %s: %r", panel_id, exc)
            return {"panel": panel_id, "status": "error", "warning": repr(exc),
                    "key": key, "cached": False}
        self._records[key] = {k: v for k, v in rec.items() if k != "path"}
        return {**self._records[key], "key": key, "cached": False}

    def clear(self) -> int:
        n = 0
        for p in self.dir.glob("*.png"):
            p.unlink()
            n += 1
        self._records.clear()
        return n
