"""Server-side raster PNG rendering for the plate-viewer overview.

Caching guide §2: the plate *overview* is a pre-rasterized image, not a vector
figure. Drawing every spike into a small bitmap *is* the rasterization — dense
spikes collapse onto shared pixels with zero information loss at that display
size — so the browser receives ~24 small PNGs instead of hundreds of thousands
of Scattergl points. The interactive, full-resolution view is reserved for a
single drilled-in well (see ``build_single_well_figure``).

Rendering uses the matplotlib **object API** (``Figure`` + ``FigureCanvasAgg``),
never ``pyplot`` — pyplot keeps a process-global figure registry that is not
safe to touch from a long-running Dash server handling concurrent callbacks.
Each render builds a throwaway ``Figure`` and drops it; nothing global is held.

All functions are **read-only** over ``WellRecord`` data: the plate-viewer cache
hands out shared ``WellRecord`` objects, so mutating ``spike_times`` /
``plot_signals`` here would corrupt the next Load.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:  # avoid a hard import cycle at module load
    from yuxin_mea.analysis.plate_raster_synchrony import WellRecord

_RASTER_SUBDIR = "raster_png"


def render_well_png(
    well_record: "WellRecord",
    out_path: str | Path,
    *,
    w_px: int = 480,
    h_px: int = 300,
    dpi: int = 100,
    marker_lw: float = 0.4,
) -> Path:
    """Render one well's raster (all spikes) + a thin participation overlay to PNG.

    Every spike is drawn (``rasterized=True``); the bitmap does the downsampling.
    A faint participation line is overlaid in the top band so the overview keeps
    the plate-level synchrony-at-a-glance the interactive grid had. Read-only.
    """
    spike_times = well_record.spike_times or {}
    plot_signals = well_record.plot_signals or {}

    fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    FigureCanvasAgg(fig)  # bind an Agg canvas without touching pyplot globals
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Raster: one lane per unit, busiest units first (matches the interactive
    # figure's ordering so the overview and drill-down read the same).
    units = sorted(spike_times.items(), key=lambda kv: -len(kv[1]))
    xmax = 0.0
    for row, (_uid, times) in enumerate(units):
        arr = np.asarray(times, dtype=float)
        if arr.size:
            ax.scatter(
                arr, np.full(arr.shape, row), s=0.8, marker="|",
                linewidths=marker_lw, c="#161b22", rasterized=True,
            )
            xmax = max(xmax, float(arr.max()))
    n_units = max(len(units), 1)
    ax.set_ylim(-0.5, n_units - 0.5)

    # Participation overlay mapped into the top ~22% band of the y-range.
    t = plot_signals.get("t")
    part = plot_signals.get("participation_signal")
    if t is not None and part is not None:
        t_arr = np.asarray(t, dtype=float)
        p_arr = np.asarray(part, dtype=float)
        if t_arr.size and t_arr.size == p_arr.size:
            finite = np.isfinite(p_arr)
            pmax = float(np.max(p_arr[finite])) if finite.any() else 0.0
            if pmax > 0:
                top = n_units - 0.5
                band = n_units * 0.22
                y = top - band + np.clip(p_arr, 0.0, None) / pmax * band
                ax.plot(t_arr, y, lw=0.7, color="#c0392b", alpha=0.85)
                xmax = max(xmax, float(t_arr.max()))

    ax.set_xlim(0.0, xmax if xmax > 0 else 1.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, transparent=True)
    return out_path


def _cache_token(recording_key: str, source: str, well_id: str, well_sig: str) -> str:
    """Short, filename-safe hash keying a PNG to its well's staleness signature.

    Namespacing recording_key + source inside the hash means one well's re-run
    (new signature) re-renders exactly that PNG, never the other 23.
    """
    raw = "\x00".join((str(recording_key), str(source), str(well_id), str(well_sig)))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def render_overview_pngs(
    well_records: list["WellRecord"],
    cache_root: str | Path,
    *,
    well_sigs: dict[str, str],
    recording_key: str,
    source: str,
    **render_kw,
) -> dict[str, Path]:
    """Render (or reuse cached) raster PNGs for every ``status=="ok"`` well.

    Returns ``{well_id: png_path}``. A PNG whose signature-keyed filename already
    exists is reused untouched (the cache hit — no re-render). ``well_sigs`` maps
    each well to a cheap stat-based signature string computed by the caller.
    """
    cache_dir = Path(cache_root) / _RASTER_SUBDIR
    out: dict[str, Path] = {}
    for wr in well_records:
        if wr.status != "ok":
            continue
        token = _cache_token(recording_key, source, wr.well_id, well_sigs.get(wr.well_id, ""))
        png_path = cache_dir / f"{wr.well_id}_{token}.png"
        if not png_path.exists():
            try:
                render_well_png(wr, png_path, **render_kw)
            except Exception:  # noqa: BLE001 — one bad well shouldn't sink the grid
                continue
        out[wr.well_id] = png_path
    return out


def png_to_data_uri(path: str | Path) -> str:
    """Return a ``data:image/png;base64,…`` URI for embedding in an ``<img>``."""
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")
