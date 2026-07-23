"""Viewer library for the ActivityScan dashboard page.

Pure loaders + Plotly builders + a server-side PNG rasterizer for the
``Activity scan`` page (:mod:`yuxin_mea.dashboard.pages.activity_scan`). Like
:mod:`yuxin_mea.analysis.network_inspector` it imports **no Dash and no
SpikeInterface / torch** — everything is ``pd.read_parquet`` / ``np.load`` /
``json`` off what :func:`yuxin_mea.analysis.activity_scan.write_activity_scan`
wrote, plus matplotlib (object API only) for the plate thumbnails.

On-disk layout produced by ``scripts/run_activity_scan.py`` (note: no
``rec_name`` level — a well is assembled across all its configurations)::

    activity_scan_data/<recording_key>/
        manifest.json
        <well_id>/
            electrodes.parquet   maps.npz   stats.json   diagnostics.json
            yuxin_provenance.json

The four maps live on a (120, 220) electrode grid, NaN where the checkerboard
sweep never routed that electrode. Only ~49 % of the array is covered, so an
uncovered cell must read as *background*, never as zero — every heatmap and
thumbnail paints NaN in the paper colour, not the low end of the ramp.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from yuxin_mea.analysis.activity_scan import (
    GRID_COLS,
    GRID_ROWS,
    PITCH_UM,
    active_area_curve,
    load_activity_scan,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_SUBDIR = "activity_scan_data"

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
# Warm-paper design tokens (mirror theme.py / assets/styles.css).
_PAPER = "#f4f1ea"
_SURFACE = "#fbf9f3"
_INK = "#1c1a15"
_INK3 = "#84807a"
_LINE = "#d9d3c5"
_GREY_FILL = "#d9d3c5"

# Single-hue sequential ramps, light (paper) -> dark. Each MAP uses exactly one
# ramp, and the metric selector swaps between them — they are never shown side
# by side, so the relevant property is lightness monotonicity (satisfied by
# construction), not categorical CVD separation. NaN cells are painted with the
# paper colour underneath, so the ramp's light end is never mistaken for "no
# data": the ramp starts one step above paper.
SEQ_ACTIVITY = [  # firing rate — sage
    [0.0, "#e9efe7"], [0.25, "#a9cbb4"], [0.5, "#6fae86"],
    [0.75, "#3d8862"], [1.0, "#1f5a3f"],
]
SEQ_AMP = [  # spike amplitude magnitude — rust
    [0.0, "#f0e6df"], [0.25, "#e0b39a"], [0.5, "#cf7d54"],
    [0.75, "#a9512a"], [1.0, "#6f2f16"],
]
SEQ_NOISE = [  # electrode noise — amber
    [0.0, "#f1ead9"], [0.25, "#e0c58a"], [0.5, "#cda148"],
    [0.75, "#a17a2a"], [1.0, "#6a4f16"],
]

# Per-electrode maps offered by the metric selector. Each carries its ramp, a
# label, a unit, and how to turn the stored grid into the displayed scalar.
MAP_SPECS: dict[str, dict] = {
    "firing_rate": {"label": "Firing rate", "unit": "Hz", "scale": SEQ_ACTIVITY,
                    "source": "firing_rate", "transform": "identity"},
    "amplitude": {"label": "Spike amplitude", "unit": "µV", "scale": SEQ_AMP,
                  "source": "amp_median", "transform": "abs"},
    "noise": {"label": "Noise (MAD)", "unit": "µV", "scale": SEQ_NOISE,
              "source": "noise", "transform": "identity"},
    "active": {"label": "Active electrodes", "unit": "", "scale": SEQ_ACTIVITY,
               "source": "firing_rate", "transform": "active_mask"},
}
MAP_ORDER = ("firing_rate", "amplitude", "noise", "active")

# Well-level scalars offered to the plate grid, in menu order.
PLATE_METRICS = (
    "active_frac", "active_area_mm2", "fr_mean_active_hz", "fr_median_active_hz",
    "total_rate_hz", "amp_median_uv", "noise_median_uv", "fr_gini",
    "n_active", "coverage_frac", "largest_active_component",
)
METRIC_LABELS = {
    "active_frac": "active fraction",
    "active_area_mm2": "active area (mm²)",
    "fr_mean_active_hz": "mean FR over active (Hz)",
    "fr_median_active_hz": "median FR over active (Hz)",
    "fr_mean_all_hz": "mean FR, all electrodes (Hz)",
    "fr_max_hz": "max FR (Hz)",
    "total_rate_hz": "total spike rate (Hz)",
    "amp_median_uv": "median amplitude (µV)",
    "amp_p10_uv": "amplitude p10 (µV)",
    "noise_median_uv": "median noise (µV)",
    "fr_gini": "firing-rate Gini",
    "n_active": "active electrodes",
    "n_scanned": "electrodes scanned",
    "coverage_frac": "array coverage",
    "largest_active_component": "largest active blob",
    "activity_centroid_um": "activity centroid (µm)",
    "n_configs": "configurations",
    "duration_total_s": "total duration (s)",
    "online_total_spikes": "vendor online spikes",
}
# Plate metrics whose sequential ramp should be the noise/amber one.
_NOISE_METRICS = {"noise_median_uv"}
_AMP_METRICS = {"amp_median_uv", "amp_p10_uv"}


# --------------------------------------------------------------------------- #
# Discovery / loaders
# --------------------------------------------------------------------------- #
def output_root(analysis_root: Path | str, override: str | None = None) -> Path:
    if override:
        return Path(override)
    return Path(analysis_root) / DEFAULT_OUTPUT_SUBDIR


def recordings_with_activity_scan(analysis_root: Path | str,
                                  override: str | None = None) -> list[str]:
    """Recording keys that have at least one written well, from the output tree.

    A well counts once its ``stats.json`` exists. Discovery walks the output
    root rather than ``experiment_cache.json`` so the page only ever lists
    recordings that were actually processed.
    """
    root = output_root(analysis_root, override)
    if not root.exists():
        return []
    found: set[str] = set()
    # recording_key is <sample>/<date>/<plate>/<scan>/<run>, five levels; a well
    # dir sits one deeper. Glob for the stats marker and trim the two trailing
    # segments (well_id, filename).
    for stats in root.glob("*/*/*/*/*/well*/stats.json"):
        rec = stats.relative_to(root).parent.parent
        found.add(rec.as_posix())
    return sorted(found)


def wells_in_recording(analysis_root: Path | str, recording_key: str,
                       override: str | None = None) -> list[str]:
    """Written well ids for a recording (``well000`` … ), sorted."""
    rec_dir = output_root(analysis_root, override) / recording_key
    if not rec_dir.exists():
        return []
    return sorted(d.name for d in rec_dir.glob("well*")
                  if (d / "stats.json").exists())


def well_dir(analysis_root: Path | str, recording_key: str, well_id: str,
             override: str | None = None) -> Path:
    return output_root(analysis_root, override) / recording_key / well_id


def load_manifest(analysis_root: Path | str, recording_key: str,
                  override: str | None = None) -> dict | None:
    path = output_root(analysis_root, override) / recording_key / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def load_well(analysis_root: Path | str, recording_key: str, well_id: str,
              override: str | None = None):
    """Load one well's :class:`WellActivityResults`, or None if absent."""
    d = well_dir(analysis_root, recording_key, well_id, override)
    if not (d / "stats.json").exists():
        return None
    try:
        return load_activity_scan(d)
    except Exception as exc:  # noqa: BLE001 — a corrupt well must not break the page
        logger.warning("could not load %s: %s", d, exc)
        return None


def well_index_to_position(well_id: str) -> tuple[int, int]:
    """``well000`` -> (row 0, col 0); 24 wells in a 4×6 row-major plate."""
    n = int(str(well_id).replace("well", ""))
    return n // 6, n % 6


def well_name(well_id: str) -> str:
    """Plate label ``A1`` … ``D6`` from the well index (row letter, 1-based col)."""
    row, col = well_index_to_position(well_id)
    return f"{chr(ord('A') + row)}{col + 1}"


# --------------------------------------------------------------------------- #
# Map extraction
# --------------------------------------------------------------------------- #
def displayed_map(results, metric: str,
                  active_min_rate_hz: float | None = None) -> np.ndarray:
    """The (120, 220) array shown for ``metric``, NaN off coverage."""
    spec = MAP_SPECS[metric]
    grid = np.asarray(results.maps[spec["source"]], dtype=float).copy()
    coverage = np.asarray(results.maps["coverage"], dtype=float)
    off = ~(coverage > 0)
    transform = spec["transform"]
    if transform == "abs":
        grid = np.abs(grid)
    elif transform == "active_mask":
        thr = (active_min_rate_hz if active_min_rate_hz is not None
               else results.stats.get("active_min_rate_hz", 0.1))
        mask = np.where(off, np.nan, (grid >= thr).astype(float))
        return mask
    grid[off] = np.nan
    return grid


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _empty_figure(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(color=_INK3, size=13), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8))
    return fig


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "—"
        if abs(v) >= 1000 or (v != 0 and abs(v) < 0.01):
            return f"{v:.3g}"
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def fig_map(results, metric: str, active_min_rate_hz: float | None = None,
            height: int = 460) -> go.Figure:
    """Full-resolution electrode heatmap for one metric.

    Axes are in microns (the physical plate), not bin indices — so the activity
    centroid marker lands where it should. NaN cells render transparent over the
    paper background rather than at the ramp's low end.
    """
    if results is None:
        return _empty_figure("no activity-scan output for this well", height)
    if metric not in MAP_SPECS:
        return _empty_figure(f"unknown metric {metric!r}", height)

    spec = MAP_SPECS[metric]
    grid = displayed_map(results, metric, active_min_rate_hz)
    if not np.isfinite(grid).any():
        return _empty_figure("no covered electrodes", height)

    x = np.arange(GRID_COLS) * PITCH_UM
    y = np.arange(GRID_ROWS) * PITCH_UM
    is_mask = spec["transform"] == "active_mask"

    if is_mask:
        colorscale = [[0.0, _GREY_FILL], [1.0, SEQ_ACTIVITY[-1][1]]]
        zmin, zmax = 0.0, 1.0
        colorbar = dict(tickvals=[0, 1], ticktext=["inactive", "active"],
                        thickness=12, len=0.7)
        hover = "electrode %{x:.0f},%{y:.0f} µm<br>%{z:.0f}<extra></extra>"
    else:
        finite = grid[np.isfinite(grid)]
        zmin = float(np.nanpercentile(finite, 1))
        zmax = float(np.nanpercentile(finite, 99))
        if zmax <= zmin:
            zmax = zmin + 1e-6
        colorscale = spec["scale"]
        colorbar = dict(title=spec["unit"], thickness=12, len=0.7)
        hover = ("x %{x:.0f} µm · y %{y:.0f} µm<br>"
                 + spec["label"] + " %{z:.3g} " + spec["unit"] + "<extra></extra>")

    fig = go.Figure(go.Heatmap(
        z=grid, x=x, y=y, colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar=colorbar, hovertemplate=hover, hoverongaps=False,
    ))

    centroid = results.stats.get("activity_centroid_um")
    if centroid and all(np.isfinite(centroid)):
        fig.add_trace(go.Scatter(
            x=[centroid[0]], y=[centroid[1]], mode="markers",
            marker=dict(symbol="x", size=11, color=_INK, line=dict(width=2)),
            name="activity centroid",
            hovertemplate="activity centroid<br>%{x:.0f}, %{y:.0f} µm<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        height=height, margin=dict(l=48, r=16, t=28, b=40),
        title=dict(text=f"{spec['label']}"
                        + (f" ({spec['unit']})" if spec["unit"] else ""),
                   font=dict(size=13)),
    )
    fig.update_xaxes(title="x (µm)", constrain="domain", range=[0, GRID_COLS * PITCH_UM])
    fig.update_yaxes(title="y (µm)", scaleanchor="x", scaleratio=1,
                     range=[0, GRID_ROWS * PITCH_UM])
    return fig


def fig_histogram(results, metric: str, height: int = 260) -> go.Figure:
    """Distribution across electrodes for firing rate / amplitude / noise."""
    if results is None or len(results.electrodes) == 0:
        return _empty_figure("no electrodes", height)
    col_map = {"firing_rate": ("firing_rate_hz", SEQ_ACTIVITY, "Firing rate (Hz)"),
               "amplitude": ("amp_median_uv", SEQ_AMP, "Spike amplitude (µV)"),
               "noise": ("noise_uv", SEQ_NOISE, "Noise (µV)")}
    if metric not in col_map:
        return _empty_figure(f"no histogram for {metric!r}", height)
    col, scale, label = col_map[metric]
    v = results.electrodes[col].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if metric == "amplitude":
        v = np.abs(v)
        label = "Spike amplitude |µV|"
    if v.size == 0:
        return _empty_figure("no finite values", height)

    fig = go.Figure(go.Histogram(x=v, nbinsx=60, marker=dict(
        color=scale[3][1], line=dict(color=_SURFACE, width=0.5))))
    fig.update_layout(height=height, margin=dict(l=48, r=16, t=24, b=40),
                      bargap=0.02, showlegend=False,
                      title=dict(text=label, font=dict(size=12)))
    fig.update_xaxes(title=label)
    fig.update_yaxes(title="electrodes")
    return fig


def fig_active_area_curve(results, height: int = 260) -> go.Figure:
    """Active fraction vs the firing-rate threshold, with the default cut marked.

    The 0.1 Hz default is 3 events in 30 s, so 'active' is a choice, not a fact;
    this makes the sensitivity to that choice visible.
    """
    if results is None or len(results.electrodes) == 0:
        return _empty_figure("no electrodes", height)
    thr, frac = active_area_curve(results.electrodes)
    fig = go.Figure(go.Scatter(x=thr, y=frac, mode="lines",
                               line=dict(color=SEQ_ACTIVITY[3][1], width=2)))
    default_cut = results.stats.get("active_min_rate_hz", 0.1)
    fig.add_vline(x=default_cut, line=dict(color=_INK3, width=1, dash="dot"),
                  annotation_text=f"{default_cut:g} Hz", annotation_font_size=10)
    fig.update_layout(height=height, margin=dict(l=48, r=16, t=24, b=40),
                      showlegend=False,
                      title=dict(text="Active fraction vs threshold",
                                 font=dict(size=12)))
    fig.update_xaxes(title="firing-rate cut (Hz)", type="log")
    fig.update_yaxes(title="fraction active", range=[0, 1])
    return fig


def fig_config_coverage(results, height: int = 220) -> go.Figure:
    """Per-configuration electrode count + noise, from diagnostics."""
    if results is None:
        return _empty_figure("no diagnostics", height)
    per = (results.diagnostics or {}).get("per_config") or []
    rows = [p for p in per if "n_channels" in p]
    if not rows:
        return _empty_figure("no per-configuration record", height)
    labels = [p["store"] for p in rows]
    n_ch = [p["n_channels"] for p in rows]
    noise = [p.get("noise_median_uv", np.nan) for p in rows]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=n_ch, marker_color=SEQ_ACTIVITY[2][1],
                         name="routed electrodes",
                         hovertemplate="%{x}<br>%{y} electrodes<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=noise, yaxis="y2", mode="markers+lines",
                             marker=dict(color=SEQ_NOISE[3][1], size=7),
                             line=dict(color=SEQ_NOISE[3][1], width=1),
                             name="median noise (µV)",
                             hovertemplate="%{x}<br>%{y:.2f} µV<extra></extra>"))
    fig.update_layout(
        height=height, margin=dict(l=48, r=48, t=24, b=60),
        title=dict(text="Per-configuration coverage", font=dict(size=12)),
        legend=dict(orientation="h", y=1.18, x=0),
        yaxis=dict(title="electrodes"),
        yaxis2=dict(title="noise µV", overlaying="y", side="right",
                    showgrid=False),
    )
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    return fig


def well_stats_table(results) -> list[dict[str, str]]:
    """[{metric, value}] rows for the drill-in stats table."""
    if results is None:
        return []
    stats = results.stats
    rows: list[dict[str, str]] = []
    order = ["n_scanned", "coverage_frac", "n_active", "active_frac",
             "active_area_mm2", "largest_active_component", "fr_mean_active_hz",
             "fr_median_active_hz", "fr_max_hz", "total_rate_hz", "fr_gini",
             "amp_median_uv", "noise_median_uv", "n_configs",
             "duration_total_s", "online_total_spikes"]
    for key in order:
        if key not in stats:
            continue
        val = stats[key]
        if key == "activity_centroid_um" and isinstance(val, list):
            val = f"({_fmt(val[0])}, {_fmt(val[1])})"
        rows.append({"metric": METRIC_LABELS.get(key, key), "value": _fmt(val)})
    return rows


# --------------------------------------------------------------------------- #
# Plate-level summary
# --------------------------------------------------------------------------- #
def plate_scalars(analysis_root: Path | str, recording_key: str, metric: str,
                  override: str | None = None) -> dict[str, float]:
    """{well_id: scalar} for a plate metric, over every written well."""
    out: dict[str, float] = {}
    for well_id in wells_in_recording(analysis_root, recording_key, override):
        d = well_dir(analysis_root, recording_key, well_id, override)
        try:
            stats = json.loads((d / "stats.json").read_text())
        except (OSError, ValueError):
            continue
        val = stats.get(metric)
        if isinstance(val, (int, float)) and np.isfinite(val):
            out[well_id] = float(val)
    return out


def _scale_for_metric(metric: str) -> list:
    if metric in _NOISE_METRICS:
        return SEQ_NOISE
    if metric in _AMP_METRICS:
        return SEQ_AMP
    return SEQ_ACTIVITY


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _sample_scale(t: float, scale: list) -> str:
    t = float(min(1.0, max(0.0, t)))
    for i in range(len(scale) - 1):
        t0, c0 = scale[i]
        t1, c1 = scale[i + 1]
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            r0, g0, b0 = _hex_to_rgb(c0)
            r1, g1, b1 = _hex_to_rgb(c1)
            return "#%02x%02x%02x" % (round(r0 + (r1 - r0) * f),
                                      round(g0 + (g1 - g0) * f),
                                      round(b0 + (b1 - b0) * f))
    return scale[-1][1]


def plate_colors(values: dict[str, float], metric: str) -> dict[str, str]:
    """{well_id: hex} normalising a plate metric across its wells (amp on |v|)."""
    if not values:
        return {}
    scale = _scale_for_metric(metric)
    vals = {k: (abs(v) if metric in _AMP_METRICS else v) for k, v in values.items()}
    lo = min(vals.values())
    hi = max(vals.values())
    span = hi - lo if hi > lo else 1.0
    return {k: _sample_scale((v - lo) / span, scale) for k, v in vals.items()}


# --------------------------------------------------------------------------- #
# Server-side plate thumbnail (matplotlib object API, no pyplot)
# --------------------------------------------------------------------------- #
_RASTER_SUBDIR = "activity_png"


def _thumbnail_path(cache_dir: Path, results_dir: Path, metric: str,
                    active_min_rate_hz: float) -> Path:
    stat = results_dir / "maps.npz"
    try:
        sig = stat.stat()
        token = f"{results_dir}|{metric}|{active_min_rate_hz}|{sig.st_mtime_ns}|{sig.st_size}"
    except OSError:
        token = f"{results_dir}|{metric}|{active_min_rate_hz}"
    name = hashlib.sha1(token.encode()).hexdigest()[:16]
    return cache_dir / _RASTER_SUBDIR / f"{name}.png"


def render_well_png(results, metric: str, out_path: str | Path, *,
                    active_min_rate_hz: float | None = None,
                    w_px: int = 220, h_px: int = 130) -> Path:
    """Rasterise one well's map to a small PNG for the plate grid (read-only).

    Uses the matplotlib object API (``Figure`` + ``FigureCanvasAgg``), never
    pyplot — pyplot's global figure registry is unsafe on a Dash server serving
    concurrent callbacks. NaN cells are left transparent so the page background
    shows through as 'unscanned'.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spec = MAP_SPECS[metric]
    grid = displayed_map(results, metric, active_min_rate_hz)
    masked = np.ma.masked_invalid(grid)

    cmap = LinearSegmentedColormap.from_list(
        f"seq_{metric}", [c for _, c in spec["scale"]])
    cmap = cmap.copy()
    cmap.set_bad(alpha=0.0)

    fig = Figure(figsize=(w_px / 100, h_px / 100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    if masked.count():
        finite = grid[np.isfinite(grid)]
        vmin = float(np.nanpercentile(finite, 2))
        vmax = float(np.nanpercentile(finite, 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap,
                  vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.savefig(out_path, transparent=True, dpi=100)
    return out_path


def well_png_data_uri(results, metric: str, cache_dir: Path, results_dir: Path,
                      active_min_rate_hz: float | None = None) -> str:
    """Cached base64 PNG for a well thumbnail; empty string when nothing to draw."""
    if results is None:
        return ""
    thr = (active_min_rate_hz if active_min_rate_hz is not None
           else results.stats.get("active_min_rate_hz", 0.1))
    path = _thumbnail_path(Path(cache_dir), Path(results_dir), metric, thr)
    if not path.exists():
        try:
            render_well_png(results, metric, path, active_min_rate_hz=thr)
        except Exception as exc:  # noqa: BLE001 — a bad thumbnail must not break the grid
            logger.warning("thumbnail render failed for %s: %s", results_dir, exc)
            return ""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
