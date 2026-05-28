"""Per-well interactive HTML for inspecting the ML burst detector.

Produces one standalone Plotly HTML per well where ``ml_burst_detection`` has
completed. Each HTML lets you re-judge a well manually: it ties the spike
raster, the three event-hierarchy lanes (burstlets / network_bursts /
superbursts), the driving signals that fed HDBSCAN, the per-bin HDBSCAN
cluster labels, and a 2-D feature-space scatter all together — so you can
tell *why* a stretch was (or was not) called as a burst, and judge whether
visually-subtle real bursts were missed.

Discovery mirrors ``visualize_bursts.py``: reads ``pipeline_cache.json`` for
wells where auto_curation AND ml_burst_detection are complete, and
``experiment_cache.json`` for well_name / groupname. Per-well dir is taken
from the pipeline cache entry (``tasks.ml_burst_detection.output_path``),
not reconstructed.

Files consumed per well:
  - <ml_burst_dir>/{burstlets,network_bursts,superbursts}.pkl
  - <ml_burst_dir>/plot_signals.npy
  - <ml_burst_dir>/diagnostics.json
  - <ml_burst_dir>/metrics.json
  - <ml_burst_dir>/debug_trace.pkl   (optional; skips cluster panels if absent)
  - <auto_curation_dir>/curated_spike_times.npy

Output: {output_dir}/{recording_key}/ml_burst_inspect/{well_name}_{well_id}.html
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger("inspect_ml_bursts")

ML_TASK = "ml_burst_detection"
CUR_TASK = "auto_curation"

PLATE_COLS = 6

EVENT_TRACKS = [
    ("burstlets", "Burstlets", "rgba(31, 119, 180, 0.55)"),
    ("network_bursts", "Network bursts", "rgba(255, 127, 14, 0.75)"),
    ("superbursts", "Superbursts", "rgba(148, 103, 189, 0.55)"),
]
RASTER_COLOR = "rgba(40, 40, 40, 0.85)"
MAX_RASTER_POINTS_PER_WELL = 12_000

# Distinguishable hues for non-burst, non-noise HDBSCAN clusters.
CLUSTER_PALETTE = [
    "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
    "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]
NOISE_COLOR = "rgba(180, 180, 180, 0.55)"
BURST_COLOR = "#d62728"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class WellBundle:
    rec_key: str
    rec_name: str
    well_id: str
    well_name: str
    groupname: str
    spike_times: dict[str, np.ndarray] | None
    plot_signals: dict[str, Any] | None
    diagnostics: dict[str, Any] | None
    metrics: dict[str, Any] | None
    events: dict[str, list[dict[str, Any]]]
    trace: Any | None  # MLBurstTrace or None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _load_pipeline_cache(analysis_root: Path) -> dict[str, Any]:
    path = analysis_root / "pipeline_cache.json"
    if not path.exists():
        raise SystemExit(f"pipeline_cache.json not found at {path}")
    with path.open() as fh:
        return json.load(fh)


def _load_experiment_cache(analysis_root: Path) -> dict[str, Any]:
    path = analysis_root / "experiment_cache.json"
    if not path.exists():
        logger.warning("experiment_cache.json not found at %s; using fallback names", path)
        return {}
    with path.open() as fh:
        return json.load(fh)


def _make_resolver(search_roots: list[Path]):
    def _resolve(rel: str | None) -> Path | None:
        if not rel:
            return None
        p = Path(rel)
        if p.is_absolute():
            return p if p.exists() else None
        for base in search_roots:
            cand = (base / p).resolve()
            if cand.exists():
                return cand
        return None
    return _resolve


def _read_event_table(pkl_path: Path) -> list[dict[str, Any]]:
    """Load an event DataFrame and return a list of plain dicts.

    Keeps any (str, int, float, bool) column, so ML-specific scores
    (posterior_peak, posterior_mean, llr_aggregate, ff_peak, participation,
    fragment_count, total_spikes, etc.) survive for hover display.
    """
    if not pkl_path.exists():
        return []
    try:
        df = pd.read_pickle(pkl_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", pkl_path, exc)
        return []
    if df is None or df.empty or "start" not in df or "end" not in df:
        return []
    rows: list[dict[str, Any]] = []
    for r in df.to_dict(orient="records"):
        try:
            s = float(r["start"])
            e = float(r["end"])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(s) and np.isfinite(e)) or e <= s:
            continue
        clean: dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, float) and not np.isfinite(v):
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
        clean["start"], clean["end"] = s, e
        rows.append(clean)
    return rows


def _load_trace(pkl_path: Path):
    if not pkl_path.exists():
        return None
    try:
        with pkl_path.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load debug_trace %s: %s", pkl_path, exc)
        return None


def _load_well_bundle(
    rec_key: str,
    rec_name: str,
    well_id: str,
    entry: dict[str, Any],
    well_meta: dict[str, Any],
    resolve,
) -> WellBundle | None:
    tasks = entry.get("tasks", {})
    ml_dir = resolve(tasks.get(ML_TASK, {}).get("output_path"))
    if ml_dir is None:
        logger.warning("Missing ml_burst_detection output for %s/%s/%s", rec_key, rec_name, well_id)
        return None
    cur_dir = resolve(tasks.get(CUR_TASK, {}).get("output_path"))

    spike_times = None
    if cur_dir is not None:
        sp = cur_dir / "curated_spike_times.npy"
        if sp.exists():
            try:
                spike_times = np.load(sp, allow_pickle=True).item()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load spike times %s: %s", sp, exc)

    plot_signals = None
    ps_path = ml_dir / "plot_signals.npy"
    if ps_path.exists():
        try:
            plot_signals = np.load(ps_path, allow_pickle=True).item()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load plot_signals %s: %s", ps_path, exc)

    events = {key: _read_event_table(ml_dir / f"{key}.pkl") for key, *_ in EVENT_TRACKS}
    diagnostics = _load_json(ml_dir / "diagnostics.json")
    metrics = _load_json(ml_dir / "metrics.json")
    trace = _load_trace(ml_dir / "debug_trace.pkl")

    return WellBundle(
        rec_key=rec_key,
        rec_name=rec_name,
        well_id=well_id,
        well_name=well_meta.get("well_name", _fallback_well_name(well_id)),
        groupname=well_meta.get("groupname", "?"),
        spike_times=spike_times,
        plot_signals=plot_signals,
        diagnostics=diagnostics,
        metrics=metrics,
        events=events,
        trace=trace,
    )


def _fallback_well_name(well_id: str) -> str:
    try:
        n = int(well_id.replace("well", ""))
    except ValueError:
        return "?"
    row = chr(ord("A") + n // PLATE_COLS)
    col = (n % PLATE_COLS) + 1
    return f"{row}{col}"


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def _firing_rate(arr: np.ndarray, t_end: float) -> float:
    return float(arr.size / t_end) if t_end > 0 and arr.size else 0.0


def _raster_traces(spike_times: dict[str, np.ndarray]) -> tuple[list[go.Scattergl], float, int]:
    items: list[tuple[str, np.ndarray]] = []
    for uid, st in spike_times.items():
        arr = np.asarray(st, dtype=float)
        arr = arr[np.isfinite(arr)]
        items.append((str(uid), arr))
    t_end = max((float(a.max()) if a.size else 0.0 for _, a in items), default=0.0)
    items.sort(key=lambda kv: _firing_rate(kv[1], t_end), reverse=True)

    n_units = len(items) or 1
    per_unit_cap = max(40, MAX_RASTER_POINTS_PER_WELL // n_units)
    traces: list[go.Scattergl] = []
    for rank, (uid, spikes) in enumerate(items):
        if spikes.size == 0:
            continue
        if spikes.size > per_unit_cap:
            stride = int(np.ceil(spikes.size / per_unit_cap))
            spikes = spikes[::stride]
        rate = _firing_rate(spikes, t_end)
        cd = np.column_stack([
            np.full(spikes.size, uid, dtype=object),
            np.full(spikes.size, rate, dtype=float),
        ])
        traces.append(go.Scattergl(
            x=spikes,
            y=np.full(spikes.size, rank, dtype=float),
            mode="markers",
            marker=dict(size=4.0, color=RASTER_COLOR, symbol="line-ns-open",
                        line=dict(width=0.6, color=RASTER_COLOR)),
            customdata=cd,
            hovertemplate=(
                "unit %{customdata[0]} (FR %{customdata[1]:.2f} Hz)"
                "<br>t = %{x:.4f}s<extra></extra>"
            ),
            name=uid,
            showlegend=False,
        ))
    return traces, t_end, len(items)


def _event_traces(events: dict[str, list[dict[str, Any]]]) -> list[go.Scatter]:
    """Three lanes (y=0,1,2). Each event drawn as filled rectangle with hover
    showing ML-specific scores."""
    traces: list[go.Scatter] = []
    for lane_idx, (key, label, color) in enumerate(EVENT_TRACKS):
        evs = events.get(key) or []
        if not evs:
            traces.append(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(width=0, color=color),
                name=f"{label} (0)",
                legendgroup=key, showlegend=True,
            ))
            continue
        y_center = float(lane_idx)
        y_lo, y_hi = y_center - 0.4, y_center + 0.4
        xs: list[float | None] = []
        ys: list[float | None] = []
        cd: list[list[Any]] = []
        for ev in evs:
            s, e = ev["start"], ev["end"]
            xs.extend([s, e, e, s, s, None])
            ys.extend([y_lo, y_lo, y_hi, y_hi, y_lo, None])
            base = [
                label,
                s, e,
                ev.get("duration_s", e - s),
                ev.get("participation", float("nan")),
                ev.get("total_spikes", float("nan")),
                ev.get("peak_synchrony", float("nan")),
                ev.get("posterior_peak", float("nan")),
                ev.get("posterior_mean", float("nan")),
                ev.get("llr_aggregate", float("nan")),
                ev.get("ff_peak", float("nan")),
            ]
            cd.extend([base, base, base, base, base, [None] * len(base)])
        traces.append(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            line=dict(width=0.5, color=color), fillcolor=color,
            customdata=cd,
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>start=%{customdata[1]:.3f}s  end=%{customdata[2]:.3f}s"
                "<br>duration=%{customdata[3]:.3f}s"
                "<br>participation=%{customdata[4]:.3f}"
                "<br>total_spikes=%{customdata[5]}"
                "<br>peak_synchrony=%{customdata[6]:.3f}"
                "<br>posterior_peak=%{customdata[7]:.3f}"
                "<br>posterior_mean=%{customdata[8]:.3f}"
                "<br>llr_aggregate=%{customdata[9]:.3f}"
                "<br>ff_peak=%{customdata[10]:.3f}"
                "<extra></extra>"
            ),
            name=f"{label} ({len(evs)})",
            legendgroup=key, showlegend=True,
        ))
    return traces


def _signal_traces(
    plot_signals: dict[str, Any] | None,
    merge_threshold: float | None,
) -> list[go.Scatter]:
    if not plot_signals:
        return []
    t = np.asarray(plot_signals.get("t"), dtype=float) if plot_signals.get("t") is not None else None
    if t is None or t.size < 2:
        return []
    out: list[go.Scatter] = []
    series = [
        ("ranking_signal", "ranking_signal", "#d62728", True),
        ("posterior_matrix_mean", "post_mean (units)", "#1f77b4", True),
        ("participation_signal", "participation", "#2ca02c", "legendonly"),
        ("ff_signal", "FF1", "#ff7f0e", "legendonly"),
        ("llr_signal", "LLR_mean", "#9467bd", "legendonly"),
    ]
    for key, label, color, visible in series:
        v = plot_signals.get(key)
        if v is None:
            continue
        arr = np.asarray(v, dtype=float)
        if arr.size != t.size:
            continue
        out.append(go.Scatter(
            x=t, y=arr, mode="lines", name=label,
            line=dict(color=color, width=1.2),
            visible=visible,
            legendgroup="signals",
            hovertemplate=f"{label}<br>t=%{{x:.3f}}s<br>v=%{{y:.4f}}<extra></extra>",
        ))
    if merge_threshold is not None and np.isfinite(merge_threshold):
        out.append(go.Scatter(
            x=[float(t.min()), float(t.max())],
            y=[merge_threshold, merge_threshold],
            mode="lines",
            line=dict(color="#d62728", width=1.0, dash="dash"),
            name=f"merge_threshold ({merge_threshold:.3f})",
            legendgroup="signals",
            hoverinfo="skip",
        ))
    return out


def _hdbscan_strip(trace, diagnostics: dict[str, Any] | None) -> go.Heatmap | None:
    if trace is None or getattr(trace, "hdbscan_labels", None) is None:
        return None
    labels = np.asarray(trace.hdbscan_labels)
    t = np.asarray(trace.t_centers, dtype=float)
    if labels.size == 0 or t.size != labels.size:
        return None
    burst_label = None
    if diagnostics is not None:
        bl = diagnostics.get("cluster_burst_label")
        if bl is not None and bl != -1:
            burst_label = int(bl)

    unique = sorted(set(int(v) for v in labels.tolist()))
    # Assign a categorical colour per label. Map labels -> small integer codes
    # so we can use a discrete colorscale.
    code_of: dict[int, int] = {}
    colors: list[str] = []
    palette_iter = iter(CLUSTER_PALETTE)
    for lbl in unique:
        code_of[lbl] = len(code_of)
        if lbl == -1:
            colors.append(NOISE_COLOR)
        elif burst_label is not None and lbl == burst_label:
            colors.append(BURST_COLOR)
        else:
            try:
                colors.append(next(palette_iter))
            except StopIteration:
                colors.append("#666666")

    z = np.array([[code_of[int(v)] for v in labels.tolist()]])
    n_codes = len(colors)
    if n_codes == 1:
        colorscale = [[0.0, colors[0]], [1.0, colors[0]]]
    else:
        colorscale = []
        for i, c in enumerate(colors):
            colorscale.append([i / n_codes, c])
            colorscale.append([(i + 1) / n_codes, c])

    inv_code = {v: k for k, v in code_of.items()}
    cd = np.array([[inv_code[c] for c in row] for row in z])

    return go.Heatmap(
        x=t, y=[0],
        z=z, customdata=cd,
        colorscale=colorscale,
        zmin=-0.5, zmax=n_codes - 0.5,
        showscale=False,
        hovertemplate="t=%{x:.3f}s<br>cluster=%{customdata}<extra></extra>",
        name="HDBSCAN label",
    )


def _cluster_color_map(
    labels: np.ndarray, burst_label: int | None
) -> tuple[dict[int, str], dict[int, str]]:
    """Return (color_by_label, display_name_by_label) for HDBSCAN labels.

    Noise (-1) is gray; the cluster matching ``burst_label`` is the burst red;
    other clusters draw sequentially from ``CLUSTER_PALETTE``.
    """
    color_by: dict[int, str] = {}
    name_by: dict[int, str] = {}
    palette_iter = iter(CLUSTER_PALETTE)
    for lbl in sorted({int(v) for v in labels.tolist()}):
        if lbl == -1:
            color_by[lbl] = NOISE_COLOR
            name_by[lbl] = "noise (-1)"
        elif burst_label is not None and lbl == burst_label:
            color_by[lbl] = BURST_COLOR
            name_by[lbl] = f"cluster {lbl} (burst)"
        else:
            try:
                color_by[lbl] = next(palette_iter)
            except StopIteration:
                color_by[lbl] = "#666666"
            name_by[lbl] = f"cluster {lbl}"
    return color_by, name_by


def _znorm_from_trace(trace) -> np.ndarray | None:
    """Reconstruct the z-normed feature matrix HDBSCAN actually saw.

    Uses ``trace.scaler_mean`` / ``trace.scaler_std`` saved by the detector.
    Falls back to whole-matrix z-norm if those aren't present (only happens
    when consuming traces from a very early debug build).
    """
    X = np.asarray(trace.feature_matrix, dtype=float) if trace.feature_matrix is not None else None
    if X is None or X.ndim != 2 or X.shape[0] == 0:
        return None
    mu = getattr(trace, "scaler_mean", None)
    sd = getattr(trace, "scaler_std", None)
    if mu is None or sd is None:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
    mu = np.asarray(mu, float)
    sd = np.asarray(sd, float)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (X - mu) / sd


def _compute_pca_axes(trace) -> tuple[np.ndarray, str, str] | None:
    """Return (coords_n×2, x_label, y_label) for the PCA(2) view, or None."""
    Z = _znorm_from_trace(trace)
    if Z is None or Z.shape[0] < 3:
        return None
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return None
    n_comp = min(2, Z.shape[1])
    if n_comp < 2:
        return None
    pca = PCA(n_components=2, random_state=0).fit(Z)
    coords = pca.transform(Z)
    fnames = list(trace.feature_names or [])

    def _label(pc: int) -> str:
        comp = pca.components_[pc]
        ev = float(pca.explained_variance_ratio_[pc]) * 100
        order = np.argsort(np.abs(comp))[::-1][:3]
        if fnames and all(0 <= int(i) < len(fnames) for i in order):
            tops = ", ".join(f"{fnames[int(i)]}{comp[int(i)]:+.2f}" for i in order)
            return f"PC{pc+1} ({ev:.1f}%): {tops}"
        return f"PC{pc+1} ({ev:.1f}%)"

    return coords, _label(0), _label(1)


def _compute_umap_axes(
    trace,
    max_bins: int = 5000,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
) -> tuple[np.ndarray, str, str, np.ndarray] | None:
    """Return (coords_n×2, x_label, y_label, kept_indices) for UMAP(2), or None.

    Subsamples to ``max_bins`` rows when the input is larger so per-well UMAP
    stays under a few seconds. Returns None when ``umap-learn`` is not
    importable, so callers can degrade gracefully.
    """
    Z = _znorm_from_trace(trace)
    if Z is None or Z.shape[0] < 10:
        return None
    try:
        import umap  # type: ignore
    except ImportError:
        return None
    n = Z.shape[0]
    if n > max_bins:
        stride = int(np.ceil(n / max_bins))
        kept = np.arange(0, n, stride)[:max_bins]
    else:
        kept = np.arange(n)
    Zsub = Z[kept]
    nn = max(2, min(int(n_neighbors), Zsub.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2, n_neighbors=nn, min_dist=float(min_dist),
        metric="euclidean", random_state=42,
    )
    coords = reducer.fit_transform(Zsub)
    return coords, "UMAP-1", "UMAP-2", kept


def _projection_scatter_traces(
    coords: np.ndarray,        # (n, 2)
    labels: np.ndarray,        # (n,)
    t_centers: np.ndarray,     # (n,)
    burst_label: int | None,
    x_axis_label: str,
    y_axis_label: str,
    legend_group: str,         # "pca" or "umap" — separate so per-panel legends don't fight
) -> list[go.Scattergl]:
    color_by, name_by = _cluster_color_map(labels, burst_label)
    traces: list[go.Scattergl] = []
    for lbl in sorted(color_by.keys()):
        mask = labels == lbl
        if not mask.any():
            continue
        cd = (t_centers[mask]
              if t_centers is not None and t_centers.size == coords.shape[0]
              else np.zeros(int(mask.sum())))
        traces.append(go.Scattergl(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers",
            marker=dict(
                size=5 if lbl != -1 else 3,
                color=color_by[lbl],
                opacity=0.85 if lbl != -1 else 0.35,
                line=dict(width=0),
            ),
            customdata=cd,
            hovertemplate=(
                f"{name_by[lbl]}<br>{x_axis_label}=%{{x:.3f}}<br>{y_axis_label}=%{{y:.3f}}"
                "<br>t=%{customdata:.3f}s<extra></extra>"
            ),
            name=f"{name_by[lbl]} (n={int(mask.sum())})",
            legendgroup=legend_group,
            showlegend=True,
        ))
    return traces


def _ranking_hist_traces(
    trace, diagnostics: dict[str, Any] | None,
) -> tuple[list[go.Histogram], str, float | None]:
    """Return (per-cluster histograms over the ranking feature, x_label,
    threshold-for-vline).

    Shows the 1-D distribution the detector actually ranks clusters on, so the
    "which cluster has the highest mean ranking score" decision is visible
    independent of any 2-D projection.
    """
    if trace is None or trace.feature_matrix is None or trace.hdbscan_labels is None:
        return [], "", None
    X = np.asarray(trace.feature_matrix, float)
    fnames = list(trace.feature_names or [])
    if not fnames:
        return [], "", None
    rank_name = (diagnostics or {}).get("ranking_feature", "post_frac_gt_0_5")
    if rank_name not in fnames:
        rank_name = fnames[0]
    col = X[:, fnames.index(rank_name)]
    labels = np.asarray(trace.hdbscan_labels)
    burst_label = None
    if diagnostics is not None:
        bl = diagnostics.get("cluster_burst_label")
        if bl is not None and bl != -1:
            burst_label = int(bl)
    color_by, name_by = _cluster_color_map(labels, burst_label)
    finite = col[np.isfinite(col)]
    if not finite.size:
        return [], rank_name, None
    nbins = 40
    edges_lo, edges_hi = float(finite.min()), float(finite.max())
    bin_size = (edges_hi - edges_lo) / nbins if edges_hi > edges_lo else 1.0
    traces: list[go.Histogram] = []
    for lbl in sorted(color_by.keys()):
        mask = labels == lbl
        if not mask.any():
            continue
        traces.append(go.Histogram(
            x=col[mask],
            xbins=dict(start=edges_lo, end=edges_hi, size=bin_size),
            opacity=0.55,
            marker=dict(color=color_by[lbl]),
            name=f"{name_by[lbl]} (n={int(mask.sum())})",
            legendgroup="hist",
            showlegend=True,
        ))
    threshold: float | None = None
    if diagnostics is not None:
        for key in ("merge_threshold", "fallback_threshold"):
            v = diagnostics.get(key)
            try:
                if v is not None:
                    threshold = float(v)
                    break
            except (TypeError, ValueError):
                pass
    return traces, rank_name, threshold


def _cluster_table(diagnostics: dict[str, Any] | None, trace) -> go.Table | None:
    if diagnostics is None:
        return None
    ranking = diagnostics.get("cluster_ranking") or {}
    if not ranking:
        return None
    burst_label = diagnostics.get("cluster_burst_label")
    burst_label_int = None if burst_label in (None, -1) else int(burst_label)

    n_bins_by_label: dict[int, int] = {}
    if trace is not None and getattr(trace, "hdbscan_labels", None) is not None:
        arr = np.asarray(trace.hdbscan_labels).astype(int)
        for lbl in np.unique(arr):
            n_bins_by_label[int(lbl)] = int((arr == lbl).sum())

    rows = []
    for lbl_str, score in ranking.items():
        try:
            lbl = int(lbl_str)
        except (TypeError, ValueError):
            continue
        is_burst = "yes" if burst_label_int is not None and lbl == burst_label_int else ""
        rows.append((lbl, n_bins_by_label.get(lbl, 0), float(score), is_burst))
    rows.sort(key=lambda r: -r[2])

    if not rows:
        return None
    return go.Table(
        header=dict(
            values=["<b>cluster</b>", "<b>n_bins</b>", "<b>mean rank score</b>", "<b>burst?</b>"],
            fill_color="#eeeeee", align="left", font=dict(size=11),
        ),
        cells=dict(
            values=[
                [r[0] for r in rows],
                [r[1] for r in rows],
                [f"{r[2]:.4f}" for r in rows],
                [r[3] for r in rows],
            ],
            align="left", font=dict(size=11), height=22,
        ),
    )


def _metrics_table(metrics: dict[str, Any] | None) -> go.Table | None:
    if not metrics:
        return None
    rows = []
    for level in ("burstlets", "network_bursts", "superbursts"):
        m = metrics.get(level) or {}
        if not m:
            continue
        count = m.get("count", 0)
        rate = m.get("rate", float("nan"))
        dur = (m.get("duration") or {}).get("mean", float("nan"))
        part = (m.get("participation") or {}).get("mean", float("nan"))
        ps = (m.get("peak_synchrony") or {}).get("mean", float("nan"))
        rows.append((level, count, rate, dur, part, ps))
    if not rows:
        return None

    def _fmt(v):
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return "—"

    return go.Table(
        header=dict(
            values=["<b>level</b>", "<b>count</b>", "<b>rate/min</b>",
                    "<b>mean dur (s)</b>", "<b>mean part.</b>",
                    "<b>mean peak sync</b>"],
            fill_color="#eeeeee", align="left", font=dict(size=11),
        ),
        cells=dict(
            values=[
                [r[0] for r in rows],
                [r[1] for r in rows],
                [_fmt(r[2]) for r in rows],
                [_fmt(r[3]) for r in rows],
                [_fmt(r[4]) for r in rows],
                [_fmt(r[5]) for r in rows],
            ],
            align="left", font=dict(size=11), height=22,
        ),
    )


def build_well_figure(well: WellBundle) -> go.Figure:
    has_trace = well.trace is not None
    has_signals = bool(well.plot_signals)
    has_strip = has_trace and getattr(well.trace, "hdbscan_labels", None) is not None
    has_hist = (
        has_trace
        and getattr(well.trace, "feature_matrix", None) is not None
        and getattr(well.trace, "hdbscan_labels", None) is not None
    )

    # Compute projections up front so layout knows which panels to allocate.
    pca_result = _compute_pca_axes(well.trace) if has_trace else None
    umap_result = _compute_umap_axes(well.trace) if has_trace else None
    has_pca = pca_result is not None
    has_umap = umap_result is not None

    # Row layout. Time-domain rows (raster/events/signals/strip) share an x-axis;
    # the projections rows are independent xy; the hist row is independent xy;
    # tables come last.
    rows_spec = []
    row_heights = []
    row_meanings = []

    rows_spec.append([{"colspan": 2, "type": "xy"}, None])
    row_heights.append(0.26); row_meanings.append("raster")

    rows_spec.append([{"colspan": 2, "type": "xy"}, None])
    row_heights.append(0.11); row_meanings.append("events")

    if has_signals:
        rows_spec.append([{"colspan": 2, "type": "xy"}, None])
        row_heights.append(0.18); row_meanings.append("signals")

    if has_strip:
        rows_spec.append([{"colspan": 2, "type": "heatmap"}, None])
        row_heights.append(0.05); row_meanings.append("strip")

    if has_pca and has_umap:
        rows_spec.append([{"type": "xy"}, {"type": "xy"}])
        row_heights.append(0.22); row_meanings.append("pca_umap")
    elif has_pca:
        rows_spec.append([{"colspan": 2, "type": "xy"}, None])
        row_heights.append(0.20); row_meanings.append("pca_only")
    elif has_umap:
        rows_spec.append([{"colspan": 2, "type": "xy"}, None])
        row_heights.append(0.20); row_meanings.append("umap_only")

    if has_hist:
        rows_spec.append([{"colspan": 2, "type": "xy"}, None])
        row_heights.append(0.14); row_meanings.append("hist")

    rows_spec.append([{"type": "table"}, {"type": "table"}])
    row_heights.append(0.18); row_meanings.append("tables")

    n_rows = len(rows_spec)
    # Normalise row_heights so plotly accepts.
    row_heights = [h / sum(row_heights) for h in row_heights]

    # subplot_titles is one entry per *actual* subplot cell, in row-major order;
    # None positions in `specs` are skipped, so colspan=2 rows contribute one
    # title each and 2-cell rows (pca_umap, tables) contribute two.
    SINGLE_TITLES = {
        "raster":    "Spike raster (units ranked by firing rate)",
        "events":    "Burst events (lanes: burstlets / network_bursts / superbursts)",
        "signals":   "Driving signals (ranking_signal drives HDBSCAN ranking)",
        "strip":     "HDBSCAN cluster label per bin",
        "pca_only":  "PCA(2) of z-normed features — what HDBSCAN saw",
        "umap_only": "UMAP(2) of z-normed features",
        "hist":      "Ranking-feature distribution per HDBSCAN cluster",
    }
    flat_titles: list[str] = []
    for name in row_meanings:
        if name == "tables":
            flat_titles.append("HDBSCAN cluster ranking")
            flat_titles.append("Per-level metrics")
        elif name == "pca_umap":
            flat_titles.append("PCA(2) of z-normed features — what HDBSCAN saw")
            flat_titles.append("UMAP(2) of z-normed features")
        else:
            flat_titles.append(SINGLE_TITLES.get(name, ""))

    fig = make_subplots(
        rows=n_rows, cols=2,
        specs=rows_spec,
        row_heights=row_heights,
        vertical_spacing=0.045,
        horizontal_spacing=0.06,
        subplot_titles=flat_titles,
    )

    row_idx = {name: i + 1 for i, name in enumerate(row_meanings)}

    # ----- Raster -----
    t_end = 0.0
    n_units = 0
    if well.spike_times:
        traces, t_end, n_units = _raster_traces(well.spike_times)
        for tr in traces:
            fig.add_trace(tr, row=row_idx["raster"], col=1)
        fig.update_yaxes(title_text="unit rank", row=row_idx["raster"], col=1)
    else:
        fig.add_annotation(text="no spike times", showarrow=False,
                           xref="x domain", yref="y domain", x=0.5, y=0.5,
                           row=row_idx["raster"], col=1)

    # ----- Events -----
    for tr in _event_traces(well.events):
        fig.add_trace(tr, row=row_idx["events"], col=1)
    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=[k for k, *_ in EVENT_TRACKS],
        range=[-0.6, 2.6],
        showgrid=False,
        row=row_idx["events"], col=1,
    )

    # ----- Signals -----
    merge_threshold = None
    if well.plot_signals is not None:
        mt = well.plot_signals.get("merge_threshold")
        if mt is not None:
            try:
                merge_threshold = float(mt)
            except (TypeError, ValueError):
                merge_threshold = None
    if "signals" in row_idx:
        for tr in _signal_traces(well.plot_signals, merge_threshold):
            fig.add_trace(tr, row=row_idx["signals"], col=1)
        fig.update_yaxes(title_text="value", row=row_idx["signals"], col=1)

    # ----- HDBSCAN strip -----
    if "strip" in row_idx:
        strip = _hdbscan_strip(well.trace, well.diagnostics)
        if strip is not None:
            fig.add_trace(strip, row=row_idx["strip"], col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                         row=row_idx["strip"], col=1)

    # ----- PCA + UMAP projections -----
    burst_label_int: int | None = None
    if well.diagnostics is not None:
        bl = well.diagnostics.get("cluster_burst_label")
        if bl is not None and bl != -1:
            burst_label_int = int(bl)
    labels_full = (
        np.asarray(well.trace.hdbscan_labels)
        if has_trace and getattr(well.trace, "hdbscan_labels", None) is not None
        else None
    )
    t_full = (
        np.asarray(well.trace.t_centers, dtype=float)
        if has_trace and getattr(well.trace, "t_centers", None) is not None
        else None
    )

    def _add_projection(row_name: str, col_int: int, result, kept_indices=None):
        if result is None or labels_full is None or t_full is None:
            return
        coords = result[0]
        xl = result[1]
        yl = result[2]
        if kept_indices is None:
            lbl_in = labels_full
            t_in = t_full
        else:
            lbl_in = labels_full[kept_indices]
            t_in = t_full[kept_indices]
        for tr in _projection_scatter_traces(coords, lbl_in, t_in, burst_label_int,
                                             xl, yl, row_name):
            fig.add_trace(tr, row=row_idx[row_name], col=col_int)
        fig.update_xaxes(title_text=xl, row=row_idx[row_name], col=col_int)
        fig.update_yaxes(title_text=yl, row=row_idx[row_name], col=col_int)

    if "pca_umap" in row_idx:
        _add_projection("pca_umap", 1, pca_result)
        _add_projection("pca_umap", 2, umap_result, kept_indices=umap_result[3] if umap_result else None)
    if "pca_only" in row_idx:
        _add_projection("pca_only", 1, pca_result)
    if "umap_only" in row_idx:
        _add_projection("umap_only", 1, umap_result,
                        kept_indices=umap_result[3] if umap_result else None)

    # ----- Ranking-feature histogram per cluster -----
    if "hist" in row_idx:
        hist_traces, rank_name, thr = _ranking_hist_traces(well.trace, well.diagnostics)
        for tr in hist_traces:
            fig.add_trace(tr, row=row_idx["hist"], col=1)
        if thr is not None and np.isfinite(thr):
            fig.add_vline(
                x=thr,
                line=dict(color="#d62728", width=1.0, dash="dash"),
                annotation_text=f"thr={thr:.3f}",
                annotation_position="top right",
                annotation_font_size=10,
                row=row_idx["hist"], col=1,
            )
        if hist_traces:
            fig.update_xaxes(title_text=rank_name, row=row_idx["hist"], col=1)
            fig.update_yaxes(title_text="bin count", row=row_idx["hist"], col=1)

    # ----- Tables -----
    ct = _cluster_table(well.diagnostics, well.trace)
    mt = _metrics_table(well.metrics)
    if ct is not None:
        fig.add_trace(ct, row=row_idx["tables"], col=1)
    if mt is not None:
        fig.add_trace(mt, row=row_idx["tables"], col=2)

    # ----- Link time axes -----
    # Use the raster x-axis as the anchor; link events/signals/strip to it.
    time_rows = [name for name in row_meanings if name in {"raster", "events", "signals", "strip"}]
    anchor_row = row_idx["raster"]
    anchor_axis = "x" if anchor_row == 1 else f"x{(anchor_row - 1) * 2 + 1}"
    # NB: colspan=2 still occupies the first column's axis slot; the second
    # column slot is empty (specs=None), so axis numbering steps by 2 per row
    # for time rows. Compute axis number robustly via fig.layout.
    def _row_xaxis(row: int) -> str:
        ref = fig.get_subplot(row=row, col=1)
        if ref is None or not hasattr(ref, "xaxis"):
            return "x"
        name = ref.xaxis.plotly_name  # "xaxis" or "xaxis2", ...
        suffix = name[len("xaxis"):]
        return "x" + suffix

    anchor_axis = _row_xaxis(anchor_row)
    for name in time_rows:
        if name == "raster":
            continue
        r = row_idx[name]
        fig.update_xaxes(matches=anchor_axis, row=r, col=1)
    if t_end > 0:
        fig.update_xaxes(range=[0, t_end], row=anchor_row, col=1)
    # Time-axis label only on the lowest time row
    last_time = [name for name in row_meanings if name in {"raster", "events", "signals", "strip"}][-1]
    fig.update_xaxes(title_text="time (s)", row=row_idx[last_time], col=1)

    # ----- Title / layout -----
    diag = well.diagnostics or {}
    n_bursts = 0
    if well.events.get("network_bursts"):
        n_bursts = len(well.events["network_bursts"])
    decision = diag.get("cluster_decision", "?")
    n_units_fit = diag.get("n_units_fit", "?")
    title = (
        f"<b>{well.well_name}</b> ({well.well_id}) — group <b>{well.groupname}</b> — "
        f"rec {well.rec_name} — n_units={n_units} (HMM fit: {n_units_fit}) — "
        f"network_bursts={n_bursts} — cluster_decision={decision}"
    )
    if not has_trace:
        title += "  <i>[no debug_trace.pkl: cluster panels omitted]</i>"

    # Extra vertical space per optional panel so panels don't crowd each other.
    proj_h = 320 if "pca_umap" in row_idx else (260 if ("pca_only" in row_idx or "umap_only" in row_idx) else 0)
    hist_h = 200 if "hist" in row_idx else 0
    strip_h = 60 if has_strip else 0
    fig.update_layout(
        title=dict(text=title, x=0.005, xanchor="left", y=0.995, yanchor="top",
                   font=dict(size=13)),
        height=900 + proj_h + hist_h + strip_h,
        width=1500,
        margin=dict(l=60, r=20, t=60, b=60),
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=11),
        barmode="overlay",
        showlegend=True,
        legend=dict(orientation="v", x=1.005, y=0.95, xanchor="left",
                    bgcolor="rgba(255,255,255,0.85)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, ticks="outside")
    fig.update_yaxes(showgrid=False, zeroline=False, ticks="outside")
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#333")
    return fig


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def _collect_ml_wells(pipeline_cache: dict[str, Any]) -> list[tuple[str, str, str, dict]]:
    """Return list of (recording_key, rec_name, well_id, entry) for wells where
    auto_curation AND ml_burst_detection are complete."""
    out = []
    for entry in pipeline_cache.values():
        tasks = entry.get("tasks") or {}
        if tasks.get(CUR_TASK, {}).get("status") != "complete":
            continue
        if tasks.get(ML_TASK, {}).get("status") != "complete":
            continue
        rec_key = entry.get("recording_key")
        compound = entry.get("well_id", "")
        if not rec_key or "/" not in compound:
            continue
        rec_name, well_id = compound.split("/", 1)
        out.append((rec_key, rec_name, well_id, entry))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="Path to pipeline_config.json (read for analysis_root + figure_root).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output directory. Default: {figure_root}.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s — %(message)s")

    with args.config.open() as fh:
        cfg = json.load(fh)
    analysis_root = Path(cfg["global"]["analysis_root"])
    figure_root = Path(args.output_dir or cfg["global"].get("figure_root") or analysis_root)
    figure_root.mkdir(parents=True, exist_ok=True)

    pipeline_cache = _load_pipeline_cache(analysis_root)
    experiment_cache = _load_experiment_cache(analysis_root)

    config_dir = args.config.resolve().parent
    resolve = _make_resolver([config_dir, analysis_root, Path.cwd().resolve()])

    wells = _collect_ml_wells(pipeline_cache)
    if not wells:
        logger.warning("No wells with completed ml_burst_detection found in %s/pipeline_cache.json",
                       analysis_root)
        return 0
    logger.info("Found %d wells with completed ml_burst_detection", len(wells))

    n_written = 0
    for rec_key, rec_name, well_id, entry in wells:
        well_meta = (
            experiment_cache.get(rec_key, {})
            .get("wells", {}).get(well_id, {}).get("metadata", {})
        )
        bundle = _load_well_bundle(rec_key, rec_name, well_id, entry, well_meta, resolve)
        if bundle is None:
            continue
        try:
            fig = build_well_figure(bundle)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to build figure for %s/%s/%s: %s",
                             rec_key, rec_name, well_id, exc)
            continue

        out_dir = figure_root / rec_key / "ml_burst_inspect"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{bundle.well_name}_{rec_name}_{well_id}.html"
        out_path = out_dir / fname
        fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
        n_written += 1
        logger.info("Wrote %s", out_path)

    logger.info("Done. %d HTML file(s) written under %s", n_written, figure_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
