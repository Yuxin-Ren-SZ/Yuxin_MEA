"""Shared-axis bin-level UMAP/PCA overlay across one physical well's recordings.

Every ``(well x recording)`` is normally embedded by its *own* UMAP fit
(``inspect_ml_bursts._compute_umap_axes``) with its own normalization and its
own learned manifold, so two recordings' axes are not comparable. This script
instead pools the per-bin 26-D feature matrices from *all* recordings of one
physical well, applies a single shared normalization, and fits ONE embedding so
every recording shares the same axes/loadings. You can then overlay the
recordings and see whether the resting-state and bursting-state clusters drift
across days.

Output: a single self-contained Plotly HTML. A dropdown selects the
``well x method`` view; within a view, **color = recording** (ordered by date,
so color doubles as a time gradient), **marker shape = burst vs non-burst**
(circle vs diamond), marker opacity 0.5.

Data consumed per recording (must have run with ``debug=True``):
  <ml_root>/<sample>/<date>/<plate>/<scan>/<run_id>/<rec_name>/<well_id>/
      ml_burst_detection/debug_trace.pkl   (feature_matrix, hdbscan_labels, t_centers, ...)
      ml_burst_detection/diagnostics.json  (cluster_burst_labels)

Examples:
    python scripts/well_recording_umap_overlay.py \
        --config pipeline_config.json --well "CX118|T003346|well000"

    # structural-only view (remove per-recording baseline via background z-norm):
    python scripts/well_recording_umap_overlay.py \
        --config pipeline_config.json --well "CX118|T003346|well000" --norm per-recording
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors

# Reuse the loaders/helpers from the per-well inspector (import-safe: its
# module-level imports are only stdlib + numpy/pandas/plotly).
import inspect_ml_bursts as iml

logger = logging.getLogger("well_recording_umap_overlay")

ML_TERMINAL = "ml_burst_detection"
NONBURST_SYMBOL = "circle"
BURST_SYMBOL = "cross"  # bold "+", very unlike a circle
MARKER_ALPHA = 0.5

# Okabe-Ito colorblind-safe qualitative (8 colors)
_OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
              "#0072B2", "#D55E00", "#CC79A7", "#000000"]


# --------------------------------------------------------------------------- #
# Config / discovery
# --------------------------------------------------------------------------- #
def _load_config(config_path: Path) -> tuple[Path, Path]:
    with config_path.open() as fh:
        cfg = json.load(fh)
    g = cfg.get("global", {})
    analysis_root = Path(g["analysis_root"])
    figure_root = Path(g.get("figure_root") or analysis_root)
    return analysis_root, figure_root


def _parse_well_uid(s: str) -> tuple[str, str, str]:
    """Parse ``sample|plate|well_id`` (``/`` also accepted) -> (sample, plate, well_id)."""
    parts = [p for p in s.replace("/", "|").split("|") if p]
    if len(parts) != 3:
        raise SystemExit(
            f"--well must be 'sample|plate|well_id' (e.g. 'CX118|T003346|well000'); got {s!r}"
        )
    return parts[0], parts[1], parts[2]


def _parse_plating_date(s: Any) -> datetime | None:
    """Parse a day-first dot/slash/dash date (e.g. '13.3.2026', '30.01.2026')."""
    if s is None:
        return None
    s = str(s).strip()
    for sep in (".", "/", "-"):
        if sep in s:
            parts = s.split(sep)
            break
    else:
        return None
    if len(parts) != 3:
        return None
    try:
        d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if y < 100:
        y += 2000
    try:
        return datetime(y, m, d)  # day-first
    except ValueError:
        return None


def _plating_lookup(experiment_cache: dict, sample: str, plate: str, well_id: str) -> datetime | None:
    """First parseable 'Plating Date' for this physical well across recordings."""
    for rec_key, rec in experiment_cache.items():
        kp = rec_key.split("/")
        if not (len(kp) >= 3 and kp[0] == sample and kp[2] == plate):
            continue
        for _wid, w in (rec.get("wells") or {}).items():
            md = w.get("metadata") or {}
            if str(w.get("well_id")) == well_id or str(_wid) == well_id:
                pd_ = _parse_plating_date(md.get("Plating Date"))
                if pd_ is not None:
                    return pd_
    return None


def _discover_recordings(
    ml_root: Path, sample: str, plate: str, well_id: str, scan_type: str
) -> list[Path]:
    """All ``debug_trace.pkl`` for one physical well, sorted by (date, run_id, rec_name)."""
    pattern = f"{sample}/*/{plate}/{scan_type}/*/*/{well_id}/{ML_TERMINAL}/debug_trace.pkl"
    hits = sorted(ml_root.glob(pattern))
    return hits


# --------------------------------------------------------------------------- #
# Per-recording load
# --------------------------------------------------------------------------- #
class Recording:
    """One recording's pooled inputs for the shared embedding."""

    __slots__ = ("date", "run_id", "rec_name", "div", "label",
                 "X_raw", "Z_local", "is_burst", "t_centers", "labels")

    def __init__(self, date, run_id, rec_name, div, label,
                 X_raw, Z_local, is_burst, t_centers, labels):
        self.date = date
        self.run_id = run_id
        self.rec_name = rec_name
        self.div = div
        self.label = label
        self.X_raw = X_raw
        self.Z_local = Z_local
        self.is_burst = is_burst
        self.t_centers = t_centers
        self.labels = labels


def _stride_keep(n: int, max_bins: int) -> np.ndarray:
    if n <= max_bins:
        return np.arange(n)
    stride = int(np.ceil(n / max_bins))
    return np.arange(0, n, stride)[:max_bins]


def _load_recording(
    trace_path: Path,
    ml_root: Path,
    plating: datetime | None,
    max_bins: int,
    feat_ref: list[str] | None,
) -> tuple[Recording | None, list[str] | None]:
    """Load + subsample one recording. Returns (Recording or None, feature_names)."""
    rel = trace_path.relative_to(ml_root).parts
    # rel = (sample, date, plate, scan, run_id, rec_name, well_id, terminal, file)
    date, run_id, rec_name = rel[1], rel[4], rel[5]

    trace = iml._load_trace(trace_path)
    if trace is None:
        logger.warning("skip %s: trace failed to load", date)
        return None, feat_ref
    X = getattr(trace, "feature_matrix", None)
    labels = getattr(trace, "hdbscan_labels", None)
    t = getattr(trace, "t_centers", None)
    fnames = list(getattr(trace, "feature_names", None) or [])
    if X is None or labels is None:
        logger.warning("skip %s/%s: missing feature_matrix/hdbscan_labels", date, rec_name)
        return None, feat_ref
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels).astype(int)
    t = np.asarray(t, dtype=float) if t is not None else np.arange(X.shape[0], dtype=float)
    if X.ndim != 2 or X.shape[0] < 10:
        logger.warning("skip %s/%s: too few bins (%s)", date, rec_name, X.shape)
        return None, feat_ref
    if not (X.shape[0] == labels.shape[0] == t.shape[0]):
        logger.warning("skip %s/%s: misaligned rows X=%s labels=%s t=%s",
                       date, rec_name, X.shape[0], labels.shape[0], t.shape[0])
        return None, feat_ref
    if feat_ref is not None and fnames and fnames != feat_ref:
        logger.warning("skip %s/%s: feature_names differ from reference", date, rec_name)
        return None, feat_ref
    if feat_ref is None and fnames:
        feat_ref = fnames

    diagnostics = iml._load_json(trace_path.parent / "diagnostics.json")
    burst_set = iml._burst_label_set(diagnostics)
    is_burst_full = np.isin(labels, list(burst_set)) if burst_set else np.zeros(X.shape[0], bool)

    # per-recording (background) z-norm for the --norm per-recording option
    Z_local = iml._znorm_from_trace(trace)

    keep = _stride_keep(X.shape[0], max_bins)
    div = None
    rec_dt = _parse_yymmdd(date)
    if plating is not None and rec_dt is not None:
        div = (rec_dt - plating).days

    # include run_id so same-day recordings stay distinct (unique legend entry per recording)
    label = f"{date}" + (f" DIV{div}" if div is not None else "") + f" {run_id}"
    rec = Recording(
        date=date, run_id=run_id, rec_name=rec_name, div=div, label=label,
        X_raw=X[keep],
        Z_local=(Z_local[keep] if Z_local is not None else None),
        is_burst=is_burst_full[keep],
        t_centers=t[keep],
        labels=labels[keep],
    )
    return rec, feat_ref


def _parse_yymmdd(s: str) -> datetime | None:
    s = str(s).strip()
    if len(s) != 6 or not s.isdigit():
        return None
    try:
        return datetime.strptime("20" + s, "%Y%m%d")
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def _pool_matrix(recs: list[Recording], norm: str) -> np.ndarray:
    if norm == "per-recording":
        if any(r.Z_local is None for r in recs):
            raise SystemExit("--norm per-recording needs scaler stats in every trace; some missing")
        return np.vstack([r.Z_local for r in recs])
    # shared: one StandardScaler over pooled raw features
    from sklearn.preprocessing import StandardScaler
    X = np.vstack([r.X_raw for r in recs])
    return StandardScaler().fit_transform(X)


def _fit_umap(Z: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    try:
        import umap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"umap-learn not importable: {exc}")
    nn = max(2, min(int(n_neighbors), Z.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=float(min_dist),
                        metric="euclidean", random_state=seed)
    return np.asarray(reducer.fit_transform(Z))


def _fit_pca(Z: np.ndarray, feat_names: list[str], seed: int) -> tuple[np.ndarray, str, str]:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=seed).fit(Z)
    coords = pca.transform(Z)

    def _lab(pc: int) -> str:
        comp = pca.components_[pc]
        ev = float(pca.explained_variance_ratio_[pc]) * 100
        order = np.argsort(np.abs(comp))[::-1][:3]
        if feat_names and all(0 <= int(i) < len(feat_names) for i in order):
            tops = ", ".join(f"{feat_names[int(i)]}{comp[int(i)]:+.2f}" for i in order)
            return f"PC{pc+1} ({ev:.1f}%): {tops}"
        return f"PC{pc+1} ({ev:.1f}%)"

    return coords, _lab(0), _lab(1)


# --------------------------------------------------------------------------- #
# Plotly figure
# --------------------------------------------------------------------------- #
_QUAL_PALETTES = {
    "dark24": pcolors.qualitative.Dark24,   # 24 high-contrast hues, default
    "safe": pcolors.qualitative.Safe,       # 11 colorblind-safe
    "okabe": _OKABE_ITO,                     # 8 colorblind-safe (gold standard)
}
_SEQ_PALETTES = {"cividis": "Cividis", "turbo": "Turbo", "viridis": "Viridis"}
PALETTE_CHOICES = list(_QUAL_PALETTES) + list(_SEQ_PALETTES)


def _rec_colors(n: int, palette: str = "dark24") -> list[str]:
    """n distinct recording colors. Qualitative palettes cycle; sequential ones
    sample a gradient (so color still tracks day order)."""
    if palette in _QUAL_PALETTES:
        base = _QUAL_PALETTES[palette]
        return [base[i % len(base)] for i in range(n)]
    name = _SEQ_PALETTES.get(palette, "Cividis")
    if n == 1:
        return list(pcolors.sample_colorscale(name, [0.5]))
    return list(pcolors.sample_colorscale(name, list(np.linspace(0, 1, n))))


def _marker(state_is_burst: bool, color: str) -> dict:
    """Per-state marker style. Burst = bold cross with a thick dark border so it
    stands out from the soft rest circles."""
    if state_is_burst:
        return dict(color=color, symbol=BURST_SYMBOL, size=9, opacity=MARKER_ALPHA,
                    line=dict(width=1.6, color="#111"))
    return dict(color=color, symbol=NONBURST_SYMBOL, size=5, opacity=MARKER_ALPHA,
                line=dict(width=0))


def _build_multi_figure(blocks: list[tuple[str, list["Recording"], dict]], norm: str,
                        palette: str = "dark24") -> go.Figure:
    """Build one figure over many wells. Each block = (well_uid, recs, method_coords).

    Every well has its OWN pooled fit (own coordinate space) — overlaying two
    wells on one axis is meaningless — so the ``well·method`` dropdown switches
    which view is ``visible``. A separate Both/Burst/Rest button set filters
    state via ``marker.opacity`` (a channel the dropdown doesn't touch), so the
    two controls never fight and the state filter persists across method switches.
    """
    fig = go.Figure()
    multi = len(blocks) > 1
    methods = list(blocks[0][2].keys())
    first_combo = f"{blocks[0][0]}|{methods[0]}"

    trace_combo: list[str] = []          # which (well|method) each trace belongs to
    trace_is_burst: list[bool | None] = []  # per-trace state; None = symbol-legend dummy

    for well_uid, recs, method_coords in blocks:
        colors = _rec_colors(len(recs), palette)
        sizes = [r.X_raw.shape[0] for r in recs]
        offsets = np.cumsum([0] + sizes)
        wshort = well_uid.split("|")[-1]
        for method in method_coords:
            coords, _xl, _yl = method_coords[method]
            combo = f"{well_uid}|{method}"
            for ri, r in enumerate(recs):
                sl = slice(int(offsets[ri]), int(offsets[ri + 1]))
                xy = coords[sl]
                for is_burst, tag in ((False, "rest"), (True, "burst")):
                    m = r.is_burst == is_burst
                    if not m.any():
                        continue
                    hover = [
                        f"{wshort} {r.label}<br>state={tag}<br>t={tc:.1f}s<br>cluster={cl}"
                        for tc, cl in zip(r.t_centers[m], r.labels[m])
                    ]
                    fig.add_trace(go.Scattergl(
                        x=xy[m, 0], y=xy[m, 1], mode="markers",
                        marker=_marker(is_burst, colors[ri]),
                        name=(f"{wshort} {r.label}" if multi else r.label),
                        legendgroup=f"{well_uid}|{r.label}",  # burst+rest share group -> one recording entry
                        showlegend=(not is_burst),            # single legend entry per recording (by color)
                        hovertext=hover, hoverinfo="text",
                        visible=(combo == first_combo),
                    ))
                    trace_combo.append(combo)
                    trace_is_burst.append(is_burst)

    # symbol-legend dummies (always visible) — show the two glyph meanings
    for is_burst, nm in ((False, "○ rest"), (True, "✚ burst")):
        fig.add_trace(go.Scattergl(
            x=[None], y=[None], mode="markers",
            marker=_marker(is_burst, "#333"),
            name=nm, legendgroup="_symbol", showlegend=True, hoverinfo="skip", visible=True,
        ))
        trace_combo.append("_always")
        trace_is_burst.append(None)

    # well·method dropdown: owns `visible`
    wm_buttons = []
    for well_uid, recs, method_coords in blocks:
        wshort = well_uid.split("|")[-1]
        for method in method_coords:
            combo = f"{well_uid}|{method}"
            _coords, xl, yl = method_coords[method]
            vis = [(tc == combo) or (tc == "_always") for tc in trace_combo]
            wm_buttons.append(dict(
                label=(f"{wshort} · {method.upper()}" if multi else method.upper()),
                method="update",
                args=[{"visible": vis},
                      {"xaxis.title.text": xl, "yaxis.title.text": yl,
                       "title.text": _title(well_uid, method, recs, norm)}],
            ))

    # state buttons: own `marker.opacity` (per-trace list restyle over ALL traces)
    def _op(mode: str) -> list[float]:
        out = []
        for b in trace_is_burst:
            if b is None:
                out.append(1.0)                                   # dummies always solid
            elif mode == "both":
                out.append(MARKER_ALPHA)
            elif mode == "burst":
                out.append(MARKER_ALPHA if b else 0.0)
            else:  # rest
                out.append(MARKER_ALPHA if not b else 0.0)
        return out

    state_buttons = [
        dict(label="Both", method="restyle", args=[{"marker.opacity": _op("both")}]),
        dict(label="Burst only", method="restyle", args=[{"marker.opacity": _op("burst")}]),
        dict(label="Rest only", method="restyle", args=[{"marker.opacity": _op("rest")}]),
    ]

    w0, recs0, mc0 = blocks[0]
    _, xl0, yl0 = mc0[methods[0]]
    # Header split into non-overlapping bands: title (container-anchored, very top),
    # controls row (dropdown hard-left / state buttons hard-right), plot + right legend,
    # caption below the x-axis. Title on yref="container" sits above the paper-anchored menus.
    fig.update_layout(
        title=dict(text=_title(w0, methods[0], recs0, norm),
                   x=0.0, xanchor="left", y=0.985, yref="container", yanchor="top",
                   font=dict(size=15)),
        updatemenus=[
            dict(active=0, buttons=wm_buttons, x=0.0, xanchor="left",
                 y=1.11, yanchor="top", direction="down",
                 pad=dict(t=2, r=8), showactive=True),
            dict(type="buttons", active=0, buttons=state_buttons, x=1.0, xanchor="right",
                 y=1.11, yanchor="top", direction="right",
                 pad=dict(t=2, l=8), showactive=True),
        ],
        xaxis_title=xl0, yaxis_title=yl0,
        legend=dict(title="recording", x=1.02, xanchor="left", y=1.0, yanchor="top",
                    itemsizing="constant", itemclick="toggle", itemdoubleclick="toggleothers"),
        template="plotly_white", height=800, width=1200,
        margin=dict(l=70, r=220, t=150, b=110),
    )
    fig.add_annotation(
        text=("color = recording (date) · ○ rest / ✚ burst · "
              "legend: click = toggle recording, double-click = isolate · "
              "PCA axes show loadings; UMAP distances not metric."),
        xref="paper", yref="paper", x=0.0, xanchor="left", y=-0.12, yanchor="top",
        showarrow=False, font=dict(size=11, color="#666"), align="left",
    )
    return fig


def _title(well_uid: str, method: str, recs: list[Recording], norm: str) -> str:
    nb = sum(int(r.X_raw.shape[0]) for r in recs)
    return (f"{well_uid} — {method.upper()} shared axis (norm={norm}) · "
            f"{len(recs)} recordings · {nb} bins")


def _build_single_animated(block: tuple[str, list["Recording"], dict], norm: str,
                           palette: str = "dark24") -> go.Figure:
    """Single-well figure with a time-lapse animation over recordings (DIV order).

    Only used when exactly one well is rendered — each well is its own
    coordinate space, so revealing "recordings ≤ k" is meaningful only within
    one well. Three orthogonal control channels, so nothing collides:
      • method (UMAP/PCA)     → x/y  (restyle: one trace set, swap coordinates)
      • state (Both/Burst/Rest) → marker.opacity
      • animation (time)      → visible ('legendonly' keeps the legend a stable
                                 DIV timeline while a recording is hidden)
    Two static frame groups: cum_k (leave previous) and cur_k (current only).
    """
    well_uid, recs, method_coords = block
    methods = list(method_coords.keys())          # [--method first, then the other]
    default_m = methods[0]
    colors = _rec_colors(len(recs), palette)
    sizes = [r.X_raw.shape[0] for r in recs]
    offsets = np.cumsum([0] + sizes)
    K = len(recs)

    fig = go.Figure()
    # per-trace coordinates for each projection (for the method x/y restyle)
    px: dict[str, list] = {m: [] for m in methods}
    py: dict[str, list] = {m: [] for m in methods}
    trace_rank: list[int | None] = []     # recording rank per trace (None = dummy)
    trace_is_burst: list[bool | None] = []

    for ri, r in enumerate(recs):
        for is_burst, tag in ((False, "rest"), (True, "burst")):
            m = r.is_burst == is_burst
            if not m.any():
                continue
            for name in methods:
                xy = method_coords[name][0][int(offsets[ri]):int(offsets[ri + 1])][m]
                px[name].append(xy[:, 0])
                py[name].append(xy[:, 1])
            xy0 = method_coords[default_m][0][int(offsets[ri]):int(offsets[ri + 1])][m]
            hover = [
                f"{r.label}<br>state={tag}<br>t={tc:.1f}s<br>cluster={cl}"
                for tc, cl in zip(r.t_centers[m], r.labels[m])
            ]
            fig.add_trace(go.Scattergl(
                x=xy0[:, 0], y=xy0[:, 1], mode="markers",
                marker=_marker(is_burst, colors[ri]),
                name=r.label, legendgroup=f"rec|{r.label}",
                showlegend=(not is_burst),
                hovertext=hover, hoverinfo="text", visible=True,
            ))
            trace_rank.append(ri)
            trace_is_burst.append(is_burst)

    for is_burst, nm in ((False, "○ rest"), (True, "✚ burst")):
        fig.add_trace(go.Scattergl(
            x=[None], y=[None], mode="markers", marker=_marker(is_burst, "#333"),
            name=nm, legendgroup="_symbol", showlegend=True, hoverinfo="skip", visible=True,
        ))
        for name in methods:
            px[name].append([None])
            py[name].append([None])
        trace_rank.append(None)
        trace_is_burst.append(None)

    n_traces = len(trace_rank)

    # method: x/y restyle (+ axis titles) — frees `visible` for the animation
    method_buttons = []
    for name in methods:
        _c, xl, yl = method_coords[name]
        method_buttons.append(dict(
            label=name.upper(), method="update",
            args=[{"x": px[name], "y": py[name]},
                  {"xaxis.title.text": xl, "yaxis.title.text": yl,
                   "title.text": _title(well_uid, name, recs, norm)}],
        ))

    # state: marker.opacity restyle (per-trace list)
    def _op(mode: str) -> list[float]:
        out = []
        for b in trace_is_burst:
            if b is None:
                out.append(1.0)
            elif mode == "both":
                out.append(MARKER_ALPHA)
            elif mode == "burst":
                out.append(MARKER_ALPHA if b else 0.0)
            else:
                out.append(MARKER_ALPHA if not b else 0.0)
        return out

    state_buttons = [
        dict(label="Both", method="restyle", args=[{"marker.opacity": _op("both")}]),
        dict(label="Burst only", method="restyle", args=[{"marker.opacity": _op("burst")}]),
        dict(label="Rest only", method="restyle", args=[{"marker.opacity": _op("rest")}]),
    ]

    # animation: two static frame groups over `visible`
    def _vis(k: int, cumulative: bool) -> list:
        out = []
        for rk in trace_rank:
            if rk is None:
                out.append(True)                                  # dummies always shown
            elif (rk <= k) if cumulative else (rk == k):
                out.append(True)
            else:
                out.append("legendonly")                          # hidden but keeps legend slot
        return out

    frames = []
    for pref, cumulative in (("cum", True), ("cur", False)):
        for k in range(K):
            frames.append(go.Frame(
                name=f"{pref}{k}",
                data=[{"visible": v} for v in _vis(k, cumulative)],
                traces=list(range(n_traces)),
            ))
    fig.frames = frames

    def _seq(pref: str) -> list[str]:
        return [f"{pref}{k}" for k in range(K)]

    play_opts = {"frame": {"duration": 600, "redraw": True},
                 "transition": {"duration": 0}, "mode": "immediate", "fromcurrent": True}
    pause_opts = {"frame": {"duration": 0, "redraw": False},
                  "transition": {"duration": 0}, "mode": "immediate"}
    step_opts = {"frame": {"duration": 0, "redraw": True},
                 "transition": {"duration": 0}, "mode": "immediate"}

    anim_buttons = dict(
        type="buttons", direction="right", x=0.0, xanchor="left", y=-0.03, yanchor="top",
        pad=dict(t=6, r=8), showactive=False,
        buttons=[
            dict(label="▶ Leave previous", method="animate", args=[_seq("cum"), play_opts]),
            dict(label="▶ Current only", method="animate", args=[_seq("cur"), play_opts]),
            dict(label="⏸ Pause", method="animate", args=[[None], pause_opts]),
        ],
    )
    slider = dict(
        active=K - 1, x=0.06, len=0.9, y=-0.14, yanchor="top", pad=dict(t=8, b=8),
        currentvalue=dict(prefix="cumulative up to: ", visible=True, xanchor="right"),
        steps=[dict(method="animate", label=recs[k].label, args=[[f"cum{k}"], step_opts])
               for k in range(K)],
    )

    _, xl0, yl0 = method_coords[default_m]
    fig.update_layout(
        title=dict(text=_title(well_uid, default_m, recs, norm),
                   x=0.0, xanchor="left", y=0.985, yref="container", yanchor="top",
                   font=dict(size=15)),
        updatemenus=[
            dict(type="buttons", active=0, buttons=method_buttons, x=0.0, xanchor="left",
                 y=1.11, yanchor="top", direction="right", pad=dict(t=2, r=8), showactive=True),
            dict(type="buttons", active=0, buttons=state_buttons, x=1.0, xanchor="right",
                 y=1.11, yanchor="top", direction="right", pad=dict(t=2, l=8), showactive=True),
            anim_buttons,
        ],
        sliders=[slider],
        xaxis_title=xl0, yaxis_title=yl0,
        legend=dict(title="recording", x=1.02, xanchor="left", y=1.0, yanchor="top",
                    itemsizing="constant", itemclick="toggle", itemdoubleclick="toggleothers"),
        template="plotly_white", height=860, width=1200,
        margin=dict(l=70, r=220, t=150, b=200),
    )
    fig.add_annotation(
        text=("time-lapse: <b>▶ Leave previous</b> = accumulate · <b>▶ Current only</b> = one at a time · "
              "slider scrubs cumulative &nbsp;|&nbsp; UMAP/PCA (top-left) · Both/Burst/Rest (top-right) · "
              "color = recording (date) · ○ rest / ✚ burst · PCA axes show loadings."),
        xref="paper", yref="paper", x=0.0, xanchor="left", y=-0.30, yanchor="top",
        showarrow=False, font=dict(size=11, color="#666"), align="left",
    )
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--well", type=str, required=True, nargs="+",
                    help="one or more physical wells 'sample|plate|well_id' "
                         "(each becomes a dropdown entry with its own shared fit)")
    ap.add_argument("--scan-type", default="Network")
    ap.add_argument("--ml-dirname", default="ml_burst_data_umap")
    ap.add_argument("--norm", choices=["shared", "per-recording"], default="shared")
    ap.add_argument("--method", choices=["umap", "pca"], default="umap",
                    help="primary view; the other is also emitted in the dropdown")
    ap.add_argument("--palette", choices=PALETTE_CHOICES, default="dark24",
                    help="recording colors: dark24 (high contrast, default); "
                         "safe/okabe (colorblind-safe, fewer colors); "
                         "cividis/turbo/viridis (ordered gradient by day)")
    ap.add_argument("--max-bins-per-rec", type=int, default=1500)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--animate", dest="animate", action="store_true", default=True,
                    help="single-well only: add a time-lapse over recordings (default on)")
    ap.add_argument("--no-animate", dest="animate", action="store_false",
                    help="static overlay (no time-lapse); the only mode for multi-well")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(message)s")

    analysis_root, figure_root = _load_config(args.config)
    ml_root = analysis_root / args.ml_dirname
    try:
        experiment_cache = iml._load_json(analysis_root / "experiment_cache.json") or {}
    except Exception:  # noqa: BLE001
        experiment_cache = {}

    blocks: list[tuple[str, list[Recording], dict]] = []
    for well_str in args.well:
        block = _embed_one_well(ml_root, experiment_cache, well_str, args)
        if block is not None:
            blocks.append(block)
    if not blocks:
        raise SystemExit("no wells produced a usable embedding")

    grand_bins = sum(sum(int(r.X_raw.shape[0]) for r in recs) for _, recs, _ in blocks)
    if grand_bins > 120_000:
        logger.warning("%d total plotted points across %d wells — HTML will be large/heavy; "
                       "consider fewer wells or a smaller --max-bins-per-rec", grand_bins, len(blocks))

    if args.animate and len(blocks) == 1:
        fig = _build_single_animated(blocks[0], args.norm, args.palette)
    else:
        if args.animate and len(blocks) > 1:
            logger.info("animation is single-well only; rendering static overlay for %d wells", len(blocks))
        fig = _build_multi_figure(blocks, args.norm, args.palette)

    if args.output:
        out = args.output
    elif len(blocks) == 1:
        s, p, w = blocks[0][0].split("|")
        out = figure_root / "well_umap_overlay" / f"{s}_{p}_{w}.html"
    else:
        tag = "-".join(b[0].split("|")[-1] for b in blocks)[:60]
        out = figure_root / "well_umap_overlay" / f"overlay_{len(blocks)}wells_{tag}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="inline", full_html=True)
    logger.info("wrote %s", out)
    print(out)
    return 0


def _embed_one_well(ml_root: Path, experiment_cache: dict, well_str: str, args) -> tuple[str, list["Recording"], dict] | None:
    """Discover, load, pool, and fit (UMAP+PCA) one physical well. Returns a figure block or None."""
    sample, plate, well_id = _parse_well_uid(well_str)
    well_uid = f"{sample}|{plate}|{well_id}"

    traces = _discover_recordings(ml_root, sample, plate, well_id, args.scan_type)
    if not traces:
        logger.warning("no debug_trace.pkl found for %s under %s — skipping", well_uid, ml_root)
        return None
    logger.info("[%s] found %d recording traces", well_uid, len(traces))

    plating = _plating_lookup(experiment_cache, sample, plate, well_id)
    logger.info("[%s] plating date: %s", well_uid, plating.date() if plating else "unknown (DIV unavailable)")

    recs: list[Recording] = []
    feat_ref: list[str] | None = None
    for tp in traces:
        rec, feat_ref = _load_recording(tp, ml_root, plating, args.max_bins_per_rec, feat_ref)
        if rec is not None:
            recs.append(rec)
    if not recs:
        logger.warning("[%s] no usable recordings after loading — skipping", well_uid)
        return None
    if len(recs) < 2:
        logger.warning("[%s] only %d usable recording — overlay needs >=2 to show drift", well_uid, len(recs))

    recs.sort(key=lambda r: (r.date, r.run_id, r.rec_name))
    total_bins = sum(int(r.X_raw.shape[0]) for r in recs)
    logger.info("[%s] pooled %d recordings, %d bins (max %d/rec)",
                well_uid, len(recs), total_bins, args.max_bins_per_rec)

    Z = _pool_matrix(recs, args.norm)
    logger.info("[%s] pooled feature matrix %s (norm=%s)", well_uid, Z.shape, args.norm)

    # compute both projections; order puts --method first (dropdown default)
    all_coords = {
        "umap": (_fit_umap(Z, args.n_neighbors, args.min_dist, args.seed), "UMAP-1", "UMAP-2"),
        "pca": _fit_pca(Z, feat_ref or [], args.seed),
    }
    method_coords: dict[str, tuple[np.ndarray, str, str]] = {}
    for name in ([args.method] + [m for m in ("umap", "pca") if m != args.method]):
        method_coords[name] = all_coords[name]
    return well_uid, recs, method_coords


if __name__ == "__main__":
    sys.exit(main())
