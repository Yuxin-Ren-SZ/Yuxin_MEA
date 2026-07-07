"""Shared-axis bin-level UMAP/PCA overlay — the reusable core.

Pools the per-bin 26-D feature matrices of several "series" (e.g. all recordings
of one physical well, or all wells of one recording) into ONE shared
normalization + UMAP/PCA fit, so every series sits on the same axes/loadings.
Renders an interactive Plotly figure: color = series, marker shape = burst vs
non-burst, method (UMAP/PCA) and Both/Burst/Rest toggles, optional time-lapse
animation over the series.

Used by both ``scripts/well_recording_umap_overlay.py`` (standalone CLI,
series = recordings of one well) and the in-pipeline cluster-overlay aggregate
tasks (series = wells of one recording, or recordings of one well).

``color_by`` ("recording" | "well") only changes legend/title/caption wording;
the embedding and per-series coloring are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.colors as pcolors
import plotly.graph_objects as go

from .ml_trace_io import burst_label_set, load_json, load_trace, znorm_from_trace

NONBURST_SYMBOL = "circle"
BURST_SYMBOL = "cross"  # bold "+", very unlike a circle
MARKER_ALPHA = 0.5

_OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
              "#0072B2", "#D55E00", "#CC79A7", "#000000"]
_QUAL_PALETTES = {
    "dark24": pcolors.qualitative.Dark24,
    "safe": pcolors.qualitative.Safe,
    "okabe": _OKABE_ITO,
}
_SEQ_PALETTES = {"cividis": "Cividis", "turbo": "Turbo", "viridis": "Viridis"}
PALETTE_CHOICES = list(_QUAL_PALETTES) + list(_SEQ_PALETTES)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class SeriesTrace:
    """One series (a recording, or a well) pooled into the shared embedding."""

    label: str
    X_raw: np.ndarray
    Z_local: np.ndarray | None
    is_burst: np.ndarray
    t_centers: np.ndarray
    labels: np.ndarray


def stride_keep(n: int, max_bins: int) -> np.ndarray:
    if n <= max_bins:
        return np.arange(n)
    stride = int(np.ceil(n / max_bins))
    return np.arange(0, n, stride)[:max_bins]


def load_series_trace(
    trace_dir: Path, label: str, max_bins: int, feat_ref: list[str] | None
) -> tuple[SeriesTrace | None, list[str] | None]:
    """Load + subsample one series from ``trace_dir`` (an ``ml_burst_detection`` dir).

    Reads ``debug_trace.pkl`` + ``diagnostics.json``. Returns ``(SeriesTrace | None,
    feature_names)``; the feature-name reference is threaded so all series in a
    pool are asserted to share the same 26 columns.
    """
    trace_dir = Path(trace_dir)
    trace = load_trace(trace_dir / "debug_trace.pkl")
    if trace is None:
        return None, feat_ref
    X = getattr(trace, "feature_matrix", None)
    labels = getattr(trace, "hdbscan_labels", None)
    t = getattr(trace, "t_centers", None)
    fnames = list(getattr(trace, "feature_names", None) or [])
    if X is None or labels is None:
        return None, feat_ref
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels).astype(int)
    t = np.asarray(t, dtype=float) if t is not None else np.arange(X.shape[0], dtype=float)
    if X.ndim != 2 or X.shape[0] < 10:
        return None, feat_ref
    if not (X.shape[0] == labels.shape[0] == t.shape[0]):
        return None, feat_ref
    if feat_ref is not None and fnames and fnames != feat_ref:
        return None, feat_ref
    if feat_ref is None and fnames:
        feat_ref = fnames

    diagnostics = load_json(trace_dir / "diagnostics.json")
    burst_set = burst_label_set(diagnostics)
    is_burst_full = np.isin(labels, list(burst_set)) if burst_set else np.zeros(X.shape[0], bool)
    Z_local = znorm_from_trace(trace)

    keep = stride_keep(X.shape[0], max_bins)
    st = SeriesTrace(
        label=label,
        X_raw=X[keep],
        Z_local=(Z_local[keep] if Z_local is not None else None),
        is_burst=is_burst_full[keep],
        t_centers=t[keep],
        labels=labels[keep],
    )
    return st, feat_ref


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def pool_matrix(series: list[SeriesTrace], norm: str) -> np.ndarray:
    if norm == "per-recording":
        if any(s.Z_local is None for s in series):
            raise ValueError("--norm per-recording needs scaler stats in every trace; some missing")
        return np.vstack([s.Z_local for s in series])
    from sklearn.preprocessing import StandardScaler
    X = np.vstack([s.X_raw for s in series])
    return StandardScaler().fit_transform(X)


def fit_umap(Z: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    try:
        import umap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"umap-learn not importable: {exc}")
    nn = max(2, min(int(n_neighbors), Z.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=float(min_dist),
                        metric="euclidean", random_state=seed)
    return np.asarray(reducer.fit_transform(Z))


def fit_pca(Z: np.ndarray, feat_names: list[str], seed: int) -> tuple[np.ndarray, str, str]:
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


def pool_and_fit(
    series: list[SeriesTrace], norm: str, feat_names: list[str] | None,
    method: str = "umap", n_neighbors: int = 30, min_dist: float = 0.1, seed: int = 42,
) -> dict[str, tuple[np.ndarray, str, str]]:
    """Pool + fit both UMAP and PCA; return ``{method: (coords, xlabel, ylabel)}``
    with ``method`` (the primary view) first."""
    Z = pool_matrix(series, norm)
    all_coords = {
        "umap": (fit_umap(Z, n_neighbors, min_dist, seed), "UMAP-1", "UMAP-2"),
        "pca": fit_pca(Z, feat_names or [], seed),
    }
    ordered: dict[str, tuple[np.ndarray, str, str]] = {}
    for name in ([method] + [m for m in ("umap", "pca") if m != method]):
        ordered[name] = all_coords[name]
    return ordered


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def series_colors(n: int, palette: str = "dark24") -> list[str]:
    if palette in _QUAL_PALETTES:
        base = _QUAL_PALETTES[palette]
        return [base[i % len(base)] for i in range(n)]
    name = _SEQ_PALETTES.get(palette, "Cividis")
    if n == 1:
        return list(pcolors.sample_colorscale(name, [0.5]))
    return list(pcolors.sample_colorscale(name, list(np.linspace(0, 1, n))))


def _marker(state_is_burst: bool, color: str) -> dict:
    if state_is_burst:
        return dict(color=color, symbol=BURST_SYMBOL, size=9, opacity=MARKER_ALPHA,
                    line=dict(width=1.6, color="#111"))
    return dict(color=color, symbol=NONBURST_SYMBOL, size=5, opacity=MARKER_ALPHA,
                line=dict(width=0))


def _fig_title(title_label: str, method: str, series: list[SeriesTrace],
               norm: str, unit: str) -> str:
    nb = sum(int(s.X_raw.shape[0]) for s in series)
    return (f"{title_label} — {method.upper()} shared axis (norm={norm}) · "
            f"{len(series)} {unit} · {nb} bins")


def build_multi_figure(blocks, norm: str, palette: str = "dark24",
                       color_by: str = "recording") -> go.Figure:
    """Static overlay over one or more blocks. Each block = (title_label, series,
    method_coords). With >1 block a title·method dropdown switches which block is
    visible (each has its own coordinate space)."""
    unit = f"{color_by}s"
    fig = go.Figure()
    multi = len(blocks) > 1
    methods = list(blocks[0][2].keys())
    first_combo = f"{blocks[0][0]}|{methods[0]}"

    trace_combo: list[str] = []
    trace_is_burst: list[bool | None] = []

    for title_label, series, method_coords in blocks:
        colors = series_colors(len(series), palette)
        sizes = [s.X_raw.shape[0] for s in series]
        offsets = np.cumsum([0] + sizes)
        short = title_label.split("|")[-1]
        for method in method_coords:
            coords, _xl, _yl = method_coords[method]
            combo = f"{title_label}|{method}"
            for si, s in enumerate(series):
                sl = slice(int(offsets[si]), int(offsets[si + 1]))
                xy = coords[sl]
                for is_burst, tag in ((False, "rest"), (True, "burst")):
                    m = s.is_burst == is_burst
                    if not m.any():
                        continue
                    hover = [
                        f"{short} {s.label}<br>state={tag}<br>t={tc:.1f}s<br>cluster={cl}"
                        for tc, cl in zip(s.t_centers[m], s.labels[m])
                    ]
                    fig.add_trace(go.Scattergl(
                        x=xy[m, 0], y=xy[m, 1], mode="markers",
                        marker=_marker(is_burst, colors[si]),
                        name=(f"{short} {s.label}" if multi else s.label),
                        legendgroup=f"{title_label}|{s.label}",
                        showlegend=(not is_burst),
                        hovertext=hover, hoverinfo="text",
                        visible=(combo == first_combo),
                    ))
                    trace_combo.append(combo)
                    trace_is_burst.append(is_burst)

    for is_burst, nm in ((False, "○ rest"), (True, "✚ burst")):
        fig.add_trace(go.Scattergl(
            x=[None], y=[None], mode="markers", marker=_marker(is_burst, "#333"),
            name=nm, legendgroup="_symbol", showlegend=True, hoverinfo="skip", visible=True,
        ))
        trace_combo.append("_always")
        trace_is_burst.append(None)

    wm_buttons = []
    for title_label, series, method_coords in blocks:
        short = title_label.split("|")[-1]
        for method in method_coords:
            combo = f"{title_label}|{method}"
            _coords, xl, yl = method_coords[method]
            vis = [(tc == combo) or (tc == "_always") for tc in trace_combo]
            wm_buttons.append(dict(
                label=(f"{short} · {method.upper()}" if multi else method.upper()),
                method="update",
                args=[{"visible": vis},
                      {"xaxis.title.text": xl, "yaxis.title.text": yl,
                       "title.text": _fig_title(title_label, method, series, norm, unit)}],
            ))

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

    t0, series0, mc0 = blocks[0]
    _, xl0, yl0 = mc0[methods[0]]
    fig.update_layout(
        title=dict(text=_fig_title(t0, methods[0], series0, norm, unit),
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
        legend=dict(title=color_by, x=1.02, xanchor="left", y=1.0, yanchor="top",
                    itemsizing="constant", itemclick="toggle", itemdoubleclick="toggleothers"),
        template="plotly_white", height=800, width=1200,
        margin=dict(l=70, r=220, t=150, b=110),
    )
    fig.add_annotation(
        text=(f"color = {color_by} · ○ rest / ✚ burst · "
              f"legend: click = toggle {color_by}, double-click = isolate · "
              "PCA axes show loadings; UMAP distances not metric."),
        xref="paper", yref="paper", x=0.0, xanchor="left", y=-0.12, yanchor="top",
        showarrow=False, font=dict(size=11, color="#666"), align="left",
    )
    return fig


def build_single_animated(block, norm: str, palette: str = "dark24",
                          color_by: str = "recording") -> go.Figure:
    """Single-block figure with a time-lapse over the series (block = (title, series,
    method_coords)). method→x/y restyle, state→marker.opacity, animation→visible."""
    unit = f"{color_by}s"
    title_label, series, method_coords = block
    methods = list(method_coords.keys())
    default_m = methods[0]
    colors = series_colors(len(series), palette)
    sizes = [s.X_raw.shape[0] for s in series]
    offsets = np.cumsum([0] + sizes)
    K = len(series)

    fig = go.Figure()
    px: dict[str, list] = {m: [] for m in methods}
    py: dict[str, list] = {m: [] for m in methods}
    trace_rank: list[int | None] = []
    trace_is_burst: list[bool | None] = []

    for si, s in enumerate(series):
        for is_burst, tag in ((False, "rest"), (True, "burst")):
            m = s.is_burst == is_burst
            if not m.any():
                continue
            for name in methods:
                xy = method_coords[name][0][int(offsets[si]):int(offsets[si + 1])][m]
                px[name].append(xy[:, 0])
                py[name].append(xy[:, 1])
            xy0 = method_coords[default_m][0][int(offsets[si]):int(offsets[si + 1])][m]
            hover = [
                f"{s.label}<br>state={tag}<br>t={tc:.1f}s<br>cluster={cl}"
                for tc, cl in zip(s.t_centers[m], s.labels[m])
            ]
            fig.add_trace(go.Scattergl(
                x=xy0[:, 0], y=xy0[:, 1], mode="markers",
                marker=_marker(is_burst, colors[si]),
                name=s.label, legendgroup=f"series|{s.label}",
                showlegend=(not is_burst),
                hovertext=hover, hoverinfo="text", visible=True,
            ))
            trace_rank.append(si)
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

    method_buttons = []
    for name in methods:
        _c, xl, yl = method_coords[name]
        method_buttons.append(dict(
            label=name.upper(), method="update",
            args=[{"x": px[name], "y": py[name]},
                  {"xaxis.title.text": xl, "yaxis.title.text": yl,
                   "title.text": _fig_title(title_label, name, series, norm, unit)}],
        ))

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

    def _vis(k: int, cumulative: bool) -> list:
        out = []
        for rk in trace_rank:
            if rk is None:
                out.append(True)
            elif (rk <= k) if cumulative else (rk == k):
                out.append(True)
            else:
                out.append("legendonly")
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
        steps=[dict(method="animate", label=series[k].label, args=[[f"cum{k}"], step_opts])
               for k in range(K)],
    )

    _, xl0, yl0 = method_coords[default_m]
    fig.update_layout(
        title=dict(text=_fig_title(title_label, default_m, series, norm, unit),
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
        legend=dict(title=color_by, x=1.02, xanchor="left", y=1.0, yanchor="top",
                    itemsizing="constant", itemclick="toggle", itemdoubleclick="toggleothers"),
        template="plotly_white", height=860, width=1200,
        margin=dict(l=70, r=220, t=150, b=200),
    )
    fig.add_annotation(
        text=(f"time-lapse: <b>▶ Leave previous</b> = accumulate · <b>▶ Current only</b> = one at a time · "
              f"slider scrubs cumulative &nbsp;|&nbsp; UMAP/PCA (top-left) · Both/Burst/Rest (top-right) · "
              f"color = {color_by} · ○ rest / ✚ burst · PCA axes show loadings."),
        xref="paper", yref="paper", x=0.0, xanchor="left", y=-0.30, yanchor="top",
        showarrow=False, font=dict(size=11, color="#666"), align="left",
    )
    return fig


def build_overlay_figure(blocks, *, color_by: str = "recording", animate: bool = True,
                         palette: str = "dark24", norm: str = "shared") -> go.Figure:
    """Dispatch: single block + animate → time-lapse; else static overlay."""
    if len(blocks) == 1 and animate:
        return build_single_animated(blocks[0], norm, palette, color_by)
    return build_multi_figure(blocks, norm, palette, color_by)
