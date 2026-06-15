"""Per-well diagnostic library for burst detectors.

Pure-library figure builders used by
``yuxin_mea.dashboard.pages.burst_inspector``. No Dash imports — so the same
figures can be rendered in notebooks/HTML reports.

Supports three methods:
- **iterative** — full diagnostic traces (iterations, LDA, GMM)
- **traditional** — standard BurstResults only (no debug trace)
- **ml** — standard BurstResults + optional MLBurstTrace

Public API:
- ``InspectorBundle`` — dataclass holding everything one panel needs.
- ``load_inspector_bundle`` — disk-first loader with optional on-demand fallback.
- ``fig_*`` — one Plotly figure per UI panel.
- ``summary_card`` — returns a dict of KV rows (rendered as HTML by the page).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from yuxin_mea.analysis.iterative_burst_detector import (
    IterativeBurstConfig,
    IterativeBurstTrace,
    compute_iterative_bursts,
)

# Method → terminal directory name inside each well's output path.
METHOD_TERMINALS: dict[str, str] = {
    "traditional": "burst_detection",
    "iterative": "iterative_burst_detection",
    "ml": "ml_burst_detection",
}

# Method → pipeline config task name (for output_root lookup).
METHOD_TASK_NAMES: dict[str, str] = {
    "traditional": "burst_detection",
    "iterative": "iterative_burst_detection",
    "ml": "ml_burst_detection",
}

# Method → conventional subdir under analysis_root (fallback).
METHOD_SUBDIRS: dict[str, str] = {
    "traditional": "burst_detection_data",
    "iterative": "iterative_burst_data",
    "ml": "ml_burst_data",
}


# ---------------------------------------------------------------------------
# Bundle + loader
# ---------------------------------------------------------------------------


@dataclass
class InspectorBundle:
    """Everything one well's diagnostic panels need.

    ``source`` lets the page show a "disk" vs "on-demand" status badge so the
    user knows whether they're looking at persisted task output or a live
    in-process re-run.

    ``method`` indicates which detector produced the data. Algorithm-specific
    tabs are rendered only when the matching method is active.

    For non-iterative methods, ``trace`` and ``config`` are ``None`` and
    ``results`` holds the standard ``BurstResults``.
    """

    trace: IterativeBurstTrace | None
    spike_times: dict[str, np.ndarray]
    burstlets: pd.DataFrame
    config: IterativeBurstConfig | None
    recording_key: str
    rec_name: str
    well_id: str
    source: Literal["disk", "on_demand"]
    method: str = "iterative"
    output_dir: Path | None = None
    results: Any = None  # BurstResults — stored for non-iterative summary_card


def _try_load_disk(
    output_dir: Path,
) -> tuple[IterativeBurstTrace, dict, pd.DataFrame, IterativeBurstConfig | None] | None:
    """Read ``debug_trace.pkl``, ``debug_spike_times.npy``, ``burstlets.pkl``.

    Also reads ``debug_config.json`` if present so the inspector can show
    the *run-time* config the trace was produced with, not whatever the
    user's pipeline_config currently says. Returns ``None`` if any of the
    three primary artifacts is missing; ``debug_config.json`` is optional
    (back-compat with traces written before that file was added).
    """
    trace_path = output_dir / "debug_trace.pkl"
    spikes_path = output_dir / "debug_spike_times.npy"
    burstlets_path = output_dir / "burstlets.pkl"
    if not (trace_path.exists() and spikes_path.exists() and burstlets_path.exists()):
        return None
    with open(trace_path, "rb") as fh:
        trace = pickle.load(fh)
    spike_times = np.load(spikes_path, allow_pickle=True).item()
    burstlets = pd.read_pickle(burstlets_path)

    config: IterativeBurstConfig | None = None
    config_path = output_dir / "debug_config.json"
    if config_path.exists():
        import json
        try:
            with open(config_path) as fh:
                raw = json.load(fh)
            # JSON round-trip turned the tuple into a list — restore.
            if "ff_scale_multipliers" in raw and isinstance(raw["ff_scale_multipliers"], list):
                raw["ff_scale_multipliers"] = tuple(raw["ff_scale_multipliers"])
            config = IterativeBurstConfig(**raw)
        except (TypeError, ValueError):
            # Unknown / renamed fields → fall back to defaults rather than crash.
            config = None
    return trace, spike_times, burstlets, config


def load_inspector_bundle(
    output_root: Path,
    recording_key: str,
    rec_name: str,
    well_id: str,
    *,
    on_demand_spike_times: dict | None = None,
    on_demand_config: IterativeBurstConfig | None = None,
) -> InspectorBundle:
    """Disk-first load; fall back to recompute when ``debug_trace.pkl`` is missing.

    ``output_root`` follows the IterativeBurstDetectionTask layout:
    ``<output_root>/<recording_key>/<rec_name>/<well_id>/iterative_burst_detection/``.

    If the disk path is missing but ``on_demand_spike_times`` is provided,
    runs ``compute_iterative_bursts`` in-process with a fresh trace and
    returns ``source="on_demand"`` — without writing anything to disk.
    """
    output_dir = (
        Path(output_root)
        / recording_key
        / rec_name
        / well_id
        / "iterative_burst_detection"
    )

    disk = _try_load_disk(output_dir)
    if disk is not None:
        trace, spike_times, burstlets, disk_config = disk
        # Prefer the config that was actually used at run time (from
        # debug_config.json). Fall back to caller-provided / defaults only
        # for back-compat with traces written before debug_config.json existed.
        config = disk_config or on_demand_config or IterativeBurstConfig()
        return InspectorBundle(
            trace=trace,
            spike_times=spike_times,
            burstlets=burstlets,
            config=config,
            recording_key=recording_key,
            rec_name=rec_name,
            well_id=well_id,
            source="disk",
            output_dir=output_dir,
        )

    if on_demand_spike_times is None:
        raise FileNotFoundError(
            f"No debug artifacts at {output_dir}. Either re-run "
            "IterativeBurstDetectionTask with debug=True, or pass "
            "on_demand_spike_times to recompute in-process."
        )

    config = on_demand_config or IterativeBurstConfig()
    trace = IterativeBurstTrace()
    results = compute_iterative_bursts(
        on_demand_spike_times, config=config, trace=trace
    )
    return InspectorBundle(
        trace=trace,
        spike_times=on_demand_spike_times,
        burstlets=results.burstlets,
        config=config,
        recording_key=recording_key,
        rec_name=rec_name,
        well_id=well_id,
        source="on_demand",
        output_dir=output_dir if output_dir.exists() else None,
    )


def load_generic_bundle(
    output_root: Path,
    recording_key: str,
    rec_name: str,
    well_id: str,
    method: str,
) -> InspectorBundle:
    """Load a BurstResults bundle for any detector method.

    Used by the dashboard when the user picks traditional or ML. Reads the
    standard pickle output (burstlets.pkl, metrics.json, etc.) without
    requiring a debug trace.
    """
    from yuxin_mea.analysis.burst_output import PickleBurstOutputWriter

    terminal = METHOD_TERMINALS[method]
    output_dir = Path(output_root) / recording_key / rec_name / well_id / terminal
    if not output_dir.exists():
        raise FileNotFoundError(f"No output at {output_dir}")

    results = PickleBurstOutputWriter().read(output_dir)

    spike_times: dict[str, np.ndarray] = {}
    spikes_path = output_dir / "debug_spike_times.npy"
    if spikes_path.exists():
        spike_times = np.load(spikes_path, allow_pickle=True).item()

    return InspectorBundle(
        trace=None,
        spike_times=spike_times,
        burstlets=results.burstlets,
        config=None,
        recording_key=recording_key,
        rec_name=rec_name,
        well_id=well_id,
        source="disk",
        method=method,
        output_dir=output_dir,
        results=results,
    )


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------


_FINAL = "final"


def _resolve_iter_index(trace: IterativeBurstTrace, iteration: int | str) -> int:
    """Map a user-facing iteration selector to a valid index in ``trace.iterations``.

    ``"final"`` resolves to the last iteration. Out-of-range integers are
    clamped to the valid window so the slider never raises.
    """
    n = len(trace.iterations)
    if n == 0:
        raise ValueError("trace has no recorded iterations — was debug=True?")
    if iteration == _FINAL or iteration is None:
        return n - 1
    idx = int(iteration)
    if idx < 0:
        idx = 0
    if idx >= n:
        idx = n - 1
    return idx


def _empty_figure(message: str) -> go.Figure:
    """A single annotation centered in a blank figure — used for missing data states."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#888"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    return fig


# ---------------------------------------------------------------------------
# fig_raster
# ---------------------------------------------------------------------------


def _spike_times_by_str_uid(
    spike_times: dict[Any, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Re-key spike_times so lookup tolerates int↔str mismatch with trace.unit_ids.

    The detector stores ``unit_ids`` as ``list[str]`` on the trace but
    ``debug_spike_times.npy`` keeps them as the original ints. Always
    return a ``{str: array}`` view so callers can ``get(str(uid))``.
    """
    if not spike_times:
        return {}
    return {str(k): v for k, v in spike_times.items()}


def fig_raster(
    bundle: InspectorBundle,
    *,
    iteration: int | str = _FINAL,
    max_units: int = 64,
) -> go.Figure:
    """Per-unit spike raster + candidate-window overlay for one iteration.

    ``max_units`` caps how many units are plotted — large wells get
    subsampled to keep the figure interactive.
    """
    trace = bundle.trace
    spike_times = bundle.spike_times
    if not spike_times:
        return _empty_figure("No spike times available.")

    sp_by_uid = _spike_times_by_str_uid(spike_times)
    unit_ids = [str(u) for u in (trace.unit_ids or sorted(spike_times.keys()))]
    if len(unit_ids) > max_units:
        step = len(unit_ids) // max_units
        unit_ids = unit_ids[::step][:max_units]

    fig = go.Figure()
    for row, uid in enumerate(unit_ids):
        spikes = np.asarray(sp_by_uid.get(uid, []), dtype=float)
        if spikes.size == 0:
            continue
        fig.add_trace(
            go.Scattergl(
                x=spikes,
                y=np.full(spikes.size, row),
                mode="markers",
                marker=dict(size=2.5, color="#222", line=dict(width=0)),
                hoverinfo="skip",
                showlegend=False,
                name=uid,
            )
        )

    if trace.iterations:
        idx = _resolve_iter_index(trace, iteration)
        candidates = trace.iterations[idx].get("candidates", [])
        for c in candidates:
            fig.add_vrect(
                x0=float(c["start"]),
                x1=float(c["end"]),
                fillcolor="rgba(220, 80, 60, 0.12)",
                line_width=0,
                layer="below",
            )

    fig.update_layout(
        height=max(220, 12 * len(unit_ids)),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="time (s)",
        yaxis_title="unit",
        plot_bgcolor="white",
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(unit_ids))),
        ticktext=unit_ids,
        autorange="reversed",
    )
    return fig


# ---------------------------------------------------------------------------
# fig_per_bin_gmm_strip
# ---------------------------------------------------------------------------


# Qualitative palette for inner-GMM components. Cycles if k > 8.
_GMM_CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#17becf", "#bcbd22",
]


def fig_per_bin_gmm_strip(
    bundle: InspectorBundle, iteration: int | str = _FINAL
) -> go.Figure:
    """Horizontal heatmap of the inner GMM's per-bin component ids vs time.

    Shows what ``_fit_gmm_em`` decided each bin's burst-vs-background-vs-
    intermediate identity was, alongside which component(s) were called
    burst. Two rows, x-axis shared with the raster above:
      • top    — single-row band marking burst-group bins (red on white).
      • bottom — per-bin GMM component id, qualitative colormap, k_chosen
                 distinct colours.

    Falls back to the binary burst/background band alone when
    ``bin_component_id`` isn't on the trace (debug artifact written before
    that key was persisted).
    """
    trace = bundle.trace
    if not trace.iterations or trace.t_centers is None:
        return _empty_figure("No iterations recorded.")

    idx = _resolve_iter_index(trace, iteration)
    entry = trace.iterations[idx]
    t = np.asarray(trace.t_centers, dtype=float)
    mask = np.asarray(entry["candidate_mask"], dtype=bool)
    k_chosen = int(entry.get("k_chosen", 2))

    bin_cid = entry.get("bin_component_id")
    if bin_cid is None:
        bin_cid = mask.astype(np.int16)
        k_chosen = 2
        legacy = True
    else:
        bin_cid = np.asarray(bin_cid, dtype=int)
        legacy = False

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.35, 0.65],
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            "burst-group bins (mask)",
            "per-bin GMM cluster"
            + (
                f" (k={k_chosen}, burst components = "
                f"{entry.get('burst_group_members', [])})"
                if not legacy
                else " (legacy trace — burst/background only)"
            ),
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=t,
            z=mask.astype(int).reshape(1, -1),
            colorscale=[[0, "#f6f6f6"], [1, "#c62828"]],
            showscale=False,
            hovertemplate="t=%{x:.2f}s<br>burst=%{z}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Qualitative colorscale: split [0,1] into k_chosen equal bands.
    if k_chosen <= 1:
        cscale = [[0.0, _GMM_CLUSTER_COLORS[0]], [1.0, _GMM_CLUSTER_COLORS[0]]]
    else:
        cscale = []
        for i in range(k_chosen):
            color = _GMM_CLUSTER_COLORS[i % len(_GMM_CLUSTER_COLORS)]
            cscale.append([i / k_chosen, color])
            cscale.append([(i + 1) / k_chosen, color])

    fig.add_trace(
        go.Heatmap(
            x=t,
            z=bin_cid.reshape(1, -1),
            zmin=0,
            zmax=max(1, k_chosen - 1),
            colorscale=cscale,
            showscale=False,
            hovertemplate="t=%{x:.2f}s<br>component=%{z}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_yaxes(showticklabels=False, zeroline=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, zeroline=False, showgrid=False, row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_layout(
        height=160,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# fig_composite_with_threshold
# ---------------------------------------------------------------------------


def fig_composite_with_threshold(
    bundle: InspectorBundle, iteration: int | str = _FINAL
) -> go.Figure:
    """Composite trace with threshold/baseline lines and LDA-weight inset."""
    trace = bundle.trace
    if not trace.iterations or trace.t_centers is None:
        return _empty_figure("No iterations recorded.")

    idx = _resolve_iter_index(trace, iteration)
    entry = trace.iterations[idx]
    composite = np.asarray(entry["composite"], dtype=float)
    threshold = float(entry["composite_threshold"])
    baseline = float(entry.get("composite_baseline", 0.0))
    t = np.asarray(trace.t_centers, dtype=float)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.78, 0.22],
        subplot_titles=(
            f"composite signal — iter {idx} (Δ={entry.get('convergence_delta', 0):.4f})",
            "LDA weights w",
        ),
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scattergl(x=t, y=composite, mode="lines",
                     line=dict(color="#1f3a93", width=1.2), name="composite"),
        row=1, col=1,
    )
    fig.add_hline(y=threshold, line=dict(color="#c62828", width=1, dash="dash"),
                  annotation_text=f"threshold {threshold:.2f}",
                  annotation_position="top right", row=1, col=1)
    fig.add_hline(y=baseline, line=dict(color="#888", width=1, dash="dot"),
                  row=1, col=1)

    for c in entry.get("candidates", []):
        fig.add_vrect(
            x0=float(c["start"]), x1=float(c["end"]),
            fillcolor="rgba(220, 80, 60, 0.10)", line_width=0,
            layer="below", row=1, col=1,
        )

    w = np.asarray(entry.get("w", []), dtype=float)
    names = trace.feature_names or [f"F{i}" for i in range(w.size)]
    if w.size:
        colors = ["#1f3a93" if v >= 0 else "#c62828" for v in w]
        fig.add_trace(
            go.Bar(x=w, y=names, orientation="h", marker_color=colors,
                   showlegend=False),
            row=1, col=2,
        )
        fig.update_xaxes(title_text="weight", row=1, col=2)

    fig.update_xaxes(title_text="time (s)", row=1, col=1)
    fig.update_yaxes(title_text="composite", row=1, col=1)
    fig.update_layout(height=360, margin=dict(l=50, r=20, t=50, b=40),
                      plot_bgcolor="white", showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# fig_iteration_trajectory
# ---------------------------------------------------------------------------


def fig_iteration_trajectory(bundle: InspectorBundle) -> go.Figure:
    """n_candidates, convergence_delta, composite_threshold across iterations."""
    trace = bundle.trace
    if not trace.iterations:
        return _empty_figure("No iterations recorded.")

    iters = [int(e["iter"]) for e in trace.iterations]
    n_cands = [int(e["n_candidates"]) for e in trace.iterations]
    deltas = [float(e.get("convergence_delta", 0)) for e in trace.iterations]
    thrs = [float(e["composite_threshold"]) for e in trace.iterations]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("n_candidates", "convergence Δ", "composite threshold"),
        horizontal_spacing=0.10,
    )
    fig.add_trace(go.Scatter(x=iters, y=n_cands, mode="lines+markers",
                             line=dict(color="#1f3a93")), row=1, col=1)
    fig.add_trace(go.Scatter(x=iters, y=deltas, mode="lines+markers",
                             line=dict(color="#c62828")), row=1, col=2)
    fig.add_trace(go.Scatter(x=iters, y=thrs, mode="lines+markers",
                             line=dict(color="#2e7d32")), row=1, col=3)

    eps = bundle.config.convergence_eps
    fig.add_hline(y=eps, line=dict(color="#888", width=1, dash="dot"),
                  annotation_text=f"ε = {eps}", annotation_position="top right",
                  row=1, col=2)

    for c in range(1, 4):
        fig.update_xaxes(title_text="iteration", row=1, col=c)
    fig.update_layout(height=280, margin=dict(l=50, r=20, t=50, b=40),
                      showlegend=False, plot_bgcolor="white")
    return fig


# ---------------------------------------------------------------------------
# fig_pca_feature_space
# ---------------------------------------------------------------------------


def fig_pca_feature_space(
    bundle: InspectorBundle, iteration: int | str = _FINAL
) -> go.Figure:
    """PCA(2) on X_norm coloured by candidate_mask + PC1/PC2 loadings.

    Left: bins in PC1/PC2 space, coloured by the algorithm's burst/background
    label at end of the iteration.
    Right: feature loadings on PC1 and PC2 — surfaces which features
    contributed to the variance the algorithm separates on.
    """
    from sklearn.decomposition import PCA

    trace = bundle.trace
    if not trace.iterations:
        return _empty_figure("No iterations recorded.")

    idx = _resolve_iter_index(trace, iteration)
    entry = trace.iterations[idx]
    X = np.asarray(entry["X_norm"], dtype=float)
    mask = np.asarray(entry["candidate_mask"], dtype=bool)
    names = trace.feature_names or [f"F{i}" for i in range(X.shape[1])]

    if X.size == 0:
        return _empty_figure("Empty X_norm.")

    n_components = min(2, X.shape[1])
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=(
            f"X_norm PCA — iter {idx} ({int(mask.sum())} burst / {int((~mask).sum())} bg bins)",
            "feature loadings",
        ),
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Scattergl(
            x=Z[~mask, 0], y=Z[~mask, 1] if n_components > 1 else np.zeros((~mask).sum()),
            mode="markers", name="background",
            marker=dict(size=4, color="rgba(120,120,120,0.45)", line=dict(width=0)),
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=Z[mask, 0], y=Z[mask, 1] if n_components > 1 else np.zeros(mask.sum()),
            mode="markers", name="burst",
            marker=dict(size=5, color="#c62828", line=dict(width=0)),
        ), row=1, col=1,
    )

    loadings = pca.components_  # shape (n_components, n_features)
    fig.add_trace(
        go.Bar(name="PC1", x=loadings[0], y=names, orientation="h",
               marker_color="#1f3a93"),
        row=1, col=2,
    )
    if n_components > 1:
        fig.add_trace(
            go.Bar(name="PC2", x=loadings[1], y=names, orientation="h",
                   marker_color="#2e7d32"),
            row=1, col=2,
        )

    fig.update_xaxes(title_text=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                     row=1, col=1)
    if n_components > 1:
        fig.update_yaxes(title_text=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                         row=1, col=1)
    fig.update_xaxes(title_text="loading", row=1, col=2)
    fig.update_layout(height=420, margin=dict(l=60, r=20, t=50, b=50),
                      barmode="group", plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.18))
    return fig


# ---------------------------------------------------------------------------
# fig_event_gmm_clusters
# ---------------------------------------------------------------------------


def fig_event_gmm_clusters(bundle: InspectorBundle) -> go.Figure:
    """PCA(2) of per-event GMM features, side-by-side raw labels vs kept/killed.

    Surfaces the post-convergence event GMM (``trace.gmm``) — the step
    that most often drops real bursts on hyperactive wells.
    """
    from sklearn.decomposition import PCA

    g = bundle.trace.gmm
    if g is None:
        return _empty_figure("trace.gmm missing.")
    if "skipped" in g:
        return _empty_figure(
            f"GMM clustering skipped: {g['skipped']} (n_events={g.get('n_events', 0)})"
        )

    X = np.asarray(g["X"], dtype=float)
    if X.shape[0] < 2:
        return _empty_figure("Too few events for PCA.")
    labels = np.asarray(g["labels"], dtype=int)
    kept = np.asarray(g["kept_event_mask"], dtype=bool)
    names = list(g.get("feature_names", []))

    n_components = min(2, X.shape[1])
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(np.asarray(g["X_scaled"], dtype=float))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"raw GMM clusters (k={int(g.get('n_initial_components', 0))})",
            f"kept vs killed ({int(kept.sum())}/{kept.size})",
        ),
        horizontal_spacing=0.10,
    )

    palette = [
        "#1f3a93", "#c62828", "#2e7d32", "#f9a825",
        "#6a1b9a", "#00838f", "#ef6c00", "#5d4037",
    ]
    for lb in np.unique(labels):
        m = labels == lb
        fig.add_trace(
            go.Scattergl(
                x=Z[m, 0],
                y=Z[m, 1] if n_components > 1 else np.zeros(m.sum()),
                mode="markers", name=f"cluster {int(lb)}",
                marker=dict(size=7, color=palette[int(lb) % len(palette)],
                            line=dict(width=0.5, color="#fff")),
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scattergl(
            x=Z[~kept, 0],
            y=Z[~kept, 1] if n_components > 1 else np.zeros((~kept).sum()),
            mode="markers", name="killed",
            marker=dict(size=7, color="rgba(180,180,180,0.6)",
                        symbol="x", line=dict(width=1, color="#666")),
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scattergl(
            x=Z[kept, 0],
            y=Z[kept, 1] if n_components > 1 else np.zeros(kept.sum()),
            mode="markers", name="kept",
            marker=dict(size=8, color="#c62828", line=dict(width=0.5, color="#fff")),
        ),
        row=1, col=2,
    )

    xtitle = (
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
        + (f" · features: {', '.join(names)}" if names else "")
    )
    fig.update_xaxes(title_text=xtitle, row=1, col=1)
    fig.update_xaxes(title_text="PC1", row=1, col=2)
    if n_components > 1:
        fig.update_yaxes(
            title_text=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            row=1, col=1,
        )
    fig.update_layout(height=420, margin=dict(l=50, r=20, t=50, b=60),
                      plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.18))
    return fig


# ---------------------------------------------------------------------------
# fig_label_comparison_table
# ---------------------------------------------------------------------------


_KEY_FIELDS = ("start", "end", "peak_time")


def _event_key(ev: dict) -> tuple[float, float, float]:
    return tuple(float(ev.get(k, np.nan)) for k in _KEY_FIELDS)


def _classify_events(trace: IterativeBurstTrace) -> pd.DataFrame:
    """Per-event row with raw GMM cluster, final keep/kill, and kill reason.

    ``kill_reason`` is the first stage that dropped the event:
    ``participation`` → ``BMI`` → ``GMM`` → ``—`` (survived all gates).
    """
    pre = trace.burstlets_pre_gates or []
    pg = trace.participation_gate or {}
    bmi = trace.bmi_gate or {}
    gmm = trace.gmm or {}

    dropped_by_participation = {_event_key(e) for e in pg.get("dropped_events", [])}
    bmi_pre_events = bmi.get("pre_events", [])
    bmi_pre_keys = [_event_key(e) for e in bmi_pre_events]
    bmi_threshold = float(bmi.get("threshold", 0.0))
    bmi_enabled = bool(bmi.get("enabled", False))

    gmm_kept_mask = np.asarray(gmm.get("kept_event_mask", []), dtype=bool)
    gmm_labels = np.asarray(gmm.get("labels", []), dtype=int)
    gmm_skipped = "skipped" in gmm

    bmi_to_gmm_idx: dict[tuple, int] = {}
    if not gmm_skipped:
        kept_post_bmi = []
        for ev in bmi_pre_events:
            ag = float(ev.get("llr_aggregate", 0.0))
            if not bmi_enabled or ag >= bmi_threshold:
                kept_post_bmi.append(ev)
        for i, ev in enumerate(kept_post_bmi):
            bmi_to_gmm_idx[_event_key(ev)] = i

    rows = []
    for ev in pre:
        k = _event_key(ev)
        gmm_cluster_id: int | str = "—"
        kept = True
        kill_reason = "—"

        if k in dropped_by_participation:
            kept = False
            kill_reason = "participation"
        elif bmi_enabled and k in bmi_pre_keys:
            ag = float(ev.get("llr_aggregate", 0.0))
            if ag < bmi_threshold:
                kept = False
                kill_reason = "BMI"

        if kept and not gmm_skipped and k in bmi_to_gmm_idx:
            gi = bmi_to_gmm_idx[k]
            if gi < gmm_labels.size:
                gmm_cluster_id = int(gmm_labels[gi])
            if gi < gmm_kept_mask.size and not bool(gmm_kept_mask[gi]):
                kept = False
                kill_reason = "GMM"

        rows.append({
            "peak_time_s": float(ev.get("peak_time", np.nan)),
            "start_s": float(ev.get("start", np.nan)),
            "end_s": float(ev.get("end", np.nan)),
            "composite_peak": float(ev.get("composite_peak", np.nan)),
            "llr_aggregate": float(ev.get("llr_aggregate", np.nan)),
            "participation": float(ev.get("participation", np.nan)),
            "gmm_cluster_id": gmm_cluster_id,
            "kept": "Y" if kept else "N",
            "kill_reason": kill_reason,
        })

    return pd.DataFrame(rows)


def fig_label_comparison_table(bundle: InspectorBundle) -> go.Figure:
    """Plotly table — one row per pre-gate event with final + GMM labels."""
    df = _classify_events(bundle.trace)
    if df.empty:
        return _empty_figure("No pre-gate events recorded.")

    columns = list(df.columns)
    cell_values = [
        [
            (f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else str(v))
            for v in df[col]
        ]
        for col in columns
    ]

    kept_color = ["#e7f5e9" if k == "Y" else "#fcecec" for k in df["kept"]]
    fill_colors = [kept_color] * len(columns)

    fig = go.Figure(data=[go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in columns],
                    fill_color="#f0f1f3",
                    align="left", font=dict(size=11)),
        cells=dict(values=cell_values, fill_color=fill_colors,
                   align="left", font=dict(size=10, family="monospace"),
                   height=22),
    )])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=400)
    return fig


# ---------------------------------------------------------------------------
# summary_card — KV dict for the page, not a figure
# ---------------------------------------------------------------------------


def summary_card(bundle: InspectorBundle) -> dict[str, Any]:
    """Compact summary of detector state — rendered as a KV card by the page."""
    if bundle.method != "iterative" or bundle.trace is None:
        return _summary_card_generic(bundle)

    trace = bundle.trace
    df = _classify_events(trace)
    kill_breakdown = (
        df["kill_reason"].value_counts(dropna=False).to_dict()
        if not df.empty else {}
    )

    last = trace.iterations[-1] if trace.iterations else {}
    converged = bool(last.get("converged", False))
    n_iter = len(trace.iterations)
    delta = float(last.get("convergence_delta", float("nan")))

    return {
        "well": f"{bundle.recording_key} / {bundle.rec_name} / {bundle.well_id}",
        "source": bundle.source,
        "method": bundle.method,
        "n_units": len(bundle.spike_times),
        "n_iterations_run": n_iter,
        "converged": converged,
        "final_convergence_delta": delta,
        "convergence_eps": bundle.config.convergence_eps,
        "max_iterations": bundle.config.max_iterations,
        "n_pre_gate_events": int(len(df)),
        "n_final_burstlets": int(len(bundle.burstlets)),
        "kill_breakdown": kill_breakdown,
        "min_burst_modulation": bundle.config.min_burst_modulation,
        "cluster_events_enabled": bool(bundle.config.cluster_events),
        "bin_size_s": trace.bin_size,
    }


def _summary_card_generic(bundle: InspectorBundle) -> dict[str, Any]:
    """Summary card for traditional / ML bundles (no iterative trace)."""
    card: dict[str, Any] = {
        "well": f"{bundle.recording_key} / {bundle.rec_name} / {bundle.well_id}",
        "source": bundle.source,
        "method": bundle.method,
        "n_units": len(bundle.spike_times) if bundle.spike_times else "n/a",
        "n_burstlets": int(len(bundle.burstlets)),
    }
    if bundle.results is not None:
        card["n_network_bursts"] = int(len(bundle.results.network_bursts))
        card["n_superbursts"] = int(len(bundle.results.superbursts))
        for key in ("n_units", "adaptive_bin_ms", "burst_activity_detected"):
            val = bundle.results.diagnostics.get(key)
            if val is not None:
                card[key] = val
    return card


# ---------------------------------------------------------------------------
# Basic figures for non-iterative methods
# ---------------------------------------------------------------------------


def fig_raster_basic(bundle: InspectorBundle, max_units: int = 64) -> go.Figure:
    """Spike raster with burstlet event overlays — no iteration trace needed."""
    spike_times = bundle.spike_times
    if not spike_times:
        return _empty_figure("No spike times available.")

    sp_by_uid = _spike_times_by_str_uid(spike_times)
    unit_ids = sorted(sp_by_uid.keys())
    if len(unit_ids) > max_units:
        step = len(unit_ids) // max_units
        unit_ids = unit_ids[::step][:max_units]

    fig = go.Figure()
    for row, uid in enumerate(unit_ids):
        spikes = np.asarray(sp_by_uid.get(uid, []), dtype=float)
        if spikes.size == 0:
            continue
        fig.add_trace(go.Scattergl(
            x=spikes, y=np.full(spikes.size, row),
            mode="markers",
            marker=dict(size=2.5, color="#222", line=dict(width=0)),
            hoverinfo="skip", showlegend=False, name=uid,
        ))

    for _, evt in bundle.burstlets.iterrows():
        fig.add_vrect(
            x0=float(evt["start"]), x1=float(evt["end"]),
            fillcolor="rgba(220, 80, 60, 0.12)", line_width=0, layer="below",
        )

    fig.update_layout(
        height=max(220, 12 * len(unit_ids)),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="time (s)", yaxis_title="unit", plot_bgcolor="white",
    )
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(len(unit_ids))),
        ticktext=unit_ids, autorange="reversed",
    )
    return fig


def fig_composite_basic(bundle: InspectorBundle) -> go.Figure:
    """Composite signal from plot_data with burstlet event overlays."""
    if bundle.results is None or not bundle.results.plot_data:
        return _empty_figure("No plot data available.")

    pd_data = bundle.results.plot_data
    t = np.asarray(pd_data.get("t", []), dtype=float)
    if t.size == 0:
        return _empty_figure("Empty time axis in plot data.")

    fig = go.Figure()
    for key, label, color in [
        ("participation_signal", "participation", "#1f77b4"),
        ("rate_signal", "PFR", "#ff7f0e"),
    ]:
        y = pd_data.get(key)
        if y is not None:
            fig.add_trace(go.Scattergl(
                x=t, y=np.asarray(y, dtype=float),
                mode="lines", name=label,
                line=dict(width=1.5, color=color),
            ))

    for _, evt in bundle.burstlets.iterrows():
        fig.add_vrect(
            x0=float(evt["start"]), x1=float(evt["end"]),
            fillcolor="rgba(220, 80, 60, 0.12)", line_width=0, layer="below",
        )

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="time (s)", yaxis_title="signal",
        plot_bgcolor="white", legend=dict(orientation="h", y=1.02),
    )
    return fig
