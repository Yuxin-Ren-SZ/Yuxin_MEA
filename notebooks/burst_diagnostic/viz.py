"""Burst detector diagnostic plotting + Dash dashboard.

Public API:
- ``BatchResults`` dataclass holding all detector outputs.
- ``run_batch`` to execute the iterative burst detector on every Kilosort
  source under a root directory, with two configs (default + no-gate).
- ``fig_*`` functions, each returning a ``plotly.graph_objects.Figure``
  for one diagnostic plot.
- ``save_html`` / ``save_all_section_htmls`` to write the figures to disk.
- ``build_dashboard_app`` to assemble a Dash web app from a ``BatchResults``.

This module is the extraction of the plotting/data-loading code that used
to live inline in ``notebooks/07_iterative_burst_detector_diagnostic.ipynb``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from pipeline_tasks.analysis import (
    IterativeBurstConfig,
    IterativeBurstTrace,
    compute_iterative_bursts,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BatchResults:
    """All detector outputs for every recording in a batch.

    Two configs are run per recording:
    - **default**: full pipeline with BMI gate (``min_burst_modulation=0.1``)
      and post-event GMM clustering enabled.
    - **no_gate**: BMI gate and event clustering disabled — useful for
      before/after comparisons.

    ``recording_names`` returns a sorted list for stable iteration order.
    """

    spike_times: dict[str, dict] = field(default_factory=dict)
    traces: dict[str, IterativeBurstTrace] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    traces_no_gate: dict[str, IterativeBurstTrace] = field(default_factory=dict)
    results_no_gate: dict[str, Any] = field(default_factory=dict)

    @property
    def recording_names(self) -> list[str]:
        return sorted(self.traces.keys())

    def trace(self, name: str, kind: str = "default") -> IterativeBurstTrace:
        return self.traces[name] if kind == "default" else self.traces_no_gate[name]

    def result(self, name: str, kind: str = "default") -> Any:
        return self.results[name] if kind == "default" else self.results_no_gate[name]


# ---------------------------------------------------------------------------
# Kilosort loaders
# ---------------------------------------------------------------------------


def is_kilosort_dir(path: Path) -> bool:
    """A Kilosort output directory has spike_times.npy, spike_clusters.npy, and params.py."""
    return path.is_dir() and all(
        (path / fn).exists()
        for fn in ("spike_times.npy", "spike_clusters.npy", "params.py")
    )


def discover_real_spike_sources(root: Path | None) -> list[Path]:
    """Discover all Kilosort sources (or curated spike-time files) under ``root``."""
    if root is None or not root.exists():
        return []
    if root.is_file() or is_kilosort_dir(root):
        return [root]
    curated = sorted(root.rglob("curated_spike_times.npy"))
    ks_dirs = sorted(
        {p.parent for p in root.rglob("spike_times.npy") if is_kilosort_dir(p.parent)}
    )
    curated_parents = {p.parent for p in curated}
    sources = curated + [p for p in ks_dirs if p not in curated_parents]
    return sorted(set(sources), key=lambda p: str(p))


def _read_kilosort_sample_rate(params_path: Path) -> float:
    values: dict[str, object] = {}
    exec(params_path.read_text(), {}, values)
    for key in ("sample_rate", "fs", "sampling_rate"):
        if key in values:
            return float(values[key])
    raise ValueError(f"Could not find sample_rate/fs/sampling_rate in {params_path}")


def _read_kilosort_keep_clusters(
    ks_dir: Path, labels: set[str] | None
) -> set[int] | None:
    if labels is None:
        return None
    for filename in ("cluster_group.tsv", "cluster_KSLabel.tsv"):
        label_path = ks_dir / filename
        if not label_path.exists():
            continue
        table = pd.read_csv(label_path, sep="\t")
        if "cluster_id" not in table.columns:
            continue
        label_col = next(
            (c for c in ("group", "KSLabel", "label") if c in table.columns), None
        )
        if label_col is None:
            continue
        keep = table[
            table[label_col].astype(str).str.lower().isin({l.lower() for l in labels})
        ]
        return set(keep["cluster_id"].astype(int).tolist())
    return None


def load_kilosort_spike_times(
    ks_dir: Path,
    labels: set[str] | None = frozenset({"good"}),
) -> dict[str, np.ndarray]:
    """Load spike times from a Kilosort directory, optionally filtered by labels."""
    ks_dir = Path(ks_dir)
    sample_rate = _read_kilosort_sample_rate(ks_dir / "params.py")
    spike_samples = np.load(ks_dir / "spike_times.npy").reshape(-1).astype(float)
    spike_clusters = np.load(ks_dir / "spike_clusters.npy").reshape(-1).astype(int)
    keep_clusters = _read_kilosort_keep_clusters(ks_dir, labels)
    out: dict[str, np.ndarray] = {}
    for cid in np.unique(spike_clusters):
        if keep_clusters is not None and int(cid) not in keep_clusters:
            continue
        times = spike_samples[spike_clusters == cid] / sample_rate
        times = times[np.isfinite(times) & (times >= 0.0)]
        if times.size:
            out[f"cluster_{int(cid):03d}"] = np.sort(times.astype(float))
    if not out:
        raise ValueError(f"No clusters found in {ks_dir}")
    return out


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_batch(
    sources: list[Path],
    config_default: IterativeBurstConfig | None = None,
    config_no_gate: IterativeBurstConfig | None = None,
    labels: set[str] | None = frozenset({"good"}),
    verbose: bool = True,
) -> BatchResults:
    """Run both detector configs on every source; return cached results.

    Spike-time loading is done exactly once per source.
    """
    if config_default is None:
        config_default = IterativeBurstConfig()
    if config_no_gate is None:
        config_no_gate = IterativeBurstConfig(
            min_burst_modulation=0.0, cluster_events=False
        )

    batch = BatchResults()
    for source in sources:
        name = source.name if source.is_dir() else source.parent.name
        st = load_kilosort_spike_times(source, labels=labels)
        batch.spike_times[name] = st

        tr = IterativeBurstTrace()
        res = compute_iterative_bursts(
            st, config=config_default, debug=verbose, trace=tr
        )
        batch.traces[name] = tr
        batch.results[name] = res

        tr_ng = IterativeBurstTrace()
        res_ng = compute_iterative_bursts(
            st, config=config_no_gate, debug=False, trace=tr_ng
        )
        batch.traces_no_gate[name] = tr_ng
        batch.results_no_gate[name] = res_ng

        if verbose:
            print(
                f"{name}  default: {len(res.burstlets)} burstlets  "
                f"no-gate: {len(res_ng.burstlets)} burstlets"
            )
    return batch


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------


def save_html(fig: go.Figure, path: Path, offline: bool = False) -> Path:
    """Write ``fig`` to ``path`` as an HTML file (Plotly JS from CDN by default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotlyjs = True if offline else "cdn"
    fig.write_html(str(path), include_plotlyjs=plotlyjs)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attribution_rows(batch: BatchResults) -> list[dict]:
    """Per-recording per-stage survivor / dropped counts."""
    rows = []
    for name, tr in batch.traces.items():
        iters = tr.iterations
        if not iters:
            continue
        n0 = iters[0]["n_candidates"]
        nc = iters[-1]["n_candidates"]
        npf = len(tr.burstlets_pre_gates)
        npp = tr.participation_gate["n_post"] if tr.participation_gate else npf
        npm = tr.bmi_gate["n_pre"] if tr.bmi_gate else npp
        npb = tr.bmi_gate["n_post"] if tr.bmi_gate else npm
        if tr.gmm and "kept_event_mask" in tr.gmm:
            npg = int(tr.gmm["kept_event_mask"].sum())
        else:
            npg = npb
        for stage, surv, drop in [
            ("iter trim", npf, max(0, nc - npf)),
            ("participation", npp, npf - npp),
            ("BMI/LLR", npb, npm - npb),
            ("GMM", npg, max(0, npb - npg)),
        ]:
            rows.append(
                {
                    "recording": name,
                    "stage": stage,
                    "survivors": surv,
                    "dropped": drop,
                }
            )
    return rows


def _select_trace(
    batch: BatchResults, name: str, kind: str
) -> tuple[IterativeBurstTrace, Any]:
    if kind == "no_gate":
        return batch.traces_no_gate[name], batch.results_no_gate[name]
    return batch.traces[name], batch.results[name]


# ---------------------------------------------------------------------------
# Kill-stage plots (Stages 1–4 + cross-stage flow)
# ---------------------------------------------------------------------------


def fig_kill_attribution(batch: BatchResults) -> go.Figure:
    """Survivors vs dropped at each kill stage, one subplot per recording."""
    attr_df = pd.DataFrame(_attribution_rows(batch))
    stages_order = ["iter trim", "participation", "BMI/LLR", "GMM"]
    names = sorted(attr_df["recording"].unique())

    fig = make_subplots(
        rows=1, cols=len(names), subplot_titles=names, shared_yaxes=True
    )
    for col, rec in enumerate(names, 1):
        sub = (
            attr_df[attr_df["recording"] == rec]
            .set_index("stage")
            .reindex(stages_order)
        )
        fig.add_trace(
            go.Bar(
                name="survived",
                x=stages_order,
                y=sub["survivors"].values,
                marker_color="steelblue",
                hovertemplate="%{x}<br>survived=%{y}<br>rec=" + rec,
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Bar(
                name="dropped",
                x=stages_order,
                y=sub["dropped"].values,
                base=sub["survivors"].values,
                marker_color="crimson",
                hovertemplate="%{x}<br>dropped=%{y}<br>rec=" + rec,
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )
    fig.update_layout(
        barmode="stack",
        title="Kill-stage attribution across recordings",
        height=420,
        legend_title="count type",
    )
    return fig


def fig_stage1_composite_slider(
    batch: BatchResults, recording: str, trace_kind: str = "default"
) -> go.Figure:
    """Composite signal with iteration slider for one recording."""
    tr, _ = _select_trace(batch, recording, trace_kind)
    t = tr.t_centers
    iters = tr.iterations

    frames = []
    for k, snap in enumerate(iters):
        comp = snap["composite"]
        thr = snap["composite_threshold"]
        base = snap["composite_baseline"]
        cands = snap["candidates"]
        n = snap["n_candidates"]

        shade_x: list = []
        shade_y: list = []
        ymin = float(comp.min()) - 0.05
        ymax = float(comp.max()) + 0.05
        for c in cands:
            shade_x += [c["start"], c["start"], c["end"], c["end"], None]
            shade_y += [ymin, ymax, ymax, ymin, None]

        frame_data = [
            go.Scatter(
                x=t,
                y=comp,
                mode="lines",
                line=dict(color="steelblue", width=0.8),
                name="composite",
                hovertemplate="t=%{x:.2f}s  composite=%{y:.3f}",
            ),
            go.Scatter(
                x=[t[0], t[-1]],
                y=[thr, thr],
                mode="lines",
                line=dict(color="red", dash="dash", width=1),
                name=f"thr={thr:.2f}",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[t[0], t[-1]],
                y=[base, base],
                mode="lines",
                line=dict(color="gray", dash="dot", width=0.8),
                name=f"base={base:.2f}",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=shade_x,
                y=shade_y,
                fill="toself",
                fillcolor="rgba(255,165,0,0.12)",
                line=dict(width=0),
                name="candidates",
                hoverinfo="skip",
            ),
        ]
        frames.append(
            go.Frame(
                data=frame_data,
                name=str(k),
                layout=go.Layout(
                    title_text=(
                        f"{recording} ({trace_kind}) — iter {k}  "
                        f"n_candidates={n}  "
                        f"delta={snap['convergence_delta']:.4f}"
                    ),
                    xaxis_autorange=True,
                    yaxis_autorange=True,
                ),
            )
        )

    slider_steps = [
        dict(
            args=[[f.name], dict(mode="immediate", frame=dict(duration=0))],
            label=f.name,
            method="animate",
        )
        for f in frames
    ]

    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames,
        layout=go.Layout(
            title=f"{recording} — composite signal (slider to step through iterations)",
            xaxis_title="time (s)",
            yaxis_title="composite score",
            hovermode="x unified",
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.15,
                    x=0.05,
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=600),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(mode="immediate", frame=dict(duration=0)),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=slider_steps,
                    currentvalue=dict(prefix="iteration: ", font=dict(size=13)),
                    pad=dict(t=50),
                )
            ],
            height=500,
        ),
    )
    return fig


def fig_stage2_participation(batch: BatchResults) -> go.Figure:
    """Participation floor facet scatter across all recordings."""
    rows = []
    for name, tr in batch.traces.items():
        gate = tr.participation_gate or {}
        floor = float(gate.get("floor", 0.0))
        for ev in tr.burstlets_pre_gates:
            rows.append(
                {
                    "recording": name,
                    "peak_synchrony": float(ev.get("peak_synchrony", 0)),
                    "duration_s": float(ev.get("duration_s", 0)),
                    "peak_time": float(ev.get("peak_time", 0)),
                    "kept": "kept"
                    if float(ev.get("peak_synchrony", 0)) >= floor
                    else "killed",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure().add_annotation(
            text="no pre-floor events found", showarrow=False
        )

    fig = px.scatter(
        df,
        x="peak_synchrony",
        y="duration_s",
        color="peak_time",
        symbol="kept",
        facet_col="recording",
        facet_col_wrap=3,
        color_continuous_scale="Viridis",
        symbol_map={"kept": "circle", "killed": "x"},
        hover_data=["recording", "peak_time", "peak_synchrony", "duration_s"],
        labels={"peak_time": "peak time (s)"},
        title="Stage 2 — Participation floor (color = timestamp, shape = kept/killed)",
        height=420,
    )
    floor_val = next(
        (
            float((tr.participation_gate or {}).get("floor", 0))
            for tr in batch.traces.values()
            if tr.participation_gate
        ),
        0.0,
    )
    fig.add_vline(
        x=floor_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"floor={floor_val:.3f}",
        annotation_position="top right",
    )
    return fig


def fig_stage3_bmi(batch: BatchResults) -> go.Figure:
    """BMI/LLR gate facet scatter across all recordings."""
    rows = []
    for name, tr in batch.traces.items():
        bmi = tr.bmi_gate or {}
        thr = float(bmi.get("threshold", 0.0))
        for ev in bmi.get("pre_events", []):
            rows.append(
                {
                    "recording": name,
                    "llr_aggregate": float(ev.get("llr_aggregate", 0)),
                    "participation": float(ev.get("participation", 0)),
                    "start": float(ev.get("start", 0)),
                    "kept": "kept"
                    if float(ev.get("llr_aggregate", 0)) >= thr
                    else "killed",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure().add_annotation(
            text="no BMI gate events found", showarrow=False
        )

    fig = px.scatter(
        df,
        x="llr_aggregate",
        y="participation",
        color="start",
        symbol="kept",
        facet_col="recording",
        facet_col_wrap=3,
        color_continuous_scale="Viridis",
        symbol_map={"kept": "circle", "killed": "x"},
        hover_data=["recording", "start", "llr_aggregate", "participation"],
        labels={"start": "event start (s)"},
        title="Stage 3 — BMI/LLR gate (color = event start time, shape = kept/killed)",
        height=420,
    )
    thr_val = next(
        (
            float((tr.bmi_gate or {}).get("threshold", 0))
            for tr in batch.traces.values()
            if tr.bmi_gate
        ),
        0.1,
    )
    fig.add_vline(
        x=thr_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"min_bmi={thr_val:.2f}",
        annotation_position="top right",
    )
    return fig


def fig_stage4_gmm_pca(batch: BatchResults, recording: str) -> go.Figure:
    """GMM event clustering — 2x2 PCA panels for one recording."""
    tr = batch.traces[recording]
    gmm = tr.gmm
    if not gmm or "X" not in gmm:
        reason = (gmm or {}).get("skipped", "no trace")
        return go.Figure().add_annotation(
            text=f"{recording}: GMM skipped — {reason}", showarrow=False
        )

    Xs = gmm["X_scaled"]
    labels = gmm["labels"]
    centers = gmm["component_means_scaled"]
    mgroups = gmm["merged_groups"]
    cscores = np.asarray(gmm["cluster_scores"])
    kept = gmm["kept_event_mask"]

    pca = PCA(n_components=2).fit(Xs)
    Z = pca.transform(Xs)
    Zc = pca.transform(centers)
    ev = pca.explained_variance_ratio_

    comp_to_grp: dict[int, int] = {}
    for gi, g in enumerate(mgroups):
        for m in g["members"]:
            comp_to_grp[int(m)] = gi
    grp_id = np.array([comp_to_grp.get(int(lb), -1) for lb in labels])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"(a) Initial GMM components (n={len(centers)})",
            f"(b) Merged groups (n={len(mgroups)})",
            "(c) Kept vs killed (color=event index)",
            "(d) Cluster score (blue=burst)",
        ],
    )

    cmap_a = px.colors.qualitative.Plotly
    for k in np.unique(labels):
        m = labels == k
        fig.add_trace(
            go.Scatter(
                x=Z[m, 0],
                y=Z[m, 1],
                mode="markers",
                marker=dict(color=cmap_a[int(k) % len(cmap_a)], size=6, opacity=0.7),
                name=f"comp {k}",
                hovertemplate=f"comp {k}<br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=Zc[:, 0],
            y=Zc[:, 1],
            mode="markers",
            marker=dict(symbol="x", size=12, color="black"),
            name="centroids",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    cmap_b = px.colors.qualitative.Set1
    for gi, _ in enumerate(mgroups):
        m = grp_id == gi
        sc = cscores[gi] if gi < len(cscores) else 0
        fig.add_trace(
            go.Scatter(
                x=Z[m, 0],
                y=Z[m, 1],
                mode="markers",
                marker=dict(color=cmap_b[gi % len(cmap_b)], size=6, opacity=0.7),
                name=f"grp{gi} score={sc:+.2f}",
                hovertemplate=(
                    f"group {gi}  score={sc:+.2f}<br>"
                    "PC1=%{x:.2f}<br>PC2=%{y:.2f}"
                ),
            ),
            row=1,
            col=2,
        )

    ev_idx = np.arange(len(Z))
    fig.add_trace(
        go.Scatter(
            x=Z[kept, 0],
            y=Z[kept, 1],
            mode="markers",
            marker=dict(
                color=ev_idx[kept],
                colorscale="Viridis",
                symbol="circle",
                size=6,
                opacity=0.8,
                colorbar=dict(title="event idx", len=0.45, y=0.75),
            ),
            name="kept",
            hovertemplate="kept<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=Z[~kept, 0],
            y=Z[~kept, 1],
            mode="markers",
            marker=dict(
                color=ev_idx[~kept],
                colorscale="Viridis",
                symbol="x",
                size=7,
                opacity=0.8,
                showscale=False,
            ),
            name="killed",
            hovertemplate="killed<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}",
        ),
        row=2,
        col=1,
    )

    per_ev_score = np.array(
        [cscores[g] if g >= 0 else np.nan for g in grp_id]
    )
    abs_max = (
        float(np.nanmax(np.abs(per_ev_score)))
        if np.isfinite(per_ev_score).any()
        else 1.0
    )
    fig.add_trace(
        go.Scatter(
            x=Z[:, 0],
            y=Z[:, 1],
            mode="markers",
            marker=dict(
                color=per_ev_score,
                colorscale="RdBu_r",
                cmin=-abs_max,
                cmax=abs_max,
                size=6,
                colorbar=dict(title="score", len=0.45, y=0.25),
            ),
            name="score",
            showlegend=False,
            hovertemplate=(
                "score=%{marker.color:.2f}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}"
            ),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=(
            f"Stage 4 GMM clustering — {recording}  "
            f"(PC1+PC2 = {ev.sum() * 100:.1f}% var)"
        ),
        height=700,
    )
    fig.update_xaxes(title_text=f"PC1 ({ev[0] * 100:.1f}%)")
    fig.update_yaxes(title_text=f"PC2 ({ev[1] * 100:.1f}%)")
    return fig


def fig_cross_stage_flow(batch: BatchResults) -> go.Figure:
    """Stacked-bar cross-stage flow with one stack per recording."""
    attr_df = pd.DataFrame(_attribution_rows(batch))
    stages_ord = ["iter trim", "participation", "BMI/LLR", "GMM"]
    names_all = sorted(attr_df["recording"].unique())
    fig = go.Figure()
    for stage in stages_ord:
        sub = (
            attr_df[attr_df["stage"] == stage]
            .set_index("recording")
            .reindex(names_all)
        )
        fig.add_trace(
            go.Bar(
                name=f"{stage} survived",
                x=names_all,
                y=sub["survivors"].values,
                marker_color="steelblue",
                hovertemplate="%{x}<br>stage=" + stage + "<br>survived=%{y}",
            )
        )
        fig.add_trace(
            go.Bar(
                name=f"{stage} dropped",
                x=names_all,
                y=sub["dropped"].values,
                marker_color="crimson",
                hovertemplate="%{x}<br>stage=" + stage + "<br>dropped=%{y}",
            )
        )
    fig.update_layout(
        barmode="stack",
        title="Cross-stage kill flow (per recording)",
        xaxis_title="recording",
        yaxis_title="events",
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# Sections C–G (LDA deep-dive)
# ---------------------------------------------------------------------------


def fig_section_c_lda_pca(
    batch: BatchResults,
    recording: str,
    trace_kind: str = "default",
    plot_all_iters: bool = False,
) -> go.Figure:
    """Per-iteration LDA PCA with an iteration slider."""
    tr, _ = _select_trace(batch, recording, trace_kind)
    iters = tr.iterations
    t_centers = tr.t_centers
    fnames = tr.feature_names

    if plot_all_iters:
        show_iters = list(range(len(iters)))
    else:
        show_iters = sorted({0, len(iters) // 2, len(iters) - 1})

    htmpl = (
        "t=%{customdata[0]:.2f}s<br>"
        "PFR=%{customdata[1]:.1f}<br>"
        "P=%{customdata[2]:.2f}<br>"
        "LLR=%{customdata[7]:.2f}<extra></extra>"
    )
    frames = []
    for k in show_iters:
        snap = iters[k]
        Xn = snap["X_norm"]
        mask = snap["candidate_mask"].astype(bool)
        w = snap["w"]
        pca = PCA(n_components=2).fit(Xn)
        Z = pca.transform(Xn)
        ev = pca.explained_variance_ratio_
        w_pc = pca.components_ @ w
        scale = float(np.abs(Z).max()) * 0.5 / max(
            float(np.linalg.norm(w_pc)), 1e-9
        )

        cdata_bg = np.column_stack([t_centers[~mask], Xn[~mask]])
        cdata_bst = np.column_stack([t_centers[mask], Xn[mask]])

        top3 = sorted(zip(fnames, w), key=lambda x: -abs(x[1]))[:3]
        wstr = "  ".join(f"{n}={v:+.2f}" for n, v in top3)

        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=Z[~mask, 0],
                        y=Z[~mask, 1],
                        mode="markers",
                        marker=dict(
                            color=t_centers[~mask],
                            colorscale="Viridis",
                            symbol="circle",
                            size=3,
                            opacity=0.35,
                            colorbar=dict(title="time (s)", len=0.6),
                        ),
                        name="background",
                        customdata=cdata_bg,
                        hovertemplate=htmpl,
                    ),
                    go.Scatter(
                        x=Z[mask, 0],
                        y=Z[mask, 1],
                        mode="markers",
                        marker=dict(
                            color=t_centers[mask],
                            colorscale="Viridis",
                            symbol="diamond",
                            size=5,
                            opacity=0.7,
                            showscale=False,
                        ),
                        name="burst",
                        customdata=cdata_bst,
                        hovertemplate=htmpl,
                    ),
                    go.Scatter(
                        x=[0, w_pc[0] * scale],
                        y=[0, w_pc[1] * scale],
                        mode="lines+markers",
                        line=dict(color="darkorange", width=2),
                        marker=dict(
                            symbol="arrow",
                            size=10,
                            color="darkorange",
                            angleref="previous",
                        ),
                        name="Fisher w",
                        hoverinfo="skip",
                    ),
                ],
                name=str(k),
                layout=go.Layout(
                    title_text=(
                        f"{recording} — iter {k}  "
                        f"n_cand={snap['n_candidates']}  "
                        f"thr={snap['composite_threshold']:.2f}  "
                        f"Δ={snap['convergence_delta']:.4f}<br>"
                        f"top w: {wstr}  "
                        f"PC1={ev[0] * 100:.1f}%  PC2={ev[1] * 100:.1f}%"
                    ),
                    xaxis_autorange=True,
                    yaxis_autorange=True,
                ),
            )
        )

    slider_steps = [
        dict(
            args=[[f.name], dict(mode="immediate", frame=dict(duration=0))],
            label=f"iter {f.name}",
            method="animate",
        )
        for f in frames
    ]

    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames,
        layout=go.Layout(
            title=f"{recording} — LDA PCA per iteration (trace={trace_kind})",
            xaxis_title="PC1",
            yaxis_title="PC2",
            hovermode="closest",
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.1,
                    x=0.05,
                    buttons=[
                        dict(
                            label="▶",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=700),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                ),
                            ],
                        ),
                        dict(
                            label="⏸",
                            method="animate",
                            args=[
                                [None],
                                dict(mode="immediate", frame=dict(duration=0)),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=slider_steps,
                    currentvalue=dict(prefix="iteration: ", font=dict(size=12)),
                    pad=dict(t=50),
                )
            ],
            height=580,
        ),
    )
    return fig


def fig_section_d_boundary_shift(
    batch: BatchResults, recording: str, trace_kind: str = "default"
) -> go.Figure:
    """Two-column input vs output PCA panels for selected iterations."""
    tr, _ = _select_trace(batch, recording, trace_kind)
    iters = tr.iterations
    t_centers = tr.t_centers
    show_io = sorted({0, len(iters) // 2, len(iters) - 1})

    fig = make_subplots(
        rows=len(show_io),
        cols=2,
        column_titles=["Input labels", "Output labels"],
        row_titles=[f"iter {k}" for k in show_io],
        shared_xaxes=False,
        shared_yaxes=False,
    )
    htmpl = "t=%{customdata:.2f}s<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>"

    for row, k in enumerate(show_io, 1):
        snap = iters[k]
        Xn = snap["X_norm"]
        pca = PCA(n_components=2).fit(Xn)
        Z = pca.transform(Xn)
        for col, (mask, _) in enumerate(
            [
                (snap["candidate_mask_in"].astype(bool), "input"),
                (snap["candidate_mask"].astype(bool), "output"),
            ],
            1,
        ):
            show_lg = row == 1 and col == 1
            fig.add_trace(
                go.Scatter(
                    x=Z[~mask, 0],
                    y=Z[~mask, 1],
                    mode="markers",
                    marker=dict(
                        color=t_centers[~mask],
                        colorscale="Viridis",
                        symbol="circle",
                        size=3,
                        opacity=0.3,
                        showscale=(col == 1 and row == 1),
                        colorbar=dict(title="time (s)", len=0.3, y=0.85),
                    ),
                    name="background",
                    legendgroup="bg",
                    showlegend=show_lg,
                    customdata=t_centers[~mask],
                    hovertemplate=htmpl,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=Z[mask, 0],
                    y=Z[mask, 1],
                    mode="markers",
                    marker=dict(
                        color=t_centers[mask],
                        colorscale="Viridis",
                        symbol="diamond",
                        size=5,
                        opacity=0.7,
                        showscale=False,
                    ),
                    name="burst",
                    legendgroup="burst",
                    showlegend=show_lg,
                    customdata=t_centers[mask],
                    hovertemplate=htmpl,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=f"{recording} — iteration boundary shift (trace={trace_kind})",
        height=380 * len(show_io),
    )
    return fig


def fig_section_e_3d_pca(
    batch: BatchResults, recording: str, trace_kind: str = "default"
) -> go.Figure:
    """3D PCA at the converged iteration."""
    tr, _ = _select_trace(batch, recording, trace_kind)
    snap = tr.iterations[-1]
    Xn = snap["X_norm"]
    labels = snap["candidate_mask"].astype(bool)
    t_c = tr.t_centers

    pca3 = PCA(n_components=3).fit(Xn)
    Z3 = pca3.transform(Xn)
    ev = pca3.explained_variance_ratio_

    htmpl = (
        "t=%{customdata:.2f}s<br>"
        "PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=Z3[~labels, 0],
            y=Z3[~labels, 1],
            z=Z3[~labels, 2],
            mode="markers",
            marker=dict(
                color=t_c[~labels],
                colorscale="Viridis",
                size=2,
                opacity=0.3,
                colorbar=dict(title="time (s)", len=0.6, x=1.05),
            ),
            name="background",
            customdata=t_c[~labels],
            hovertemplate=htmpl,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=Z3[labels, 0],
            y=Z3[labels, 1],
            z=Z3[labels, 2],
            mode="markers",
            marker=dict(
                color=t_c[labels],
                colorscale="Viridis",
                symbol="diamond",
                size=4,
                opacity=0.8,
                showscale=False,
            ),
            name="burst",
            customdata=t_c[labels],
            hovertemplate=htmpl,
        )
    )
    fig.update_layout(
        title=f"{recording} — 3D PCA (converged iter, trace={trace_kind})",
        scene=dict(
            xaxis_title=f"PC1 ({ev[0] * 100:.1f}%)",
            yaxis_title=f"PC2 ({ev[1] * 100:.1f}%)",
            zaxis_title=f"PC3 ({ev[2] * 100:.1f}%)",
        ),
        height=620,
    )
    return fig


def _fit_gmm_bic_sweep(
    Xn: np.ndarray, ks: tuple[int, ...] = (2, 3, 4, 5)
) -> tuple[dict[int, tuple[GaussianMixture, np.ndarray]], pd.DataFrame, int]:
    """Fit GMMs for each k; return fits, BIC dataframe, and BIC-preferred k."""
    fits = {}
    bic_rows = []
    for k in ks:
        gm = GaussianMixture(
            n_components=k, n_init=5, random_state=42, reg_covar=1e-6
        ).fit(Xn)
        cl = gm.predict(Xn)
        bic_rows.append(
            {"k": k, "BIC": float(gm.bic(Xn)), "AIC": float(gm.aic(Xn))}
        )
        fits[k] = (gm, cl)
    bic_df = pd.DataFrame(bic_rows).set_index("k")
    best_k = int(bic_df["BIC"].idxmin())
    return fits, bic_df, best_k


def fig_section_f_gmm_bic_sweep(
    batch: BatchResults, recording: str, trace_kind: str = "default"
) -> go.Figure:
    """Multi-k GMM BIC sweep on the converged-iteration features."""
    tr, _ = _select_trace(batch, recording, trace_kind)
    snap = tr.iterations[-1]
    Xn = snap["X_norm"]
    t_c = tr.t_centers
    pca2 = PCA(n_components=2).fit(Xn)
    Z2 = pca2.transform(Xn)
    ev2 = pca2.explained_variance_ratio_

    ks = (2, 3, 4, 5)
    fits, bic_df, _best_k = _fit_gmm_bic_sweep(Xn, ks)

    fig = make_subplots(
        rows=1,
        cols=len(ks),
        subplot_titles=[
            f'k={k}  BIC={bic_df.loc[k, "BIC"]:.0f}' for k in ks
        ],
    )
    for ci, k in enumerate(ks, 1):
        _, cl = fits[k]
        for c in range(k):
            m = cl == c
            fig.add_trace(
                go.Scatter(
                    x=Z2[m, 0],
                    y=Z2[m, 1],
                    mode="markers",
                    marker=dict(
                        color=t_c[m],
                        colorscale="Viridis",
                        size=3,
                        opacity=0.55,
                        showscale=(ci == 1 and c == 0),
                        colorbar=dict(title="time (s)", len=0.6),
                    ),
                    name=f"k={k} c{c}",
                    customdata=t_c[m],
                    hovertemplate=(
                        "t=%{customdata:.2f}s<br>"
                        "PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>"
                    ),
                ),
                row=1,
                col=ci,
            )

    fig.update_layout(
        title=(
            f"{recording} — bin-level GMM k-sweep on converged X_norm "
            f"(trace={trace_kind})"
        ),
        height=420,
        showlegend=False,
    )
    fig.update_xaxes(title_text=f"PC1 ({ev2[0] * 100:.1f}%)")
    fig.update_yaxes(title_text=f"PC2 ({ev2[1] * 100:.1f}%)", col=1)
    return fig


def fig_section_g_time_strip(
    batch: BatchResults, recording: str, trace_kind: str = "default"
) -> go.Figure:
    """Bin-level cluster assignment as a time strip + participation + composite."""
    tr, res = _select_trace(batch, recording, trace_kind)
    snap = tr.iterations[-1]
    t = tr.t_centers
    comp = snap["composite"]
    thr = snap["composite_threshold"]
    Xn = snap["X_norm"]

    fits, _, best_k = _fit_gmm_bic_sweep(Xn)
    _, cl_best = fits[best_k]
    part = res.plot_data["participation_signal"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.18, 0.36, 0.46],
        subplot_titles=[
            f"GMM cluster assignment (k={best_k})",
            "Participation signal",
            "Composite signal + LDA burst",
        ],
        vertical_spacing=0.06,
    )
    fig.add_trace(
        go.Heatmap(
            x=t,
            y=[0],
            z=[cl_best.astype(float)],
            colorscale="Plotly3",
            showscale=True,
            colorbar=dict(
                title="cluster",
                len=0.18,
                y=0.93,
                tickvals=list(range(best_k)),
            ),
            hovertemplate="t=%{x:.2f}s  cluster=%{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=part,
            mode="lines",
            line=dict(color="steelblue", width=0.8),
            name="participation",
            hovertemplate="t=%{x:.2f}s  P=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=comp,
            mode="lines",
            line=dict(color="black", width=0.8),
            name="composite",
            hovertemplate="t=%{x:.2f}s  composite=%{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=thr,
        line_dash="dash",
        line_color="red",
        annotation_text=f"thr={thr:.2f}",
        annotation_position="bottom right",
        row=3,
        col=1,
    )
    for c in snap.get("candidates", []):
        fig.add_vrect(
            x0=c["start"],
            x1=c["end"],
            fillcolor="crimson",
            opacity=0.08,
            layer="below",
            line_width=0,
            row="all",
            col=1,
        )

    fig.update_layout(
        title=f"{recording} — GMM time strip (trace={trace_kind})",
        hovermode="x unified",
        height=560,
        xaxis3=dict(
            rangeslider=dict(visible=True, thickness=0.06),
            title="time (s)",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Convenience batch saver
# ---------------------------------------------------------------------------


def save_all_section_htmls(
    batch: BatchResults,
    output_dir: Path,
    trace_kind: str = "default",
    plot_all_iters: bool = False,
    offline: bool = False,
) -> list[Path]:
    """Produce every individual HTML file for the batch."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Summary plots across all recordings
    saved.append(save_html(fig_kill_attribution(batch), output_dir / "summary_kill_stages.html", offline))
    saved.append(save_html(fig_stage2_participation(batch), output_dir / "summary_stage2_participation.html", offline))
    saved.append(save_html(fig_stage3_bmi(batch), output_dir / "summary_stage3_bmi.html", offline))
    saved.append(save_html(fig_cross_stage_flow(batch), output_dir / "summary_cross_stage_flow.html", offline))

    # Per-recording plots
    for name in batch.recording_names:
        saved.append(save_html(
            fig_stage1_composite_slider(batch, name, trace_kind),
            output_dir / f"{name}_stage1_composite_slider.html",
            offline,
        ))
        saved.append(save_html(
            fig_stage4_gmm_pca(batch, name),
            output_dir / f"{name}_stage4_gmm.html",
            offline,
        ))
        saved.append(save_html(
            fig_section_c_lda_pca(batch, name, trace_kind, plot_all_iters),
            output_dir / f"{name}_sectionC_lda_pca.html",
            offline,
        ))
        saved.append(save_html(
            fig_section_d_boundary_shift(batch, name, trace_kind),
            output_dir / f"{name}_sectionD_boundary_shift.html",
            offline,
        ))
        saved.append(save_html(
            fig_section_e_3d_pca(batch, name, trace_kind),
            output_dir / f"{name}_sectionE_3d_pca.html",
            offline,
        ))
        saved.append(save_html(
            fig_section_f_gmm_bic_sweep(batch, name, trace_kind),
            output_dir / f"{name}_sectionF_gmm_bic_sweep.html",
            offline,
        ))
        saved.append(save_html(
            fig_section_g_time_strip(batch, name, trace_kind),
            output_dir / f"{name}_sectionG_timestrip.html",
            offline,
        ))
    return saved


# ---------------------------------------------------------------------------
# Dash dashboard
# ---------------------------------------------------------------------------


def build_dashboard_app(
    batch: BatchResults,
    initial_recording: str | None = None,
    initial_trace_kind: str = "default",
):
    """Assemble a Dash app that renders every plot for the batch.

    Layout: a top control bar (recording + trace dropdowns) followed by three
    tabs (Summary, Kill stages, LDA deep-dive). Dropdown changes trigger
    callbacks that re-render the per-recording figures.
    """
    from dash import Dash, dcc, html, Input, Output

    if not batch.recording_names:
        raise ValueError("BatchResults is empty — nothing to display.")

    if initial_recording is None or initial_recording not in batch.recording_names:
        initial_recording = batch.recording_names[0]

    total_burstlets = sum(len(r.burstlets) for r in batch.results.values())
    total_net = sum(len(r.network_bursts) for r in batch.results.values())

    app = Dash(__name__)
    app.title = "Burst Detector Diagnostic"

    # Cross-recording figures are pre-computed — they don't depend on dropdowns.
    fig_kill = fig_kill_attribution(batch)
    fig_xflow = fig_cross_stage_flow(batch)
    fig_part = fig_stage2_participation(batch)
    fig_bmi = fig_stage3_bmi(batch)

    app.layout = html.Div(
        style={"fontFamily": "sans-serif", "padding": "16px"},
        children=[
            html.Div(
                [
                    html.H2(
                        "Iterative Burst Detector — Diagnostic Dashboard",
                        style={"margin": "0"},
                    ),
                    html.Div(
                        f"{len(batch.recording_names)} recordings  •  "
                        f"{total_burstlets} burstlets  •  {total_net} network bursts",
                        style={"color": "#555", "fontSize": "13px"},
                    ),
                ]
            ),
            html.Hr(),
            html.Div(
                [
                    html.Label("Recording: ", style={"marginRight": "6px"}),
                    dcc.Dropdown(
                        id="recording-dropdown",
                        options=[
                            {"label": n, "value": n} for n in batch.recording_names
                        ],
                        value=initial_recording,
                        clearable=False,
                        style={"width": "260px", "display": "inline-block"},
                    ),
                    html.Span(style={"display": "inline-block", "width": "24px"}),
                    html.Label("Trace: ", style={"marginRight": "6px"}),
                    dcc.Dropdown(
                        id="trace-dropdown",
                        options=[
                            {"label": "default", "value": "default"},
                            {"label": "no_gate", "value": "no_gate"},
                        ],
                        value=initial_trace_kind,
                        clearable=False,
                        style={"width": "150px", "display": "inline-block"},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            dcc.Tabs(
                id="tabs",
                value="summary",
                children=[
                    dcc.Tab(
                        label="Summary",
                        value="summary",
                        children=[
                            dcc.Graph(figure=fig_kill),
                            dcc.Graph(figure=fig_xflow),
                        ],
                    ),
                    dcc.Tab(
                        label="Kill stages",
                        value="kill",
                        children=[
                            html.H4("Stage 1 — Composite signal (per recording)"),
                            dcc.Graph(id="stage1-graph"),
                            html.H4("Stage 2 — Participation floor"),
                            dcc.Graph(figure=fig_part),
                            html.H4("Stage 3 — BMI / LLR gate"),
                            dcc.Graph(figure=fig_bmi),
                            html.H4("Stage 4 — GMM event clustering"),
                            dcc.Graph(id="stage4-graph"),
                        ],
                    ),
                    dcc.Tab(
                        label="LDA deep-dive",
                        value="lda",
                        children=[
                            html.H4("Section C — LDA PCA per iteration"),
                            dcc.Graph(id="section-c-graph"),
                            html.H4("Section D — Boundary shift"),
                            dcc.Graph(id="section-d-graph"),
                            html.H4("Section E — 3D PCA at converged iteration"),
                            dcc.Graph(id="section-e-graph"),
                            html.H4("Section F — Multi-k GMM BIC sweep"),
                            dcc.Graph(id="section-f-graph"),
                            html.H4("Section G — GMM cluster time strip"),
                            dcc.Graph(id="section-g-graph"),
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("stage1-graph", "figure"),
        Output("stage4-graph", "figure"),
        Output("section-c-graph", "figure"),
        Output("section-d-graph", "figure"),
        Output("section-e-graph", "figure"),
        Output("section-f-graph", "figure"),
        Output("section-g-graph", "figure"),
        Input("recording-dropdown", "value"),
        Input("trace-dropdown", "value"),
    )
    def _update_per_recording(recording: str, trace_kind: str):
        return (
            fig_stage1_composite_slider(batch, recording, trace_kind),
            fig_stage4_gmm_pca(batch, recording),
            fig_section_c_lda_pca(batch, recording, trace_kind, False),
            fig_section_d_boundary_shift(batch, recording, trace_kind),
            fig_section_e_3d_pca(batch, recording, trace_kind),
            fig_section_f_gmm_bic_sweep(batch, recording, trace_kind),
            fig_section_g_time_strip(batch, recording, trace_kind),
        )

    return app
