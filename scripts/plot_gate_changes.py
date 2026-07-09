#!/usr/bin/env python
"""Plotly HTML visualisations of the ML burst-gate change (mean-LLR -> posterior_peak>=0.4).

Two standalone HTML deliverables, from the enriched candidate tables produced by
scratchpad/extract_enriched.py:

  detail_gate_changes.html   facet grid of the per-well recordings the new gate changes MOST;
                             each panel = spike-raster PNG (all spikes, Agg-rasterized) overlaid
                             with burst spans coloured by change-class (recovered / removed / kept).
  summary_gate_changes.html  dataset-wide summary: old-vs-new per-well scatter, delta histogram,
                             per-sample totals, gained/unchanged/lost breakdown.

OLD gate = llr_aggregate >= 0.1 (reproduces shipped network_bursts counts).
NEW gate = posterior_peak >= 0.4.

Raster PNG rendering adapts src/yuxin_mea/analysis/raster_image.py (Figure + FigureCanvasAgg,
every spike rasterized=True). Run with PYTHONPATH=<worktree>/src (editable install points at the
main repo, not this worktree).
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from plotly.subplots import make_subplots

ROOT = "/mnt/Vol20tb2/SadeghLab/yuxin/analysis/ml_burst_data_umap"
VAL_DIR = "/mnt/Vol20tb2/SadeghLab/yuxin/analysis/burst_gate_validation"
OUT_DIR = os.path.join(VAL_DIR, "plots")
OLD_THR, NEW_THR = 0.1, 0.4
TOP_N = 50          # per direction (recoveries + removals); min(TOP_N, available)
MAX_SPANS = 120     # cap spans drawn per class per panel (dense over-detection wells)
GRID_COLS = 6

SAMPLE_COLORS = {"CX118": "#0072B2", "CX138": "#D55E00", "CX169": "#009E73"}  # Okabe-Ito
CLASS_STYLE = {
    "recovered": dict(fill="rgba(46,160,67,0.28)", line="rgba(46,160,67,0.95)", label="recovered (new only)"),
    "removed":   dict(fill="rgba(214,39,40,0.24)", line="rgba(214,39,40,0.95)", label="removed (old only)"),
    "kept":      dict(fill="rgba(120,120,120,0.20)", line="rgba(120,120,120,0.85)", label="kept (both)"),
}


# --------------------------------------------------------------------------- raster PNG
def _render_raster_bytes(spike_times: dict, tmax: float, *, w_px=300, h_px=180, dpi=100):
    """All spikes -> transparent PNG bytes. Adapted from raster_image.render_well_png.

    Returns (png_bytes, n_units). Lanes ordered busiest-first (matches interactive views).
    """
    fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    units = sorted(spike_times.items(), key=lambda kv: -len(np.asarray(kv[1])))
    for row, (_uid, times) in enumerate(units):
        arr = np.asarray(times, dtype=float)
        if arr.size:
            ax.scatter(arr, np.full(arr.shape, row), s=0.8, marker="|",
                       linewidths=0.4, c="#161b22", rasterized=True)
    n_units = max(len(units), 1)
    ax.set_ylim(-0.5, n_units - 0.5)
    ax.set_xlim(0.0, tmax if tmax > 0 else 1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True)
    return buf.getvalue(), n_units


def raster_cached(sp_path: str, tmax: float, cache_dir: str, keystr: str):
    """Return (data_uri, n_units) with a disk cache keyed by well.

    Rendering 100 rasters is the cost; n_units is embedded in the filename so a cache
    hit needs no np.load. Cache miss renders + writes the PNG.
    """
    hits = glob.glob(os.path.join(cache_dir, f"{keystr}_u*.png"))
    if hits:
        data = open(hits[0], "rb").read()
        n_units = int(re.search(r"_u(\d+)\.png$", hits[0]).group(1))
    else:
        spikes = np.load(sp_path, allow_pickle=True).item()
        data, n_units = _render_raster_bytes(spikes, tmax)
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, f"{keystr}_u{n_units}.png"), "wb") as fh:
            fh.write(data)
    uri = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    return uri, n_units


def find_spike_path(sample, date, run, rec, well):
    # pandas parses the zero-padded run/date dirs ("000035") as ints (35); re-pad the run.
    run_s = f"{int(run):06d}"
    hits = glob.glob(f"{ROOT}/{sample}/{date}/*/Network/{run_s}/{rec}/{well}/ml_burst_detection/debug_spike_times.npy")
    return hits[0] if hits else None


def classify(row):
    old = row.llr_aggregate >= OLD_THR
    new = row.posterior_peak >= NEW_THR
    if old and new:
        return "kept"
    if new and not old:
        return "recovered"
    if old and not new:
        return "removed"
    return None  # dropped by both — not drawn


# --------------------------------------------------------------------------- detail grid
CLS_ORDER = ("kept", "removed", "recovered")
# 4 view modes -> which change-classes are shown (rasters are always shown).
MODES = {
    "diff": {"kept": True, "removed": True, "recovered": True},
    "old method": {"kept": True, "removed": True, "recovered": False},   # old detections
    "new method": {"kept": True, "removed": False, "recovered": True},   # new detections
    "just raster": {"kept": False, "removed": False, "recovered": False},
}


def build_detail(summary: pd.DataFrame, cand: pd.DataFrame, out_path: str, cache_dir: str):
    recov = summary[summary.delta > 0].sort_values("delta", ascending=False).head(TOP_N)
    remov = summary[summary.delta < 0].sort_values("delta", ascending=True).head(TOP_N)
    n_recov_total = int((summary.delta > 0).sum())
    n_remov_total = int((summary.delta < 0).sum())
    n_recov, n_remov = len(recov), len(remov)

    cols = GRID_COLS
    recov_rows = int(np.ceil(n_recov / cols)) if n_recov else 0
    total_rows = recov_rows + (int(np.ceil(n_remov / cols)) if n_remov else 0)
    # panel -> grid index: recoveries fill top rows, removals start on a fresh row.
    panels = [(i, wr, "recov") for i, (_, wr) in enumerate(recov.iterrows())]
    panels += [(recov_rows * cols + j, wr, "remov") for j, (_, wr) in enumerate(remov.iterrows())]

    grid_titles = [""] * (total_rows * cols)
    for gi, wr, kind in panels:
        arrow = "↑" if kind == "recov" else "↓"
        grid_titles[gi] = (f"{wr.well_label} {wr['sample']} r{int(wr.run):06d}"
                           f"<br>{wr.old}→{wr.new} {arrow}{abs(int(wr.delta))}")
    fig = make_subplots(rows=total_rows, cols=cols, subplot_titles=grid_titles,
                        horizontal_spacing=0.008, vertical_spacing=0.030)

    key_cols = ["sample", "date", "run", "rec", "well"]
    cand_by = {k: v for k, v in cand.groupby(key_cols)}
    trace_class: list[str] = []   # parallel to fig.data — used to build visibility masks
    n_capped = 0

    for gi, wr, kind in panels:
        r, c = gi // cols + 1, gi % cols + 1
        key = tuple(wr[k] for k in key_cols)
        wc = cand_by.get(key)
        tmax = float(wc.end.max()) if wc is not None and len(wc) else 1.0
        n_units = 1
        sp = find_spike_path(*key)
        if sp:
            keystr = f"{wr['sample']}_{wr.date}_r{int(wr.run):06d}_{wr.rec}_{wr.well}"
            uri, n_units = raster_cached(sp, tmax, cache_dir, keystr)
            fig.add_layout_image(dict(source=uri, xref="x domain", yref="y domain",
                                      x=0, y=1, sizex=1, sizey=1, xanchor="left", yanchor="top",
                                      sizing="stretch", layer="below"), row=r, col=c)
        fig.update_xaxes(range=[0, tmax], row=r, col=c, showticklabels=False)
        fig.update_yaxes(range=[0, n_units], row=r, col=c, showticklabels=False)
        if wc is None:
            continue
        wc = wc.assign(_cls=wc.apply(classify, axis=1))
        for cl in CLS_ORDER:
            grp = wc[wc._cls == cl]
            if not len(grp):
                continue
            if len(grp) > MAX_SPANS:
                n_capped += 1
            drawn = grp.head(MAX_SPANS)
            st = CLASS_STYLE[cl]
            xs: list = []
            ys: list = []
            cd: list = []
            for _, ev in drawn.iterrows():
                xs += [ev.start, ev.end, ev.end, ev.start, ev.start, None]
                ys += [0, 0, n_units, n_units, 0, None]
                base = [ev.posterior_peak, ev.llr_aggregate, ev.peak_synchrony,
                        ev.participation, ev.duration_s, int(ev.total_spikes)]
                cd += [base, base, base, base, base, [None] * 6]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", fill="toself",
                line=dict(width=0.4, color=st["line"]), fillcolor=st["fill"],
                customdata=cd, showlegend=False,
                hovertemplate=("post=%{customdata[0]:.3f} llr=%{customdata[1]:.3f} "
                               "sync=%{customdata[2]:.3f}<br>partic=%{customdata[3]:.2f} "
                               "dur=%{customdata[4]:.2f}s spikes=%{customdata[5]}<extra>" + cl + "</extra>"),
            ), row=r, col=c)
            trace_class.append(cl)

    # static legend color key (always visible)
    for cl in CLS_ORDER:
        st = CLASS_STYLE[cl]
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(width=8, color=st["line"]), name=st["label"],
                                 showlegend=True))
        trace_class.append("legend")

    # 4-mode toggle: each button sets a visibility mask over every trace (images stay).
    def mask_for(m):
        return [True if tc == "legend" else m[tc] for tc in trace_class]

    def title_for(mode_label):
        return (f"ML burst-gate change per recording (mean-LLR≥0.1 → posterior_peak≥0.4) — "
                f"top {n_recov} recoveries ↑ / {n_remov} removals ↓ (of {n_recov_total}/{n_remov_total} total)"
                f"<br><sub>raster = all spikes; bands = burst windows — "
                f"green=recovered (new only), red=removed (old only), gray=kept (both). "
                f"spans capped {MAX_SPANS}/class/panel.  <b>MODE: {mode_label}</b></sub>")

    buttons = [dict(label=label, method="update",
                    args=[{"visible": mask_for(m)}, {"title.text": title_for(label)}])
               for label, m in MODES.items()]

    for tr in fig.data:              # default = diff (all visible)
        tr.visible = True
    # Header layout: reserve a fixed px band above the grid so the title (pinned to the
    # figure top via yref="container"), the mode buttons and the legend (both pinned just
    # above the plot area) get a real gap and never overlap — even on a very tall grid.
    header_px = 340
    plot_px = max(520, 200 * total_rows + header_px) - header_px - 40   # plot-area height (b=40)
    ctrl_y = 1.0 + 100.0 / plot_px   # buttons/legend ~100px above the grid, clear of subplot titles
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="right", showactive=True, active=0,
                          x=0.0, xanchor="left", y=ctrl_y, yanchor="bottom",
                          bgcolor="#eeeeee", bordercolor="#bbbbbb", borderwidth=1,
                          buttons=buttons, pad=dict(t=2, b=2, l=3, r=3))],
        title=dict(text=title_for("diff"), x=0.0, xanchor="left",
                   y=0.99, yref="container", font=dict(size=13)),
        height=max(520, 200 * total_rows + header_px), width=1550, template="plotly_white",
        legend=dict(orientation="h", y=ctrl_y, yanchor="bottom", x=0.62, xanchor="left",
                    bgcolor="rgba(255,255,255,0.75)", bordercolor="#cccccc", borderwidth=1),
        margin=dict(t=header_px, l=42, r=20, b=40),
    )
    for ann in fig.layout.annotations:
        ann.font.size = 9
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"wrote {out_path}  ({len(panels)} panels, {len(trace_class)} traces, "
          f"{n_capped} class-panels capped at {MAX_SPANS})", flush=True)
    return fig, trace_class


# --------------------------------------------------------------------------- summary
def build_summary(summary: pd.DataFrame, out_path: str):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("per-well OLD vs NEW burst count (log, +1)",
                        "Δ = NEW − OLD  (bursts per well)",
                        "total bursts per sample",
                        "wells gained / unchanged / lost"),
        specs=[[{"type": "scatter"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        vertical_spacing=0.14, horizontal_spacing=0.12,
    )
    # (1,1) scatter
    for s, col in SAMPLE_COLORS.items():
        d = summary[summary["sample"] == s]
        fig.add_trace(go.Scatter(
            x=d.old + 1, y=d.new + 1, mode="markers", name=s,
            marker=dict(size=6, color=col, opacity=0.55, line=dict(width=0.3, color="white")),
            customdata=np.stack([d.well_label, d.date, d.run, d.old, d.new, d.delta], axis=-1),
            hovertemplate=("%{customdata[0]} %{customdata[1]} %{customdata[2]}<br>"
                           "old=%{customdata[3]} new=%{customdata[4]} Δ=%{customdata[5]}<extra>" + s + "</extra>"),
        ), row=1, col=1)
    mx = float(max(summary.old.max(), summary.new.max())) + 1
    fig.add_trace(go.Scatter(x=[1, mx], y=[1, mx], mode="lines", line=dict(color="black", dash="dash", width=1),
                             name="no change", showlegend=False), row=1, col=1)
    fig.update_xaxes(type="log", title_text="OLD + 1", row=1, col=1)
    fig.update_yaxes(type="log", title_text="NEW + 1", row=1, col=1)

    # (1,2) delta histogram (clip extreme negatives so bulk is visible)
    dclip = summary.delta.clip(lower=-100, upper=100)
    fig.add_trace(go.Histogram(x=dclip, nbinsx=80, marker_color="#6c5ce7", showlegend=False), row=1, col=2)
    fig.add_vline(x=0, line=dict(color="black", dash="dash", width=1), row=1, col=2)
    fig.update_xaxes(title_text="Δ (clipped ±100)", row=1, col=2)
    fig.update_yaxes(title_text="# wells", row=1, col=2)

    # (2,1) per-sample totals
    samples = list(SAMPLE_COLORS)
    tot_old = [int(summary[summary["sample"] == s].old.sum()) for s in samples]
    tot_new = [int(summary[summary["sample"] == s].new.sum()) for s in samples]
    fig.add_trace(go.Bar(x=samples, y=tot_old, name="OLD", marker_color="#b2bec3"), row=2, col=1)
    fig.add_trace(go.Bar(x=samples, y=tot_new, name="NEW", marker_color="#2d3436"), row=2, col=1)
    fig.update_yaxes(title_text="total bursts", row=2, col=1)

    # (2,2) gained/unchanged/lost
    gained = int((summary.delta > 0).sum())
    unch = int((summary.delta == 0).sum())
    lost = int((summary.delta < 0).sum())
    fig.add_trace(go.Bar(x=["gained (↑)", "unchanged", "lost (↓)"], y=[gained, unch, lost],
                         marker_color=["#00b894", "#b2bec3", "#d63031"], showlegend=False), row=2, col=2)
    fig.update_yaxes(title_text="# wells", row=2, col=2)

    n = len(summary)
    fig.update_layout(
        title=(f"ML burst-gate change summary (mean-LLR≥0.1 → posterior_peak≥0.4) — "
               f"{n} wells, CX118/CX138/CX169<br>"
               f"<sub>total bursts {int(summary.old.sum())} → {int(summary.new.sum())} "
               f"({int(summary.new.sum()-summary.old.sum()):+d}); "
               f"gained {gained}, unchanged {unch}, lost {lost}</sub>"),
        template="plotly_white", height=820, width=1200, barmode="group", margin=dict(t=110),
    )
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", default=VAL_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(args.val_dir, "well_change_summary.csv"))
    cand = pd.read_csv(os.path.join(args.val_dir, "candidates_enriched.csv.gz"))
    print(f"loaded {len(summary)} wells, {len(cand)} candidates", flush=True)

    build_summary(summary, os.path.join(args.out_dir, "summary_gate_changes.html"))
    build_detail(summary, cand, os.path.join(args.out_dir, "detail_gate_changes.html"),
                 cache_dir=os.path.join(args.out_dir, "raster_png"))
    print("PLOTS DONE", flush=True)


if __name__ == "__main__":
    main()
