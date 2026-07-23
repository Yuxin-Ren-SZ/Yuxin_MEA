"""F7b — Node-level graph metrics & directed (transfer-entropy) connectivity.

Companion to F7 (graph-level STTC). Panels:

* **A** Node cartography (Guimerà-Amaral / MEA-NAP) — participation coefficient P
  vs within-module degree z-score for one well; nodes coloured by role
  (connector/provincial hub, peripheral, leaf); z=2.5 and P=0.62 guides.
* **B** Directed TE graph — effective transfer-entropy edges (arrows) on the
  electrode plane for the same well (top edges shown). NB: binary spike-train TE
  is inflated by network-burst co-activation, so the graph is dense; flow
  hierarchy / degree asymmetry are the interpretable readouts, not edge density.
* **C** Node-metric response vs Control maturation (DiD) — hub/leaf fractions,
  participation, betweenness, rich-club, assortativity.
* **D** Directed-metric response vs Control (DiD) — mean TE, degree asymmetry,
  reciprocity, flow hierarchy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load, stats as S
from .report_style import (
    METRIC_LABELS, caption, group_color, ordered_groups, save_fig)

FOCUS_ARMS = ["IVH_Early", "IVH_Late"]
NODE_METRICS = ["hub_fraction", "leaf_fraction", "participation_mean",
                "mean_betweenness", "rich_club", "assortativity"]
DIR_METRICS = ["mean_te", "degree_asymmetry", "reciprocity", "flow_hierarchy"]
_ROLE_COLORS = {"connector_hub": "#d62728", "provincial_hub": "#ff7f0e",
                "peripheral": "#7f7f7f", "leaf": "#1f77b4"}


def _pick(tidy, group="IVH_Late", metric="mean_sttc", lo=14, hi=24):
    sub = tidy[(tidy.canonical_group == group) & (tidy.DIV >= lo)
               & (tidy.DIV <= hi) & tidy[metric].notna()]
    if sub.empty:
        return None
    med = sub[metric].median()
    return sub.iloc[(sub[metric] - med).abs().to_numpy().argsort()[0]]


def _pick_pair(tidy, treated="IVH_Late", metric="mean_sttc", lo=14, hi=24):
    """Representative Control + treated wells on the same chip at matched DIV."""
    win = tidy[(tidy.DIV >= lo) & (tidy.DIV <= hi) & tidy[metric].notna()]
    best = None
    for chip in win.sample_id.unique():
        c = win[(win.sample_id == chip) & (win.canonical_group == "Control")]
        t = win[(win.sample_id == chip) & (win.canonical_group == treated)]
        if c.empty or t.empty:
            continue
        t_row = t.iloc[(t[metric] - t[metric].median()).abs().to_numpy().argsort()[0]]
        c_row = c.iloc[(c.DIV - t_row.DIV).abs().to_numpy().argsort()[0]]
        gap = abs(int(c_row.DIV) - int(t_row.DIV))
        if best is None or gap < best[0]:
            best = (gap, c_row, t_row)
    if best is None:
        return _pick(tidy, "Control"), _pick(tidy, treated)
    return best[1], best[2]


def _panel_cartography(ax, analysis_root, row, title):
    ax.set_title(title, fontsize=7)
    if row is None:
        return
    p = load.artifact_path(analysis_root, "connectivity_data", row,
                           "connectivity", "node_metrics.parquet")
    if not p.exists():
        ax.text(0.5, 0.5, "no node metrics", ha="center", va="center",
                transform=ax.transAxes)
        return
    nm = pd.read_parquet(p)
    for role, c in _ROLE_COLORS.items():
        d = nm[nm.role == role]
        if len(d):
            ax.scatter(d.participation, d.z_within, s=10, color=c, alpha=0.7,
                       lw=0, label=role.replace("_", " "))
    ax.axhline(2.5, ls="--", color="0.5", lw=0.8)
    ax.axvline(0.62, ls=":", color="0.6", lw=0.8)
    ax.set_xlabel("participation P", fontsize=7)
    if title.startswith("Control"):
        ax.set_ylabel("within-module degree z", fontsize=7)
        ax.legend(fontsize=5, loc="lower left")


def _panel_directed_graph(ax, analysis_root, row, top_frac=0.15):
    ax.set_title("B  Directed TE graph", loc="left", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    if row is None:
        return
    dbase = load.artifact_path(analysis_root, "directed_connectivity_data", row,
                               "directed", "te_matrix.npy")
    if not dbase.exists():
        ax.text(0.5, 0.5, "no directed data", ha="center", va="center",
                transform=ax.transAxes)
        return
    TE = np.load(dbase)
    uids = np.load(dbase.parent / "unit_ids.npy")
    qm = pd.read_pickle(load.quality_metrics_path(analysis_root, row))
    pos = {int(u): (float(r["loc_x"]), float(r["loc_y"]))
           for u, r in qm.iterrows() if "loc_x" in qm.columns}
    xy = np.array([pos.get(int(u), (np.nan, np.nan)) for u in uids])
    ok = ~np.isnan(xy[:, 0])
    ax.scatter(xy[ok, 0], xy[ok, 1], s=6, color="0.4", zorder=3)
    if TE.size:
        thr = np.quantile(TE[TE > 0], 1 - top_frac) if (TE > 0).any() else np.inf
        n = TE.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and TE[i, j] >= thr and ok[i] and ok[j]:
                    x0, y0 = xy[i]; x1, y1 = xy[j]
                    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                arrowprops=dict(arrowstyle="->", color="#b2182b",
                                                alpha=0.35, lw=0.4), zorder=2)
    ax.set_aspect("equal")


def _stars(q):
    if q is None or np.isnan(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def _forest(ax, tidy, metrics, title):
    from .forest import plot_did_forest

    resp = S.well_response(tidy, metrics=metrics)
    resp = resp[resp.arm.isin(["Control"] + FOCUS_ARMS)]
    plot_did_forest(ax, resp, metrics, FOCUS_ARMS,
                    xlabel="response (log2 post/pre; Δ for bounded)",
                    title=title, legend_loc="upper left", footnote=False)


def build_f7b(tidy, analysis_root):
    import matplotlib.pyplot as plt

    from .trajectory import pick_example_well, pick_early_late, plot_trajectory

    ex = pick_example_well(tidy, "IVH_Late", metric="mean_sttc")
    e_row, l_row = pick_early_late(tidy, ex) if ex else (None, None)

    def _lab(row, when):
        return when if row is None else f"{when} DIV{int(row.DIV)} tau{int(row.tau):+d}"

    fig = plt.figure(figsize=(9.2, 10.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85],
                          hspace=0.5, wspace=0.28)

    # Row 0: A cartography early | late  +  B directed graph early | late
    gsa = gs[0, 0].subgridspec(1, 2, wspace=0.08)
    axa0 = fig.add_subplot(gsa[0, 0]); axa1 = fig.add_subplot(gsa[0, 1])
    _panel_cartography(axa0, analysis_root, e_row, _lab(e_row, "early"))
    _panel_cartography(axa1, analysis_root, l_row, _lab(l_row, "late"))
    axa1.sharey(axa0)
    axa0.annotate("A  Node cartography — early vs late (P vs z)", xy=(0, 1.15),
                  xycoords="axes fraction", fontweight="bold", fontsize=9)
    gsb = gs[0, 1].subgridspec(1, 2, wspace=0.08)
    axb0 = fig.add_subplot(gsb[0, 0]); axb1 = fig.add_subplot(gsb[0, 1])
    _panel_directed_graph(axb0, analysis_root, e_row)
    _panel_directed_graph(axb1, analysis_root, l_row)
    for a, when in [(axb0, "early"), (axb1, "late")]:
        a.set_title("", loc="left"); a.set_title(when, fontsize=7)
    axb0.annotate("B  Directed TE graph — early vs late", xy=(0, 1.15),
                  xycoords="axes fraction", fontweight="bold", fontsize=9)

    # Row 1: C node forest | D directed forest
    _forest(fig.add_subplot(gs[1, 0]), tidy, NODE_METRICS,
            "C  Node-metric response vs Control (DiD, chip-level)")
    _forest(fig.add_subplot(gs[1, 1]), tidy, DIR_METRICS,
            "D  Directed-metric response vs Control (DiD, chip-level)")

    # Row 2: E trajectories vs tau
    arms = ["Control"] + FOCUS_ARMS
    gse = gs[2, :].subgridspec(1, 3, wspace=0.35)
    for ci, metric in enumerate(["participation_mean", "rich_club", "flow_hierarchy"]):
        axe = fig.add_subplot(gse[0, ci])
        plot_trajectory(axe, tidy, metric, arms, legend=(ci == 0))
        if ci == 0:
            axe.annotate("E  Node/directed metrics vs treatment day", xy=(0, 1.15),
                         xycoords="axes fraction", fontweight="bold", fontsize=9)
    fig.suptitle("Figure 7b — Node-level & directed (TE) connectivity; "
                 "chip = biological replicate"
                 + ("" if ex is None else f"  ·  {ex.split('|')[0]} (IVH_Late)"),
                 fontsize=9, y=1.0)
    caption(fig,
        "Node-level & directed connectivity. (A) Node cartography (participation "
        "coefficient P vs within-module degree z) of one representative treated "
        "well, early vs late; dashed lines = hub thresholds. (B) Directed "
        "transfer-entropy (TE) graph for the same well early vs late (arrows = "
        "inferred information flow). (C, D) Difference-in-differences vs Control "
        "for node-level (C) and directed (D) metrics at the biological-replicate "
        "level: bold points = 3 per-chip means, faint = wells; marker = mean of "
        "chip means, whisker = 95% t-CI over 3 chips (df=2); ★ = FDR q<0.05, △ = "
        "suggestive (consistent 3/3 chips, p<0.05 uncorrected). (E) Group "
        "trajectories vs treatment day (tau): line = per-arm median across wells, "
        "ribbon = 95% bootstrap CI (well-level). TE = Schreiber (2000) transfer "
        "entropy, effective/bias-corrected. Unit: chip (n=3 per arm).")
    return fig, {"example": ex}


def render(tidy, analysis_root, figure_root):
    fig, meta = build_f7b(tidy, analysis_root)
    return save_fig(fig, "F7b_node_directed", figure_root, subdir="main"), meta
