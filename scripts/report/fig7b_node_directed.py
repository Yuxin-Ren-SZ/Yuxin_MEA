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


def _forest(ax, tidy, metrics, title, *, highlight_metric=None,
            robust_xlim=False, effect_labels=False):
    from .forest import plot_did_forest

    resp = S.well_response(tidy, metrics=metrics)
    resp = resp[resp.arm.isin(["Control"] + FOCUS_ARMS)]
    return plot_did_forest(
        ax, resp, metrics, FOCUS_ARMS,
        xlabel="response (log2 post/pre; Δ for bounded)",
        title=title, legend_loc="upper left", footnote=False,
        highlight_metric=highlight_metric, robust_xlim=robust_xlim,
        effect_labels=effect_labels)


def build_f7b(tidy, analysis_root):
    import matplotlib.pyplot as plt

    from .report_style import MUTED
    from .trajectory import pick_example_well, pick_early_late, plot_trajectory

    ex = pick_example_well(tidy, "IVH_Late", metric="mean_sttc")
    e_row, l_row = pick_early_late(tidy, ex) if ex else (None, None)

    def _lab(row, when):
        return when if row is None else f"{when} DIV{int(row.DIV)} tau{int(row.tau):+d}"

    fig = plt.figure(figsize=(9.4, 10.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 0.9, 0.8], top=0.94,
                          bottom=0.06, hspace=0.58, wspace=0.3)

    # Row 0 (LEAD): the inference. Two forests, independent x-scales; the node
    # forest clips the participation-coeff runaway CI; flow hierarchy highlighted.
    _forest(fig.add_subplot(gs[0, 0]), tidy, NODE_METRICS,
            "A  Node-metric DiD vs Control", robust_xlim=True)
    _forest(fig.add_subplot(gs[0, 1]), tidy, DIR_METRICS,
            "B  Directed-metric DiD — flow hierarchy highlighted",
            highlight_metric="flow_hierarchy", effect_labels=True, robust_xlim=True)

    # Row 1: C trajectories — the interpretable readouts (participation + flow).
    arms = ["Control"] + FOCUS_ARMS
    gsc = gs[1, :].subgridspec(1, 2, wspace=0.3)
    for ci, metric in enumerate(["participation_mean", "flow_hierarchy"]):
        axc = fig.add_subplot(gsc[0, ci])
        plot_trajectory(axc, tidy, metric, arms, legend=(ci == 0))
        if ci == 0:
            axc.annotate("C  Node/directed metrics vs treatment day", xy=(0, 1.14),
                         xycoords="axes fraction", fontweight="bold", fontsize=9)

    # Row 2: D demoted illustrative strip — cartography + TE graph (single well).
    gsd = gs[2, :].subgridspec(1, 4, wspace=0.14)
    axd = [fig.add_subplot(gsd[0, i]) for i in range(4)]
    _panel_cartography(axd[0], analysis_root, e_row, _lab(e_row, "early"))
    _panel_cartography(axd[1], analysis_root, l_row, _lab(l_row, "late"))
    axd[1].sharey(axd[0])
    _panel_directed_graph(axd[2], analysis_root, e_row)
    _panel_directed_graph(axd[3], analysis_root, l_row)
    # Override each title at its OWN loc so the sub-panel's baked-in title is
    # replaced, not stacked (cartography = center loc; directed = left loc).
    axd[0].set_title(_lab(e_row, "early"), fontsize=7, color=MUTED)
    axd[1].set_title(_lab(l_row, "late"), fontsize=7, color=MUTED)
    axd[2].set_title("TE early", loc="left", fontsize=7, color=MUTED)
    axd[3].set_title("TE late", loc="left", fontsize=7, color=MUTED)
    axd[0].annotate("D  Illustrative — single well: node cartography (P vs z) + "
                    "directed TE graph, early vs late", xy=(0, 1.20),
                    xycoords="axes fraction", fontweight="bold", fontsize=8,
                    color=MUTED, annotation_clip=False)

    fig.suptitle("Figure 7b — Node-level & directed (TE) connectivity; "
                 "chip = biological replicate"
                 + ("" if ex is None else f"  ·  {ex.split('|')[0]} (IVH_Late)"),
                 fontsize=9, y=0.985)
    caption(fig,
        "Node-level & directed connectivity; the point of interest is directed-"
        "flow reorganisation (flow hierarchy). (A, B) Difference-in-differences vs "
        "Control for node-level (A) and directed (B) metrics, each on its own "
        "x-scale; bold points = 3 per-chip means (biological replicates), faint = "
        "wells; marker = mean of chip means, whisker = 95% t-CI (df=2); the "
        "participation-coefficient CI is clipped to the axis (⟩ = off-scale) so "
        "one wild interval does not flatten the rest; the flow-hierarchy row is "
        "highlighted with its effect size Δ. ★ = FDR q<0.05 (none survive at "
        "n=3); △ = suggestive (consistent 3/3 chips, p<0.05 uncorrected). (C) "
        "Group trajectories vs treatment day (tau): participation coefficient and "
        "flow hierarchy; line = per-arm median across wells, ribbon = 95% "
        "bootstrap CI. (D) Demoted illustrative strip (single well): node "
        "cartography (participation P vs within-module degree z; dashed lines = "
        "hub thresholds) and directed transfer-entropy graph (arrows = inferred "
        "flow), early vs late. TE = Schreiber (2000), effective/bias-corrected. "
        "Unit: chip (n=3 per arm).")
    return fig, {"example": ex}


def render(tidy, analysis_root, figure_root):
    fig, meta = build_f7b(tidy, analysis_root)
    return save_fig(fig, "F7b_node_directed", figure_root, subdir="main"), meta
