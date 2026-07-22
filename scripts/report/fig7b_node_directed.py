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
from .report_style import METRIC_LABELS, group_color, ordered_groups, save_fig

FOCUS_ARMS = ["IVH_Early", "IVH_Late", "H2O2_20uM"]
NODE_METRICS = ["hub_fraction", "leaf_fraction", "participation_mean",
                "mean_betweenness", "rich_club", "assortativity"]
DIR_METRICS = ["mean_te", "degree_asymmetry", "reciprocity", "flow_hierarchy"]
_ROLE_COLORS = {"connector_hub": "#d62728", "provincial_hub": "#ff7f0e",
                "peripheral": "#7f7f7f", "leaf": "#1f77b4"}


def _pick(tidy, metric="mean_sttc"):
    sub = tidy[(tidy.canonical_group == "IVH_Late") & (tidy.DIV >= 14)
               & (tidy.DIV <= 24) & tidy[metric].notna()]
    if sub.empty:
        return None
    med = sub[metric].median()
    return sub.iloc[(sub[metric] - med).abs().to_numpy().argsort()[0]]


def _panel_cartography(ax, analysis_root, row):
    ax.set_title("A  Node cartography", loc="left", fontweight="bold")
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
    ax.set_xlabel("participation coefficient P")
    ax.set_ylabel("within-module degree z")
    ax.legend(fontsize=5.5, loc="upper left")


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
    resp = S.well_response(tidy, metrics=metrics)
    resp = resp[resp.arm.isin(["Control"] + FOCUS_ARMS)]
    summ = S.arm_response_summary(resp) if not resp.empty else pd.DataFrame()
    vc = S.arm_response_vs_control(resp) if not resp.empty else pd.DataFrame()
    if summ.empty:
        ax.text(0.5, 0.5, "no paired wells", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title, loc="left", fontweight="bold")
        return
    rng = np.random.default_rng(0)
    treated = [a for a in ordered_groups(summ.arm.unique()) if a in FOCUS_ARMS]
    arms = ["Control"] + treated
    arm_off = np.linspace(0.3, -0.3, len(arms))
    vc_idx = vc.set_index(["arm", "metric"]) if len(vc) else None
    yticks, ylabels = [], []
    for mi, metric in enumerate(metrics):
        y0 = -mi; yticks.append(y0); ylabels.append(METRIC_LABELS.get(metric, metric))
        for ai, arm in enumerate(arms):
            r = summ[(summ.metric == metric) & (summ.arm == arm)]
            if r.empty:
                continue
            r = r.iloc[0]; y = y0 + arm_off[ai]; is_ctrl = arm == "Control"
            pts = resp.loc[(resp.metric == metric) & (resp.arm == arm),
                           "response"].to_numpy(float)
            if len(pts):
                jit = (rng.random(len(pts)) - 0.5) * 0.16
                ax.scatter(pts, np.full(len(pts), y) + jit, s=4,
                           color=group_color(arm), alpha=0.28, lw=0, zorder=2)
            xerr = np.array([[max(r.median_response - r.ci_low, 0)],
                             [max(r.ci_high - r.median_response, 0)]])
            ax.errorbar(r.median_response, y, xerr=xerr,
                        fmt=("D" if is_ctrl else "o"), ms=(4 if is_ctrl else 3.5),
                        color=group_color(arm), capsize=2, elinewidth=0.8,
                        markerfacecolor=group_color(arm), markeredgecolor=group_color(arm))
            if not is_ctrl and vc_idx is not None and (arm, metric) in vc_idx.index:
                s = _stars(vc_idx.loc[(arm, metric), "q_bh"])
                if s:
                    ax.text(r.ci_high + 0.02, y, s, va="center", ha="left",
                            fontsize=6.5, color=group_color(arm))
    ax.axvline(0, ls="--", color="0.5", lw=0.8)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_ylim(min(yticks) - 0.6, max(yticks) + 0.6)
    if len(summ):     # clip to CI range so outliers don't squash the medians
        lo = float(summ.ci_low.min()); hi = float(summ.ci_high.max())
        pad = 0.25 * (hi - lo) + 0.1
        ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("response (log2 post/pre; Δ for bounded)", fontsize=7)
    ax.set_title(title, loc="left", fontweight="bold")


def build_f7b(tidy, analysis_root):
    import matplotlib.pyplot as plt

    row = _pick(tidy)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
    _panel_cartography(axes[0, 0], analysis_root, row)
    _panel_directed_graph(axes[0, 1], analysis_root, row)
    _forest(axes[1, 0], tidy, NODE_METRICS, "C  Node-metric response vs Control")
    _forest(axes[1, 1], tidy, DIR_METRICS, "D  Directed-metric response vs Control")
    tag = "" if row is None else f"  ·  example {row.well_uid.split('|')[0]} DIV{int(row.DIV)}"
    fig.suptitle(f"Figure 7b — Node-level & directed (TE) connectivity, "
                 f"well_uid unit{tag}", fontsize=9, y=1.0)
    fig.tight_layout()
    return fig, {"example": None if row is None else row.well_uid}


def render(tidy, analysis_root, figure_root):
    fig, meta = build_f7b(tidy, analysis_root)
    return save_fig(fig, "F7b_node_directed", figure_root, subdir="main"), meta
