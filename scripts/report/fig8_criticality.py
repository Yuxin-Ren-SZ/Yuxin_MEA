"""F8 — Neuronal-avalanche criticality.

* **A** Avalanche size & duration distributions (log-log) for a representative
  well, with the MLE power-law exponents τ (size) and α (duration).
* **B** Crackling-noise scaling — mean avalanche size vs duration; the fitted γ
  and the criticality-predicted γ=(α−1)/(τ−1) give the DCC.
* **C** Branching ratio per group (well_uid unit) — MR estimator (subsampling-
  robust) with the naive estimator for comparison; dashed line at the critical
  m=1.
* **D** Criticality response vs Control maturation (difference-in-differences)
  for branching ratio, DCC and the avalanche exponents, focus arms only.

References: Beggs & Plenz 2003; Ma et al. 2019 (DCC); Wilting & Priesemann 2018
(MR estimator); powerlaw (Alstott/Clauset).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load, stats as S
from .report_style import METRIC_LABELS, group_color, ordered_groups, save_fig

FOCUS_ARMS = ["IVH_Early", "IVH_Late", "H2O2_20uM"]
FOREST_METRICS = ["branching_ratio_mr", "dcc", "aval_tau", "aval_alpha"]
_B_GROUPS = ["Control", "IVH_Early", "IVH_Late", "H2O2_20uM"]


def _avalanches(analysis_root, row):
    p = load.artifact_path(analysis_root, "criticality_data", row,
                           "criticality", "avalanches.parquet")
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _loglog_pdf(ax, x, color, label, fit_exp=None):
    x = np.asarray(x); x = x[x > 0]
    if len(x) < 10:
        return
    lo, hi = x.min(), x.max()
    bins = np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), 25)).astype(int))
    bins = bins[bins > 0]
    if len(bins) < 3:
        return
    h, edges = np.histogram(x, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    m = h > 0
    ax.plot(centers[m], h[m], "o", ms=3, color=color, label=label, alpha=0.8)
    if fit_exp and np.isfinite(fit_exp):
        xs = centers[m]
        ys = xs.astype(float) ** (-fit_exp)
        ys = ys / ys[0] * h[m][0]
        ax.plot(xs, ys, "-", color=color, lw=1.0, alpha=0.7)


def _pick_example(tidy):
    """Representative well: near-median branching ratio, DIV 14-24, focus arm."""
    sub = tidy[(tidy.DIV >= 14) & (tidy.DIV <= 24)
               & tidy.branching_ratio_mr.notna()
               & tidy.canonical_group.isin(_B_GROUPS)]
    if sub.empty:
        return None
    med = sub.branching_ratio_mr.median()
    return sub.iloc[(sub.branching_ratio_mr - med).abs().to_numpy().argsort()[0]]


def _panel_distributions(ax, analysis_root, row):
    ax.set_title("A  Avalanche distributions", loc="left", fontweight="bold")
    if row is None:
        ax.text(0.5, 0.5, "no example", ha="center", va="center", transform=ax.transAxes)
        return
    av = _avalanches(analysis_root, row)
    if av.empty:
        ax.text(0.5, 0.5, "no avalanches", ha="center", va="center", transform=ax.transAxes)
        return
    tau = float(row.get("aval_tau", np.nan))
    alpha = float(row.get("aval_alpha", np.nan))
    _loglog_pdf(ax, av["size"], "#333333", f"size (τ={tau:.2f})", tau)
    _loglog_pdf(ax, av["duration"], "#d6604d", f"duration (α={alpha:.2f})", alpha)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("avalanche size / duration"); ax.set_ylabel("PDF")
    ax.legend(fontsize=6, loc="lower left")
    tag = f"{row.well_uid.split('|')[0]} DIV{int(row.DIV)}"
    ax.text(0.98, 0.98, tag, transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color="0.5")


def _panel_crackling(ax, analysis_root, row):
    ax.set_title("B  Crackling-noise scaling", loc="left", fontweight="bold")
    if row is None:
        return
    av = _avalanches(analysis_root, row)
    if av.empty:
        return
    df = av.groupby("duration")["size"].mean()
    df = df[df.index > 0]
    ax.loglog(df.index, df.values, "o", ms=3, color="#4477aa", alpha=0.8)
    g_fit = float(row.get("gamma_fit", np.nan))
    g_pred = float(row.get("gamma_pred", np.nan)) if "gamma_pred" in row else np.nan
    dcc = float(row.get("dcc", np.nan))
    xs = np.array(sorted(df.index))
    if np.isfinite(g_fit):
        ax.loglog(xs, (xs ** g_fit) / (xs[0] ** g_fit) * df.values[0], "-",
                  color="#4477aa", lw=1.0, label=f"γ_fit={g_fit:.2f}")
    ax.set_xlabel("duration (bins)"); ax.set_ylabel("mean size")
    ax.legend(fontsize=6, loc="upper left")
    ax.text(0.98, 0.02, f"DCC={dcc:.2f}", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7, color="0.2",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))


_TAU_EDGES = list(range(-6, 22, 4))   # treatment-relative day bins


def _boot_ci(x, seed=0, n=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _panel_branching(ax, tidy):
    """Branching-ratio trajectory vs treatment day (tau), per arm — so the
    pre→post change is visible (pooling all days would hide it)."""
    ax.set_title("C  Branching ratio vs treatment day", loc="left", fontweight="bold")
    wi = S.well_index(tidy)[["well_uid", "arm"]]
    d = tidy.merge(wi, on="well_uid")
    d = d[d.arm.isin(_B_GROUPS) & d.branching_ratio_mr.notna()].copy()
    cats = pd.cut(d.tau, _TAU_EDGES, right=False)
    d["tauc"] = cats.map(lambda c: (c.left + c.right) / 2 if pd.notna(c) else np.nan)
    # per well_uid x arm x tau-bin median (kills pseudo-replication)
    pw = d.groupby(["arm", "well_uid", "tauc"])["branching_ratio_mr"].median().reset_index()
    for g in [a for a in ordered_groups(d.arm.unique()) if a in _B_GROUPS]:
        sub = pw[pw.arm == g]
        xs, ys, los, his = [], [], [], []
        for tc, grp in sub.groupby("tauc"):
            v = grp.branching_ratio_mr.to_numpy()
            if np.isnan(tc) or len(v) < 2:
                continue
            lo, hi = _boot_ci(v)
            xs.append(tc); ys.append(np.median(v)); los.append(lo); his.append(hi)
        if xs:
            order = np.argsort(xs); xs = np.array(xs)[order]
            ys = np.array(ys)[order]; los = np.array(los)[order]; his = np.array(his)[order]
            ax.plot(xs, ys, "-o", ms=3, color=group_color(g), label=g, zorder=3)
            ax.fill_between(xs, los, his, color=group_color(g), alpha=0.15, lw=0)
    ax.axhline(1.0, ls="--", color="0.5", lw=0.8)
    ax.axvline(0.0, ls=":", color="0.5", lw=0.8)
    ax.set_xlabel("treatment day (tau)")
    ax.set_ylabel("branching ratio (MR)")
    ax.legend(fontsize=6, loc="lower left", ncol=2)
    ax.text(0.98, 0.02, "-- m=1 critical · : treatment", transform=ax.transAxes,
            fontsize=5.5, color="0.45", va="bottom", ha="right")


def _stars(q):
    if q is None or np.isnan(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def _panel_forest(ax, summ, vc):
    treated = [a for a in ordered_groups(summ.arm.unique()) if a in FOCUS_ARMS]
    arms = ["Control"] + treated
    arm_off = np.linspace(0.3, -0.3, len(arms))
    vc_idx = vc.set_index(["arm", "metric"]) if len(vc) else None
    yticks, ylabels = [], []
    for mi, metric in enumerate(FOREST_METRICS):
        y0 = -mi; yticks.append(y0); ylabels.append(METRIC_LABELS.get(metric, metric))
        for ai, arm in enumerate(arms):
            r = summ[(summ.metric == metric) & (summ.arm == arm)]
            if r.empty:
                continue
            r = r.iloc[0]; y = y0 + arm_off[ai]; is_ctrl = arm == "Control"
            xerr = np.array([[max(r.median_response - r.ci_low, 0)],
                             [max(r.ci_high - r.median_response, 0)]])
            ax.errorbar(r.median_response, y, xerr=xerr,
                        fmt=("D" if is_ctrl else "o"), ms=(4.5 if is_ctrl else 4),
                        color=group_color(arm), capsize=2, elinewidth=0.9,
                        markerfacecolor=group_color(arm), markeredgecolor=group_color(arm),
                        zorder=(4 if is_ctrl else 3))
            if not is_ctrl and vc_idx is not None and (arm, metric) in vc_idx.index:
                s = _stars(vc_idx.loc[(arm, metric), "q_bh"])
                if s:
                    ax.text(r.ci_high + 0.02, y, s, va="center", ha="left",
                            fontsize=7, color=group_color(arm))
    ax.axvline(0, ls="--", color="0.5", lw=0.8)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.set_ylim(min(yticks) - 0.6, max(yticks) + 0.6)
    ax.set_xlabel("response  (log2 post/pre; Δ for BR/DCC)")
    ax.set_title("D  Criticality response vs Control", loc="left", fontweight="bold")
    handles = [ax.plot([], [], marker="D", ls="", color="k", label="Control")[0]]
    handles += [ax.plot([], [], marker="o", ls="", color=group_color(a), label=a)[0]
                for a in treated]
    ax.legend(handles=handles, loc="lower right", fontsize=6)


def build_f8(tidy, analysis_root):
    import matplotlib.pyplot as plt

    resp = S.well_response(tidy, metrics=FOREST_METRICS)
    resp = resp[resp.arm.isin(["Control"] + FOCUS_ARMS)]
    summ = S.arm_response_summary(resp) if not resp.empty else pd.DataFrame()
    vc = S.arm_response_vs_control(resp) if not resp.empty else pd.DataFrame()
    ex = _pick_example(tidy)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.0))
    _panel_distributions(axes[0, 0], analysis_root, ex)
    _panel_crackling(axes[0, 1], analysis_root, ex)
    _panel_branching(axes[1, 0], tidy)
    if summ.empty:
        axes[1, 1].text(0.5, 0.5, "no paired wells", ha="center", va="center",
                        transform=axes[1, 1].transAxes)
    else:
        _panel_forest(axes[1, 1], summ, vc)
    fig.suptitle("Figure 8 — Neuronal-avalanche criticality (branching ratio, DCC), "
                 "well_uid unit", fontsize=9, y=1.0)
    fig.tight_layout()
    return fig, {"response": resp, "summary": summ, "vs_control": vc}


def render(tidy, analysis_root, figure_root):
    fig, data = build_f8(tidy, analysis_root)
    return save_fig(fig, "F8_criticality", figure_root, subdir="main"), data
