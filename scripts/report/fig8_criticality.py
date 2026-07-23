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
from .report_style import (
    METRIC_LABELS, caption, group_color, ordered_groups, save_fig)

FOCUS_ARMS = ["IVH_Early", "IVH_Late"]
FOREST_METRICS = ["branching_ratio_mr", "dcc", "aval_tau", "aval_alpha"]
_B_GROUPS = ["Control", "IVH_Early", "IVH_Late"]


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


_EARLY_C, _LATE_C = "#4477aa", "#cc3311"    # early vs late timepoint colours


def _panel_distributions(ax, analysis_root, e_row, l_row):
    ax.set_title("A  Avalanche distributions (early vs late)", loc="left",
                 fontweight="bold")
    for row, col, lab in [(e_row, _EARLY_C, "early"), (l_row, _LATE_C, "late")]:
        if row is None:
            continue
        av = _avalanches(analysis_root, row)
        if av.empty:
            continue
        tau = int(row.get("tau", 0))
        _loglog_pdf(ax, av["size"], col, f"{lab} (τ{tau:+d})",
                    float(row.get("aval_tau", np.nan)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("avalanche size"); ax.set_ylabel("PDF")
    ax.legend(fontsize=6, loc="lower left")
    if e_row is not None:
        ax.text(0.98, 0.98, e_row.well_uid.split("|")[0], transform=ax.transAxes,
                ha="right", va="top", fontsize=6, color="0.5")


def _panel_crackling(ax, analysis_root, e_row, l_row):
    ax.set_title("B  Crackling scaling (early vs late)", loc="left",
                 fontweight="bold")
    for row, col, lab in [(e_row, _EARLY_C, "early"), (l_row, _LATE_C, "late")]:
        if row is None:
            continue
        av = _avalanches(analysis_root, row)
        if av.empty:
            continue
        df = av.groupby("duration")["size"].mean()
        df = df[df.index > 0]
        dcc = float(row.get("dcc", np.nan))
        ax.loglog(df.index, df.values, "o", ms=3, color=col, alpha=0.7,
                  label=f"{lab} DCC={dcc:.2f}")
    ax.set_xlabel("duration (bins)"); ax.set_ylabel("mean size")
    ax.legend(fontsize=6, loc="upper left")


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


def build_f8(tidy, analysis_root):
    import matplotlib.pyplot as plt

    from .forest import plot_did_forest
    from .trajectory import plot_trajectory, pick_early_late, pick_example_well

    resp = S.well_response(tidy, metrics=FOREST_METRICS)
    resp = resp[resp.arm.isin(["Control"] + FOCUS_ARMS)]
    ex_well = pick_example_well(tidy, "IVH_Late", metric="branching_ratio_mr")
    e_row, l_row = pick_early_late(tidy, ex_well) if ex_well else (None, None)
    arms = ["Control"] + FOCUS_ARMS

    fig = plt.figure(figsize=(9.2, 9.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1], hspace=0.5, wspace=0.3)
    # Row 0: early-vs-late avalanche example (one treated well)
    _panel_distributions(fig.add_subplot(gs[0, 0]), analysis_root, e_row, l_row)
    _panel_crackling(fig.add_subplot(gs[0, 1]), analysis_root, e_row, l_row)
    # Row 1: group trajectories vs treatment day
    axc = fig.add_subplot(gs[1, 0])
    plot_trajectory(axc, tidy, "branching_ratio_mr", arms, ref=1.0,
                    ylabel="branching ratio (MR)", title="C  Branching ratio vs tau")
    axd = fig.add_subplot(gs[1, 1])
    plot_trajectory(axd, tidy, "dcc", arms, legend=False,
                    ylabel="DCC (distance to criticality)", title="D  DCC vs tau")
    # Row 2: DiD forest (chip-level, well-nested-in-chip LMM)
    axe = fig.add_subplot(gs[2, :])
    fdata = plot_did_forest(
        axe, resp, FOREST_METRICS, FOCUS_ARMS,
        xlabel="response  (log2 post/pre; Δ for BR/DCC)",
        title="E  Criticality response vs Control (DiD, chip-level)",
        legend_loc="lower right")
    fig.suptitle("Figure 8 — Neuronal-avalanche criticality (branching ratio, DCC); "
                 "chip = biological replicate", fontsize=9, y=1.0)
    caption(fig,
        "Neuronal-avalanche criticality. (A) Avalanche-size distribution (log–log "
        "PDF) of one representative treated well (CX169), early (τ+0) vs late "
        "(τ+14); solid line = fitted power law (exponent τ). (B) Crackling-noise "
        "scaling — mean avalanche size vs duration for the same well early vs "
        "late; DCC (distance to criticality) per timepoint in the legend. "
        "(C, D) Group trajectories vs treatment day (tau): branching ratio (MR "
        "estimator; dashed line = critical value 1) and DCC. Line = per-arm "
        "median across wells, ribbon = 95% bootstrap CI (well-level), dotted "
        "vertical = treatment day (tau 0). (E) Difference-in-differences vs "
        "Control at the biological-replicate level: marker = mean of the 3 "
        "per-chip mean responses, whisker = 95% t-CI over those 3 chips (df=2); "
        "bold dots = per-chip means, faint dots = wells; Control diamond = "
        "maturation baseline. ★ = FDR q<0.05 (none here); △ = suggestive (same "
        "direction on all 3 chips, uncorrected p<0.05, does not survive FDR at "
        "n=3). Response = log2(post/pre), or Δ(post−pre) for branching ratio and "
        "DCC. Unit of replication: chip (n=3 per arm).")
    return fig, {"response": resp, "example": ex_well,
                 **(fdata or {})}


def render(tidy, analysis_root, figure_root):
    fig, data = build_f8(tidy, analysis_root)
    return save_fig(fig, "F8_criticality", figure_root, subdir="main"), data
