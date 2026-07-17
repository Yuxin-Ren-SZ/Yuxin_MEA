"""F5 — Network-burst phenotype: IVH & oxidative stress vs Control maturation.

Focus arms: **IVH_Early, IVH_Late** (intraventricular-haemorrhage CSF) and
**H2O2_20uM** (oxidative stress); Control is the developmental reference. NPH,
AraC and H2O2_10uM are excluded here (see S4 for the full cross-group view).

The pipeline records each treated well as ``Control`` before treatment
(``tau < 0``) and its treatment name after (``tau > 0``); the naive within-well
``log2(post/pre)`` is confounded with development. Control wells (never treated)
give the maturation-only change, so treatment is read as a **difference-in-
differences**: each arm's response vs Control's response (Mann-Whitney + BH-FDR).

* **A. Response vs maturation** — per metric, each arm's within-well pre→post
  response (median ± 95% bootstrap CI). Control (black diamond) is the
  developmental baseline; treated arms are read relative to it. Stars = DiD
  significant vs Control.
* **B. ROS-Glo assay (IVH_Early vs IVH_Late)** — oxidative-stress readout.
  Placeholder until the assay table is supplied (`--rosglo-table`).

Note (statistics): paired n is modest (IVH×2, H2O2_20uM = 12-13 wells / 3 chips);
several DiD contrasts sit near p≈0.05 and are power-limited — reported as trends,
not claims, pending more N.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S
from .report_style import (
    METRIC_LABELS, group_color, ordered_groups, save_fig,
)

# Arms shown in F5 (Control is always the reference; not listed here).
FOCUS_ARMS = ["IVH_Early", "IVH_Late", "H2O2_20uM"]
FOREST_METRICS = [
    "nb_rate", "nb_duration_mean", "nb_spikes_per_burst_mean",
    "nb_ibi_mean", "median_firing_rate",
]


def _stars(q: float) -> str:
    if np.isnan(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def _panel_forest(ax, summ, vc) -> None:
    treated = [a for a in ordered_groups(summ.arm.unique())
               if a in FOCUS_ARMS]
    arms = ["Control"] + treated
    arm_off = np.linspace(0.30, -0.30, len(arms))
    vc_idx = vc.set_index(["arm", "metric"]) if len(vc) else None
    yticks, ylabels = [], []
    for mi, metric in enumerate(FOREST_METRICS):
        y0 = -mi
        yticks.append(y0)
        ylabels.append(METRIC_LABELS.get(metric, metric))
        for ai, arm in enumerate(arms):
            r = summ[(summ.metric == metric) & (summ.arm == arm)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = y0 + arm_off[ai]
            is_ctrl = arm == "Control"
            xerr = np.array([[max(r.median_response - r.ci_low, 0)],
                             [max(r.ci_high - r.median_response, 0)]])
            ax.errorbar(
                r.median_response, y, xerr=xerr,
                fmt=("D" if is_ctrl else "o"), ms=(4.5 if is_ctrl else 4),
                color=group_color(arm), capsize=2, elinewidth=0.9,
                markerfacecolor=group_color(arm), markeredgecolor=group_color(arm),
                zorder=(4 if is_ctrl else 3),
            )
            if not is_ctrl and vc_idx is not None and (arm, metric) in vc_idx.index:
                s = _stars(vc_idx.loc[(arm, metric), "q_bh"])
                if s:
                    ax.text(r.ci_high + 0.05, y, s, va="center", ha="left",
                            fontsize=7, color=group_color(arm))
    ax.axvline(0, ls="--", color="0.5", lw=0.8, zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(min(yticks) - 0.6, max(yticks) + 0.6)
    ax.set_xlabel("within-well response  log2(post / pre)")
    ax.set_title("A  Response vs developmental maturation", loc="left",
                 fontweight="bold")
    handles = [ax.plot([], [], marker="D", ls="", color="k",
                       label="Control (maturation)")[0]]
    handles += [ax.plot([], [], marker="o", ls="", color=group_color(a), label=a)[0]
                for a in treated]
    ax.legend(handles=handles, loc="upper left", fontsize=6, ncol=1)


def _load_rosglo(path) -> pd.DataFrame:
    path = str(path)
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) \
        else pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    gcol = next((low[c] for c in ("group", "condition", "arm") if c in low), None)
    vcol = next((low[c] for c in ("value", "rlu", "luminescence", "ros",
                                  "signal") if c in low), None)
    if gcol is None or vcol is None:
        raise ValueError(f"ROS-Glo table needs group + value columns; got {list(df.columns)}")
    out = df[[gcol, vcol]].copy()
    out.columns = ["group", "value"]
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna()


def _panel_rosglo(ax, table=None) -> None:
    arms = ["IVH_Early", "IVH_Late"]
    if table is None:
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linestyle((0, (4, 4))); sp.set_color("0.6")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "ROS-Glo assay\nIVH_Early vs IVH_Late\n\n(placeholder —\n"
                "supply --rosglo-table)", ha="center", va="center",
                fontsize=7, color="0.4", transform=ax.transAxes)
    else:
        qp = _load_rosglo(table)
        for i, arm in enumerate(arms):
            v = qp.loc[qp.group == arm, "value"].to_numpy(float)
            if len(v) == 0:
                continue
            m = np.nanmean(v)
            sem = np.nanstd(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            ax.bar(i, m, yerr=sem, width=0.6, color=group_color(arm),
                   alpha=0.85, capsize=3)
            jit = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.25
            ax.scatter(np.full(len(v), i) + jit, v, s=10, color="0.2",
                       zorder=3, lw=0)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel("ROS (RLU)")
    ax.set_title("B  ROS-Glo (oxidative stress)", loc="left", fontweight="bold")


def build_f5(tidy, rosglo_table=None):
    import matplotlib.pyplot as plt

    resp = S.well_response(tidy, metrics=FOREST_METRICS)
    resp = resp[resp.arm.isin(["Control"] + FOCUS_ARMS)]
    summ = S.arm_response_summary(resp)
    vc = S.arm_response_vs_control(resp)

    fig, (axf, axr) = plt.subplots(
        1, 2, figsize=(8.6, 4.2), gridspec_kw={"width_ratios": [2.3, 0.8]})
    _panel_forest(axf, summ, vc)
    _panel_rosglo(axr, rosglo_table)
    fig.suptitle(
        "Figure 5 — IVH & oxidative-stress network-burst change vs Control "
        "maturation (difference-in-differences, well_uid unit)",
        fontsize=8.5, y=1.02)
    fig.tight_layout()
    return fig, {"response": resp, "summary": summ, "vs_control": vc}


def render(tidy, figure_root, rosglo_table=None):
    fig, data = build_f5(tidy, rosglo_table)
    paths = save_fig(fig, "F5_burst_phenotype", figure_root, subdir="main")
    return paths, data
