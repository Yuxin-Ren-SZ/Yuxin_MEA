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
    METRIC_LABELS, caption, group_color, ordered_groups, save_fig,
)

# Arms shown in F5 (Control is always the reference; not listed here).
FOCUS_ARMS = ["IVH_Early", "IVH_Late"]
FOREST_METRICS = [
    "nb_rate", "nb_duration_mean", "nb_spikes_per_burst_mean",
    "nb_ibi_mean", "median_firing_rate",
]


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


# Metrics shown as tau-trajectories (row B).
TRAJ_METRICS = ["nb_rate", "nb_duration_mean", "median_firing_rate"]
_ALL_ARMS = ["Control"] + FOCUS_ARMS


def build_f5(tidy, rosglo_table=None):
    import matplotlib.pyplot as plt

    from .forest import plot_did_forest
    from .trajectory import plot_trajectory

    resp = S.well_response(tidy, metrics=FOREST_METRICS)
    resp = resp[resp.arm.isin(_ALL_ARMS)]

    fig = plt.figure(figsize=(9.2, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.32)
    # Row A: DiD forest (wide) + ROS-Glo
    axf = fig.add_subplot(gs[0, :2]); axr = fig.add_subplot(gs[0, 2])
    fdata = plot_did_forest(
        axf, resp, FOREST_METRICS, FOCUS_ARMS,
        xlabel="within-well response  log2(post / pre)",
        title="A  Response vs developmental maturation (DiD, chip-level)")
    _panel_rosglo(axr, rosglo_table)
    # Row B: tau-trajectories (the time-course the forest's pooled 'post' hides)
    for ci, metric in enumerate(TRAJ_METRICS):
        ax = fig.add_subplot(gs[1, ci])
        plot_trajectory(ax, tidy, metric, _ALL_ARMS, legend=(ci == 0))
        if ci == 0:
            ax.annotate("C  Network-burst trajectories vs treatment day",
                        xy=(0, 1.12), xycoords="axes fraction", fontweight="bold",
                        fontsize=9)
    fig.suptitle(
        "Figure 5 — IVH & oxidative-stress network-burst change vs Control "
        "maturation (DiD + tau-trajectories; chip = biological replicate)",
        fontsize=8.5, y=1.0)
    caption(fig,
        "Network-burst phenotype. (A) Within-well difference-in-differences vs "
        "Control: each treated arm's pre→post change (log2 post/pre) minus "
        "Control's maturation change, per metric. Bold points = the 3 per-chip "
        "means (biological replicates), faint = wells; marker = mean of chip "
        "means, whisker = 95% t-CI over the 3 chips (df=2); Control diamond = "
        "maturation baseline. ★ = FDR q<0.05 (none); △ = suggestive (same "
        "direction on all 3 chips, uncorrected p<0.05, not FDR-significant at "
        "n=3). (B) ROS-Glo oxidative-stress assay placeholder (supply "
        "--rosglo-table). (C) Group trajectories vs treatment day (tau) for the "
        "raw metric: line = per-arm median across wells, ribbon = 95% bootstrap "
        "CI (well-level), dotted vertical = treatment day (tau 0). Unit of "
        "replication: chip (n=3 per arm); IVH_Early / IVH_Late vs Control.")
    return fig, {"response": resp, **(fdata or {})}


def render(tidy, figure_root, rosglo_table=None):
    fig, data = build_f5(tidy, rosglo_table)
    paths = save_fig(fig, "F5_burst_phenotype", figure_root, subdir="main")
    return paths, data
