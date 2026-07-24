"""F5 — Network-burst phenotype: IVH & oxidative stress vs Control maturation.

Focus arms: **IVH_Early, IVH_Late** (intraventricular-haemorrhage CSF) and
**H2O2_20uM** (oxidative stress); Control is the developmental reference. NPH,
AraC and H2O2_10uM are excluded here (see S4 for the full cross-group view).

The pipeline records each treated well as ``Control`` before treatment
(``tau < 0``) and its treatment name after (``tau > 0``); the naive within-well
``log2(post/pre)`` is confounded with development. Control wells (never treated)
give the maturation-only change, so treatment is read as a **difference-in-
differences**: each arm's response vs Control's response (Mann-Whitney + BH-FDR).

* **A. DiD forest, story-ordered** — per metric, each arm's within-well pre→post
  response vs Control's maturation. Control (diamond) is the developmental
  baseline; the headline metric (median firing rate) is on top in a highlight
  band with its DiD effect size Δ. ★ = FDR q<0.05 (none at n=3); △ = suggestive.
* **B. Headline-metric trajectory** — median firing rate vs treatment day (tau),
  per arm (median + 95% bootstrap CI), replacing the three cramped trajectories.
* **C. ROS-Glo assay** — added only when `--rosglo-table` is supplied; the empty
  dashed placeholder is gone (never ship empty boxes).

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
# Story-ordered: the headline metric (median firing rate — the one suggestive
# effect) sits on top, highlighted; the four null metrics follow.
HEADLINE_METRIC = "median_firing_rate"
FOREST_METRICS = [
    HEADLINE_METRIC,
    "nb_rate", "nb_duration_mean", "nb_spikes_per_burst_mean", "nb_ibi_mean",
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
    ax.set_title("C  ROS-Glo (oxidative stress)", loc="left", fontweight="bold")


_ALL_ARMS = ["Control"] + FOCUS_ARMS
# Grid positions for the 5 metric panels (headline first, then row-major).
_METRIC_CELLS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]


def build_f5(tidy, rosglo_table=None):
    import matplotlib.pyplot as plt

    from .forest import group_columns_grid
    from .report_style import preliminary_tag
    from .trajectory import plot_trajectory

    resp = S.well_response(tidy, metrics=FOREST_METRICS)
    resp = resp[resp.arm.isin(_ALL_ARMS)]

    # ROS-Glo panel is added ONLY when the assay table exists — no empty dashed
    # placeholder ever ships (Figure Redesign Handoff).
    have_ros = rosglo_table is not None
    if have_ros:
        fig = plt.figure(figsize=(9.8, 8.2))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.85], top=0.88,
                              bottom=0.08, hspace=0.62, wspace=0.36)
    else:
        fig = plt.figure(figsize=(9.8, 6.0))
        gs = fig.add_gridspec(2, 3, top=0.86, bottom=0.10,
                              hspace=0.58, wspace=0.36)

    # A: grouped-column DiD small multiples — one panel per metric, groups on x.
    metric_axes = [fig.add_subplot(gs[r, c]) for r, c in _METRIC_CELLS]
    axt = fig.add_subplot(gs[1, 2])                     # headline trajectory (B)
    fdata = group_columns_grid(
        metric_axes, resp, FOREST_METRICS, FOCUS_ARMS,
        highlight_metric=HEADLINE_METRIC, effect_labels=True,
        ylabel_axes=[metric_axes[0], metric_axes[3]])   # left column only

    # B: one enlarged trajectory of the headline metric (replaces 3 tiny panels).
    plot_trajectory(axt, tidy, HEADLINE_METRIC, _ALL_ARMS, legend=True,
                    title="B  Headline metric trajectory")
    preliminary_tag(axt)

    if have_ros:
        _panel_rosglo(fig.add_subplot(gs[2, 0]), rosglo_table)

    fig.suptitle(
        "Figure 5 — IVH & oxidative-stress network-burst change vs Control "
        "maturation (chip-level DiD; chip = biological replicate)",
        fontsize=8.5, y=0.985)
    fig.text(0.02, 0.93,
             "A  Difference-in-differences vs Control — one panel per metric",
             fontweight="bold", fontsize=9, ha="left", va="top")
    ros_note = ("" if have_ros else
                " ROS-Glo oxidative-stress assay pending — supply --rosglo-table "
                "to add its panel.")
    caption(fig,
        "Network-burst phenotype, chip-level difference-in-differences vs Control "
        "maturation. (A) One panel per metric with the groups side-by-side "
        "(Control = maturation baseline, IVH_Early / IVH_Late read against it): "
        "bold = the 3 per-chip means (biological replicates), faint = wells, "
        "bar = mean of chip means, whisker = 95% t-CI (df=2), dashed line = no "
        "change; the headline metric (median firing rate) panel is highlighted "
        "and carries the per-arm DiD effect size Δ. ★ = FDR q<0.05 (none survive "
        "at n=3); △ = suggestive (same direction 3/3 chips, uncorrected p<0.05). "
        "(B) Headline-metric trajectory vs treatment day (tau): line = per-arm "
        "median across wells, ribbon = 95% bootstrap CI (well-level), dotted "
        "vertical = treatment day." + ros_note +
        " Unit of replication: chip (n=3 per arm); IVH_Early / IVH_Late vs Control.")
    return fig, {"response": resp, **(fdata or {})}


def render(tidy, figure_root, rosglo_table=None):
    fig, data = build_f5(tidy, rosglo_table)
    paths = save_fig(fig, "F5_burst_phenotype", figure_root, subdir="main")
    return paths, data
