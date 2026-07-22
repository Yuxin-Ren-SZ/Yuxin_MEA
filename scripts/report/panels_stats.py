"""Statistical panels — the forest plots and matched-DIV comparisons.

F5, F7, F7b and F8 each carried their own ``_panel_forest``: four copies of the
same difference-in-differences forest that had drifted apart (only two drew the
raw per-well point cloud, only one distinguished exploratory arms, only one
clipped the x-range). This is the single implementation, parameterised by the
metric list, so every figure gets the full treatment and a fix lands once.

Reading a forest
----------------
Each row is one metric. The black diamond is Control — the *maturation-only*
pre→post change over the same interval. Coloured circles are treated arms.
Stars mark the treated-vs-Control difference-in-differences (Mann-Whitney on
per-well responses, BH-FDR), never the arm's own change from zero: a
significant within-arm change that Control also shows is development, not
treatment. Open markers are exploratory arms (single chip or <5 wells).
"""
from __future__ import annotations

import numpy as np

from .panels import (
    Bool, Choice, Int, MultiChoice, PanelDataMissing, PanelSpec, Text, register,
    simple_panel, METRIC_LABELS, group_color, ordered_groups,
)
from . import stats as S

BURST_METRICS = ["nb_rate", "nb_duration_mean", "nb_spikes_per_burst_mean",
                 "nb_ibi_mean", "median_firing_rate", "n_curated"]
CONNECTIVITY_METRICS = ["mean_sttc", "edge_density", "modularity",
                        "small_worldness", "mean_prop_speed_um_ms", "activity_gini"]
NODE_METRICS = ["hub_fraction", "leaf_fraction", "participation_mean",
                "mean_betweenness", "rich_club", "assortativity"]
DIRECTED_METRICS = ["mean_te", "degree_asymmetry", "reciprocity", "flow_hierarchy"]
CRITICALITY_METRICS = ["branching_ratio_mr", "dcc", "aval_tau", "aval_alpha"]

ALL_FOREST_METRICS = (BURST_METRICS + CONNECTIVITY_METRICS + NODE_METRICS
                      + DIRECTED_METRICS + CRITICALITY_METRICS)


def _stars(q) -> str:
    if q is None or (isinstance(q, float) and np.isnan(q)):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def _response_axis_label(metrics) -> str:
    ratio = [m for m in metrics if m in S.RATIO_METRICS]
    diff = [m for m in metrics if m in S.DIFF_METRICS]
    if ratio and diff:
        return "within-well response  (log2 post/pre; Δ for bounded metrics)"
    if ratio:
        return "within-well response  log2(post / pre)"
    return "within-well response  (post − pre)"


def draw_forest(ax, ctx, metrics, show_points=True, clip_to_ci=True,
                show_legend=True, legend_loc="lower right", title="") -> None:
    metrics = [m for m in metrics if m in ctx.tidy.columns]
    if not metrics:
        raise PanelDataMissing("none of the selected metrics are in tidy_long")

    resp = ctx.response(metrics)
    if resp.empty:
        raise PanelDataMissing("no paired pre/post wells for the selected groups")
    summ = ctx.summary(metrics)
    vc = ctx.vs_control(metrics)
    if summ.empty:
        raise PanelDataMissing("no arm summaries (too few paired wells)")

    rng = np.random.default_rng(ctx.seed)
    treated = [a for a in ordered_groups(summ.arm.unique()) if a != "Control"]
    arms = ["Control"] + treated
    arm_off = np.linspace(0.32, -0.32, max(len(arms), 2))
    vc_idx = vc.set_index(["arm", "metric"]) if len(vc) else None

    yticks, ylabels = [], []
    for mi, metric in enumerate(metrics):
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
            exploratory = str(getattr(r, "tier", "confirmatory")) == "exploratory"

            if show_points:
                pts = resp.loc[(resp.metric == metric) & (resp.arm == arm),
                               "response"].to_numpy(float)
                if len(pts):
                    jit = (rng.random(len(pts)) - 0.5) * 0.16
                    ax.scatter(pts, np.full(len(pts), y) + jit, s=4,
                               color=group_color(arm), alpha=0.28, lw=0, zorder=2)

            xerr = np.array([[max(r.median_response - r.ci_low, 0)],
                             [max(r.ci_high - r.median_response, 0)]])
            ax.errorbar(
                r.median_response, y, xerr=xerr,
                fmt=("D" if is_ctrl else "o"), ms=(4.5 if is_ctrl else 4),
                color=group_color(arm), capsize=2, elinewidth=0.9,
                markerfacecolor=("none" if exploratory else group_color(arm)),
                markeredgecolor=group_color(arm), zorder=(4 if is_ctrl else 3),
            )
            if not is_ctrl and vc_idx is not None and (arm, metric) in vc_idx.index:
                s = _stars(vc_idx.loc[(arm, metric), "q_bh"])
                if s:
                    ax.text(r.ci_high + 0.05, y, s, va="center", ha="left",
                            fontsize=7, color=group_color(arm))

    ax.axvline(0, ls="--", color="0.5", lw=0.8, zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_ylim(min(yticks) - 0.6, max(yticks) + 0.6)
    if clip_to_ci and len(summ):
        lo, hi = float(summ.ci_low.min()), float(summ.ci_high.max())
        pad = 0.25 * (hi - lo) + 0.15
        ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(_response_axis_label(metrics), fontsize=7)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")

    if show_legend:
        handles = [ax.plot([], [], marker="D", ls="", color="k",
                           label="Control (maturation)")[0]]
        handles += [ax.plot([], [], marker="o", ls="", color=group_color(a), label=a)[0]
                    for a in treated]
        if (summ.get("tier") == "exploratory").any():
            handles.append(ax.plot([], [], marker="o", ls="", mfc="none",
                                   mec="0.4", color="0.4",
                                   label="open = exploratory (1 chip / <5 wells)")[0])
        # An opaque frame, because a pinned cell has no spare margin: the legend
        # sits over the plot rather than pushing into the neighbouring panel.
        ax.legend(handles=handles, loc=legend_loc, fontsize=5.5, ncol=1,
                  frameon=True, framealpha=0.85, edgecolor="0.85",
                  borderpad=0.3, handletextpad=0.4)


def _forest_factory(metrics, section, families, pid, title, desc):
    @simple_panel
    def _panel(ax, ctx, metrics=tuple(metrics), show_points=True,
               clip_to_ci=True, show_legend=True, legend_loc="lower right",
               title=""):
        draw_forest(ax, ctx, list(metrics), show_points=show_points,
                    clip_to_ci=clip_to_ci, show_legend=show_legend,
                    legend_loc=legend_loc, title=title)

    return register(PanelSpec(
        id=pid, title=title, section=section, fn=_panel,
        default_w=6, default_h=4, families=tuple(families), description=desc,
        params={
            "metrics": MultiChoice(metrics, ALL_FOREST_METRICS, "Metrics (rows)"),
            "show_points": Bool(True, "Raw per-well points"),
            "clip_to_ci": Bool(True, "Clip x-range to bootstrap CIs"),
            "show_legend": Bool(True, "Legend"),
            "legend_loc": Choice("lower right",
                                 ["lower right", "upper right", "lower left",
                                  "upper left", "center right", "best"],
                                 "Legend position"),
            "title": Text("", "Panel title"),
        },
    ))


_DID_DESC = ("Difference-in-differences vs Control maturation. Stars are the "
             "treated-vs-Control comparison (Mann-Whitney + BH-FDR), not the "
             "arm's own pre/post change.")

_forest_factory(BURST_METRICS, "Bursting", ["burst", "activity"],
                "forest.burst", "Burst response vs Control", _DID_DESC)
_forest_factory(CONNECTIVITY_METRICS, "Connectivity", ["connectivity", "spatial"],
                "forest.connectivity", "Connectivity response vs Control", _DID_DESC)
_forest_factory(NODE_METRICS, "Connectivity", ["node"],
                "forest.node", "Node-metric response vs Control", _DID_DESC)
_forest_factory(DIRECTED_METRICS, "Connectivity", ["directed"],
                "forest.directed", "Directed-metric response vs Control", _DID_DESC)
_forest_factory(CRITICALITY_METRICS, "Criticality", ["criticality"],
                "forest.criticality", "Criticality response vs Control", _DID_DESC)


# --------------------------------------------------------------------------- #
# Matched-DIV cross-group comparison (S4)
# --------------------------------------------------------------------------- #
@simple_panel
def _panel_matched_div(ax, ctx, metric="nb_rate", div_lo=14, div_hi=24,
                       show_stats=True, title=""):
    if metric not in ctx.tidy.columns:
        raise PanelDataMissing(f"{metric} not in tidy_long")
    matched = ctx.matched(int(div_lo), int(div_hi), metrics=[metric])
    matched = matched[matched.arm.isin(ctx.groups)]
    if matched.empty or matched[metric].notna().sum() == 0:
        raise PanelDataMissing(f"no wells with {metric} in DIV {div_lo}-{div_hi}")

    arms = [a for a in ordered_groups(matched.arm.unique()) if a in ctx.groups]
    stats = S.mwu_vs_control(matched, metric) if show_stats else None
    qlook = ({r.arm: r.q_bh for r in stats.itertuples()}
             if stats is not None and not stats.empty else {})
    rng = np.random.default_rng(ctx.seed)

    for i, arm in enumerate(arms):
        v = matched.loc[matched.arm == arm, metric].dropna().to_numpy(float)
        if not len(v):
            continue
        if len(v) >= 5:
            bp = ax.boxplot(v, positions=[i], widths=0.6, patch_artist=True,
                            showfliers=False)
            bp["boxes"][0].set(facecolor=group_color(arm), alpha=0.35, lw=0.8)
            bp["medians"][0].set(color=group_color(arm), lw=1.2)
        jit = (rng.random(len(v)) - 0.5) * 0.3
        ax.scatter(np.full(len(v), i) + jit, v, s=6, color=group_color(arm),
                   alpha=0.7, lw=0, zorder=3)
        s = _stars(qlook.get(arm, np.nan))
        if s:
            ax.text(i, v.max(), s, ha="center", va="bottom", fontsize=7,
                    color=group_color(arm))

    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=7)
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)}  ·  DIV {div_lo}-{div_hi}",
                 loc="left", fontweight="bold")
    if show_stats:
        ax.set_xlabel("well = unit; MWU vs Control, BH-FDR", fontsize=5.5)


register(PanelSpec(
    id="matched.div", title="Cross-group at matched DIV", section="Comparison",
    fn=_panel_matched_div, default_w=4, default_h=4,
    families=(), description=(
        "One metric across arms inside a fixed DIV window, so groups are compared "
        "at matched developmental stage rather than matched calendar time."),
    params={
        "metric": Choice("nb_rate", ALL_FOREST_METRICS, "Metric"),
        "div_lo": Int(14, "DIV from", min=0, max=200),
        "div_hi": Int(24, "DIV to", min=0, max=200),
        "show_stats": Bool(True, "MWU vs Control + stars"),
        "title": Text("", "Panel title"),
    },
))
