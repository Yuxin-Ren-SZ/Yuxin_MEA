"""Artifact-backed panels — rasters, spatial maps, graphs, avalanches, waveforms.

These wrap the per-figure drawing code that already exists in ``fig*.py`` rather
than duplicating it: the figure modules stay the authority on how a raster or an
STTC graph looks, and ``make_report`` keeps producing byte-identical output.
What changes here is *who chooses the well* — the batch figures hard-code a
representative recording, while a composer panel takes it from
:meth:`RenderContext.example` so the user can retarget it from the UI.

Every panel raises :class:`PanelDataMissing` (drawn as an in-cell note, and
recorded in the export manifest) when its artifact is absent, because a missing
``debug_trace.pkl`` for one well must not abort a 20-panel composition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load as L
from .panels import (
    Bool, Choice, Float, Int, PanelDataMissing, PanelSpec, RenderContext, Text,
    register, simple_panel, group_color, ordered_groups,
)

_DIV_WINDOW = (14, 24)


def _example_or_raise(ctx: RenderContext, metric: str, group: str | None,
                      div_lo: int, div_hi: int, what: str):
    grp = group or None
    if grp in ("", "auto"):
        grp = None
    row = ctx.example(metric=metric, group=grp, div=(int(div_lo), int(div_hi)))
    if row is None:
        raise PanelDataMissing(
            f"no {what} well with {metric} in DIV {div_lo}-{div_hi}"
            + (f" for {grp}" if grp else ""))
    return row


def _well_tag(row) -> str:
    return f"{str(row.well_uid).split('|')[0]} {row.well_name} DIV{int(row.DIV)}"


# --------------------------------------------------------------------------- #
# Rasters (F4 row A, S3)
# --------------------------------------------------------------------------- #
@simple_panel
def _panel_raster(ax, ctx, group="auto", div_lo=14, div_hi=24, max_units=40,
                  window_s=60.0, title=""):
    from .fig4_activity_dev import _panel_raster as draw

    row = _example_or_raise(ctx, "median_firing_rate", group, div_lo, div_hi, "raster")
    lab = title or f"{_well_tag(row)}"
    if group not in ("auto", "", None):
        lab = f"{group} — {lab}"
    draw(ax, ctx.analysis_root, row, lab, max_units=int(max_units),
         t_win=float(window_s))
    if group not in ("auto", "", None):
        ax.title.set_color(group_color(group))


register(PanelSpec(
    id="raster.well", title="Spike raster", section="Activity",
    fn=_panel_raster, default_w=6, default_h=3, families=("activity",),
    description="Curated-unit raster for one representative well-recording.",
    params={
        "group": Choice("auto", ["auto", "Control", "IVH_Early", "IVH_Late",
                                 "NPH", "AraC", "H2O2_10uM", "H2O2_20uM"], "Arm"),
        "div_lo": Int(14, "DIV from", min=0, max=200),
        "div_hi": Int(24, "DIV to", min=0, max=200),
        "max_units": Int(40, "Units shown", min=5, max=400),
        "window_s": Float(60.0, "Window (s)", min=1.0, max=600.0),
        "title": Text("", "Panel title"),
    },
))


# --------------------------------------------------------------------------- #
# Developmental trajectory (F4 row B/C)
# --------------------------------------------------------------------------- #
@simple_panel
def _panel_trajectory(ax, ctx, metric="median_firing_rate", title=""):
    from .fig4_activity_dev import _panel_trajectory as draw

    if metric not in ctx.tidy.columns:
        raise PanelDataMissing(f"{metric} not in tidy_long")
    draw(ax, ctx.tidy, metric)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")


register(PanelSpec(
    id="trajectory.metric", title="Developmental trajectory", section="Activity",
    fn=_panel_trajectory, default_w=6, default_h=4, families=("activity", "burst"),
    description="Per-arm median trajectory of one metric against DIV.",
    params={
        "metric": Choice("median_firing_rate",
                         ["median_firing_rate", "nb_rate", "n_curated",
                          "nb_duration_mean", "nb_ibi_mean", "mean_sttc",
                          "edge_density", "branching_ratio_mr", "dcc"], "Metric"),
        "title": Text("", "Panel title"),
    },
))


# --------------------------------------------------------------------------- #
# Spatial activity map + propagation + STTC graph (F7)
# --------------------------------------------------------------------------- #
@simple_panel
def _panel_activity_field(ax, ctx, group="auto", div_lo=14, div_hi=24, title=""):
    from .fig7_spatial_connectivity import _panel_field as draw

    row = _example_or_raise(ctx, "activity_gini", group, div_lo, div_hi, "spatial-map")
    draw(ax, ctx.analysis_root, row, title or _well_tag(row))


@simple_panel
def _panel_propagation(ax, ctx, group="auto", div_lo=14, div_hi=24, title=""):
    from .fig7_spatial_connectivity import _panel_propagation as draw

    row = _example_or_raise(ctx, "mean_prop_speed_um_ms", group, div_lo, div_hi,
                            "propagation")
    draw(ax, ctx.analysis_root, row)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")


@simple_panel
def _panel_sttc_graph(ax, ctx, group="auto", div_lo=14, div_hi=24, title=""):
    from .fig7_spatial_connectivity import _panel_graph as draw

    row = _example_or_raise(ctx, "mean_sttc", group, div_lo, div_hi, "connectivity")
    draw(ax, ctx.analysis_root, row, title or _well_tag(row))


@simple_panel
def _panel_sttc_distance(ax, ctx, max_wells=18, title=""):
    from .fig7_spatial_connectivity import _panel_sttc_distance as draw

    draw(ax, ctx.analysis_root, ctx.tidy, max_wells=int(max_wells), seed=ctx.seed)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")


_SPATIAL_PARAMS = {
    "group": Choice("auto", ["auto", "Control", "IVH_Early", "IVH_Late",
                             "NPH", "AraC", "H2O2_10uM", "H2O2_20uM"], "Arm"),
    "div_lo": Int(14, "DIV from", min=0, max=200),
    "div_hi": Int(24, "DIV to", min=0, max=200),
    "title": Text("", "Panel title"),
}

register(PanelSpec(
    id="spatial.field", title="Activity field", section="Spatial",
    fn=_panel_activity_field, default_w=4, default_h=4, families=("spatial",),
    description="Electrode-plane firing-rate field for one well.",
    params=dict(_SPATIAL_PARAMS)))

register(PanelSpec(
    id="spatial.propagation", title="Burst propagation", section="Spatial",
    fn=_panel_propagation, default_w=4, default_h=4, families=("spatial",),
    description="Per-electrode first-spike latency across network bursts.",
    params=dict(_SPATIAL_PARAMS)))

register(PanelSpec(
    id="connectivity.graph", title="STTC graph", section="Connectivity",
    fn=_panel_sttc_graph, default_w=4, default_h=4, families=("connectivity",),
    description="Significant STTC edges drawn on the electrode plane.",
    params=dict(_SPATIAL_PARAMS)))

register(PanelSpec(
    id="connectivity.sttc_distance", title="STTC vs distance", section="Connectivity",
    fn=_panel_sttc_distance, default_w=6, default_h=4, families=("connectivity",),
    description="Edge STTC against inter-unit distance, pooled over sampled wells.",
    params={"max_wells": Int(18, "Wells pooled", min=2, max=200),
            "title": Text("", "Panel title")}))


# --------------------------------------------------------------------------- #
# Node cartography + directed TE graph (F7b)
# --------------------------------------------------------------------------- #
@simple_panel
def _panel_cartography(ax, ctx, group="auto", div_lo=14, div_hi=24, title=""):
    from .fig7b_node_directed import _panel_cartography as draw

    row = _example_or_raise(ctx, "mean_sttc", group, div_lo, div_hi, "node-metrics")
    draw(ax, ctx.analysis_root, row)
    ax.set_title(title or "Node cartography", loc="left", fontweight="bold")


@simple_panel
def _panel_directed_graph(ax, ctx, group="auto", div_lo=14, div_hi=24,
                          top_frac=0.15, title=""):
    from .fig7b_node_directed import _panel_directed_graph as draw

    row = _example_or_raise(ctx, "mean_sttc", group, div_lo, div_hi, "directed")
    draw(ax, ctx.analysis_root, row, top_frac=float(top_frac))
    ax.set_title(title or "Directed TE graph", loc="left", fontweight="bold")


register(PanelSpec(
    id="node.cartography", title="Node cartography", section="Connectivity",
    fn=_panel_cartography, default_w=6, default_h=4, families=("node",),
    description=("Guimera-Amaral participation P against within-module degree z. "
                 "Note the z>2.5 hub threshold rarely fires at these graph sizes."),
    params=dict(_SPATIAL_PARAMS)))

register(PanelSpec(
    id="directed.graph", title="Directed TE graph", section="Connectivity",
    fn=_panel_directed_graph, default_w=6, default_h=4, families=("directed",),
    description=("Effective transfer-entropy edges on the electrode plane. Binary "
                 "spike-train TE is inflated by co-bursting, so read flow "
                 "hierarchy and degree asymmetry, not edge density."),
    params={**_SPATIAL_PARAMS,
            "top_frac": Float(0.15, "Top edge fraction", min=0.01, max=1.0)}))


# --------------------------------------------------------------------------- #
# Criticality (F8)
# --------------------------------------------------------------------------- #
@simple_panel
def _panel_avalanche_dist(ax, ctx, group="auto", div_lo=14, div_hi=24, title=""):
    from .fig8_criticality import _panel_distributions as draw

    row = _example_or_raise(ctx, "branching_ratio_mr", group, div_lo, div_hi,
                            "criticality")
    draw(ax, ctx.analysis_root, row)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")


@simple_panel
def _panel_crackling(ax, ctx, group="auto", div_lo=14, div_hi=24, title=""):
    from .fig8_criticality import _panel_crackling as draw

    row = _example_or_raise(ctx, "branching_ratio_mr", group, div_lo, div_hi,
                            "criticality")
    draw(ax, ctx.analysis_root, row)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")


@simple_panel
def _panel_branching(ax, ctx, title=""):
    if "branching_ratio_mr" not in ctx.tidy.columns:
        raise PanelDataMissing("branching_ratio_mr not in tidy_long")
    per_well = ctx.tidy.groupby("well_uid").agg(
        arm=("canonical_group", lambda s: next((g for g in s if g != "Control"),
                                               "Control")),
        mr=("branching_ratio_mr", "median")).reset_index()
    arms = [a for a in ordered_groups(per_well.arm.unique()) if a in ctx.groups]
    if not arms:
        raise PanelDataMissing("no wells with a branching ratio")
    rng = np.random.default_rng(ctx.seed)
    for i, arm in enumerate(arms):
        d = per_well.loc[per_well.arm == arm, "mr"].dropna().to_numpy(float)
        if not len(d):
            continue
        if len(d) >= 8:
            bp = ax.boxplot(d, positions=[i], widths=0.6, patch_artist=True,
                            showfliers=False)
            bp["boxes"][0].set(facecolor=group_color(arm), alpha=0.35, lw=0.8)
            bp["medians"][0].set(color=group_color(arm), lw=1.2)
        jit = (rng.random(len(d)) - 0.5) * 0.3
        ax.scatter(np.full(len(d), i) + jit, d, s=6, color=group_color(arm),
                   alpha=0.7, lw=0)
    ax.axhline(1.0, ls="--", color="0.5", lw=0.8)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("branching ratio (MR)", fontsize=7)
    ax.text(0.02, 0.02, "dashed = critical m=1", transform=ax.transAxes,
            fontsize=5.5, color="0.45", va="bottom")
    ax.set_title(title or "Branching ratio by group", loc="left", fontweight="bold")


register(PanelSpec(
    id="criticality.distributions", title="Avalanche distributions",
    section="Criticality", fn=_panel_avalanche_dist, default_w=6, default_h=4,
    families=("criticality",),
    description="Avalanche size and duration PDFs (log-log) with MLE exponents.",
    params=dict(_SPATIAL_PARAMS)))

register(PanelSpec(
    id="criticality.crackling", title="Crackling-noise scaling",
    section="Criticality", fn=_panel_crackling, default_w=6, default_h=4,
    families=("criticality",),
    description="Mean avalanche size against duration; gamma_fit vs the "
                "criticality prediction gives the DCC.",
    params=dict(_SPATIAL_PARAMS)))

register(PanelSpec(
    id="criticality.branching", title="Branching ratio by group",
    section="Criticality", fn=_panel_branching, default_w=6, default_h=4,
    families=("criticality",),
    description="Per-well MR branching ratio by arm; dashed line at critical m=1.",
    params={"title": Text("", "Panel title")}))


# --------------------------------------------------------------------------- #
# Unit QC (S2)
# --------------------------------------------------------------------------- #
@simple_panel
def _panel_unit_qc(ax, ctx, metric="presence_ratio", log=False, n_wells=40, title=""):
    from .fig_s2_units import _hist_qc as draw

    units = ctx.units(int(n_wells))
    if units.empty or metric not in units.columns:
        raise PanelDataMissing(f"no unit {metric} available")
    draw(ax, units, metric, log=bool(log))
    if title:
        ax.set_title(title, loc="left", fontweight="bold")


@simple_panel
def _panel_waveforms(ax, ctx, title=""):
    from .fig_s2_units import _panel_waveforms as draw

    best = ctx.tidy.sort_values("n_curated", ascending=False).iloc[0]
    draw(ax, ctx.analysis_root, best)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")


@simple_panel
def _panel_celltype(ax, ctx, n_wells=40, title=""):
    from .fig_s2_units import _panel_celltype as draw

    units = ctx.units(int(n_wells))
    if units.empty:
        raise PanelDataMissing("no curated units loaded")
    draw(ax, units)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")


register(PanelSpec(
    id="units.qc_hist", title="Unit QC histogram", section="Units",
    fn=_panel_unit_qc, default_w=4, default_h=3, families=(),
    description="Pooled single-unit quality-metric distribution.",
    params={
        "metric": Choice("presence_ratio",
                         ["presence_ratio", "rp_contamination", "amplitude_median",
                          "firing_rate", "snr", "isi_violations_ratio"], "QC metric"),
        "log": Bool(False, "Log x-axis"),
        "n_wells": Int(40, "Wells pooled", min=1, max=400),
        "title": Text("", "Panel title"),
    }))

register(PanelSpec(
    id="units.waveforms", title="Unit waveforms", section="Units",
    fn=_panel_waveforms, default_w=4, default_h=3, families=(),
    description="Average waveform templates for the best-yield well.",
    params={"title": Text("", "Panel title")}))

register(PanelSpec(
    id="units.celltype", title="Cell-type scatter", section="Units",
    fn=_panel_celltype, default_w=4, default_h=3, families=(),
    description="Waveform-shape scatter used for putative cell-type separation.",
    params={"n_wells": Int(40, "Wells pooled", min=1, max=400),
            "title": Text("", "Panel title")}))


# --------------------------------------------------------------------------- #
# ML burst characterisation (F6)
# --------------------------------------------------------------------------- #
def _panel_ml_posterior(fig, subspec, ctx, group="auto", div_lo=14, div_hi=24,
                        t_start=0.0, t_end=60.0, title=""):
    """Composite: HMM posterior heatmap + population trace + colourbar."""
    from .fig6_ml import _panel_posterior as draw

    row = _example_or_raise(ctx, "burst_modulation_index", group, div_lo, div_hi,
                            "ML-trace")
    try:
        tr = ctx.ml_trace(row.well_uid)
    except FileNotFoundError as exc:
        raise PanelDataMissing(f"no debug_trace.pkl for {row.well_uid}") from exc

    inner = subspec.subgridspec(2, 2, height_ratios=[1.0, 0.30],
                                width_ratios=[1, 0.02], hspace=0.08, wspace=0.02)
    ax_post = fig.add_subplot(inner[0, 0])
    cax = fig.add_subplot(inner[0, 1])
    ax_trace = fig.add_subplot(inner[1, 0], sharex=ax_post)
    fig.add_subplot(inner[1, 1]).set_visible(False)
    draw(ax_post, ax_trace, cax, tr, t_win=(float(t_start), float(t_end)))
    ax_post.set_title(title or f"HMM posterior  ·  {_well_tag(row)}",
                      loc="left", fontweight="bold")


@simple_panel
def _panel_ml_umap(ax, ctx, group="auto", div_lo=14, div_hi=24, title=""):
    from .fig6_ml import _panel_umap as draw

    row = _example_or_raise(ctx, "burst_modulation_index", group, div_lo, div_hi,
                            "ML-trace")
    try:
        tr = ctx.ml_trace(row.well_uid)
    except FileNotFoundError as exc:
        raise PanelDataMissing(f"no debug_trace.pkl for {row.well_uid}") from exc
    draw(ax, tr, seed=ctx.seed)
    ax.set_box_aspect(1)
    ax.set_title(title or "Burst-feature UMAP", loc="left", fontweight="bold")


@simple_panel
def _panel_ml_modulation(ax, ctx, title=""):
    from .fig6_ml import _panel_modulation as draw

    if "burst_modulation_index" not in ctx.tidy.columns:
        raise PanelDataMissing("burst_modulation_index not in tidy_long")
    draw(ax, ctx.tidy)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")


register(PanelSpec(
    id="ml.posterior", title="HMM posterior + trace", section="ML bursts",
    fn=_panel_ml_posterior, default_w=12, default_h=4, families=("ml",),
    description="Composite panel: per-state posterior heatmap over a time window, "
                "the population rate trace beneath it, and a shared colourbar.",
    params={**_SPATIAL_PARAMS,
            "t_start": Float(0.0, "Window start (s)", min=0.0),
            "t_end": Float(60.0, "Window end (s)", min=1.0)}))

register(PanelSpec(
    id="ml.umap", title="Burst-feature UMAP", section="ML bursts",
    fn=_panel_ml_umap, default_w=4, default_h=4, families=("ml",),
    description="Per-bin UMAP of the ML detector's feature space, burst vs rest.",
    params=dict(_SPATIAL_PARAMS)))

register(PanelSpec(
    id="ml.modulation", title="Burst modulation index", section="ML bursts",
    fn=_panel_ml_modulation, default_w=4, default_h=4, families=("ml",),
    description="Cohort distribution of the burst modulation index by arm.",
    params={"title": Text("", "Panel title")}))


# --------------------------------------------------------------------------- #
# Method comparison (S1)
# --------------------------------------------------------------------------- #
@simple_panel
def _panel_method_compare(ax, ctx, view="counts", title=""):
    from .report_style import OKABE_ITO

    try:
        df = L.load_method_compare(ctx.figure_root)
    except FileNotFoundError as exc:
        raise PanelDataMissing("burst_method_comparison/per_well.csv not found") from exc

    if view == "counts":
        x = df.trad_count.to_numpy(float); y = df.ml_count.to_numpy(float)
        ax.scatter(x + 1, y + 1, s=5, alpha=0.3, lw=0, color=OKABE_ITO["blue"])
        lim = [1, float(np.nanmax([x, y])) + 5]
        ax.plot(lim, lim, "--", color="0.5", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("traditional burst count (+1)")
        ax.set_ylabel("ML burst count (+1)")
        default_title = "Burst-count agreement"
    elif view == "bland_altman":
        td = df.trad_duration_s.to_numpy(float); md = df.ml_duration_s.to_numpy(float)
        mean, diff = (td + md) / 2, md - td
        ok = ~np.isnan(mean) & ~np.isnan(diff)
        ax.scatter(mean[ok], diff[ok], s=5, alpha=0.3, lw=0,
                   color=OKABE_ITO["vermillion"])
        mu, sd = float(np.nanmean(diff[ok])), float(np.nanstd(diff[ok]))
        for yv, ls in ((mu, "-"), (mu + 1.96 * sd, "--"), (mu - 1.96 * sd, "--")):
            ax.axhline(yv, ls=ls, color="0.4", lw=0.8)
        ax.set_xlabel("mean burst duration (s)")
        ax.set_ylabel("ML − traditional (s)")
        default_title = "Duration Bland-Altman"
    else:
        iou = df.iou_mean.dropna().to_numpy(float)
        ax.hist(iou, bins=25, color=OKABE_ITO["green"], alpha=0.8)
        ax.axvline(float(np.median(iou)), ls="--", color="0.3", lw=0.9,
                   label=f"median {np.median(iou):.2f}")
        ax.set_xlabel("per-well mean event IoU"); ax.set_ylabel("wells")
        ax.legend(fontsize=6)
        default_title = "Event overlap"
    ax.set_title(title or default_title, loc="left", fontweight="bold")


register(PanelSpec(
    id="method.compare", title="ML vs traditional detection", section="Methods",
    fn=_panel_method_compare, default_w=4, default_h=3, families=(),
    description="Agreement between the ML and threshold burst detectors.",
    params={"view": Choice("counts", ["counts", "bland_altman", "iou"], "View"),
            "title": Text("", "Panel title")}))
