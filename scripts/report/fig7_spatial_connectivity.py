"""F7 — Spatial activity maps & functional connectivity.

Four panels, well_uid unit (chip = replicate), same style as F5/F6:

* **A** Activity field — example well's 2-D firing-rate map on the electrode
  plane, Control vs treated at matched DIV.
* **B** Burst propagation — units coloured by first-spike latency for one network
  burst; arrow = fitted planar propagation gradient.
* **C** Functional-connectivity graph — significant STTC edges drawn on the
  electrode plane (edge alpha ∝ STTC), one Control + one treated example.
* **D** Group comparison — within-well response (log2 post/pre for ratio metrics,
  Δ for STTC/modularity) forest of ``mean_sttc, edge_density, modularity,
  small_worldness``, reusing F5's response machinery.

Note on tiers: unlike F5 (which hard-excludes the single-chip arms), F7 panel D
shows every arm, drawing **open markers for exploratory arms** (``tier ==
"exploratory"`` from :func:`stats.arm_response_summary` — i.e. single-chip arms
like AraC/NPH or <5 wells) and filled markers for confirmatory arms. This
open-marker convention is introduced here (F5 has no such rendering).

Panels A–C are example illustrations built from per-well artifacts; each renders
a labelled placeholder if its outputs are not yet on disk (e.g. before the
production pass), so ``make_report --figures f7`` degrades gracefully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load
from . import stats as S
from .report_style import (
    METRIC_LABELS, group_color, ordered_groups, save_fig,
)

FOREST_METRICS = ["mean_sttc", "edge_density", "modularity", "small_worldness"]


# --------------------------------------------------------------------------- #
# Per-well artifact loaders (paths reconstructed from a tidy row)
# --------------------------------------------------------------------------- #
def _activity_field(analysis_root, row):
    import json
    fld = load.artifact_path(analysis_root, "spatial_map_data", row,
                             "spatial_map", "activity_field.npy")
    diag = load.artifact_path(analysis_root, "spatial_map_data", row,
                              "spatial_map", "diagnostics.json")
    field = np.load(fld)
    extent = None
    if diag.exists():
        d = json.loads(diag.read_text())
        extent = d.get("field_extent")
    return field, extent


def _positions(analysis_root, row):
    """{unit_id: (x, y)} from the curation quality_metrics.pkl."""
    qm = pd.read_pickle(load.quality_metrics_path(analysis_root, row))
    if "loc_x" not in qm.columns or "loc_y" not in qm.columns:
        raise KeyError("no loc_x/loc_y")
    return {uid: (float(r["loc_x"]), float(r["loc_y"])) for uid, r in qm.iterrows()}


def _edges(analysis_root, row):
    p = load.artifact_path(analysis_root, "connectivity_data", row,
                           "connectivity", "edges.parquet")
    return pd.read_parquet(p)


def _network_bursts(analysis_root, row):
    p = load.artifact_path(analysis_root, "burst_detection_data", row,
                           "burst_detection", "network_bursts.pkl")
    return pd.read_pickle(p) if p.exists() else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Example-well selection
# --------------------------------------------------------------------------- #
def _pick_examples(tidy, value_col):
    """A (control_row, treated_row) pair with non-null ``value_col``, DIV-matched.

    Non-null ``value_col`` means the well's spatial/connectivity output exists and
    rolled up, so the artifact is on disk. Prefers post-treatment treated wells.
    """
    if value_col not in tidy.columns:
        return None, None
    sub = tidy[tidy[value_col].notna()].copy()
    if sub.empty:
        return None, None
    ctrl = sub[sub.get("canonical_group") == "Control"]
    treat = sub[(sub.get("canonical_group") != "Control")
                & sub.get("canonical_group").notna()]
    if "tau" in treat.columns:
        post = treat[treat.tau > 0]
        treat = post if not post.empty else treat
    if ctrl.empty or treat.empty:
        # Fall back to any two distinct wells.
        rows = [sub.iloc[i] for i in range(min(2, len(sub)))]
        return (rows[0] if rows else None,
                rows[1] if len(rows) > 1 else (rows[0] if rows else None))
    t_row = treat.sort_values("DIV").iloc[-1] if "DIV" in treat.columns else treat.iloc[0]
    if "DIV" in ctrl.columns and "DIV" in t_row:
        c_row = ctrl.iloc[(ctrl.DIV - t_row.DIV).abs().argsort().iloc[0]]
    else:
        c_row = ctrl.iloc[0]
    return c_row, t_row


def _placeholder(ax, msg):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linestyle((0, (4, 4))); sp.set_color("0.6")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=7,
            color="0.4", transform=ax.transAxes, wrap=True)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _panel_field(ax, analysis_root, row, title):
    if row is None:
        _placeholder(ax, "no well with\nactivity field"); ax.set_title(title, fontsize=7)
        return
    try:
        field, extent = _activity_field(analysis_root, row)
        ax.imshow(field, origin="lower", cmap="magma", aspect="auto",
                  extent=extent if extent else None)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=7)
    except Exception as exc:  # noqa: BLE001
        _placeholder(ax, f"{title}\n(field n/a)")


def _panel_propagation(ax, analysis_root, row):
    ax.set_title("B  Burst propagation", loc="left", fontweight="bold")
    if row is None:
        _placeholder(ax, "no well with\npropagation data"); return
    try:
        from yuxin_mea.analysis.spatial_map import _first_latencies, _plane_fit
        pos = _positions(analysis_root, row)
        spikes = {int(u): np.asarray(s, float)
                  for u, s in load.load_curated_spikes(analysis_root, row).items()}
        bursts = _network_bursts(analysis_root, row)
        if bursts.empty:
            _placeholder(ax, "example well\nhas no bursts"); return
        # Choose the burst with the most positioned participants.
        best = None
        for _, b in bursts.iterrows():
            lat, xy = _first_latencies(spikes, list(spikes), pos,
                                       float(b["start"]), float(b["end"]))
            if best is None or lat.size > best[0].size:
                best = (lat, xy)
        lat, xy = best
        if lat.size < 3:
            _placeholder(ax, "too few units\nin burst"); return
        speed, direction, r2, ox, oy = _plane_fit(lat, xy)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=lat * 1e3, cmap="viridis",
                        s=22, edgecolors="0.3", linewidths=0.3, zorder=3)
        span = float(np.hypot(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))) or 1.0
        ax.quiver(ox, oy, np.cos(direction), np.sin(direction),
                  angles="xy", scale_units="xy",
                  scale=1.0 / (0.3 * span), color="k", width=0.012, zorder=4)
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("first-spike latency (ms)", fontsize=6)
        cb.ax.tick_params(labelsize=6)
        ax.set_xlabel("x (µm)", fontsize=6); ax.set_ylabel("y (µm)", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.text(0.02, 0.98, f"{speed:.1f} µm/ms\n$r^2$={r2:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=6,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
    except Exception:  # noqa: BLE001
        ax.clear()
        _placeholder(ax, "propagation\nn/a")
        ax.set_title("B  Burst propagation", loc="left", fontweight="bold")


def _panel_graph(ax, analysis_root, row, title):
    if row is None:
        _placeholder(ax, "no well with\nconnectivity"); ax.set_title(title, fontsize=7)
        return
    try:
        pos = _positions(analysis_root, row)
        edges = _edges(analysis_root, row)
        xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
        ax.scatter(xs, ys, s=8, color="0.4", zorder=3)
        if not edges.empty:
            smax = float(edges["sttc"].max()) or 1.0
            for _, e in edges.iterrows():
                if e["u"] in pos and e["v"] in pos:
                    x0, y0 = pos[e["u"]]; x1, y1 = pos[e["v"]]
                    a = max(0.05, min(1.0, float(e["sttc"]) / smax))
                    ax.plot([x0, x1], [y0, y1], color="#3b6fb0", alpha=a,
                            lw=0.6, zorder=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=7)
    except Exception:  # noqa: BLE001
        _placeholder(ax, f"{title}\n(graph n/a)")


def _stars(q):
    if q is None or np.isnan(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def _panel_forest(ax, summ, vc):
    treated = [a for a in ordered_groups(summ.arm.unique()) if a != "Control"]
    arms = ["Control"] + treated
    arm_off = np.linspace(0.32, -0.32, len(arms))
    vc_idx = vc.set_index(["arm", "metric"]) if len(vc) else None
    tier_idx = summ.set_index(["arm", "metric"])["tier"] if "tier" in summ else None
    yticks, ylabels = [], []
    for mi, metric in enumerate(FOREST_METRICS):
        y0 = -mi
        yticks.append(y0); ylabels.append(METRIC_LABELS.get(metric, metric))
        for ai, arm in enumerate(arms):
            r = summ[(summ.metric == metric) & (summ.arm == arm)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = y0 + arm_off[ai]
            is_ctrl = arm == "Control"
            exploratory = str(getattr(r, "tier", "confirmatory")) == "exploratory"
            xerr = np.array([[max(r.median_response - r.ci_low, 0)],
                             [max(r.ci_high - r.median_response, 0)]])
            face = "none" if exploratory else group_color(arm)
            ax.errorbar(
                r.median_response, y, xerr=xerr,
                fmt=("D" if is_ctrl else "o"), ms=(4.5 if is_ctrl else 4),
                color=group_color(arm), capsize=2, elinewidth=0.9,
                markerfacecolor=face, markeredgecolor=group_color(arm),
                zorder=(4 if is_ctrl else 3),
            )
            if not is_ctrl and vc_idx is not None and (arm, metric) in vc_idx.index:
                s = _stars(vc_idx.loc[(arm, metric), "q_bh"])
                if s:
                    ax.text(r.ci_high + 0.05, y, s, va="center", ha="left",
                            fontsize=7, color=group_color(arm))
    ax.axvline(0, ls="--", color="0.5", lw=0.8, zorder=1)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.set_ylim(min(yticks) - 0.6, max(yticks) + 0.6)
    ax.set_xlabel("within-well response  (log2 post/pre; Δ for STTC/modularity)")
    ax.set_title("D  Connectivity response vs Control maturation", loc="left",
                 fontweight="bold")
    handles = [ax.plot([], [], marker="D", ls="", color="k",
                       label="Control (maturation)")[0]]
    handles += [ax.plot([], [], marker="o", ls="", color=group_color(a), label=a)[0]
                for a in treated]
    handles.append(ax.plot([], [], marker="o", ls="", mfc="none", mec="0.4",
                           color="0.4", label="open = exploratory (single-chip/<5 wells)")[0])
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=5.5, ncol=2, frameon=False)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_f7(tidy, analysis_root):
    import matplotlib.pyplot as plt

    resp = S.well_response(tidy, metrics=FOREST_METRICS)
    summ = S.arm_response_summary(resp) if not resp.empty else pd.DataFrame()
    vc = S.arm_response_vs_control(resp) if not resp.empty else pd.DataFrame()

    c_field, t_field = _pick_examples(tidy, "activity_gini")
    c_graph, t_graph = _pick_examples(tidy, "mean_sttc")

    fig = plt.figure(figsize=(9.0, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # Panel A — two fields (Control | treated)
    gsa = gs[0, 0].subgridspec(1, 2, wspace=0.12)
    axa0 = fig.add_subplot(gsa[0, 0]); axa1 = fig.add_subplot(gsa[0, 1])
    _panel_field(axa0, analysis_root, c_field, "Control")
    _panel_field(axa1, analysis_root, t_field, "Treated")
    axa0.annotate("A  Activity field", xy=(0, 1.18), xycoords="axes fraction",
                  fontweight="bold", fontsize=9)

    # Panel B — propagation
    axb = fig.add_subplot(gs[0, 1])
    _panel_propagation(axb, analysis_root, t_field if t_field is not None else c_field)

    # Panel C — two graphs (Control | treated)
    gsc = gs[1, 0].subgridspec(1, 2, wspace=0.12)
    axc0 = fig.add_subplot(gsc[0, 0]); axc1 = fig.add_subplot(gsc[0, 1])
    _panel_graph(axc0, analysis_root, c_graph, "Control")
    _panel_graph(axc1, analysis_root, t_graph, "Treated")
    axc0.annotate("C  STTC connectivity graph", xy=(0, 1.18),
                  xycoords="axes fraction", fontweight="bold", fontsize=9)

    # Panel D — forest
    axd = fig.add_subplot(gs[1, 1])
    if summ.empty:
        _placeholder(axd, "no paired wells\nfor connectivity metrics")
        axd.set_title("D  Connectivity response vs Control", loc="left",
                      fontweight="bold")
    else:
        _panel_forest(axd, summ, vc)

    fig.suptitle(
        "Figure 7 — Spatial activity maps & functional connectivity (STTC), "
        "well_uid unit", fontsize=9, y=0.99)
    return fig, {"response": resp, "summary": summ, "vs_control": vc}


def render(tidy, analysis_root, figure_root):
    fig, data = build_f7(tidy, analysis_root)
    paths = save_fig(fig, "F7_spatial_connectivity", figure_root, subdir="main")
    return paths, data
