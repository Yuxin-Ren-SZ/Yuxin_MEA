"""F7 — Spatial activity maps & functional connectivity.

Four panels, well_uid unit (chip = replicate), same style as F5/F6:

* **A** Activity field — representative Control vs treated 2-D firing-rate maps
  on the electrode plane (equal aspect → the true wide chip geometry).
* **B** Functional-connectivity graph — significant STTC edges on the electrode
  plane (edge alpha ∝ STTC), the SAME representative Control + treated wells as A
  (same chip, matched DIV, near each group's median), so the example matches the
  quantitative result.
* **C** STTC vs inter-unit distance — per-group decay curve (all unit pairs from
  the full STTC matrix), matched DIV. Replaces an earlier burst-propagation panel
  that the data did not support (network bursts here are near-synchronous, planar-
  fit r²≈0 — no travelling wave).
* **D** Group comparison — within-well response (log2 post/pre for ratio metrics,
  Δ for STTC/modularity) forest of the STTC graph-topology metrics
  (``mean_sttc, edge_density, clustering_coeff, modularity, global_efficiency,
  small_worldness``) vs Control maturation (difference-in-differences), restricted
  to the focus arms (IVH_Early, IVH_Late, H2O2_20uM).

Panel order top-to-bottom, left-to-right: A (fields) · B (graphs) · C (distance)
· D (forest).

Panels render a labelled placeholder if their per-well outputs are not yet on
disk, so ``make_report --figures f7`` degrades gracefully before the production
pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load
from . import stats as S
from .report_style import (
    METRIC_LABELS, group_color, ordered_groups, save_fig,
)

# STTC graph-topology panel. (mean_degree omitted — ≈ edge_density·(n−1), so
# redundant, and its unbounded log2 response squashes the shared x-axis.)
FOREST_METRICS = ["mean_sttc", "edge_density", "clustering_coeff",
                  "modularity", "global_efficiency", "small_worldness"]
# Focus arms (drop NPH / AraC / H2O2_10uM), matching F5.
FOCUS_ARMS = ["IVH_Early", "IVH_Late", "H2O2_20uM"]


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


def _sttc_matrix(analysis_root, row):
    """(W, unit_ids) — full STTC matrix + the unit_id order of its rows."""
    base = load.artifact_path(analysis_root, "connectivity_data", row,
                              "connectivity", "sttc_matrix.npy")
    swp = load.artifact_path(analysis_root, "connectivity_data", row,
                             "connectivity", "sttc_sweep.npz")
    W = np.load(base)
    uid = np.load(swp, allow_pickle=True)["unit_ids"]
    return W, np.asarray(uid)


def _pairs_dist_sttc(analysis_root, row):
    """All unit-pair (distance_µm, STTC) for one well from the full matrix."""
    W, uid = _sttc_matrix(analysis_root, row)
    pos = _positions(analysis_root, row)                       # {unit_id: (x, y)}
    keep = [i for i, u in enumerate(uid) if u in pos]
    if len(keep) < 3:
        return np.array([]), np.array([])
    xy = np.array([pos[uid[i]] for i in keep], float)
    Ws = W[np.ix_(keep, keep)]
    iu, jv = np.triu_indices(len(keep), k=1)
    d = np.hypot(xy[iu, 0] - xy[jv, 0], xy[iu, 1] - xy[jv, 1])
    return d, Ws[iu, jv]


# STTC-vs-distance decay: focus groups, matched DIV to remove the maturation axis.
DIST_BINS = np.arange(0.0, 3200.0, 300.0)
_B_GROUPS = ["Control", "IVH_Early", "IVH_Late", "H2O2_20uM"]
_B_DIV = (14, 24)


def _panel_sttc_distance(ax, analysis_root, tidy, max_wells=18, seed=0):
    ax.set_title("C  STTC vs inter-unit distance", loc="left", fontweight="bold")
    win = tidy[(tidy.DIV >= _B_DIV[0]) & (tidy.DIV <= _B_DIV[1])
               & tidy.mean_sttc.notna()]
    centers = (DIST_BINS[:-1] + DIST_BINS[1:]) / 2
    any_line = False
    for g in _B_GROUPS:
        rows = win[win.canonical_group == g]
        if rows.empty:
            continue
        rows = rows.sample(min(len(rows), max_wells), random_state=seed)
        D, Sv = [], []
        for _, r in rows.iterrows():
            try:
                d, s = _pairs_dist_sttc(analysis_root, r)
            except Exception:  # noqa: BLE001
                continue
            D.append(d); Sv.append(s)
        if not D:
            continue
        D = np.concatenate(D); Sv = np.concatenate(Sv)
        idx = np.digitize(D, DIST_BINS) - 1
        means = np.array([np.nanmean(Sv[idx == b]) if np.any(idx == b) else np.nan
                          for b in range(len(centers))])
        ax.plot(centers, means, "-o", ms=2.5, lw=1.0, color=group_color(g),
                label=g)
        any_line = True
    if not any_line:
        _placeholder(ax, "no STTC matrices\non disk")
        ax.set_title("C  STTC vs inter-unit distance", loc="left", fontweight="bold")
        return
    ax.set_xlabel("inter-unit distance (µm)")
    ax.set_ylabel("mean STTC")
    ax.legend(fontsize=6, loc="upper right")
    ax.text(0.02, 0.02, f"DIV {_B_DIV[0]}–{_B_DIV[1]}, ≤{max_wells} wells/group",
            transform=ax.transAxes, fontsize=5.5, color="0.45", va="bottom")


def _pick_representative(tidy, metric, group, lo=14, hi=24):
    """Row whose ``metric`` is nearest that group's median (matched DIV) — so
    the example graph is representative, not an outlier."""
    if metric not in tidy.columns:
        return None
    sub = tidy[(tidy.canonical_group == group) & tidy[metric].notna()
               & (tidy.DIV >= lo) & (tidy.DIV <= hi)]
    if sub.empty:
        sub = tidy[(tidy.canonical_group == group) & tidy[metric].notna()]
    if sub.empty:
        return None
    med = sub[metric].median()
    return sub.iloc[(sub[metric] - med).abs().to_numpy().argsort()[0]]


def _pick_matched_pair(tidy, treated_arm, metric="mean_sttc", lo=14, hi=24):
    """A (Control, treated) recording pair on the SAME chip at matched DIV, each
    near its group's median ``metric`` — a fair like-for-like example for A & C.
    Chooses the chip that minimises the Control↔treated DIV gap.
    """
    need = [c for c in (metric, "activity_gini") if c in tidy.columns]
    win = tidy[(tidy.DIV >= lo) & (tidy.DIV <= hi)]
    for c in need:
        win = win[win[c].notna()]
    best = None
    for chip in win.sample_id.unique():
        c = win[(win.sample_id == chip) & (win.canonical_group == "Control")]
        t = win[(win.sample_id == chip) & (win.canonical_group == treated_arm)]
        if c.empty or t.empty:
            continue
        tmed = t[metric].median()
        t_row = t.iloc[(t[metric] - tmed).abs().to_numpy().argsort()[0]]
        c_row = c.iloc[(c.DIV - t_row.DIV).abs().to_numpy().argsort()[0]]
        gap = abs(int(c_row.DIV) - int(t_row.DIV))
        if best is None or gap < best[0]:
            best = (gap, c_row, t_row)
    if best is None:
        return None, None
    return best[1], best[2]


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
        # equal aspect + physical extent → renders the true wide chip (≈3800×2050 µm)
        ax.imshow(field, origin="lower", cmap="magma",
                  aspect=("equal" if extent else "auto"),
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
        ax.scatter(xs, ys, s=5, color="0.4", zorder=3)
        if not edges.empty:
            smax = float(edges["sttc"].max()) or 1.0
            for _, e in edges.iterrows():
                if e["u"] in pos and e["v"] in pos:
                    x0, y0 = pos[e["u"]]; x1, y1 = pos[e["v"]]
                    a = max(0.05, min(1.0, float(e["sttc"]) / smax))
                    ax.plot([x0, x1], [y0, y1], color="#3b6fb0", alpha=a,
                            lw=0.5, zorder=2)
        ax.set_aspect("equal")            # true wide chip geometry
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=7)
    except Exception:  # noqa: BLE001
        _placeholder(ax, f"{title}\n(graph n/a)")


def _axis_label(metrics) -> str:
    """Ratio vs difference is decided per metric in ``stats``; say which applies.

    Hard-coding this drifts: several connectivity metrics moved from log2-ratio
    to difference once it turned out they are exactly zero in a meaningful share
    of wells, and a stale label would have mislabelled the units.
    """
    ratio = [m for m in metrics if m in S.RATIO_METRICS]
    diff = [m for m in metrics if m in S.DIFF_METRICS]
    if ratio and diff:
        return "(log2 post/pre; \u0394 for bounded metrics)"
    return "log2(post / pre)" if ratio else "(post \u2212 pre)"

def _stars(q):
    if q is None or np.isnan(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def _panel_forest(ax, summ, vc, resp=None):
    rng = np.random.default_rng(0)
    treated = [a for a in ordered_groups(summ.arm.unique()) if a in FOCUS_ARMS]
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
            if resp is not None:                      # raw per-well points behind
                pts = resp.loc[(resp.metric == metric) & (resp.arm == arm),
                               "response"].to_numpy(float)
                if len(pts):
                    jit = (rng.random(len(pts)) - 0.5) * 0.16
                    ax.scatter(pts, np.full(len(pts), y) + jit, s=4,
                               color=group_color(arm), alpha=0.28, lw=0, zorder=2)
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
    # clip the view to the bootstrap-CI range so a few near-zero-baseline
    # outliers (huge log2) don't squash the medians + point cloud
    if len(summ):
        lo = float(summ.ci_low.min()); hi = float(summ.ci_high.max())
        pad = 0.25 * (hi - lo) + 0.15
        ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("within-well response  " + _axis_label(FOREST_METRICS))
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
_TREATED_EXAMPLE = "IVH_Late"     # representative treated arm for panels A/C


def build_f7(tidy, analysis_root):
    import matplotlib.pyplot as plt

    resp = S.well_response(tidy, metrics=FOREST_METRICS)
    resp = resp[resp.arm.isin(["Control"] + FOCUS_ARMS)]
    summ = S.arm_response_summary(resp) if not resp.empty else pd.DataFrame()
    vc = S.arm_response_vs_control(resp) if not resp.empty else pd.DataFrame()

    # ONE representative (Control, IVH_Late) pair — same chip, matched DIV, each
    # near its group median — shared by A and C so it's a fair comparison.
    c_row, t_row = _pick_matched_pair(tidy, _TREATED_EXAMPLE)

    def _lab(row, grp):
        if row is None:
            return grp
        return f"{grp} · {row.well_uid.split('|')[0]} DIV{int(row.DIV)}"

    c_lab = _lab(c_row, "Control")
    t_lab = _lab(t_row, _TREATED_EXAMPLE)

    # The chip is a WIDE rectangle (~3800×2050 µm), so the map/graph rows get the
    # full width and each map uses equal aspect.
    fig = plt.figure(figsize=(9.2, 9.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.7],
                          hspace=0.42, wspace=0.26)

    # Row 0 — Panel A: two wide activity fields (same pair)
    gsa = gs[0, :].subgridspec(1, 2, wspace=0.1)
    axa0 = fig.add_subplot(gsa[0, 0]); axa1 = fig.add_subplot(gsa[0, 1])
    _panel_field(axa0, analysis_root, c_row, c_lab)
    _panel_field(axa1, analysis_root, t_row, t_lab)
    axa0.annotate("A  Activity field (firing-rate map on the electrode plane)",
                  xy=(0, 1.12), xycoords="axes fraction", fontweight="bold",
                  fontsize=9)

    # Row 1 — Panel C: two wide STTC graphs (same pair as A)
    gsc = gs[1, :].subgridspec(1, 2, wspace=0.1)
    axc0 = fig.add_subplot(gsc[0, 0]); axc1 = fig.add_subplot(gsc[0, 1])
    _panel_graph(axc0, analysis_root, c_row, c_lab)
    _panel_graph(axc1, analysis_root, t_row, t_lab)
    axc0.annotate("B  STTC connectivity graph (same wells as A; edge α ∝ STTC)",
                  xy=(0, 1.12), xycoords="axes fraction", fontweight="bold",
                  fontsize=9)

    # Row 2 — Panel B (STTC-distance) | Panel D (forest)
    axb = fig.add_subplot(gs[2, 0])
    _panel_sttc_distance(axb, analysis_root, tidy)
    axd = fig.add_subplot(gs[2, 1])
    if summ.empty:
        _placeholder(axd, "no paired wells\nfor connectivity metrics")
        axd.set_title("D  Connectivity response vs Control", loc="left",
                      fontweight="bold")
    else:
        _panel_forest(axd, summ, vc, resp)

    fig.suptitle(
        "Figure 7 — Spatial activity maps & functional connectivity (STTC), "
        "well_uid unit", fontsize=9, y=1.0)
    return fig, {"response": resp, "summary": summ, "vs_control": vc}


def render(tidy, analysis_root, figure_root):
    fig, data = build_f7(tidy, analysis_root)
    paths = save_fig(fig, "F7_spatial_connectivity", figure_root, subdir="main")
    return paths, data
