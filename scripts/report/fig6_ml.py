"""F6 — ML burst characterization (methods novelty).

Panels (single representative well for A-B, whole cohort for D):

* **A. HMM burst posteriors** — per-unit P(burst) heatmap over a time window
  with the population co-burst signal (fraction of units with P>0.5) beneath.
* **B. Feature-space embedding** — 2-D UMAP of the 26-D per-bin feature matrix,
  bins coloured by whether HDBSCAN assigned them to a burst cluster. Shows the
  low-density burst manifold the detector recovers.
* **D. Burst modulation index across arms** — cohort-level distribution of the
  HMM burst-modulation index (well_uid unit), tying the ML method to biology.

(Per-well burst-type clustering moved to the cohort-wide F6c, where bursts are
pooled across wells and clustered in one shared feature space — see
``fig6c_bursttypes.py``. Panel labels here read A / B / D to match F6d output.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load as L
from .report_style import (
    METRIC_LABELS, caption, group_color, ordered_groups, save_fig,
)

# UMAP params mirror the pipeline (ml_burst_detection config), n_components=2 for viz.
_UMAP_KW = dict(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)
_BURST_RED = "#d62728"
_NOISE_GREY = "#c8c8c8"


def _pick_example(tidy: pd.DataFrame) -> pd.Series:
    cand = tidy[(tidy.burst_type_k >= 2) & (tidy.nb_count >= 8)
                & (tidy.n_curated >= 25)]
    return cand.sort_values("nb_rate", ascending=False).iloc[0]


def _panel_posterior(ax, ax_trace, cax, tr, t_win=(0.0, 60.0)) -> None:
    bins = np.asarray(tr.bins)
    ncols = tr.posterior_matrix.shape[1]
    centers = (bins[:-1] + bins[1:]) / 2 if len(bins) == ncols + 1 else bins[:ncols]
    m = (centers >= t_win[0]) & (centers <= t_win[1])
    P = tr.posterior_matrix[:, m]
    x0, x1 = float(centers[m][0]), float(centers[m][-1])
    # order units by burst-time activity for a cleaner heatmap
    order = np.argsort(-P.mean(axis=1))
    im = ax.imshow(P[order], aspect="auto", cmap="magma", vmin=0, vmax=1,
                   extent=[x0, x1, 0, P.shape[0]], origin="lower",
                   interpolation="nearest")
    ax.set_ylabel("unit")
    ax.set_xlim(x0, x1)
    ax.tick_params(labelbottom=False)
    # colorbar in its own axis so it does NOT steal width from the heatmap
    # (that width theft is what mis-aligned the heatmap vs the trace below).
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("P(burst)", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    # population co-burst trace (post_frac_gt_0_5 = feature col 0), same x-axis
    fn = list(tr.feature_names)
    co = tr.feature_matrix[:, fn.index("post_frac_gt_0_5")]
    fc = centers[: len(co)]
    mm = (fc >= x0) & (fc <= x1)
    ax_trace.plot(fc[mm], co[: len(fc)][mm], color="#333333", lw=0.8)
    ax_trace.set_ylabel("co-burst\nfrac", fontsize=6)
    ax_trace.set_xlabel("time (s)")
    ax_trace.set_xlim(x0, x1)
    ax.set_title("A  HMM burst posteriors", loc="left", fontweight="bold")


def _panel_umap(ax, tr, seed: int = 42) -> None:
    import umap

    X = np.asarray(tr.feature_matrix, dtype=float)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[sd == 0] = 1.0
    Z = np.nan_to_num((X - mu) / sd)
    emb = umap.UMAP(**_UMAP_KW).fit_transform(Z)
    labels = np.asarray(tr.hdbscan_labels)
    burst_clusters = set(getattr(tr, "burst_labels", []) or [])
    is_burst = np.array([lab in burst_clusters for lab in labels])
    ax.scatter(emb[~is_burst, 0], emb[~is_burst, 1], s=2, c=_NOISE_GREY,
               alpha=0.5, lw=0, label="baseline bins")
    ax.scatter(emb[is_burst, 0], emb[is_burst, 1], s=3, c=_BURST_RED,
               alpha=0.8, lw=0, label="burst bins")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=6, loc="best", markerscale=2)
    ax.set_title("B  Feature-space embedding", loc="left", fontweight="bold")


def _panel_modulation(ax, tidy) -> None:
    metric = "burst_modulation_index"
    per_well = tidy.groupby(["well_uid"]).agg(
        arm=("canonical_group", lambda s: next((g for g in s if g != "Control"), "Control")),
        val=(metric, "median")).reset_index()
    arms = ordered_groups(per_well.arm.unique())
    data = [per_well.loc[per_well.arm == a, "val"].dropna().to_numpy() for a in arms]
    for i, (a, d) in enumerate(zip(arms, data)):
        if len(d) >= 8:
            bp = ax.boxplot(d, positions=[i], widths=0.6, patch_artist=True,
                            showfliers=False, zorder=1)
            for box in bp["boxes"]:
                box.set(facecolor=group_color(a), alpha=0.35, lw=0.8)
            for med in bp["medians"]:
                med.set(color=group_color(a), lw=1.2)
        jitter = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.3
        ax.scatter(np.full(len(d), i) + jitter, d, s=8, color=group_color(a),
                   alpha=0.8, lw=0, zorder=3)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=40, ha="right", fontsize=6)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title("D  Burst modulation index (cohort)", loc="left", fontweight="bold")


def build_f6(tidy, analysis_root):
    import matplotlib.pyplot as plt

    from . import stats as S
    from .trajectory import pick_early_late

    def _valid(row):
        if row is None:
            return False
        try:
            tr = L.load_ml_trace(analysis_root, row)
            return getattr(tr, "posterior_matrix", None) is not None
        except Exception:  # noqa: BLE001
            return False

    # find an IVH_Late well whose EARLY and LATE recordings both have real traces
    wi = S.well_index(tidy)
    cand_wells = wi.loc[(wi.arm == "IVH_Late") & wi.paired, "well_uid"]
    ex, e_row, l_row = None, None, None
    scored = tidy[tidy.well_uid.isin(cand_wells) & (tidy.nb_count >= 10)]
    for w in scored.groupby("well_uid").nb_count.median().sort_values(ascending=False).index:
        e, l = pick_early_late(tidy, w)
        if _valid(e) and _valid(l) and int(e.tau) != int(l.tau):
            ex, e_row, l_row = w, e, l
            break
    if ex is None:
        ex = tidy[tidy.canonical_group == "IVH_Late"].well_uid.iloc[0]
        e_row, l_row = pick_early_late(tidy, ex)

    fig = plt.figure(figsize=(9.0, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.22)
    axp0 = axu0 = None
    for ci, (row, lab) in enumerate([(e_row, "early"), (l_row, "late")]):
        axp = fig.add_subplot(gs[0, ci]); axu = fig.add_subplot(gs[1, ci])
        if ci == 0:
            axp0, axu0 = axp, axu
        try:
            if row is None:
                raise ValueError("no recording")
            tr = L.load_ml_trace(analysis_root, row)
            if getattr(tr, "posterior_matrix", None) is None:
                raise ValueError("no posterior")
            _posterior_img(axp, tr)
            _panel_umap(axu, tr); axu.set_title("", loc="left"); axu.set_box_aspect(1)
            axp.set_title(f"{lab} · DIV{int(row.DIV)} tau{int(row.tau):+d}", fontsize=8)
        except Exception:  # noqa: BLE001
            for a in (axp, axu):
                a.set_xticks([]); a.set_yticks([])
                a.text(0.5, 0.5, f"{lab}: no debug trace", ha="center",
                       va="center", transform=a.transAxes, fontsize=7, color="0.5")
    if axp0 is not None:
        axp0.annotate("A  HMM burst posteriors (early vs late)", xy=(0, 1.18),
                      xycoords="axes fraction", fontweight="bold", fontsize=9)
        axu0.annotate("B  Feature-space embedding (early vs late)", xy=(0, 1.05),
                      xycoords="axes fraction", fontweight="bold", fontsize=9)
    tag = "" if ex is None else f"  ·  {ex.split('|')[0]} (IVH_Late)"
    fig.suptitle(f"Figure 6 — ML burst characterization: same well, early vs late{tag}",
                 fontsize=9, y=1.0)
    caption(fig,
        "ML burst characterization of one representative treated well, shown "
        "early (τ≈0) vs late (τ≈14) so the post-treatment change is visible. "
        "(A) HMM burst-state posteriors over time with the co-burst-fraction "
        "trace beneath (shared x-axis); shaded = detected network-burst windows. "
        "(B) Feature-space embedding (UMAP) of the well's time bins, coloured by "
        "burst vs rest state, on a shared embedding so early and late are "
        "comparable. Illustrative single-well panels (not a group statistic).")
    return fig, {"example": ex}


def _posterior_img(ax, tr, t_win=(0.0, 60.0)):
    """Compact per-unit P(burst) heatmap over a time window (no trace/colorbar)."""
    bins = np.asarray(tr.bins)
    ncols = tr.posterior_matrix.shape[1]
    centers = (bins[:-1] + bins[1:]) / 2 if len(bins) == ncols + 1 else bins[:ncols]
    m = (centers >= t_win[0]) & (centers <= t_win[1])
    P = tr.posterior_matrix[:, m]
    order = np.argsort(-P.mean(axis=1))
    ax.imshow(P[order], aspect="auto", cmap="magma", vmin=0, vmax=1,
              extent=[float(centers[m][0]), float(centers[m][-1]), 0, P.shape[0]],
              origin="lower", interpolation="nearest")
    ax.set_xlabel("time (s)", fontsize=7); ax.set_ylabel("unit", fontsize=7)
    ax.tick_params(labelsize=6)


_MOD_ARMS = ["Control", "IVH_Early", "IVH_Late"]


def build_f6_modulation(tidy):
    """Burst modulation index (HMM burst-vs-Poisson sharpness): its time-course
    (A, vs tau) + the within-well difference-in-differences vs Control (B)."""
    import matplotlib.pyplot as plt

    from . import stats as S
    from .forest import stars as _stars
    from .trajectory import plot_trajectory

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.4, 3.6),
                                   gridspec_kw={"width_ratios": [1.3, 1.0]})
    # A — trajectory vs treatment day
    plot_trajectory(axa, tidy, "burst_modulation_index", _MOD_ARMS,
                    title="A  Burst modulation vs treatment day")

    # B — within-well DiD response (post−pre), per *chip* (biological replicate):
    # bold points = the 3 chip means, wells faint behind, mean bar, LMM star.
    resp = S.well_response(tidy, metrics=["burst_modulation_index"])
    resp = resp[resp.arm.isin(_MOD_ARMS)]
    vc = (S.arm_response_vs_control_lmm(resp, focus_arms=["IVH_Early", "IVH_Late"])
          .set_index("arm") if not resp.empty else None)
    cr = S.chip_response(resp)
    rng = np.random.default_rng(0)
    for i, a in enumerate(_MOD_ARMS):
        w = resp.loc[resp.arm == a, "response"].dropna().to_numpy()   # wells (tech)
        cp = cr.loc[cr.arm == a, "response"].dropna().to_numpy()      # chips (bio)
        if len(w):
            jit = (rng.random(len(w)) - 0.5) * 0.30
            axb.scatter(np.full(len(w), i) + jit, w, s=6, color=group_color(a),
                        alpha=0.16, lw=0, zorder=2)
        if len(cp):
            m = float(np.mean(cp))
            axb.hlines(m, i - 0.28, i + 0.28, color=group_color(a), lw=1.8, zorder=4)
            jit = (rng.random(len(cp)) - 0.5) * 0.16
            axb.scatter(np.full(len(cp), i) + jit, cp, s=28,
                        facecolor=group_color(a), edgecolor="white",
                        linewidth=0.5, zorder=5)
        if a != "Control" and vc is not None and a in vc.index:
            row = vc.loc[a]
            s = _stars(row.q_bh) if bool(row.chip_consistent) else ""
            if not s and bool(row.get("suggestive", False)):
                s = "△"
            if s:
                top = max(float(w.max()) if len(w) else -np.inf,
                          float(cp.max()) if len(cp) else -np.inf)
                axb.text(i, top, s, ha="center", va="bottom", fontsize=9,
                         color=group_color(a))
    axb.axhline(0, ls="--", color="0.5", lw=0.8)
    axb.set_xticks(range(len(_MOD_ARMS)))
    axb.set_xticklabels(_MOD_ARMS, rotation=25, ha="right", fontsize=7)
    axb.set_ylabel("modulation Δ (post − pre)")
    axb.set_title("B  Response vs Control (DiD, chip-level)", loc="left",
                  fontweight="bold")
    axb.text(0.0, -0.30, "bold = per-chip means (n=3); faint = wells; "
             "△ = suggestive (consistent 3/3 chips, p<0.05 uncorr, not FDR-sig)",
             transform=axb.transAxes, fontsize=5, color="0.4", va="top")
    fig.suptitle("Figure 6d — Burst modulation index (HMM burst-vs-Poisson "
                 "sharpness); chip = biological replicate", fontsize=9, y=1.02)
    fig.tight_layout()
    caption(fig,
        "Burst modulation index (how sharply bursting departs from a Poisson "
        "process; higher = crisper bursts). (A) Group trajectories vs treatment "
        "day (tau): line = per-arm median across wells, ribbon = 95% bootstrap CI "
        "(well-level), dotted vertical = treatment day (tau 0). (B) Within-well "
        "difference-in-differences vs Control (post−pre) at the biological-"
        "replicate level: bold points = 3 per-chip means, faint = wells, bar = "
        "mean of chip means; △ = suggestive (same direction on all 3 chips, "
        "uncorrected p<0.05, does not survive FDR at n=3); ★ would mark FDR "
        "q<0.05 (none). Unit: chip (n=3 per arm).")
    return fig


def render(tidy, analysis_root, figure_root):
    fig, meta = build_f6(tidy, analysis_root)
    paths = save_fig(fig, "F6_ml_characterization", figure_root, subdir="main")
    figd = build_f6_modulation(tidy)
    paths += save_fig(figd, "F6d_burst_modulation", figure_root, subdir="main")
    return paths, meta
