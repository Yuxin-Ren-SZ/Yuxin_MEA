"""F6c — Cohort-wide burst-type clustering across groups and wells.

The pipeline's per-well burst typing (``ml_burst_typing``) assigns each burst an
integer type, but those identities are **per-well and not comparable across
wells** — a single well's clustering can't reveal cohort structure. This figure
pools every detected burst from many wells across the focus groups, embeds them
in one shared feature space, and clusters them **globally** into cohort-wide
burst archetypes.

Result (well = unit of analysis): the pipeline resolves ~4 reproducible burst
archetypes. Whether their **composition separates the groups** is *computed in
panel C* — a per-archetype Mann-Whitney of each treated arm's per-well fractions
vs Control, BH-FDR (`_composition_stats`) — never asserted: any q<0.05 is starred
and the panel title / figure suptitle switch to report the actual count. This is
deliberately not a hardcoded "n.s." label: on the current sample one contrast
(IVH_Late, archetype 1) clears FDR at q≈0.04, so the figure must report it. That
single borderline contrast is sensitive to the KMeans k / burst-sampling seed and
should be treated as exploratory, not a headline claim, pending confirmation.
Panel C uses per-well fractions, not burst-level pooling, precisely to avoid the
pseudo-replication that made a pooled bar chart look falsely separated.

Per-burst features live in each recording's ML ``network_bursts.pkl``; F6c
clusters on the raw biologically-interpretable subset (``duration_s,
within_burst_fr, participation, total_spikes, burst_peak, peak_synchrony,
synchrony_energy``) and excludes the ML-internal detector features
(``llr_*, posterior_*, ff_peak, n_distinct_clusters``).

Panels:
  * **A** global UMAP of all bursts, coloured by treatment group;
  * **B** same embedding coloured by global KMeans burst archetype;
  * **C** per-group archetype composition (fraction of that group's bursts);
  * **D** archetype feature profiles (standardised feature means) for readout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load as L
from . import stats as S
from .report_style import caption, group_color, ordered_groups, save_fig

FOCUS_GROUPS = ["Control", "IVH_Early", "IVH_Late"]
# Raw, biologically interpretable burst-level metrics (network_bursts.pkl).
# ML-internal detector features (llr_*, posterior_*, ff_peak, n_distinct_clusters)
# are intentionally excluded.
FEATURES = ["duration_s", "within_burst_fr", "participation", "total_spikes",
            "burst_peak", "peak_synchrony", "synchrony_energy"]
FEAT_LABELS = ["duration", "within-burst FR", "participation", "spikes/burst",
               "peak pop-FR", "peak synchrony", "intensity"]
# Positively-skewed rate/count/energy metrics: log-transform before z-scoring so
# clustering reflects shape, not a few high-rate outliers. (Fractions kept linear.)
LOG_FEATURES = {"within_burst_fr", "total_spikes", "burst_peak",
                "synchrony_energy", "duration_s"}
_MAX_ROWS_PER_GROUP = 40     # well-recordings sampled per group (I/O bound)
_MAX_BURSTS = 9000           # cap pooled bursts before UMAP
_ARCH_PALETTE = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#aa3377", "#66ccee"]


def _stars(q: float) -> str:
    if np.isnan(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def collect_bursts(tidy, analysis_root, seed: int = 0) -> pd.DataFrame:
    """Pool per-burst feature rows across sampled wells of the focus groups."""
    frames = []
    for grp in FOCUS_GROUPS:
        rows = tidy[(tidy.canonical_group == grp) & (tidy.nb_count >= 5)]
        if rows.empty:
            continue
        take = rows.sample(min(len(rows), _MAX_ROWS_PER_GROUP), random_state=seed)
        for _, r in take.iterrows():
            try:
                b = L.load_ml_bursts(analysis_root, r)
            except FileNotFoundError:
                continue
            if b is None or b.empty or not set(FEATURES).issubset(b.columns):
                continue
            full = b
            b = b[FEATURES].copy()
            b["start"] = full["start"] if "start" in full else np.nan
            b["end"] = full["end"] if "end" in full else np.nan
            b["group"] = grp
            b["well_uid"] = r["well_uid"]
            b["tau"] = int(r["tau"])
            b["recording_key"] = r["recording_key"]
            b["rec_name"] = r["rec_name"]
            b["well_id"] = r["well_id"]
            frames.append(b)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = out.dropna(subset=FEATURES)
    if len(out) > _MAX_BURSTS:
        out = out.sample(_MAX_BURSTS, random_state=seed).reset_index(drop=True)
    return out


def name_archetypes(prof: np.ndarray, feat_labels=FEAT_LABELS) -> list:
    """Descriptive per-cluster names from the z-scored feature profile (robust to
    KMeans relabeling and to the chosen k):
      Long (high duration) · Sparse (very low participation) ·
      Fast (high within-burst / peak firing rate) · Low-rate (the rest).
    Duplicates are disambiguated with a numeric suffix."""
    fi = {n: i for i, n in enumerate(feat_labels)}
    base = []
    for p in prof:
        dur, part = p[fi["duration"]], p[fi["participation"]]
        pfr, wfr = p[fi["peak pop-FR"]], p[fi["within-burst FR"]]
        if dur > 0.7:
            base.append("Long")
        elif part < -0.7:
            base.append("Sparse")
        elif pfr > 0.4 or wfr > 0.4:
            base.append("Fast")
        else:
            base.append("Low-rate")
    # disambiguate duplicates (e.g. two "Fast" -> "Fast-1", "Fast-2")
    out, seen = [], {}
    for n in base:
        if base.count(n) > 1:
            seen[n] = seen.get(n, 0) + 1
            out.append(f"{n}-{seen[n]}")
        else:
            out.append(n)
    return out


def _cluster(Z: np.ndarray, seed: int = 0) -> tuple[np.ndarray, int]:
    """KMeans with k chosen by silhouette (k = 2..6)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best = None
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        # silhouette on a subsample for speed
        idx = np.random.default_rng(seed).choice(
            len(Z), min(3000, len(Z)), replace=False)
        s = silhouette_score(Z[idx], km.labels_[idx])
        if best is None or s > best[0]:
            best = (s, k, km.labels_)
    return best[2], best[1]


def _composition_stats(wc: pd.DataFrame, k: int, groups: list[str]) -> pd.DataFrame:
    """Per-archetype per-well-fraction MWU: each treated group vs Control.

    ``wc`` = per-well archetype-fraction table (one row per well; integer archetype
    columns + a ``group`` column). Mann-Whitney each treated group vs Control for
    every archetype, then BH-FDR across all (archetype × arm) tests. This is what
    backs the panel's "n.s." claim — computed, not asserted.
    """
    from scipy.stats import mannwhitneyu

    ctrl_mask = wc.group == "Control"
    rows = []
    for a in range(k):
        if a not in wc.columns:
            continue
        c = wc.loc[ctrl_mask, a].dropna().to_numpy(float)
        for g in groups:
            if g == "Control":
                continue
            x = wc.loc[wc.group == g, a].dropna().to_numpy(float)
            p = np.nan
            if len(x) >= 3 and len(c) >= 3:
                try:
                    p = float(mannwhitneyu(x, c, alternative="two-sided").pvalue)
                except ValueError:
                    p = np.nan
            rows.append(dict(arch=a, arm=g, p=p, n_arm=len(x), n_ctrl=len(c)))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q"] = S._bh_fdr(df.p.to_numpy())
    return df


def _burst_raster(ax, analysis_root, brow, color, name, pad=0.5, max_units=60):
    """Curated-spike raster around one example burst: [start-pad, end+pad] s."""
    ax.set_title(name, fontsize=7.5, color=color, fontweight="bold")
    try:
        spikes = L.load_curated_spikes(analysis_root, brow)
        s0, s1 = float(brow["start"]), float(brow["end"])
    except Exception:  # noqa: BLE001
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "no raster", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="0.5")
        return
    t0, t1 = s0 - pad, s1 + pad
    items = list(spikes.items())[:max_units]
    for i, (_, st) in enumerate(items):
        st = np.asarray(st, float); st = st[(st >= t0) & (st <= t1)]
        ax.eventplot(st, lineoffsets=i, linelengths=0.85, linewidths=0.4, colors="0.15")
    ax.axvspan(s0, s1, color=color, alpha=0.18, lw=0)     # shade the burst
    ax.set_xlim(t0, t1); ax.set_ylim(-1, max(len(items), 1))
    ax.set_xticks([t0, s0, s1, t1])
    ax.set_xticklabels(["-0.5", "0", f"{s1 - s0:.1f}", f"{s1 - s0 + 0.5:.1f}"], fontsize=5.5)
    ax.set_xlabel("time from burst onset (s)", fontsize=6)
    ax.set_ylabel("unit", fontsize=6); ax.tick_params(labelsize=5.5)


def _rep_bursts(bursts, Z, labels, k):
    """Representative burst per archetype = nearest the archetype's Z-centroid."""
    reps = {}
    for a in range(k):
        idx = np.where(labels == a)[0]
        if len(idx) == 0:
            continue
        cen = Z[idx].mean(0)
        j = idx[np.argmin(((Z[idx] - cen) ** 2).sum(1))]
        reps[a] = bursts.iloc[j]
    return reps


def build_f6c(tidy, analysis_root, seed: int = 0):
    import matplotlib.pyplot as plt
    import umap
    from sklearn.preprocessing import StandardScaler

    bursts = collect_bursts(tidy, analysis_root, seed=seed)
    if bursts.empty:
        raise RuntimeError("no bursts collected for F6c")
    X = bursts[FEATURES].to_numpy(float)
    for j, f in enumerate(FEATURES):          # log-transform skewed metrics
        if f in LOG_FEATURES:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    Z = StandardScaler().fit_transform(X)
    emb = np.asarray(umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                               random_state=seed).fit_transform(Z))
    labels, k = _cluster(Z, seed=seed)
    bursts = bursts.assign(arch=labels)
    prof = np.vstack([Z[labels == a].mean(0) for a in range(k)])
    anames = name_archetypes(prof, FEAT_LABELS)     # descriptive per-archetype names

    from .trajectory import DEFAULT_TAU_EDGES

    fig = plt.figure(figsize=(10.6, 10.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[3.0, 0.95], hspace=0.45)
    gs = outer[0].subgridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.32)
    gl = gs[0, 0].subgridspec(3, 1, hspace=0.5)
    axA = fig.add_subplot(gl[0]); axB = fig.add_subplot(gl[1]); axD = fig.add_subplot(gl[2])
    groups = [g for g in ordered_groups(bursts.group.unique())]

    # A — coloured by group
    for g in groups:
        m = bursts.group.to_numpy() == g
        axA.scatter(emb[m, 0], emb[m, 1], s=3, c=group_color(g), lw=0, alpha=0.5,
                    label=g, rasterized=True)
    axA.set_xticks([]); axA.set_yticks([]); axA.set_xlabel("UMAP-1"); axA.set_ylabel("UMAP-2")
    axA.legend(fontsize=6, markerscale=2, loc="best")
    axA.set_title("A  Burst feature space by group", loc="left", fontweight="bold")

    # B — coloured by global archetype
    for a in range(k):
        m = labels == a
        axB.scatter(emb[m, 0], emb[m, 1], s=3, c=_ARCH_PALETTE[a % len(_ARCH_PALETTE)],
                    lw=0, alpha=0.6, label=anames[a], rasterized=True)
    axB.set_xticks([]); axB.set_yticks([]); axB.set_xlabel("UMAP-1")
    axB.legend(fontsize=6, markerscale=2, loc="best", title=f"archetype (k={k})",
               title_fontsize=6)
    axB.set_title("B  Global burst archetypes", loc="left", fontweight="bold")

    # D — archetype feature profiles (standardised means)
    im = axD.imshow(prof, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    axD.set_xticks(range(len(FEATURES)))
    axD.set_xticklabels(FEAT_LABELS, rotation=45, ha="right", fontsize=6)
    axD.set_yticks(range(k)); axD.set_yticklabels(anames, fontsize=7)
    cb = fig.colorbar(im, ax=axD, shrink=0.8, pad=0.02)
    cb.set_label("z-score", fontsize=6); cb.ax.tick_params(labelsize=6)
    axD.set_title("D  Archetype feature profiles", loc="left", fontweight="bold")

    # C — archetype composition vs treatment day (tau); arms stacked vertically,
    # SHARED tau x-axis. Pooling all days hid the temporal shift.
    wc0 = (bursts.groupby(["well_uid", "group", "arch"]).size()
           .unstack("arch", fill_value=0))
    wc0 = wc0.div(wc0.sum(axis=1), axis=0).reset_index()
    cstats = _composition_stats(wc0, k, groups)
    n_sig = int((cstats.q < 0.05).sum()) if not cstats.empty else 0

    armmap = S.well_index(tidy).set_index("well_uid").arm
    bt = bursts.assign(arm=bursts.well_uid.map(armmap))
    bt = bt[bt.arm.isin(FOCUS_GROUPS)].copy()
    bt["tauc"] = pd.cut(bt.tau, DEFAULT_TAU_EDGES, right=False).map(
        lambda c: (c.left + c.right) / 2 if pd.notna(c) else np.nan)
    arms_present = [a for a in ordered_groups(bt.arm.unique()) if a in FOCUS_GROUPS]
    gsc = gs[0, 1].subgridspec(len(arms_present), 1, hspace=0.15)
    prev = None
    for i, arm in enumerate(arms_present):
        axc = fig.add_subplot(gsc[i], sharex=prev)
        prev = axc
        sub = bt[bt.arm == arm]
        wtf = (sub.groupby(["well_uid", "tauc", "arch"]).size()
               .unstack("arch", fill_value=0))
        wtf = wtf.div(wtf.sum(axis=1), axis=0)
        comp = wtf.groupby("tauc").mean().sort_index()
        if not comp.empty:
            ys = [comp[a].values if a in comp.columns else np.zeros(len(comp))
                  for a in range(k)]
            axc.stackplot(comp.index.astype(float), *ys,
                          colors=[_ARCH_PALETTE[a % len(_ARCH_PALETTE)] for a in range(k)],
                          labels=anames)
        axc.axvline(0, ls=":", color="0.3", lw=0.8)
        axc.set_ylim(0, 1); axc.set_yticks([0, 0.5, 1.0])
        axc.set_ylabel(arm, fontsize=7, rotation=0, ha="right", va="center")
        axc.tick_params(labelsize=6)
        if i < len(arms_present) - 1:
            axc.tick_params(labelbottom=False)
        else:
            axc.set_xlabel("treatment day (tau)", fontsize=7)
        if i == 0:
            axc.annotate("C  Archetype composition vs treatment day",
                         xy=(0, 1.25), xycoords="axes fraction", fontweight="bold",
                         fontsize=9)
            axc.legend(fontsize=5, ncol=k, loc="lower center",
                       bbox_to_anchor=(0.5, 1.02), frameon=False)

    # E — one example burst raster per archetype (-0.5 s to +0.5 s around burst)
    reps = _rep_bursts(bursts, Z, labels, k)
    gsr = outer[1].subgridspec(1, k, wspace=0.28)
    for a in range(k):
        axr = fig.add_subplot(gsr[a])
        if a in reps:
            _burst_raster(axr, analysis_root, reps[a],
                          _ARCH_PALETTE[a % len(_ARCH_PALETTE)], anames[a])
        if a == 0:
            axr.annotate("E  Example burst raster per archetype (−0.5 to +0.5 s)",
                         xy=(0, 1.28), xycoords="axes fraction", fontweight="bold",
                         fontsize=9)

    sep = ("composition does not separate groups" if n_sig == 0
           else f"{n_sig} archetype fraction(s) differ from Control (BH-FDR)")
    fig.suptitle(f"Figure 6c — The pipeline resolves {k} reproducible burst "
                 f"archetypes; {sep} "
                 f"({len(bursts)} bursts, {bursts.well_uid.nunique()} wells)",
                 fontsize=8.5, y=1.0)
    fig.tight_layout()
    caption(fig,
        f"Cohort-wide burst archetypes. Every detected network burst from the "
        f"focus groups is pooled and embedded by its features, then clustered "
        f"into {k} reproducible archetypes (named by their properties). "
        f"(A) Burst feature space coloured by group; (B) the same space coloured "
        f"by archetype. (C) Archetype composition vs treatment day (tau), one "
        f"stacked panel per arm on a shared x-axis: each band = the group-mean "
        f"fraction of bursts in that archetype, averaged across wells (each well "
        f"weighted equally). (D) Mean feature profile of each archetype. "
        f"(E) One example raster per archetype (−0.5 to +0.5 s around the burst). "
        f"Whether composition separates groups is tested per archetype (per-well "
        f"fraction, Mann-Whitney vs Control, BH-FDR): {sep.lower()}.")
    return fig, {"n_bursts": int(len(bursts)), "k": int(k),
                 "n_wells": int(bursts.well_uid.nunique()),
                 "n_composition_sig": n_sig,
                 "composition_stats": cstats}


def render(tidy, analysis_root, figure_root):
    fig, meta = build_f6c(tidy, analysis_root)
    return save_fig(fig, "F6c_burst_archetypes", figure_root, subdir="main"), meta
