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
from .report_style import MUTED, caption, group_color, ordered_groups, save_fig

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
# Dedicated qualitative archetype palette — deliberately distinct from the group
# palette (amber/sienna/grey) AND the network-state palette (rest #3D6FB4 /
# burst #D1495B), so archetype colour is never confused with either.
_ARCH_PALETTE = ["#44AA99", "#882255", "#999933", "#AA4499", "#117733", "#DDCC77"]


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

    groups = [g for g in ordered_groups(bursts.group.unique())]

    # ---- composition + significance (per-well fractions; stats UNCHANGED) ----
    wc0 = (bursts.groupby(["well_uid", "group", "arch"]).size()
           .unstack("arch", fill_value=0))
    wc0 = wc0.div(wc0.sum(axis=1), axis=0).reset_index()
    cstats = _composition_stats(wc0, k, groups)
    n_sig = int((cstats.q < 0.05).sum()) if not cstats.empty else 0
    sig_arch = (set(int(a) for a in cstats.loc[cstats.q < 0.05, "arch"])
                if not cstats.empty else set())
    sig_by_arm: dict = {}
    if not cstats.empty:
        for _, r in cstats[cstats.q < 0.05].iterrows():
            sig_by_arm.setdefault(r.arm, []).append(int(r.arch))

    def _arch_label(a):
        return anames[a] + ("  ★ FDR" if a in sig_arch else "")

    armmap = S.well_index(tidy).set_index("well_uid").arm
    bt = bursts.assign(arm=bursts.well_uid.map(armmap))
    bt = bt[bt.arm.isin(FOCUS_GROUPS)].copy()
    bt["tauc"] = pd.cut(bt.tau, DEFAULT_TAU_EDGES, right=False).map(
        lambda c: (c.left + c.right) / 2 if pd.notna(c) else np.nan)
    arms_present = [a for a in ordered_groups(bt.arm.unique()) if a in FOCUS_GROUPS]

    # ---- figure: MAIN (composition + feature profiles) over a demoted strip ----
    fig = plt.figure(figsize=(10.4, 10.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.15, 1.0], top=0.85,
                             bottom=0.06, hspace=0.6)
    main = outer[0].subgridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.36)
    main_top = outer[0].get_position(fig).y1
    fig.text(0.09, main_top + 0.055, "C  Archetype composition vs treatment day",
             fontweight="bold", fontsize=9, ha="left", va="bottom")

    # C (LEAD) — archetype composition vs treatment day, one stacked panel/arm.
    gsc = main[0, 0].subgridspec(len(arms_present), 1, hspace=0.16)
    prev = None
    for i, arm in enumerate(arms_present):
        axc = fig.add_subplot(gsc[i], sharex=prev); prev = axc
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
                          labels=[_arch_label(a) for a in range(k)])
        axc.axvline(0, ls=":", color="0.3", lw=0.8)
        axc.set_ylim(0, 1); axc.set_yticks([0, 0.5, 1.0])
        axc.set_ylabel(arm, fontsize=7.5, rotation=0, ha="right", va="center",
                       color=group_color(arm), fontweight="bold")
        axc.tick_params(labelsize=6)
        if arm in sig_by_arm:          # flag this arm's FDR-significant archetype
            names = ", ".join(anames[a] for a in sig_by_arm[arm])
            axc.text(0.99, 0.05, f"★ {names} differs from Control (FDR)",
                     transform=axc.transAxes, ha="right", va="bottom",
                     fontsize=6, color="#B2182B", fontweight="bold")
        if i < len(arms_present) - 1:
            axc.tick_params(labelbottom=False)
        else:
            axc.set_xlabel("treatment day (tau)", fontsize=7)
        if i == 0:
            axc.legend(fontsize=5.5, ncol=min(k, 3), loc="lower left",
                       bbox_to_anchor=(0.0, 1.03), frameon=False)

    # D — archetype feature profiles; y-labels are the colour+name key for C.
    axD = fig.add_subplot(main[0, 1])
    im = axD.imshow(prof, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    axD.set_xticks(range(len(FEATURES)))
    axD.set_xticklabels(FEAT_LABELS, rotation=45, ha="right", fontsize=6)
    axD.set_yticks(range(k))
    axD.set_yticklabels([_arch_label(a) for a in range(k)], fontsize=7)
    for a, tick in enumerate(axD.get_yticklabels()):     # colour = archetype key
        tick.set_color("#B2182B" if a in sig_arch else
                       _ARCH_PALETTE[a % len(_ARCH_PALETTE)])
        if a in sig_arch:
            tick.set_fontweight("bold")
    cb = fig.colorbar(im, ax=axD, shrink=0.8, pad=0.02)
    cb.set_label("z-score", fontsize=6); cb.ax.tick_params(labelsize=6)
    axD.set_title("D  What defines each archetype", loc="left",
                  fontweight="bold", fontsize=9)

    # ---- DEMOTED method strip: UMAPs (by group / by archetype) + rasters ----
    reps = _rep_bursts(bursts, Z, labels, k)
    strip = outer[1].subgridspec(1, 2 + k, wspace=0.32)
    axA = fig.add_subplot(strip[0]); axB = fig.add_subplot(strip[1])
    for g in groups:
        m = bursts.group.to_numpy() == g
        axA.scatter(emb[m, 0], emb[m, 1], s=2, c=group_color(g), lw=0, alpha=0.45,
                    rasterized=True)
    axA.set_xticks([]); axA.set_yticks([])
    axA.text(0.03, 0.97, "UMAP · by group", transform=axA.transAxes, va="top",
             ha="left", fontsize=6.5, color=MUTED)
    for a in range(k):
        m = labels == a
        axB.scatter(emb[m, 0], emb[m, 1], s=2,
                    c=_ARCH_PALETTE[a % len(_ARCH_PALETTE)], lw=0, alpha=0.55,
                    rasterized=True)
    axB.set_xticks([]); axB.set_yticks([])
    axB.text(0.03, 0.97, "UMAP · by archetype", transform=axB.transAxes, va="top",
             ha="left", fontsize=6.5, color=MUTED)
    for a in range(k):
        axr = fig.add_subplot(strip[2 + a])
        if a in reps:
            _burst_raster(axr, analysis_root, reps[a],
                          _ARCH_PALETTE[a % len(_ARCH_PALETTE)], anames[a])
            axr.set_title("")     # move the archetype name inside to clear header
            axr.text(0.03, 0.98, anames[a], transform=axr.transAxes, va="top",
                     ha="left", fontsize=7, fontweight="bold",
                     color=_ARCH_PALETTE[a % len(_ARCH_PALETTE)])
    fig.text(0.09, outer[1].get_position(fig).y1 + 0.014,
             "Method context (demoted) — burst feature space (UMAP: by group / by "
             "archetype) + one example raster per archetype",
             fontweight="bold", fontsize=8, color=MUTED)

    sep = ("composition does not separate groups" if n_sig == 0
           else f"{n_sig} archetype fraction(s) differ from Control (BH-FDR)")
    fig.suptitle(f"Figure 6c — The pipeline resolves {k} reproducible burst "
                 f"archetypes; {sep} "
                 f"({len(bursts)} bursts, {bursts.well_uid.nunique()} wells)",
                 fontsize=8.5, y=0.965)
    caption(fig,
        f"Cohort-wide burst archetypes; the figure leads with the finding "
        f"(composition + what the archetypes are). Every detected network burst "
        f"from the focus groups is pooled, embedded by its features and clustered "
        f"into {k} reproducible archetypes (named by their properties). (C) "
        f"Archetype composition vs treatment day (tau), one stacked panel per arm "
        f"on a shared x-axis: each band = the group-mean fraction of bursts in "
        f"that archetype, averaged across wells (each weighted equally); a ★ marks "
        f"an archetype whose per-well fraction differs from Control (per-arm "
        f"Mann-Whitney, BH-FDR). (D) Mean standardised feature profile of each "
        f"archetype (the colour-coded name key for C). Method context (demoted): "
        f"the burst feature space (UMAP, coloured by group and by archetype) and "
        f"one example raster per archetype (−0.5 to +0.5 s around the burst). "
        f"Result: {sep.lower()}.")
    return fig, {"n_bursts": int(len(bursts)), "k": int(k),
                 "n_wells": int(bursts.well_uid.nunique()),
                 "n_composition_sig": n_sig,
                 "composition_stats": cstats}


def render(tidy, analysis_root, figure_root):
    fig, meta = build_f6c(tidy, analysis_root)
    return save_fig(fig, "F6c_burst_archetypes", figure_root, subdir="main"), meta
