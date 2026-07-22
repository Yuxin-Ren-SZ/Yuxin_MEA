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
from .report_style import group_color, ordered_groups, save_fig

FOCUS_GROUPS = ["Control", "IVH_Early", "IVH_Late", "H2O2_20uM"]
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
            b = b[FEATURES].copy()
            b["group"] = grp
            b["well_uid"] = r["well_uid"]
            frames.append(b)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = out.dropna(subset=FEATURES)
    if len(out) > _MAX_BURSTS:
        out = out.sample(_MAX_BURSTS, random_state=seed).reset_index(drop=True)
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

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0))
    groups = [g for g in ordered_groups(bursts.group.unique())]

    # A — coloured by group
    ax = axes[0, 0]
    for g in groups:
        m = bursts.group.to_numpy() == g
        ax.scatter(emb[m, 0], emb[m, 1], s=3, c=group_color(g), lw=0, alpha=0.5,
                   label=g, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=6, markerscale=2, loc="best")
    ax.set_title("A  Burst feature space by group", loc="left", fontweight="bold")

    # B — coloured by global archetype
    ax = axes[0, 1]
    for a in range(k):
        m = labels == a
        ax.scatter(emb[m, 0], emb[m, 1], s=3, c=_ARCH_PALETTE[a % len(_ARCH_PALETTE)],
                   lw=0, alpha=0.6, label=f"type {a}", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP-1")
    ax.legend(fontsize=6, markerscale=2, loc="best", title=f"archetype (k={k})",
              title_fontsize=6)
    ax.set_title("B  Global burst archetypes", loc="left", fontweight="bold")

    # C — per-WELL archetype composition (well = unit; grouped boxplots).
    # Burst-level pooling would be pseudo-replication; per-well shows whether the
    # groups actually overlap. The "n.s." is COMPUTED here (per-archetype MWU each
    # treated arm vs Control, BH-FDR), not asserted — any q<0.05 is starred and the
    # title/suptitle switch to reflect it.
    ax = axes[1, 0]
    wc = (bursts.groupby(["well_uid", "group", "arch"]).size()
          .unstack("arch", fill_value=0))
    wc = wc.div(wc.sum(axis=1), axis=0).reset_index()      # per-well fractions
    cstats = _composition_stats(wc, k, groups)
    qlook = {(int(r.arch), r.arm): r.q for r in cstats.itertuples()} if not cstats.empty else {}
    n_sig = int((cstats.q < 0.05).sum()) if not cstats.empty else 0
    n_g = len(groups)
    off = np.linspace(-0.3, 0.3, n_g)
    for a in range(k):
        for gi, g in enumerate(groups):
            v = wc.loc[wc.group == g, a].dropna().to_numpy() if a in wc.columns else np.array([])
            if len(v) == 0:
                continue
            pos = a + off[gi]
            bp = ax.boxplot(v, positions=[pos], widths=0.14, patch_artist=True,
                            showfliers=False)
            bp["boxes"][0].set(facecolor=group_color(g), alpha=0.5, lw=0.5)
            bp["medians"][0].set(color="k", lw=0.8)
            q = qlook.get((a, g), np.nan)
            if not np.isnan(q) and q < 0.05:
                ax.text(pos, min(v.max() + 0.04, 0.98), _stars(q), ha="center",
                        va="bottom", fontsize=7, color=group_color(g))
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"type {a}" for a in range(k)])
    ax.set_ylabel("per-well fraction")
    ax.set_ylim(0, 1)
    ax.set_xlabel("well = unit; MWU vs Control, BH-FDR"
                  + ("  (all n.s.)" if n_sig == 0 else f"  ({n_sig} q<0.05)"),
                  fontsize=6)
    handles = [ax.plot([], [], marker="s", ls="", color=group_color(g), label=g)[0]
               for g in groups]
    ax.legend(handles=handles, fontsize=5.5, loc="upper right", ncol=2)
    ax.set_title("C  Composition by well "
                 + ("(n.s.)" if n_sig == 0 else f"({n_sig} sig)"),
                 loc="left", fontweight="bold")

    # D — archetype feature profiles (standardised means)
    ax = axes[1, 1]
    prof = np.vstack([Z[labels == a].mean(0) for a in range(k)])
    im = ax.imshow(prof, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEAT_LABELS, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(k)); ax.set_yticklabels([f"type {a}" for a in range(k)], fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("z-score", fontsize=6); cb.ax.tick_params(labelsize=6)
    ax.set_title("D  Archetype feature profiles", loc="left", fontweight="bold")

    sep = ("composition does not separate groups" if n_sig == 0
           else f"{n_sig} archetype fraction(s) differ from Control (BH-FDR)")
    fig.suptitle(f"Figure 6c — The pipeline resolves {k} reproducible burst "
                 f"archetypes; {sep} "
                 f"({len(bursts)} bursts, {bursts.well_uid.nunique()} wells)",
                 fontsize=8.5, y=1.0)
    fig.tight_layout()
    return fig, {"n_bursts": int(len(bursts)), "k": int(k),
                 "n_wells": int(bursts.well_uid.nunique()),
                 "n_composition_sig": n_sig,
                 "composition_stats": cstats}


def render(tidy, analysis_root, figure_root):
    fig, meta = build_f6c(tidy, analysis_root)
    return save_fig(fig, "F6c_burst_archetypes", figure_root, subdir="main"), meta
