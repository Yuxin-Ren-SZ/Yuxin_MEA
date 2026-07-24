"""Shared burst-archetype fit for the F6 panels.

Clustering every detected burst into cohort-wide archetypes takes minutes: it
walks the per-recording burst tables across ~120 well-recordings and runs a
silhouette sweep over k. Three panels read that one fit (composition, feature
profile, example rasters), so it is computed once and cached rather than
recomputed per script.

The burst-subtype UMAP that used to accompany these panels has been dropped — it
did not separate anything interpretable — so no embedding is fitted here. The
clustering itself never used the embedding; it runs on the standardised features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Qualitative archetype colours, deliberately distinct from the group palette
#: (grey/amber/sienna) and the rest/burst state palette, so an archetype colour
#: can never be mistaken for a treatment arm.
ARCH_PALETTE = ["#44AA99", "#882255", "#999933", "#AA4499", "#117733", "#DDCC77"]


@dataclass
class Archetypes:
    bursts: pd.DataFrame          # one row per burst, with an ``arch`` label
    Z: np.ndarray                 # standardised feature matrix (rows = bursts)
    profile: np.ndarray           # (k, n_features) mean z-scored profile
    names: list                   # descriptive archetype names
    k: int
    composition: pd.DataFrame     # per-well archetype fractions
    stats: pd.DataFrame           # per-archetype arm-vs-Control MWU + BH-FDR


def _fit(ctx, seed: int = 0) -> Archetypes:
    from sklearn.preprocessing import StandardScaler

    from ..fig6c_bursttypes import (
        FEATURES, FEAT_LABELS, LOG_FEATURES, _cluster, collect_bursts,
        name_archetypes,
    )
    from ..report_style import ordered_groups

    bursts = collect_bursts(ctx.tidy, ctx.analysis_root, seed=seed)
    if bursts.empty:
        raise RuntimeError("no bursts collected for the archetype fit")
    X = bursts[FEATURES].to_numpy(float)
    for j, f in enumerate(FEATURES):          # tame the positively-skewed metrics
        if f in LOG_FEATURES:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    Z = StandardScaler().fit_transform(X)
    labels, k = _cluster(Z, seed=seed)
    bursts = bursts.assign(arch=labels)
    profile = np.vstack([Z[labels == a].mean(0) for a in range(k)])
    names = name_archetypes(profile, FEAT_LABELS)

    # ``group`` is the well's treatment arm, assigned once in ``collect_bursts``
    # from ``well_index``. The panels bin the same column, so the stars and the
    # bands are keyed off one labelling — they used to disagree, because
    # selection ran on the per-recording chip groupname and the plot relabelled
    # per well.
    comp = (bursts.groupby(["well_uid", "group", "arch"]).size()
            .unstack("arch", fill_value=0))
    comp = comp.div(comp.sum(axis=1), axis=0).reset_index()
    groups = ordered_groups(bursts.group.unique())
    stats = composition_stats_chip(comp, k, list(groups))
    return Archetypes(bursts, Z, profile, names, k, comp, stats)


def composition_stats_chip(comp: pd.DataFrame, k: int, groups: list) -> pd.DataFrame:
    """Per-archetype composition test at the **chip** level.

    The inherited `fig6c._composition_stats` compares per-*well* fractions with a
    Mann-Whitney across wells. That is the pseudo-replication the rest of the
    report removes: wells inside a chip share one plating and one CSF
    application, so a well-level test has an n of ~15 where the biology has 3.
    It also made this figure claim an FDR-significant archetype while every other
    figure correctly reported that nothing survives correction — the same ★
    meaning two different things.

    Here each chip's wells are averaged into one fraction per archetype, and each
    treated arm is compared with Control across chips (Welch t-test, n = 3 vs 3),
    BH-FDR over all archetype × arm tests. With three chips per arm this is a very
    low-powered test — which is the honest position, not a reason to go back to
    counting wells.
    """
    from scipy.stats import ttest_ind

    from .. import stats as S

    chip = comp.copy()
    chip["chip"] = chip.well_uid.str.split("|").str[0]
    per_chip = chip.groupby(["chip", "group"], as_index=False).mean(numeric_only=True)
    rows = []
    for a in range(k):
        if a not in per_chip.columns:
            continue
        c = per_chip.loc[per_chip.group == "Control", a].dropna().to_numpy(float)
        for g in groups:
            if g == "Control":
                continue
            x = per_chip.loc[per_chip.group == g, a].dropna().to_numpy(float)
            p = np.nan
            if len(x) >= 2 and len(c) >= 2:
                p = float(ttest_ind(x, c, equal_var=False).pvalue)
            rows.append(dict(arch=a, arm=g, p=p, n_chips_arm=len(x),
                             n_chips_ctrl=len(c),
                             mean_arm=float(np.mean(x)) if len(x) else np.nan,
                             mean_ctrl=float(np.mean(c)) if len(c) else np.nan))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q"] = S._bh_fdr(df.p.to_numpy())
    return df


def archetypes(ctx, seed: int = 0) -> Archetypes:
    """The cached cohort-wide burst-archetype fit."""
    return ctx.cache.get(f"archetypes|{seed}", lambda: _fit(ctx, seed))


def arch_color(a: int) -> str:
    return ARCH_PALETTE[a % len(ARCH_PALETTE)]


def representative_bursts(arch: Archetypes, a: int, n: int = 3) -> pd.DataFrame:
    """The ``n`` bursts closest to archetype ``a``'s centroid, spread over chips.

    One example per archetype invites the reader to treat a single burst as the
    definition; several show the within-archetype spread that the feature profile
    only summarises. Taking the ``n`` nearest outright defeats that: the nearest
    neighbours of a centroid tend to come from one culture, and twice from one
    well, so the reader sees one well's burst three times and reads it as the
    archetype. Prefer a different chip for each example, then a different well,
    and only then fall back on distance alone.
    """
    idx = np.where(arch.bursts.arch.to_numpy() == a)[0]
    if not len(idx):
        return arch.bursts.iloc[[]]
    centre = arch.Z[idx].mean(0)
    order = idx[np.argsort(((arch.Z[idx] - centre) ** 2).sum(1))]
    uid = arch.bursts.well_uid.to_numpy()
    chip = np.asarray([u.split("|")[0] for u in uid])

    picked: list[int] = []
    for keyed in (chip, uid, None):        # one per chip, then per well, then any
        seen = set() if keyed is None else {keyed[i] for i in picked}
        for i in order:
            if len(picked) >= n:
                break
            if i in picked:
                continue
            k = None if keyed is None else keyed[i]
            if k is not None and k in seen:
                continue
            seen.add(k)
            picked.append(i)
        if len(picked) >= n:
            break
    return arch.bursts.iloc[picked]


def arch_label(arch: Archetypes, a: int) -> str:
    """Archetype name plus its significance glyph, on the report's shared scale.

    Same two tiers as every other panel (``*`` from the BH-FDR q, ``△`` from the
    uncorrected p), read from the chip-level composition test — so a mark here
    means what a mark means everywhere else in the report.
    """
    from ..report_style import sig_glyph

    name = arch.names[a]
    if arch.stats.empty:
        return name
    sub = arch.stats[arch.stats.arch == a]
    if sub.empty:
        return name
    best = sub.loc[sub.p.idxmin()] if sub.p.notna().any() else None
    if best is None:
        return name
    glyph, _ = sig_glyph(best.p, best.q)
    return f"{name} {glyph}".strip()
