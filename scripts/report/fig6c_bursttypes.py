"""F6c — Cohort-wide burst-type clustering across groups and wells.

The pipeline's per-well burst typing (``ml_burst_typing``) assigns each burst an
integer type, but those identities are **per-well and not comparable across
wells** — a single well's clustering can't reveal cohort structure. This figure
pools every detected burst from many wells across the focus groups, embeds them
in one shared feature space, and clusters them **globally** into cohort-wide
burst archetypes.

Result (well = unit of analysis): k is chosen by silhouette rather than fixed, so
the number of archetypes is whatever the pooled bursts support — read it off the
figure, not from this docstring. It is not stable against how much data the fit
sees: on a 40-recording-per-group subsample the sweep settled on 4, and on the
full pool it settles on fewer, which is itself a reason to treat the archetype
identities as descriptive rather than as discovered cell-level categories.
Whether their **composition separates the groups** is *computed in
panel C* — a per-archetype Mann-Whitney of each treated arm's per-well fractions
vs Control, BH-FDR (`_composition_stats`) — never asserted: any q<0.05 is starred
and the panel title / figure suptitle switch to report the actual count. This is
deliberately not a hardcoded "n.s." label. An earlier subsampled fit produced one
FDR-clearing contrast (IVH_Late, archetype 1, q≈0.04); it did not survive fitting
on every recording, which is what that "sensitive to k and the sampling seed"
caveat was warning about.
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

import logging

import numpy as np
import pandas as pd

from . import load as L
from . import stats as S

logger = logging.getLogger("report.fig6c")

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
#: Runaway guard only. The fit loads every eligible well-recording (~823 rows,
#: ~18k bursts on the current cohort); this cap exists so a future cohort cannot
#: silently exhaust memory. It must stay well above the real pool, because
#: subsampling here re-introduces the very bias this module used to have: a flat
#: per-group draw made pre-treatment recordings — 6-15% of each pool — a matter of
#: luck, so one arm could show a tau<0 band and another none, from the same data.
_MAX_BURSTS = 200_000
#: Minimum network bursts in a recording for it to contribute to the fit.
_MIN_NB_COUNT = 5
# Dedicated qualitative archetype palette — deliberately distinct from the group
# palette (amber/sienna/grey) AND the network-state palette (rest #3D6FB4 /
# burst #D1495B), so archetype colour is never confused with either.
_ARCH_PALETTE = ["#44AA99", "#882255", "#999933", "#AA4499", "#117733", "#DDCC77"]


def _stars(q: float) -> str:
    if np.isnan(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def collect_bursts(tidy, analysis_root, seed: int = 0) -> pd.DataFrame:
    """Pool per-burst feature rows across **every** eligible well of the focus arms.

    Selection is on the well's *treatment identity* (``stats.arm_of`` via
    ``well_index``), not on the per-row ``canonical_group``. The two are not the
    same thing: ``canonical_group`` is the groupname the MaxWell chip carried at
    scan time, and on CX138 every well — treated or not — was scanned as
    ``Control`` until the day of treatment. Selecting on it therefore hides
    treated wells' baselines inside the Control pool, and any panel that later
    relabels by well (as F6c does) ends up plotting bursts it never asked for.

    No recording is subsampled. ``seed`` is still threaded through for the
    downstream KMeans/silhouette in :func:`_cluster`, so the fit stays
    reproducible.
    """
    wi = S.well_index(tidy)[["well_uid", "arm"]]
    d = tidy.merge(wi, on="well_uid", how="left")
    frames, skipped = [], []
    for grp in FOCUS_GROUPS:
        rows = d[(d.arm == grp) & (d.nb_count >= _MIN_NB_COUNT)]
        for _, r in rows.iterrows():
            try:
                b = L.load_ml_bursts(analysis_root, r)
            except FileNotFoundError:
                b = None
            if b is None or b.empty or not set(FEATURES).issubset(b.columns):
                skipped.append((grp, int(r["tau"])))
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
        logger.warning(
            "burst cap %d binding on a pool of %d — the fit is a subsample again; "
            "raise _MAX_BURSTS", _MAX_BURSTS, len(out))
        out = out.sample(_MAX_BURSTS, random_state=seed).reset_index(drop=True)
    out.attrs["selection"] = _selection_summary(d, out, skipped)
    return out


def _selection_summary(d: pd.DataFrame, out: pd.DataFrame, skipped: list) -> pd.DataFrame:
    """What the fit is built on, and what the eligibility gates removed.

    Written next to the panel so a reader can tell an arm that genuinely has no
    pre-treatment data from one whose baselines were filtered away.
    """
    sk = pd.DataFrame(skipped, columns=["arm", "tau"])
    key = ["well_uid", "recording_key", "rec_name"]
    rows = []
    for grp in FOCUS_GROUPS:
        pool = d[d.arm == grp]
        for pre, sub in pool.groupby(pool.tau < 0):
            used = out[(out.group == grp) & ((out.tau < 0) == pre)] if len(out) else out
            miss = sk[(sk.arm == grp) & ((sk.tau < 0) == pre)]
            rows.append(dict(
                arm=grp, pre_treatment=bool(pre),
                recs_total=len(sub),
                # NaN nb_count fails the gate too, so negate the gate rather than
                # counting rows below it.
                recs_below_nb_count=int((~(sub.nb_count >= _MIN_NB_COUNT)).sum()),
                recs_no_burst_file=len(miss),
                recs_used=int(used[key].drop_duplicates().shape[0]) if len(used) else 0,
                wells_used=int(used.well_uid.nunique()) if len(used) else 0,
                bursts=len(used)))
    return pd.DataFrame(rows)


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


_SIL_DRAWS, _SIL_SIZE = 5, 3000


def _cluster(Z: np.ndarray, seed: int = 0) -> tuple[np.ndarray, int]:
    """KMeans on every burst, with k chosen by silhouette (k = 2..6).

    The silhouette is O(n²), so it is scored on subsamples rather than the whole
    pool — but on a single draw the winning k is partly a property of that draw.
    Averaging several disjoint draws makes the choice a property of the data.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    n = min(_SIL_SIZE, len(Z))
    draws = [rng.choice(len(Z), n, replace=False) for _ in range(_SIL_DRAWS)]
    best = None
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        scores = [silhouette_score(Z[i], km.labels_[i]) for i in draws
                  if len(np.unique(km.labels_[i])) > 1]
        if not scores:
            continue
        s = float(np.mean(scores))
        logger.debug("k=%d silhouette %.4f (%d draws)", k, s, len(scores))
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
