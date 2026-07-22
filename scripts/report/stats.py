"""Correct, well-level statistics for the report.

Why not reuse ``<figure_root>/treatment_comparison/stats.csv``?
------------------------------------------------------------------
That table was produced by ``compare_treatment_groups.py`` pairing on the
per-recording ``role`` column. But this dataset uses a **within-well pre/post
design**: a treated well is labelled ``Control`` on its baseline recordings
(``tau < 0``) and its treatment name on post recordings (``tau > 0``). Pairing
on ``role`` therefore mis-files every baseline as a control well and collapses
the paired n to ~3/arm. Recomputing on ``tau`` sign recovers the real paired n
(7-13 wells, 2-3 chips) for the main arms.

Unit of analysis
----------------
The **physical well** (``well_uid``), never the recording. Every statistic
aggregates a well's recordings first (median), so recordings are not treated as
independent samples (pseudo-replication). The chip (``sample_id``) is the
biological replicate; single-chip arms (AraC, NPH) are flagged exploratory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Metric families (mirrors METRIC_SPECS in compare_treatment_groups.py).
RATIO_METRICS = [
    "nb_rate", "nb_count", "nb_duration_mean", "nb_spikes_per_burst_mean",
    "nb_ibi_mean", "median_firing_rate", "n_curated",
    # Spatial / connectivity (non-negative -> log2 ratio valid)
    "activity_gini", "mean_prop_speed_um_ms", "edge_density", "small_worldness",
    "clustering_coeff", "global_efficiency",
    # Node-level graph metrics (non-negative, comfortably > 0)
    "participation_mean", "degree_cv", "rich_club",
    # Criticality (exponents / non-negative)
    "aval_tau", "aval_alpha", "gamma_fit",
    # Directed / TE (non-negative, comfortably > 0)
    "te_edge_density", "degree_asymmetry",
]
# Bounded, can be <=0, or sit near 0 (log2 ratio unstable) -> diff, not ratio.
DIFF_METRICS = [
    "burst_modulation_index", "burst_type_k", "cluster_n_clusters",
    "mean_sttc", "modularity",
    "assortativity",                                   # [-1, 1]
    "branching_ratio_mr", "branching_ratio_naive", "dcc",   # ~1 / ~0
    "reciprocity", "flow_hierarchy",                   # [0, 1], can be 0
    # near-zero fractions / TE — log2 ratio explodes, so use post-pre difference
    "hub_fraction", "leaf_fraction", "mean_betweenness", "mean_te",
]
ALL_METRICS = RATIO_METRICS + DIFF_METRICS

_EPS = 1e-9
# Power thresholds for promoting an arm to a "confirmatory" main-figure claim.
MIN_CHIPS_CONFIRMATORY = 2
MIN_WELLS_CONFIRMATORY = 5


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaN entries pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * m / (np.arange(1, m + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]  # enforce monotonicity
    q[order] = np.clip(ranked, 0, 1)
    return q


def arm_of(groups: pd.Series) -> str:
    """A physical well's treatment identity = the non-Control group it takes."""
    non = [g for g in groups if g != "Control"]
    return non[0] if non else "Control"


def well_index(tidy: pd.DataFrame) -> pd.DataFrame:
    """One row per physical well: chip, arm, pre/post availability, n_chips."""
    g = tidy.groupby("well_uid")
    wi = g.agg(
        chip=("sample_id", "first"),
        arm=("canonical_group", arm_of),
        has_pre=("tau", lambda t: bool((t < 0).any())),
        has_post=("tau", lambda t: bool((t > 0).any())),
        n_rec=("tau", "size"),
    ).reset_index()
    wi["paired"] = wi.has_pre & wi.has_post
    return wi


# --------------------------------------------------------------------------- #
# Within-well pre/post response
# --------------------------------------------------------------------------- #
def well_response(tidy: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Per paired well × metric: baseline vs post + response.

    baseline = median over the well's ``tau < 0`` recordings,
    post     = median over its ``tau > 0`` recordings.
    response = log2((post+eps)/(baseline+eps)) for ratio metrics,
               (post - baseline)               for diff metrics.

    **Control wells are included** (their response is the maturation-only change
    over the same pre→post interval). This is essential: treatment effects must
    be read as the *difference-in-differences* vs Control (see
    :func:`arm_response_vs_control`), because the raw pre→post change is
    confounded with development. Only wells with ≥1 pre and ≥1 post are returned.
    """
    metrics = metrics or ALL_METRICS
    wi = well_index(tidy)
    paired = set(wi.loc[wi.paired, "well_uid"])
    rows = []
    for wuid, sub in tidy.groupby("well_uid"):
        if wuid not in paired:
            continue
        arm = arm_of(sub.canonical_group)
        chip = sub.sample_id.iloc[0]
        pre = sub[sub.tau < 0]
        post = sub[sub.tau > 0]
        for m in metrics:
            b = pre[m].median()
            p = post[m].median()
            if pd.isna(b) or pd.isna(p):
                continue
            if m in DIFF_METRICS:
                resp = p - b
            else:
                resp = np.log2((p + _EPS) / (b + _EPS))
            rows.append(dict(
                well_uid=wuid, chip=chip, arm=arm, metric=m,
                baseline=b, post=p, response=resp,
                n_pre=len(pre), n_post=len(post),
            ))
    return pd.DataFrame(rows)


def _wilcoxon_vs_zero(x: np.ndarray) -> float:
    """Wilcoxon signed-rank p that response ≠ 0; NaN if too few/degenerate."""
    from scipy.stats import wilcoxon

    x = x[~np.isnan(x)]
    if len(x) < 5 or np.allclose(x, 0):
        return np.nan
    try:
        return float(wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def _boot_median_ci(x: np.ndarray, seed: int = 0, n: int = 5000) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def arm_response_summary(resp: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Per arm × metric: n wells/chips, median response, Wilcoxon p, boot CI, tier."""
    rows = []
    for (arm, m), sub in resp.groupby(["arm", "metric"]):
        x = sub.response.to_numpy(dtype=float)
        n_wells = len(sub)
        n_chips = sub.chip.nunique()
        lo, hi = _boot_median_ci(x, seed=seed)
        rows.append(dict(
            arm=arm, metric=m, n_wells=n_wells, n_chips=n_chips,
            median_response=float(np.nanmedian(x)),
            ci_low=lo, ci_high=hi,
            p_wilcoxon=_wilcoxon_vs_zero(x),
            tier=("confirmatory"
                  if (n_chips >= MIN_CHIPS_CONFIRMATORY and n_wells >= MIN_WELLS_CONFIRMATORY)
                  else "exploratory"),
        ))
    out = pd.DataFrame(rows)
    # BH-FDR across confirmatory tests only (exploratory left raw).
    out["q_bh"] = np.nan
    conf = (out.tier == "confirmatory") & out.p_wilcoxon.notna()
    if conf.any():
        pv = out.p_wilcoxon.where(conf, np.nan).to_numpy()
        out["q_bh"] = _bh_fdr(pv)
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size (a vs b); NaN if either side empty."""
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum((ai > b).sum() for ai in a)
    lt = sum((ai < b).sum() for ai in a)
    return float((gt - lt) / (len(a) * len(b)))


def arm_response_vs_control(resp: pd.DataFrame) -> pd.DataFrame:
    """Difference-in-differences: each treated arm's pre→post response vs the
    Control arm's response (maturation baseline) per metric.

    This is the confound-controlled treatment test — the raw within-well change
    (:func:`arm_response_summary` Wilcoxon-vs-zero) is confounded with
    development. Mann-Whitney arm-vs-Control + Cliff's delta, BH-FDR over the
    confirmatory arms.
    """
    from scipy.stats import mannwhitneyu

    ctrl = resp[resp.arm == "Control"]
    rows = []
    for (arm, m), sub in resp.groupby(["arm", "metric"]):
        if arm == "Control":
            continue
        x = sub.response.to_numpy(float)
        c = ctrl.loc[ctrl.metric == m, "response"].to_numpy(float)
        n_wells, n_chips = len(sub), sub.chip.nunique()
        p = np.nan
        if len(x) >= 3 and len(c) >= 3:
            try:
                p = float(mannwhitneyu(x, c, alternative="two-sided").pvalue)
            except ValueError:
                p = np.nan
        rows.append(dict(
            arm=arm, metric=m, n_wells=n_wells, n_chips=n_chips,
            median_arm=float(np.nanmedian(x)) if len(x) else np.nan,
            median_ctrl=float(np.nanmedian(c)) if len(c) else np.nan,
            did=(float(np.nanmedian(x) - np.nanmedian(c))
                 if len(x) and len(c) else np.nan),
            cliffs_delta=cliffs_delta(x, c),
            p_vs_control=p,
            tier=("confirmatory"
                  if (n_chips >= MIN_CHIPS_CONFIRMATORY and n_wells >= MIN_WELLS_CONFIRMATORY)
                  else "exploratory"),
        ))
    out = pd.DataFrame(rows)
    out["q_bh"] = np.nan
    conf = (out.tier == "confirmatory") & out.p_vs_control.notna()
    if conf.any():
        pv = out.p_vs_control.where(conf, np.nan).to_numpy()
        out["q_bh"] = _bh_fdr(pv)
    return out


# --------------------------------------------------------------------------- #
# Cross-group at matched developmental stage (DIV window)
# --------------------------------------------------------------------------- #
def matched_div_wells(
    tidy: pd.DataFrame, lo: int, hi: int, metrics: list[str] | None = None,
) -> pd.DataFrame:
    """One row per physical well: median of each metric over recordings in
    [lo, hi] DIV. Well is assigned to its treatment arm (post identity)."""
    metrics = metrics or ALL_METRICS
    win = tidy[(tidy.DIV >= lo) & (tidy.DIV <= hi)]
    rows = []
    for wuid, sub in win.groupby("well_uid"):
        arm = arm_of(sub.canonical_group)
        rec = dict(well_uid=wuid, chip=sub.sample_id.iloc[0], arm=arm,
                   n_rec=len(sub))
        for m in metrics:
            rec[m] = sub[m].median()
        rows.append(rec)
    return pd.DataFrame(rows)


def mwu_vs_control(matched: pd.DataFrame, metric: str, seed: int = 0) -> pd.DataFrame:
    """Mann-Whitney each treated arm vs Control wells for one metric, BH-FDR."""
    from scipy.stats import mannwhitneyu

    ctrl = matched.loc[matched.arm == "Control", metric].dropna().to_numpy()
    rows = []
    for arm, sub in matched.groupby("arm"):
        if arm == "Control":
            continue
        x = sub[metric].dropna().to_numpy()
        n_wells, n_chips = len(x), sub.chip.nunique()
        p = np.nan
        if len(x) >= 3 and len(ctrl) >= 3:
            try:
                p = float(mannwhitneyu(x, ctrl, alternative="two-sided").pvalue)
            except ValueError:
                p = np.nan
        rows.append(dict(
            arm=arm, metric=metric, n_wells=n_wells, n_chips=n_chips,
            median_arm=float(np.nanmedian(x)) if len(x) else np.nan,
            median_ctrl=float(np.nanmedian(ctrl)) if len(ctrl) else np.nan,
            p_mwu=p,
            tier=("confirmatory"
                  if (n_chips >= MIN_CHIPS_CONFIRMATORY and n_wells >= MIN_WELLS_CONFIRMATORY)
                  else "exploratory"),
        ))
    out = pd.DataFrame(rows)
    out["q_bh"] = np.nan
    conf = (out.tier == "confirmatory") & out.p_mwu.notna()
    if conf.any():
        pv = out.p_mwu.where(conf, np.nan).to_numpy()
        out["q_bh"] = _bh_fdr(pv)
    return out
