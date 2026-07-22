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
#
# The classification rule is *not* "non-negative -> ratio". A log2 ratio needs
# values that are reliably **strictly positive**. When baseline or post is
# exactly 0 the ratio collapses onto ``_EPS`` and the response saturates at
# ~+-30 log2 units — an artifact of the guard, not an effect. So any metric that
# is exactly zero in a non-trivial share of wells belongs in DIFF_METRICS, even
# though it is non-negative.
#
# Ratio is kept where zero is *rare and meaningful* (a rate/count going to zero
# really is a floor, and log2 flooring is the intended behaviour), and for
# quantities bounded well away from zero (MLE exponents, small-worldness sigma).
RATIO_METRICS = [
    # Rates / counts / durations — zero means "activity stopped", flooring is
    # deliberate.
    "nb_rate", "nb_count", "nb_duration_mean", "nb_spikes_per_burst_mean",
    "nb_ibi_mean", "median_firing_rate", "n_curated",
    # Spatial — strictly positive in practice.
    "activity_gini", "mean_prop_speed_um_ms",
    # Sigma is itself a ratio, bounded away from 0.
    "small_worldness",
    # Criticality exponents — MLE fits, empirically 1.2-3.0.
    "aval_tau", "aval_alpha", "gamma_fit",
]
# Bounded, signed, or frequently exactly zero -> diff, not ratio.
#
# Zero-share on the current cohort (n non-null wells) is given per metric; each
# of these produced |response| up to ~30 log2 units when classified as a ratio.
DIFF_METRICS = [
    "burst_modulation_index", "burst_type_k", "cluster_n_clusters",
    "mean_sttc", "modularity",
    "assortativity",                                   # [-1, 1]
    "branching_ratio_mr", "branching_ratio_naive", "dcc",   # ~1 / ~0
    "reciprocity", "flow_hierarchy",                   # [0, 1], can be 0
    # STTC graph descriptors — all 0 when a well's thresholded graph has no
    # edges: edge_density/global_efficiency 86/1843, clustering_coeff 123/1843.
    "edge_density", "clustering_coeff", "global_efficiency",
    # Node-level graph metrics — bounded fractions/coefficients, often exactly
    # zero: hub_fraction 1364/1516, leaf_fraction 454/1516,
    # participation_mean 108/1516, degree_cv 66/1516.
    "hub_fraction", "leaf_fraction", "mean_betweenness", "participation_mean",
    "degree_cv", "rich_club",
    # Directed / TE — zero whenever the thresholded directed graph is empty.
    "mean_te", "te_edge_density", "degree_asymmetry",
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


# Columns every stats table carries, so an empty result still has a usable shape.
_SUMMARY_COLUMNS = ["arm", "metric", "n_wells", "n_chips", "median_response",
                    "ci_low", "ci_high", "p_wilcoxon", "tier", "q_bh"]
_VS_CONTROL_COLUMNS = ["arm", "metric", "n_wells", "n_chips", "median_arm",
                       "median_ctrl", "did", "cliffs_delta", "p_vs_control",
                       "tier", "q_bh"]
_MWU_COLUMNS = ["arm", "metric", "n_wells", "n_chips", "median_arm",
                "median_ctrl", "p_mwu", "tier", "q_bh"]


def _finalize(rows: list[dict], p_col: str, columns: list[str]) -> pd.DataFrame:
    """Assemble a stats table and BH-adjust the confirmatory tests.

    An empty ``rows`` is normal, not an error: it just means no arm had enough
    paired wells for this metric — e.g. an arm the node/criticality metrics were
    never computed for. Returning a correctly-shaped empty frame keeps callers
    from having to special-case it (they would otherwise hit ``.tier`` on a bare
    DataFrame).
    """
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    out["q_bh"] = np.nan
    conf = (out.tier == "confirmatory") & out[p_col].notna()
    if conf.any():
        out["q_bh"] = _bh_fdr(out[p_col].where(conf, np.nan).to_numpy())
    return out


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

    The returned frame carries an ``eps_floored`` flag: True when a *ratio*
    metric had a zero baseline or post, so its response came from the ``_EPS``
    guard rather than from the data. Those rows are legitimate floors for rates
    and counts, but a metric with many of them is misclassified — see the
    :data:`RATIO_METRICS` / :data:`DIFF_METRICS` note.
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
            is_ratio = m not in DIFF_METRICS
            if is_ratio:
                resp = np.log2((p + _EPS) / (b + _EPS))
            else:
                resp = p - b
            rows.append(dict(
                well_uid=wuid, chip=chip, arm=arm, metric=m,
                baseline=b, post=p, response=resp,
                eps_floored=bool(is_ratio and (b <= 0 or p <= 0)),
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
    return _finalize(rows, "p_wilcoxon", _SUMMARY_COLUMNS)


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
    return _finalize(rows, "p_vs_control", _VS_CONTROL_COLUMNS)


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
    return _finalize(rows, "p_mwu", _MWU_COLUMNS)
