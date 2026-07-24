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
    "nb_ibi_mean", "nb_ibi_cv", "nb_duration_cv", "median_firing_rate", "n_curated",
    # Spatial / connectivity (non-negative -> log2 ratio valid)
    "activity_gini", "mean_prop_speed_um_ms", "edge_density", "small_worldness",
    "clustering_coeff", "global_efficiency",
    # Node-level graph metrics (non-negative, comfortably > 0)
    "degree_cv",
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
    # Bounded in [0, 1] and genuinely reaching 0 in sparse wells: a log2 ratio
    # there produced −30 "responses" that swamped every other chip.
    "participation_mean", "rich_club",
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


# --------------------------------------------------------------------------- #
# Biological-replicate (chip) accounting
# --------------------------------------------------------------------------- #
# The physical well is the *technical* replicate; the chip (``sample_id``,
# CX118/CX138/CX169) is the biological replicate. Wells within a chip share one
# plating / CSF-application batch, so treating them as independent inflates n
# (pseudo-replication; Lazic 2010, BMC Neurosci). The functions below respect the
# hierarchy two ways: per-chip means for plotting, and a well-nested-in-chip
# linear mixed model for the difference-in-differences inference.
def multichip_arms(resp: pd.DataFrame) -> list[str]:
    """Non-Control arms measured on >=2 chips (the only ones a chip-level model
    can estimate). Single-chip arms (AraC, NPH) are perfectly confounded with
    their one chip, so their coefficient is meaningless — excluded."""
    per = resp[resp.arm != "Control"].groupby("arm").chip.nunique()
    return [a for a in per.index if per[a] >= MIN_CHIPS_CONFIRMATORY]


def chip_response(resp: pd.DataFrame) -> pd.DataFrame:
    """Per (chip, arm, metric): mean of the well responses = one *biological*
    point. This is what "all data points" plots (wells collapsed to their chip)."""
    return (resp.groupby(["chip", "arm", "metric"], as_index=False)
            .response.mean())


def within_chip_did(resp: pd.DataFrame) -> pd.DataFrame:
    """Per chip × treated arm × metric: the chip's own treated-vs-Control DiD
    ``(arm chip-mean − that chip's Control chip-mean)``.

    This is the transparent companion to the mixed model: its mean should match
    the LMM coefficient, and the **sign-consistency of the 2-3 per-chip DiDs is
    the real n=3 story** (a same-sign, tight set is a genuine biological effect;
    a mixed-sign set is null however small the Wald p). Only chips that carry
    both the arm and a Control well contribute.
    """
    cr = chip_response(resp)
    ctrl = (cr[cr.arm == "Control"][["chip", "metric", "response"]]
            .rename(columns={"response": "ctrl_response"}))
    tr = cr[cr.arm != "Control"].merge(ctrl, on=["chip", "metric"], how="inner")
    tr["did"] = tr.response - tr.ctrl_response
    return tr[["chip", "arm", "metric", "response", "ctrl_response", "did"]]


def arm_response_summary_chip(resp: pd.DataFrame) -> pd.DataFrame:
    """Per arm × metric at the biological-replicate (chip) level: mean of the
    chip means + a t-CI (df = n_chips − 1) over chips.

    Replaces the well-bootstrap CI in :func:`arm_response_summary`, which is
    pseudo-replicated (too tight). Column ``median_response`` is kept for
    drop-in use by the forest panels but now holds the **chip-level mean**.
    """
    from scipy.stats import t as tdist

    cr = chip_response(resp)
    rows = []
    for (arm, m), sub in cr.groupby(["arm", "metric"]):
        x = sub.response.to_numpy(float)
        x = x[~np.isnan(x)]
        n = len(x)
        mean = float(np.mean(x)) if n else np.nan
        if n >= 2:
            sem = float(np.std(x, ddof=1) / np.sqrt(n))
            h = float(tdist.ppf(0.975, n - 1)) * sem
            lo, hi = mean - h, mean + h
        else:
            lo = hi = mean
        rows.append(dict(arm=arm, metric=m, n_chips=n,
                         median_response=mean, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def _ttest_did_vs_zero(x: np.ndarray) -> float:
    """One-sample t-test that the per-chip DiDs differ from 0; NaN if < 2 chips
    or all identical. df = n_chips − 1 — the honest biological-replicate df."""
    from scipy.stats import ttest_1samp

    x = x[~np.isnan(x)]
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.nan
    return float(ttest_1samp(x, 0.0).pvalue)


def arm_response_vs_control_lmm(
    resp: pd.DataFrame, focus_arms: list[str] | None = None,
) -> pd.DataFrame:
    """Difference-in-differences at the biological-replicate (chip) level.

    **Inference = the cluster-summary test**, not a Wald p from an intercept-only
    mixed model. Treatment is applied *within* chip (every chip carries Control +
    treated wells), so ``response ~ arm + (1|chip)`` is random-**intercept** only:
    the arm effect is a pure within-chip contrast whose SE the intercept-only
    model draws from the **well-level residual df (~13), not the 3 chips** — that
    is pseudo-replication wearing a mixed-model coat (statsmodels also gives no
    small-sample df correction, so the Wald p is 40-400× too small at n≈3 here).
    Generalising to new chips needs a random *slope* ``(arm|chip)``, which 3
    groups cannot identify; its finite-sample equivalent is a **one-sample t-test
    of the per-chip DiDs against 0** (:func:`within_chip_did`, df = n_chips − 1).
    That is ``p_vs_control`` / ``q_bh`` here — the number a reviewer will expect.

    The mixed model is still fit, but only for the **point estimate**
    (``lmm_coef``, which matches ``did_chipmean``); its anti-conservative
    ``lmm_p`` is kept for transparency, never for significance. ``chip_consistent``
    = same sign on all chips; ``suggestive`` = consistent AND uncorrected
    ``p_vs_control`` < 0.05 (a pilot-level signal that does *not* survive FDR at
    n=3). Single-chip arms (AraC, NPH) are excluded. BH-FDR over confirmatory rows.
    """
    import warnings

    import statsmodels.formula.api as smf

    wc = within_chip_did(resp)
    ok_arms = set(multichip_arms(resp))
    arms = ([a for a in focus_arms if a in ok_arms] if focus_arms is not None
            else sorted(ok_arms))
    rows = []
    for m, d0 in resp.groupby("metric"):
        d = d0[d0.arm.isin(["Control"] + arms)].dropna(subset=["response"]).copy()
        coefs: dict[str, float] = {}
        lmm_ps: dict[str, float] = {}
        re_var = np.nan
        singular = True
        if d.arm.nunique() >= 2 and d.chip.nunique() >= 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = smf.mixedlm(
                        "response ~ C(arm, Treatment('Control'))",
                        d, groups=d["chip"]).fit(reml=False)
                re_var = float(res.cov_re.iloc[0, 0])
                singular = re_var <= 1e-8 * max(float(res.scale), 1e-12)
                for a in arms:
                    key = f"C(arm, Treatment('Control'))[T.{a}]"
                    if key in res.params.index:
                        coefs[a] = float(res.params[key])
                        lmm_ps[a] = float(res.pvalues[key])
            except Exception:  # singular / non-convergence -> companion carries it
                pass
        for a in arms:
            sub = d0[d0.arm == a]
            cd = wc.loc[(wc.metric == m) & (wc.arm == a), "did"].to_numpy(float)
            cd = cd[~np.isnan(cd)]
            n_wells = int(sub.response.notna().sum())
            n_chips = int(sub.chip.nunique())
            rows.append(dict(
                arm=a, metric=m, n_wells=n_wells, n_chips=n_chips,
                n_chips_paired=len(cd),
                did=float(np.mean(cd)) if len(cd) else np.nan,   # honest estimate
                did_chipmean=float(np.mean(cd)) if len(cd) else np.nan,
                lmm_coef=coefs.get(a, np.nan), lmm_p=lmm_ps.get(a, np.nan),
                chip_consistent=bool(len(cd) >= 2 and
                                     (np.all(cd > 0) or np.all(cd < 0))),
                re_var=re_var, singular=bool(singular),
                p_vs_control=_ttest_did_vs_zero(cd),      # cluster-summary p
                tier=("confirmatory"
                      if (n_chips >= MIN_CHIPS_CONFIRMATORY
                          and n_wells >= MIN_WELLS_CONFIRMATORY)
                      else "exploratory"),
            ))
    out = pd.DataFrame(rows)
    out["q_bh"] = np.nan
    if not out.empty:
        conf = (out.tier == "confirmatory") & out.p_vs_control.notna()
        if conf.any():
            out["q_bh"] = _bh_fdr(out.p_vs_control.where(conf, np.nan).to_numpy())
        out["suggestive"] = (out.chip_consistent & out.p_vs_control.notna()
                             & (out.p_vs_control < 0.05) & ~(out.q_bh < 0.05))
    return out


# --------------------------------------------------------------------------- #
# Per-time-point response — the ΔΔCq logic, applied to a time series
# --------------------------------------------------------------------------- #
# Pooling every post-treatment recording into one number (:func:`well_response`)
# dilutes effects that emerge late and mixes time points that differ a lot. The
# functions below keep the time axis:
#
#   1. each well is compared to **its own** pre-treatment baseline, per tau bin
#      (the ΔCq step),
#   2. the wells of a chip are averaged into one biological value per bin
#      (technical replicates collapse — the chip is the biological replicate),
#   3. each chip's treated value minus that chip's own Control value is its DiD
#      (the ΔΔ step),
#   4. inference is a one-sample t-test of those per-chip DiDs against 0
#      (df = n_chips − 1) at **one pre-specified endpoint**.
#
# Testing every tau bin would multiply the FDR family by the number of bins and,
# at n=3 chips, suppress everything; the full time course is still returned for
# plotting, but only the endpoint enters the family. See :func:`did_at_endpoint`.

#: Post-treatment bins, right-closed: (0,4], (4,8], … labelled by bin centre.
POST_TAU_EDGES = [0, 4, 8, 12, 16, 20, 24, 28]


def _tau_bin(tau: pd.Series, edges=POST_TAU_EDGES) -> pd.Series:
    cats = pd.cut(tau, edges, right=True)
    return cats.map(lambda c: (c.left + c.right) / 2 if pd.notna(c) else np.nan)


def well_response_by_tau(
    tidy: pd.DataFrame,
    metrics: list[str] | None = None,
    edges: list[float] = POST_TAU_EDGES,
) -> pd.DataFrame:
    """Per well × metric × post-tau bin: response vs that well's own baseline.

    baseline = median over the well's ``tau < 0`` recordings (recordings on the
    treatment day itself, ``tau == 0``, count as neither pre nor post).
    response = ``log2((post+eps)/(baseline+eps))`` for ratio metrics,
               ``post − baseline`` for diff metrics — same convention as
    :func:`well_response`, so the two agree when a single bin is used.

    Control wells are included: their response is the maturation-only change over
    the same interval, and every treated arm is read against it.
    """
    metrics = metrics or ALL_METRICS
    metrics = [m for m in metrics if m in tidy.columns]
    rows = []
    for wuid, sub in tidy.groupby("well_uid"):
        pre = sub[sub.tau < 0]
        post = sub[sub.tau > 0]
        if pre.empty or post.empty:
            continue
        arm = arm_of(sub.canonical_group)
        chip = sub.sample_id.iloc[0]
        post = post.assign(tauc=_tau_bin(post.tau, edges))
        for m in metrics:
            b = pre[m].median()
            if pd.isna(b):
                continue
            for tc, grp in post.groupby("tauc"):
                if pd.isna(tc):
                    continue
                p = grp[m].median()
                if pd.isna(p):
                    continue
                resp = (p - b) if m in DIFF_METRICS else np.log2((p + _EPS) / (b + _EPS))
                rows.append(dict(
                    well_uid=wuid, chip=chip, arm=arm, metric=m, tauc=float(tc),
                    baseline=b, post=p, response=resp,
                    n_pre=len(pre), n_post=len(grp),
                ))
    return pd.DataFrame(rows)


def chip_response_by_tau(resp: pd.DataFrame) -> pd.DataFrame:
    """Per (chip, arm, metric, tau bin): mean of the well responses = one
    *biological* value. Wells are technical replicates and collapse here."""
    if resp.empty:
        return resp
    return (resp.groupby(["chip", "arm", "metric", "tauc"], as_index=False)
            .response.mean())


def did_by_tau(resp: pd.DataFrame) -> pd.DataFrame:
    """Per chip × treated arm × metric × tau bin: that chip's own treated−Control
    difference-in-differences. Only chips carrying both contribute."""
    if resp.empty:
        return pd.DataFrame(columns=["chip", "arm", "metric", "tauc", "response",
                                     "ctrl_response", "did"])
    cr = chip_response_by_tau(resp)
    ctrl = (cr[cr.arm == "Control"][["chip", "metric", "tauc", "response"]]
            .rename(columns={"response": "ctrl_response"}))
    tr = cr[cr.arm != "Control"].merge(ctrl, on=["chip", "metric", "tauc"], how="inner")
    tr["did"] = tr.response - tr.ctrl_response
    return tr[["chip", "arm", "metric", "tauc", "response", "ctrl_response", "did"]]


def primary_endpoint(resp: pd.DataFrame, focus_arms: list[str] | None = None) -> float:
    """The latest tau bin that every chip still contributes to — the endpoint the
    DiD is tested at.

    Chips stop recording at different treatment days (CX138 ends at tau 15 while
    CX118/CX169 run to 27), so the last bin *present* is not the last bin that is
    a fair n=3 comparison. Pick the latest bin where the number of chips equals
    the study's chip count, for Control and for every focus arm. Falls back to the
    bin with the most chips when no bin is complete.
    """
    d = did_by_tau(resp)
    if d.empty:
        return float("nan")
    if focus_arms:
        d = d[d.arm.isin(focus_arms)]
        if d.empty:
            return float("nan")
    n_target = resp.chip.nunique()
    per_bin = d.groupby("tauc").chip.nunique()
    complete = per_bin[per_bin >= n_target]
    if len(complete):
        return float(complete.index.max())
    return float(per_bin.idxmax())


def did_at_endpoint(
    resp: pd.DataFrame,
    focus_arms: list[str] | None = None,
    endpoint: float | None = None,
) -> pd.DataFrame:
    """Chip-level DiD at one endpoint, with p and BH-FDR q over this metric set.

    One row per (arm, metric). ``p_vs_control`` is a one-sample t-test of the
    per-chip DiDs against 0 (df = n_chips − 1 — the honest biological-replicate
    df); ``q_bh`` corrects across **all rows of this call**, so callers must pass
    the whole metric family a figure reports, not one metric at a time.

    Significance glyphs read from these two columns only:
    ``*`` when ``q_bh < 0.05``; ``△`` when ``p_vs_control < 0.05 <= q_bh``.
    ``chip_consistent`` (same sign on every chip) is reported for context but no
    longer gates either glyph.
    """
    if resp.empty:
        return pd.DataFrame()
    arms = focus_arms or [a for a in resp.arm.unique() if a != "Control"]
    d = did_by_tau(resp)
    ep = endpoint if endpoint is not None else primary_endpoint(resp, arms)
    rows = []
    for m in sorted(resp.metric.unique()):
        for a in arms:
            sub = d[(d.metric == m) & (d.arm == a) & (d.tauc == ep)]
            cd = sub.did.to_numpy(float)
            cd = cd[~np.isnan(cd)]
            n_wells = int(resp[(resp.metric == m) & (resp.arm == a)
                               & (resp.tauc == ep)].response.notna().sum())
            rows.append(dict(
                arm=a, metric=m, endpoint_tau=ep,
                n_chips_paired=len(cd), n_wells=n_wells,
                did=float(np.mean(cd)) if len(cd) else np.nan,
                did_sd=float(np.std(cd, ddof=1)) if len(cd) > 1 else np.nan,
                chip_consistent=bool(len(cd) >= 2
                                     and (np.all(cd > 0) or np.all(cd < 0))),
                p_vs_control=_ttest_did_vs_zero(cd),
            ))
    out = pd.DataFrame(rows)
    out["q_bh"] = _bh_fdr(out.p_vs_control.to_numpy())
    return out


def arm_summary_at_endpoint(resp: pd.DataFrame, endpoint: float) -> pd.DataFrame:
    """Per arm × metric at one tau bin: mean of the per-chip means + 95% t-CI
    (df = n_chips − 1). The bar and whisker every DiD panel draws."""
    from scipy.stats import t as tdist

    cr = chip_response_by_tau(resp)
    cr = cr[cr.tauc == endpoint]
    rows = []
    for (arm, m), sub in cr.groupby(["arm", "metric"]):
        x = sub.response.to_numpy(float)
        x = x[~np.isnan(x)]
        n = len(x)
        mean = float(np.mean(x)) if n else np.nan
        if n >= 2:
            h = float(tdist.ppf(0.975, n - 1)) * float(np.std(x, ddof=1) / np.sqrt(n))
            lo, hi = mean - h, mean + h
        else:
            lo = hi = mean
        rows.append(dict(arm=arm, metric=m, n_chips=n, mean=mean,
                         ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


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
