"""Shared tau-trajectory helpers for the report figures.

Treatment effects emerge over time, so pooling all post-treatment recordings into
one number dilutes late effects. These helpers plot a metric vs **tau** (days
relative to treatment) per arm, aggregating to one value per physical well first
(well_uid — no pseudo-replication), with a bootstrap CI ribbon and a tau=0 marker.
Generalises the pattern first used in ``fig8_criticality``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S
from .report_style import METRIC_LABELS, group_color, ordered_groups

DEFAULT_TAU_EDGES = list(range(-6, 22, 4))     # treatment-day bins


def _boot_ci(x, seed=0, n=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def tau_trajectory(tidy, metric, arms, tau_edges=DEFAULT_TAU_EDGES):
    """Per arm × tau-bin: median across well_uid medians + 95% bootstrap CI."""
    wi = S.well_index(tidy)[["well_uid", "arm"]]
    d = tidy.merge(wi, on="well_uid")
    d = d[d.arm.isin(arms) & d[metric].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["arm", "tauc", "median", "lo", "hi", "n_wells"])
    cats = pd.cut(d.tau, tau_edges, right=False)
    d["tauc"] = cats.map(lambda c: (c.left + c.right) / 2 if pd.notna(c) else np.nan)
    pw = d.groupby(["arm", "well_uid", "tauc"])[metric].median().reset_index()
    rows = []
    for (arm, tc), grp in pw.groupby(["arm", "tauc"]):
        if pd.isna(tc):
            continue
        v = grp[metric].to_numpy(float)
        if len(v) < 2:
            continue
        lo, hi = _boot_ci(v)
        rows.append(dict(arm=arm, tauc=float(tc), median=float(np.median(v)),
                         lo=lo, hi=hi, n_wells=len(v)))
    return pd.DataFrame(rows)


def plot_trajectory(ax, tidy, metric, arms, tau_edges=DEFAULT_TAU_EDGES,
                    ref=None, ylabel=None, legend=True, title=None):
    """Draw metric-vs-tau per arm (line + ribbon), tau=0 marker, optional ref line."""
    traj = tau_trajectory(tidy, metric, arms, tau_edges)
    for arm in [a for a in ordered_groups(arms) if a in set(traj.arm.unique())]:
        sub = traj[traj.arm == arm].sort_values("tauc")
        if sub.empty:
            continue
        ax.plot(sub.tauc, sub["median"], "-o", ms=3, color=group_color(arm),
                label=arm, zorder=3)
        ax.fill_between(sub.tauc, sub.lo, sub.hi, color=group_color(arm),
                        alpha=0.15, lw=0)
    ax.axvline(0, ls=":", color="0.5", lw=0.8)
    if ref is not None:
        ax.axhline(ref, ls="--", color="0.5", lw=0.8)
    ax.set_xlabel("treatment day (tau)", fontsize=7)
    ax.set_ylabel(ylabel or METRIC_LABELS.get(metric, metric), fontsize=7)
    if legend:
        ax.legend(fontsize=5.5, loc="best", ncol=2)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return traj


# --------------------------------------------------------------------------- #
# Example-well selection (early vs late recordings of one well)
# --------------------------------------------------------------------------- #
def pick_early_late(tidy, well_uid, early=0, late=14):
    """Two recordings of a well nearest tau=early and tau=late (post-treatment)."""
    sub = tidy[tidy.well_uid == well_uid]
    if sub.empty:
        return None, None
    e = sub.iloc[(sub.tau - early).abs().to_numpy().argsort()[0]]
    post = sub[sub.tau > 0]
    if post.empty:
        return e, e
    l = post.iloc[(post.tau - late).abs().to_numpy().argsort()[0]]
    return e, l


def pick_example_well(tidy, arm, metric="mean_sttc", late=14, min_late=8):
    """A representative treated well of ``arm`` that has an early AND a late
    (tau≥min_late) recording, nearest the arm's median ``metric`` at late."""
    wi = S.well_index(tidy)
    wells = set(wi.loc[(wi.arm == arm) & wi.paired, "well_uid"])
    cand = tidy[(tidy.well_uid.isin(wells)) & (tidy.tau >= min_late)
                & tidy[metric].notna()]
    if cand.empty:
        cand = tidy[(tidy.well_uid.isin(wells)) & tidy[metric].notna()]
    if cand.empty:
        return None
    med = cand[metric].median()
    return cand.iloc[(cand[metric] - med).abs().to_numpy().argsort()[0]].well_uid
