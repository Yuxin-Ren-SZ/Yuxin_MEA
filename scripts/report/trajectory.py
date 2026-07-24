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

#: Treatment-day bins. Runs past tau 21 so the stated experiment window is fully
#: covered (CX118/CX169 record to tau 27). The stop is 31, not 30: ``range`` is
#: half-open, so stopping at 30 left the trailing edge unwritten and silently
#: binned every tau >= 26 to NaN — the last four days of two chips.
DEFAULT_TAU_EDGES = list(range(-6, 31, 4))

#: The stated experiment window, in treatment days. Chips differ in absolute DIV,
#: so every figure aligns on tau and marks the same two days.
EXPERIMENT_RANGE = (0, 21)


def tau_band(ax, start: float = EXPERIMENT_RANGE[0], end: float = EXPERIMENT_RANGE[1],
             *, label: bool = False) -> None:
    """Mark the experiment window: dashed verticals at tau=start and tau=end.

    ``start`` is the treatment day (tau 0) and ``end`` the last planned
    treatment day; the span between is tinted very faintly so the window reads as
    a region without competing with the data. Used by every trajectory panel so
    all figures share one time reference.
    """
    from .report_style import GRID, MUTED

    ax.axvspan(start, end, color=GRID, alpha=0.35, lw=0, zorder=0)
    for x in (start, end):
        ax.axvline(x, ls="--", color=MUTED, lw=0.8, zorder=1)
    if label:
        ax.annotate(f"treatment day {start:g}", xy=(start, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(2, -2),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=5.5, color=MUTED)
        ax.annotate(f"day {end:g}", xy=(end, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(2, -2),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=5.5, color=MUTED)


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
        # A single well gives a median but no bootstrap interval. Keep the point
        # and leave the interval NaN rather than dropping the bin: dropping it
        # shortened an arm's line without saying so, and it bit hardest at the
        # late bins where an effect would show (IVH_Early loses tau 20 and 24 on
        # nb_rate this way, so its line simply ended before Control's).
        lo, hi = _boot_ci(v) if len(v) >= 2 else (np.nan, np.nan)
        rows.append(dict(arm=arm, tauc=float(tc), median=float(np.median(v)),
                         lo=lo, hi=hi, n_wells=len(v)))
    return pd.DataFrame(rows)


def plot_trajectory(ax, tidy, metric, arms, tau_edges=DEFAULT_TAU_EDGES,
                    ref=None, ylabel=None, legend=True, title=None,
                    band: bool = True, band_label: bool = False):
    """Draw metric-vs-tau per arm: line + 95% CI ribbon, experiment-window marks.

    ``band`` draws the shared tau 0 / tau 21 dashed markers (:func:`tau_band`) so
    every trajectory in the report is aligned on treatment day, not DIV — chips
    differ in absolute culture age. ``ref`` adds a horizontal reference line
    (e.g. the critical branching ratio m=1).
    """
    traj = tau_trajectory(tidy, metric, arms, tau_edges)
    if band:
        tau_band(ax, label=band_label)
    for arm in [a for a in ordered_groups(arms) if a in set(traj.arm.unique())]:
        sub = traj[traj.arm == arm].sort_values("tauc")
        if sub.empty:
            continue
        ax.plot(sub.tauc, sub["median"], "-o", ms=3, color=group_color(arm),
                label=arm, zorder=3)
        ax.fill_between(sub.tauc, sub.lo, sub.hi, color=group_color(arm),
                        alpha=0.15, lw=0, zorder=2)
        # Single-well bins carry no interval; ring them so a point with no band
        # reads as "one well", not as a suspiciously precise estimate.
        thin = sub[sub.n_wells < 2]
        if not thin.empty:
            ax.plot(thin.tauc, thin["median"], "o", ms=5.5, mfc="none",
                    mec=group_color(arm), mew=0.8, zorder=4)
    if ref is not None:
        ax.axhline(ref, ls="--", color="0.5", lw=0.8, zorder=1)
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
