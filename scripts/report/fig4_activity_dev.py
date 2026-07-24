"""F4 — Network activity & development.

* **A. Representative rasters** — one control well at an early vs a late DIV,
  showing the emergence of coordinated network activity with maturation.
* **B/C. Developmental trajectories** — median firing rate and network-burst
  rate vs DIV, split into untreated (pre-treatment / never-treated recordings,
  ``role == control``) and treated (``role == treatment``) recordings.

Unit of analysis is the physical well: each well contributes one median per DIV
bin; the ribbon is a 95% bootstrap CI of the per-well medians in that bin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load as L
from .report_style import METRIC_LABELS, group_color, ordered_groups

DIV_BIN_EDGES = list(range(4, 37, 4))  # width-4 bins spanning DIV 4-34
TRAJ_METRICS = ["median_firing_rate", "nb_rate"]
# Developmental trajectories are drawn per treatment arm (group palette), so the
# maturation curves map directly onto the group colours used in F5-F8.
_TRAJ_ARMS = ["Control", "IVH_Early", "IVH_Late"]


def _bin_center(div: pd.Series) -> pd.Series:
    cats = pd.cut(div, DIV_BIN_EDGES, right=False)
    return cats.map(lambda c: (c.left + c.right) / 2 if pd.notna(c) else np.nan)


def _boot_median_ci(x: np.ndarray, seed: int, n: int = 2000) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _trajectory(tidy: pd.DataFrame, metric: str, seed: int = 0) -> pd.DataFrame:
    """Per arm × DIV-bin: group median + bootstrap CI across per-well medians."""
    from . import stats as S

    df = tidy.merge(S.well_index(tidy)[["well_uid", "arm"]], on="well_uid")
    df["divc"] = _bin_center(df.DIV)
    per_well = (df.groupby(["arm", "well_uid", "divc"])[metric]
                .median().reset_index())
    rows = []
    for (arm, divc), sub in per_well.groupby(["arm", "divc"]):
        x = sub[metric].to_numpy(dtype=float)
        lo, hi = _boot_median_ci(x, seed=seed)
        rows.append(dict(arm=arm, divc=divc, n_wells=len(sub),
                         median=float(np.nanmedian(x)), lo=lo, hi=hi))
    return pd.DataFrame(rows).sort_values(["arm", "divc"])


def _panel_trajectory(ax, tidy, metric: str, div_band=None) -> None:
    traj = _trajectory(tidy, metric)
    present = set(traj.arm.unique())
    for arm in [a for a in ordered_groups(present) if a in _TRAJ_ARMS]:
        sub = traj[traj.arm == arm].dropna(subset=["divc"])
        if sub.empty:
            continue
        ax.plot(sub.divc, sub["median"], "-o", ms=3, color=group_color(arm),
                label=arm, zorder=3)
        ax.fill_between(sub.divc, sub.lo, sub.hi, color=group_color(arm),
                        alpha=0.12, lw=0, zorder=1)
    if div_band is not None:
        ax.axvspan(div_band[0], div_band[1], color="0.55", alpha=0.10, lw=0,
                   zorder=0)
    ax.set_xlabel("DIV")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.legend(fontsize=6, loc="best")


def _pick_raster_well(tidy: pd.DataFrame):
    """A control well with the most curated units, present early & late."""
    from . import stats as S

    wi = S.well_index(tidy)
    ctrl = set(wi[wi.arm == "Control"].well_uid)
    best = None
    for wuid, sub in tidy.groupby("well_uid"):
        if wuid not in ctrl:
            continue
        if sub.DIV.min() > 8 or sub.DIV.max() < 26:
            continue
        score = sub.n_curated.max()
        if best is None or score > best[1]:
            best = (wuid, score)
    return None if best is None else best[0]


def _panel_raster(ax, analysis_root, row, title: str, max_units: int = 40,
                  t_win: float = 60.0) -> None:
    try:
        spikes = L.load_curated_spikes(analysis_root, row)
    except FileNotFoundError:
        ax.text(0.5, 0.5, "no curated spikes", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_title(title, fontsize=7)
        return
    units = list(spikes.items())[:max_units]
    for i, (_, st) in enumerate(units):
        st = np.asarray(st)
        st = st[st <= t_win]
        ax.eventplot(st, lineoffsets=i, linelengths=0.8, linewidths=0.4,
                     colors="0.15")
    ax.set_xlim(0, t_win)
    ax.set_ylim(-1, max(len(units), 1))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("unit")
    ax.set_title(title, fontsize=7)
