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
from .report_style import METRIC_LABELS, OKABE_ITO, save_fig

DIV_BIN_EDGES = list(range(4, 37, 4))  # width-4 bins spanning DIV 4-34
TRAJ_METRICS = ["median_firing_rate", "nb_rate"]
_ROLE_STYLE = {
    "control": dict(color=OKABE_ITO["grey"], label="untreated"),
    "treatment": dict(color=OKABE_ITO["vermillion"], label="treated"),
}


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
    """Per role × DIV-bin: group median + bootstrap CI across per-well medians."""
    df = tidy.copy()
    df["divc"] = _bin_center(df.DIV)
    # per well_uid × role × bin -> median (kills pseudo-replication)
    per_well = (df.groupby(["role", "well_uid", "divc"])[metric]
                .median().reset_index())
    rows = []
    for (role, divc), sub in per_well.groupby(["role", "divc"]):
        x = sub[metric].to_numpy(dtype=float)
        lo, hi = _boot_median_ci(x, seed=seed)
        rows.append(dict(role=role, divc=divc, n_wells=len(sub),
                         median=float(np.nanmedian(x)), lo=lo, hi=hi))
    return pd.DataFrame(rows).sort_values(["role", "divc"])


def _panel_trajectory(ax, tidy, metric: str) -> None:
    traj = _trajectory(tidy, metric)
    for role, style in _ROLE_STYLE.items():
        sub = traj[traj.role == role].dropna(subset=["divc"])
        if sub.empty:
            continue
        ax.plot(sub.divc, sub["median"], "-o", ms=3, color=style["color"],
                label=style["label"], zorder=3)
        ax.fill_between(sub.divc, sub.lo, sub.hi, color=style["color"],
                        alpha=0.18, lw=0, zorder=1)
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


def build_f4(tidy, analysis_root):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.0, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.45, wspace=0.3)

    # Row A: representative rasters (early vs late)
    wuid = _pick_raster_well(tidy)
    axr0 = fig.add_subplot(gs[0, 0])
    axr1 = fig.add_subplot(gs[0, 1])
    if wuid is not None:
        sub = tidy[tidy.well_uid == wuid].sort_values("DIV")
        r_early, r_late = sub.iloc[0], sub.iloc[-1]
        _panel_raster(axr0, analysis_root, r_early,
                      f"A  {wuid.split('|')[0]} {r_early.well_name} — DIV {int(r_early.DIV)} (immature)")
        _panel_raster(axr1, analysis_root, r_late,
                      f"   {wuid.split('|')[0]} {r_late.well_name} — DIV {int(r_late.DIV)} (mature)")

    # Row B/C: developmental trajectories
    axt0 = fig.add_subplot(gs[1, 0])
    axt1 = fig.add_subplot(gs[1, 1])
    _panel_trajectory(axt0, tidy, "median_firing_rate")
    axt0.set_title("B  Firing-rate maturation", loc="left", fontweight="bold")
    _panel_trajectory(axt1, tidy, "nb_rate")
    axt1.set_title("C  Network-burst maturation", loc="left", fontweight="bold")

    fig.suptitle("Figure 4 — Network activity & development (well_uid unit)",
                 fontsize=9, y=0.99)
    return fig, {"raster_well": wuid}


def render(tidy, analysis_root, figure_root):
    fig, meta = build_f4(tidy, analysis_root)
    paths = save_fig(fig, "F4_activity_development", figure_root, subdir="main")
    return paths, meta
