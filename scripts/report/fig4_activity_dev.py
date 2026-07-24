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
from .report_style import (
    METRIC_LABELS, MUTED, caption, group_color, ordered_groups, save_fig)

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


def build_f4(tidy, analysis_root):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.0, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.45, wspace=0.3)

    # Row A: representative rasters (early vs late) — demoted, illustrative.
    wuid = _pick_raster_well(tidy)
    axr0 = fig.add_subplot(gs[0, 0])
    axr1 = fig.add_subplot(gs[0, 1])
    if wuid is not None:
        sub = tidy[tidy.well_uid == wuid].sort_values("DIV")
        r_early, r_late = sub.iloc[0], sub.iloc[-1]
        _panel_raster(axr0, analysis_root, r_early,
                      f"illustrative · single well — DIV {int(r_early.DIV)} (immature)")
        _panel_raster(axr1, analysis_root, r_late,
                      f"illustrative · single well — DIV {int(r_late.DIV)} (mature)")
        for a in (axr0, axr1):                 # mute the demoted example titles
            a.set_title(a.get_title(), fontsize=7, color=MUTED)
            for sp in a.spines.values():
                sp.set_color("0.7")
    axr0.annotate("A  Representative rasters — illustrative, single well",
                  xy=(0, 1.14), xycoords="axes fraction", fontweight="bold",
                  fontsize=9, annotation_clip=False)

    # DIV window that the treatment DiD (F5-F8) samples = the post-treatment span.
    div_band = None
    if "tau" in tidy.columns:
        post = tidy[tidy.tau >= 0]
        if not post.empty:
            div_band = (float(post.DIV.min()), float(post.DIV.max()))

    # Row B/C: developmental trajectories (per group, group palette).
    axt0 = fig.add_subplot(gs[1, 0])
    axt1 = fig.add_subplot(gs[1, 1])
    _panel_trajectory(axt0, tidy, "median_firing_rate", div_band=div_band)
    axt0.set_title("B  Firing-rate maturation", loc="left", fontweight="bold")
    _panel_trajectory(axt1, tidy, "nb_rate", div_band=div_band)
    axt1.set_title("C  Network-burst maturation", loc="left", fontweight="bold")

    fig.suptitle("Figure 4 — Network activity & development · the maturation "
                 "baseline the treatment DiD (F5-F8) is read against",
                 fontsize=9, y=0.99)
    band_txt = ("" if div_band is None else
                f" The shaded DIV band ({int(div_band[0])}-{int(div_band[1])}) "
                "marks the post-treatment window sampled by F5-F8.")
    caption(fig,
        "Network activity and its developmental maturation across the cohort. "
        "(A) Representative spike rasters of one well, early (immature, low DIV) "
        "vs late — illustrative single-well examples, not a group statistic. "
        "(B) Firing-rate maturation and (C) network-burst maturation vs "
        "developmental age (DIV): each line is a treatment arm (group palette, "
        "Control neutral / IVH amber-sienna), tracking the median metric across "
        "wells (95% bootstrap CI ribbon) as the culture matures. Each arm pools "
        "ALL of that arm's recordings across DIV — including each treated well's "
        "pre-treatment recordings — so the lines show per-arm development, not a "
        "treatment contrast; any pre-band (DIV < treatment) separation is baseline "
        "well-to-well variation, and the treatment effect is read only as the "
        "difference-in-differences in F5-F8." + band_txt +
        " This establishes the maturation baseline against which that DiD is read. "
        "Points aggregate recordings to the well first (median).")
    return fig, {"raster_well": wuid}


def render(tidy, analysis_root, figure_root):
    fig, meta = build_f4(tidy, analysis_root)
    paths = save_fig(fig, "F4_activity_development", figure_root, subdir="main")
    return paths, meta
