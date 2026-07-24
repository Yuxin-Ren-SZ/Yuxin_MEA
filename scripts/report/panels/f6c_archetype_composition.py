"""F6c — Burst-archetype composition over treatment time, per group."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import panel_data as D
from ..panel_base import panel_main
from ..report_style import MUTED, group_color
from ..trajectory import DEFAULT_TAU_EDGES, tau_band
from . import _f6_common as A

NAME = "F6c_archetype_composition"
FIGSIZE = (4.6, 3.6)
#: Minimum wells for a bin to be drawn. One, deliberately: a bin backed by a
#: single well is weak evidence, but blanking it deletes real data and shortens
#: an arm's stack with nothing on the figure to say why — the same failure the
#: trajectory and DiD-timecourse panels had, and it landed on the same late
#: IVH_Early bins. A single-well bin is drawn as an isolated bar with its ``1``
#: printed above it, so its weight is visible instead of its absence.
MIN_WELLS_PER_BIN = 1
#: Every bin centre the tau axis can hold, so a bin an arm has no data in stays a
#: gap instead of being dropped and interpolated across.
BIN_CENTERS = [(lo + hi) / 2 for lo, hi in
               zip(DEFAULT_TAU_EDGES[:-1], DEFAULT_TAU_EDGES[1:])]
CAPTION = (
    "What kinds of burst each group produces, and how the mixture changes after "
    "treatment. Every detected network burst from the three arms is pooled and "
    "clustered on its own shape features into cohort-wide archetypes (named in "
    "F6d), so the labels mean the same thing in every well — unlike the "
    "pipeline's per-well burst typing, whose identities are not comparable across "
    "wells. Every well-recording with at least five detected network bursts "
    "contributes — nothing is subsampled beyond that gate, and the accompanying "
    "selection table records what the gate removed per arm — "
    "and a well is assigned to the arm it was treated with, not to the group label "
    "its chip happened to carry on the day of the scan. Each stacked panel is one "
    "arm: band height is the mean fraction of "
    "that arm's bursts falling in each archetype at that treatment-day bin, "
    "averaged across wells so every well counts equally rather than the "
    "burst-richest well dominating. The small number above each bin is how many "
    "wells it rests on — read a bin marked 1 as one culture, not as a cohort "
    "estimate. A bin with no data at all is left blank, and a blank bin is a real "
    "gap rather than a line drawn across missing data; a bin with no neighbour to "
    "join is drawn as an isolated bar. "
    "Vertical dashed lines and the tinted span "
    "mark the experiment window, τ0 to τ+21. An archetype marked * has a per-well "
    "fraction that differs from Control (Mann-Whitney per arm, BH-FDR)."
)


def _runs(centers: list) -> list:
    """Split kept bin centres into contiguous stretches of the full tau axis.

    Drawing one stackplot over a list that skips a bin joins the two sides with a
    straight segment, which reads as measured data across a bin that has none.
    Each contiguous stretch is drawn separately so a gap stays a gap.
    """
    pos = {c: i for i, c in enumerate(BIN_CENTERS)}
    out: list = []
    for c in centers:
        if out and pos[c] == pos[out[-1][-1]] + 1:
            out[-1].append(c)
        else:
            out.append([c])
    return out


def build_fig(fig, ctx):
    bt = arch_frame(ctx)
    arch = A.archetypes(ctx)
    arms = [a for a in D.MAIN_ARMS if a in set(bt.arm)]

    gs = fig.add_gridspec(len(arms), 1, hspace=0.22)
    prev, rows = None, []
    for i, arm in enumerate(arms):
        ax = fig.add_subplot(gs[i], sharex=prev)
        prev = ax
        tau_band(ax)
        sub = bt[bt.arm == arm]
        wtf = (sub.groupby(["well_uid", "tauc", "arch"]).size()
               .unstack("arch", fill_value=0))
        wtf = wtf.div(wtf.sum(axis=1), axis=0)
        comp = wtf.groupby("tauc").mean().reindex(BIN_CENTERS)
        nwell = sub.groupby("tauc").well_uid.nunique().reindex(BIN_CENTERS)
        nchip = (sub.assign(chip=sub.well_uid.str.split("|").str[0])
                 .groupby("tauc").chip.nunique().reindex(BIN_CENTERS))
        def count(s, c):                      # reindex leaves NaN, not 0
            v = s.get(c)
            return 0 if v is None or pd.isna(v) else int(v)

        keep = [c for c in BIN_CENTERS
                if comp.loc[c].notna().any() and count(nwell, c) >= MIN_WELLS_PER_BIN]

        first = True
        for run in _runs(keep):
            block = comp.loc[run]
            ys = [block[a].fillna(0).to_numpy() if a in block.columns
                  else np.zeros(len(run)) for a in range(arch.k)]
            colors = [A.arch_color(a) for a in range(arch.k)]
            labels = [A.arch_label(arch, a) for a in range(arch.k)] if first else None
            if len(run) > 1:
                ax.stackplot(np.asarray(run, float), *ys, colors=colors,
                             labels=labels, zorder=2)
            else:
                # A lone bin has no neighbour to interpolate towards; draw it as a
                # stacked bar so it is visible without implying a trend.
                bottom = 0.0
                for a in range(arch.k):
                    h = float(ys[a][0])
                    ax.bar(run[0], h, bottom=bottom, width=2.4, color=colors[a],
                           label=None if labels is None else labels[a], zorder=2)
                    bottom += h
            first = False

        for c in keep:                       # what each band actually rests on
            ax.annotate(f"{count(nwell, c)}", xy=(c, 1.0), xytext=(0, 1.5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=4.5, color=MUTED)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel(arm.replace("_", " "), fontsize=7, rotation=0, ha="right",
                      va="center", color=group_color(arm), fontweight="bold")
        ax.tick_params(labelsize=6)
        if i < len(arms) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("treatment day (tau)", fontsize=7)
        if i == 0:
            ax.legend(fontsize=5.5, ncol=min(arch.k, 3), loc="lower left",
                      bbox_to_anchor=(0.0, 1.04), frameon=False)

        for c in BIN_CENTERS:                 # every bin, drawn or not
            rows.append(dict(arm=arm, tauc=c, n_wells=count(nwell, c),
                             n_chips=count(nchip, c),
                             plotted=c in keep,
                             **{f"arch{a}": (float(comp.loc[c, a])
                                             if a in comp.columns
                                             and pd.notna(comp.loc[c, a]) else np.nan)
                                for a in range(arch.k)}))

    _write_bins(ctx, pd.DataFrame(rows), arch)
    fig.text(0.99, 0.01, f"{len(arch.bursts)} bursts · "
             f"{arch.bursts.well_uid.nunique()} wells", ha="right", va="bottom",
             fontsize=5.5, color=MUTED)
    return arch.stats


def arch_frame(ctx):
    """Archetype bursts restricted to the main arms, with the tau bin attached.

    ``group`` is already the well's arm (assigned in ``collect_bursts`` from
    ``well_index``), so no second, possibly-disagreeing labelling is applied here.
    """
    arch = A.archetypes(ctx)
    bt = arch.bursts.assign(arm=arch.bursts.group)
    bt = bt[bt.arm.isin(D.MAIN_ARMS)].copy()
    bt["tauc"] = pd.cut(bt.tau, DEFAULT_TAU_EDGES, right=False).map(
        lambda c: (c.left + c.right) / 2 if pd.notna(c) else np.nan)
    return bt


def _write_bins(ctx, bins: pd.DataFrame, arch) -> None:
    """Persist the plotted matrix and the selection audit next to the panel.

    ``panel_base`` writes one stats CSV per panel and that slot holds the
    significance table, so the numbers behind the bands would otherwise exist
    only inside the figure.
    """
    from ..panel_base import panels_dir

    out = panels_dir(ctx.figure_root)
    out.mkdir(parents=True, exist_ok=True)
    bins.to_csv(out / f"{NAME}_bins.csv", index=False)
    sel = arch.bursts.attrs.get("selection")
    if sel is not None and not sel.empty:
        sel.to_csv(out / f"{NAME}_selection.csv", index=False)


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
