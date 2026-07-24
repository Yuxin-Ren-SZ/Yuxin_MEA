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
CAPTION = (
    "What kinds of burst each group produces, and how the mixture changes after "
    "treatment. Every detected network burst from the three arms is pooled and "
    "clustered on its own shape features into cohort-wide archetypes (named in "
    "F6d), so the labels mean the same thing in every well — unlike the "
    "pipeline's per-well burst typing, whose identities are not comparable across "
    "wells. Each stacked panel is one arm: band height is the mean fraction of "
    "that arm's bursts falling in each archetype at that treatment-day bin, "
    "averaged across wells so every well counts equally rather than the "
    "burst-richest well dominating. Vertical dashed lines and the tinted span "
    "mark the experiment window, τ0 to τ+21. An archetype marked * has a per-well "
    "fraction that differs from Control (Mann-Whitney per arm, BH-FDR)."
)


def build_fig(fig, ctx):
    from .. import stats as S

    arch = A.archetypes(ctx)
    armmap = S.well_index(ctx.tidy).set_index("well_uid").arm
    bt = arch.bursts.assign(arm=arch.bursts.well_uid.map(armmap))
    bt = bt[bt.arm.isin(D.MAIN_ARMS)].copy()
    bt["tauc"] = pd.cut(bt.tau, DEFAULT_TAU_EDGES, right=False).map(
        lambda c: (c.left + c.right) / 2 if pd.notna(c) else np.nan)
    arms = [a for a in D.MAIN_ARMS if a in set(bt.arm)]

    gs = fig.add_gridspec(len(arms), 1, hspace=0.22)
    prev = None
    for i, arm in enumerate(arms):
        ax = fig.add_subplot(gs[i], sharex=prev)
        prev = ax
        tau_band(ax)
        sub = bt[bt.arm == arm]
        wtf = (sub.groupby(["well_uid", "tauc", "arch"]).size()
               .unstack("arch", fill_value=0))
        wtf = wtf.div(wtf.sum(axis=1), axis=0)
        comp = wtf.groupby("tauc").mean().sort_index()
        if not comp.empty:
            ys = [comp[a].values if a in comp.columns else np.zeros(len(comp))
                  for a in range(arch.k)]
            ax.stackplot(comp.index.astype(float), *ys,
                         colors=[A.arch_color(a) for a in range(arch.k)],
                         labels=[A.arch_label(arch, a) for a in range(arch.k)],
                         zorder=2)
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
    fig.text(0.99, 0.01, f"{len(arch.bursts)} bursts · "
             f"{arch.bursts.well_uid.nunique()} wells", ha="right", va="bottom",
             fontsize=5.5, color=MUTED)
    return arch.stats


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
