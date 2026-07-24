"""F5a — How the detector sees a recording: per-unit HMM burst posteriors."""
from __future__ import annotations

import numpy as np

from ..panel_base import panel_main
from ..report_style import MUTED

NAME = "F5a_hmm_posteriors"
FIGSIZE = (4.6, 2.9)
CAPTION = (
    "The algorithm's view of one recording. Top: per-unit posterior probability "
    "of being in the burst state, P(burst), from the hidden Markov model the "
    "detector fits to each unit; one row per curated unit, ordered by mean burst "
    "occupancy, colour from 0 (blue-black) to 1 (bright). Bottom, sharing the "
    "same time axis: the co-burst fraction — the proportion of units with "
    "P(burst) > 0.5 at each time bin — which is the population signal network "
    "bursts are called from. Together they show that network bursts are recovered "
    "as coincident high-posterior episodes across many units, not as a threshold "
    "on a summed firing rate. Illustrative single recording, not a group statistic."
)
#: Seconds shown. Long enough for several bursts, short enough that individual
#: burst episodes stay resolvable at panel width.
T_WIN = (0.0, 60.0)
ARM = "IVH_Late"


def build_fig(fig, ctx):
    from ..fig6_ml import _panel_posterior

    row = ctx.ml_trace_row(ARM)
    if row is None:
        raise RuntimeError(f"no {ARM} recording with an ML debug trace on disk")
    tr = ctx.ml_trace(row)
    if getattr(tr, "posterior_matrix", None) is None:
        raise RuntimeError("debug trace carries no posterior matrix")

    gs = fig.add_gridspec(2, 2, width_ratios=[30, 1], height_ratios=[3, 1],
                          hspace=0.08, wspace=0.04)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[0, 1])
    _panel_posterior(ax_heat, ax_trace, cax, tr, t_win=T_WIN)
    # _panel_posterior stamps its own composed-figure title; replace it with the
    # provenance line a standalone panel needs.
    ax_heat.set_title(
        f"{row.well_uid.split('|')[0]} · {ARM} · DIV {int(row.DIV)} · τ{int(row.tau):+d}",
        loc="left", fontsize=6.5, color=MUTED)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
