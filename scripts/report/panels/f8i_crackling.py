"""F8i — Crackling-noise scaling of one well, early vs late."""
from __future__ import annotations

import numpy as np

from ..panel_base import panel_main
from ..report_style import MUTED
from . import _f8_common as C

NAME = "F8i_crackling"
FIGSIZE = (3.4, 2.8)
CAPTION = (
    "Crackling-noise scaling of the same representative well as F8h, early versus "
    "late: mean avalanche size against avalanche duration on log–log axes. At "
    "criticality the slope of this relation equals (α−1)/(τ−1), the value "
    "predicted by the separately fitted duration and size exponents; the gap "
    "between the fitted and predicted slope is the distance-to-criticality "
    "coefficient (DCC) quoted in the legend for each time point. Illustrative "
    "single well, not a group statistic."
)


def build(ax, ctx):
    from ..fig8_criticality import _avalanches

    e_row, l_row, wuid = C.example(ctx)
    for row, colour, lab in ((e_row, C.EARLY_C, "early"), (l_row, C.LATE_C, "late")):
        if row is None:
            continue
        av = _avalanches(ctx.analysis_root, row)
        if av.empty:
            continue
        mean_size = av.groupby("duration")["size"].mean()
        mean_size = mean_size[mean_size.index > 0]
        dcc = float(row.get("dcc", np.nan))
        ax.loglog(mean_size.index, mean_size.values, "o", ms=3, color=colour,
                  alpha=0.75, label=f"{lab} (τ{int(row.tau):+d})  DCC={dcc:.2f}")
    ax.set_xlabel("avalanche duration (time bins)", fontsize=7)
    ax.set_ylabel("mean avalanche size", fontsize=7)
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax.set_title(f"{wuid.split('|')[0]} · {C.ARM}", loc="left", fontsize=6.5,
                 color=MUTED)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
