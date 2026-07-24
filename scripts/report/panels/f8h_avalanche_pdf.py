"""F8h — Avalanche-size distribution of one well, early vs late."""
from __future__ import annotations

import numpy as np

from ..panel_base import panel_main
from ..report_style import MUTED
from . import _f8_common as C

NAME = "F8h_avalanche_pdf"
FIGSIZE = (3.4, 2.8)
CAPTION = (
    "Avalanche-size distribution of one representative treated well, early "
    "(shortly after treatment) versus late, on log–log axes. Points are the "
    "empirical probability density in logarithmically spaced size bins; the line "
    "through each is the maximum-likelihood power law with the fitted size "
    "exponent τ. A straight line on these axes is the signature of scale-free "
    "avalanche statistics, which critical networks produce. Illustrative single "
    "well, not a group statistic — the group-level test is the "
    "difference-in-differences panels."
)


def build(ax, ctx):
    from ..fig8_criticality import _avalanches, _loglog_pdf

    e_row, l_row, wuid = C.example(ctx)
    for row, colour, lab in ((e_row, C.EARLY_C, "early"), (l_row, C.LATE_C, "late")):
        if row is None:
            continue
        av = _avalanches(ctx.analysis_root, row)
        if av.empty:
            continue
        _loglog_pdf(ax, av["size"], colour, f"{lab} (τ{int(row.tau):+d})",
                    float(row.get("aval_tau", np.nan)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("avalanche size (spikes)", fontsize=7)
    ax.set_ylabel("probability density", fontsize=7)
    ax.legend(fontsize=6, loc="lower left", frameon=False)
    ax.set_title(f"{wuid.split('|')[0]} · {C.ARM}", loc="left", fontsize=6.5,
                 color=MUTED)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
