"""F8b — Distance to criticality (DCC) vs treatment day."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ..trajectory import plot_trajectory

NAME = "F8b_dcc_trajectory"
FIGSIZE = (3.6, 2.7)
CAPTION = (
    "Distance to criticality (DCC) across treatment time. DCC is the gap between "
    "the crackling-noise exponent fitted from the data and the value predicted "
    "from the avalanche size and duration exponents if the network were critical; "
    "0 means the two agree and larger values mean the network sits further from "
    "criticality. Line = per-arm median across wells (one value per well per "
    "treatment-day bin), shaded band = 95% bootstrap CI across wells. Vertical "
    "dashed lines and the tinted span mark the experiment window, τ0 to τ+21."
)


def build(ax, ctx):
    plot_trajectory(ax, ctx.main, "dcc", D.MAIN_ARMS, legend=True,
                    ylabel="DCC (distance to criticality)")
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
