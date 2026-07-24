"""F7n — Burst modulation index vs treatment day."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ..trajectory import plot_trajectory

NAME = "F7n_burst_modulation"
FIGSIZE = (3.6, 2.7)
CAPTION = (
    "Burst modulation index across treatment time — how sharply the network's "
    "bursting departs from a rate-matched Poisson process, as scored by the "
    "hidden Markov model. Higher values mean crisper, better-separated bursts; a "
    "value near zero means firing is essentially random in time. Line = per-arm "
    "median across wells (one value per well per treatment-day bin), shaded band "
    "= 95% bootstrap CI across wells. Vertical dashed lines and the tinted span "
    "mark the experiment window, treatment day τ0 to τ+21."
)


def build(ax, ctx):
    plot_trajectory(ax, ctx.main, "burst_modulation_index", D.MAIN_ARMS, legend=True)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
