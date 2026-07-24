"""F4f — Network-burst duration vs treatment day."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ..trajectory import plot_trajectory

NAME = "F4f_nb_duration"
FIGSIZE = (3.4, 2.6)
CAPTION = (
    "Network-burst duration across maturation, aligned on treatment day (τ). "
    "Line = per-arm median of the well-median burst duration (one value per well "
    "per τ bin); shaded band = 95% bootstrap CI across wells. Dashed verticals "
    "and the tinted span mark the experiment window, τ0 to τ+21. Bursts shorten "
    "and sharpen as the culture matures; that developmental trend is the baseline "
    "the treatment difference-in-differences removes."
)
METRIC = "nb_duration_mean"


def build(ax, ctx):
    plot_trajectory(ax, ctx.main, METRIC, D.MAIN_ARMS, legend=True)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
