"""F4d — Network-burst rate vs treatment day."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ..trajectory import plot_trajectory

NAME = "F4d_nb_rate"
FIGSIZE = (3.4, 2.6)
CAPTION = (
    "Network-burst rate across maturation, aligned on treatment day (τ). Line = "
    "per-arm median of the well-median burst rate (one value per well per τ bin, "
    "so recordings are not treated as replicates); shaded band = 95% bootstrap CI "
    "across wells. Dashed verticals and the tinted span mark the experiment "
    "window, τ0 (treatment) to τ+21. The rise with τ is the maturation baseline "
    "every treatment contrast in F5–F8 is read against."
)
METRIC = "nb_rate"


def build(ax, ctx):
    plot_trajectory(ax, ctx.main, METRIC, D.MAIN_ARMS, legend=True)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
