"""F4e — Burst variability (inter-burst-interval CV) vs treatment day."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ..trajectory import plot_trajectory

NAME = "F4e_ibi_cv"
FIGSIZE = (3.4, 2.6)
CAPTION = (
    "Burst variability across maturation: coefficient of variation of the "
    "inter-burst intervals (CV = SD/mean), aligned on treatment day (τ). Low CV "
    "means a metronomic burst rhythm, high CV an irregular one. Line = per-arm "
    "median of the "
    "well-median CV (one value per well per τ bin); shaded band = 95% bootstrap "
    "CI across wells. Dashed verticals and the tinted span mark the experiment "
    "window, τ0 to τ+21."
)
METRIC = "nb_ibi_cv"


def build(ax, ctx):
    plot_trajectory(ax, ctx.main, METRIC, D.MAIN_ARMS, legend=True)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
