"""F7f — STTC as a function of inter-unit distance, per group."""
from __future__ import annotations

from ..panel_base import panel_main
from ..report_style import MUTED

NAME = "F7f_sttc_distance"
FIGSIZE = (3.6, 2.7)
CAPTION = (
    "How functional coupling decays with physical distance. Every unit pair from "
    "the full STTC matrix of the sampled wells is binned by the distance between "
    "the two units on the electrode plane, and the mean STTC of each bin is "
    "plotted per group. Wells are sampled at matched culture age so the curves "
    "are not confounded by maturation; note this is a cross-sectional comparison "
    "across wells and chips, not a within-well treatment contrast, so it "
    "describes the spatial structure of coupling rather than testing a treatment "
    "effect."
)


def build(ax, ctx):
    from ..fig7_spatial_connectivity import _panel_sttc_distance

    _panel_sttc_distance(ax, ctx.analysis_root, ctx.main)
    ax.set_title("")                     # composed-figure title; not wanted here
    ax.annotate("cross-sectional, matched DIV", xy=(0.99, 0.99),
                xycoords="axes fraction", ha="right", va="top",
                fontsize=5.5, color=MUTED)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
