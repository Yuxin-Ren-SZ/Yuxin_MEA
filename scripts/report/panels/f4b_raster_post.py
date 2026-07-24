"""F4b — Representative raster, 14 days after treatment day (mature network)."""
from __future__ import annotations

from ..panel_base import panel_main
from ..raster import FULL_WINDOW, load_spikes, plot_raster
from . import _f4_common as C

NAME = "F4b_raster_post"
FIGSIZE = (4.2, 2.8)
CAPTION = (
    "Spike raster of the same Control well as F4a, 14 days later (τ+14), showing "
    "the matured network: coordinated network bursts emerge against a quiet "
    "baseline. Full 300 s recording; each row is one curated single unit, ordered "
    "by firing rate (lowest at the bottom). Axes are pinned to the same unit count "
    "and time window as F4a, so the visible difference is the data, not the "
    "layout. Illustrative single-well example, not a group statistic."
)


def build(ax, ctx):
    _pre, post, wuid, n = C.pair(ctx)
    spikes = load_spikes(ctx.analysis_root, post)
    plot_raster(ax, spikes, t_win=FULL_WINDOW, max_units=C.MAX_UNITS, n_units=n,
                label=f"{C.label(post, 'post-treatment')}  ·  {wuid.split('|')[0]}")
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
