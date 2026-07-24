"""F4a — Representative raster, immediately before treatment (immature network)."""
from __future__ import annotations

from ..panel_base import panel_main
from ..raster import FULL_WINDOW, load_spikes, plot_raster
from . import _f4_common as C

NAME = "F4a_raster_pre"
FIGSIZE = (4.2, 2.8)
CAPTION = (
    "Spike raster of one Control well at the recording immediately before "
    "treatment (τ0), showing the immature network. The full 300 s recording is "
    "shown; each row is one curated single unit, ordered by firing rate (lowest "
    "at the bottom). Unit count and time axis are pinned to the same values as "
    "the matched post-treatment raster (F4b) so the two panels are directly "
    "comparable. Illustrative single-well example, not a group statistic."
)


def build(ax, ctx):
    pre, _post, wuid, n = C.pair(ctx)
    spikes = load_spikes(ctx.analysis_root, pre)
    plot_raster(ax, spikes, t_win=FULL_WINDOW, max_units=C.MAX_UNITS, n_units=n,
                label=f"{C.label(pre, 'pre-treatment')}  ·  {wuid.split('|')[0]}")
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
