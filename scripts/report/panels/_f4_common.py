"""Shared selection for the two F4 rasters.

Both raster panels must show the *same* Control well on the *same* axes, or the
"immature vs mature" comparison is confounded by the layout. They are separate
scripts, so the choice and the shared y-height live here rather than in either
one.
"""
from __future__ import annotations

from ..raster import FULL_WINDOW, load_spikes, sorted_units

#: Control well, recording immediately before treatment vs 14 days after.
PRE_TAU, POST_TAU = 0.0, 14.0
ARM = "Control"
#: Cap so a 180-unit raster stays legible at panel size; the most active units
#: are kept (see :func:`raster.plot_raster`) and the count is annotated.
MAX_UNITS = 120


def pair(ctx):
    """``(pre_row, post_row, well_uid, n_units)`` — n_units is the shared height."""
    pre, post, wuid = ctx.example_pair(ARM, PRE_TAU, POST_TAU)
    if pre is None:
        raise RuntimeError(f"no {ARM} well covering tau {PRE_TAU} and {POST_TAU}")

    def _n(row):
        return len(sorted_units(load_spikes(ctx.analysis_root, row), FULL_WINDOW))

    n = ctx.cache.get(f"f4_units|{wuid}", lambda: min(MAX_UNITS, max(_n(pre), _n(post))))
    return pre, post, wuid, n


def label(row, when: str) -> str:
    return f"{when} · DIV {int(row.DIV)} · τ{int(row.tau):+d}"
