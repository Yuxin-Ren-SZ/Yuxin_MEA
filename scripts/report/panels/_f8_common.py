"""Shared example-well selection for the F8 avalanche illustrations.

The avalanche PDF and the crackling-scaling panels must describe the *same* well
at the *same* two time points, or they cannot be read together. They are separate
scripts, so the choice lives here.
"""
from __future__ import annotations

from ..trajectory import pick_early_late, pick_example_well

ARM = "IVH_Late"
METRIC = "branching_ratio_mr"
#: Early vs late colours for the two time points (kept distinct from the group
#: palette — these are time points within one well, not treatment arms).
EARLY_C, LATE_C = "#4477AA", "#CC3311"


def example(ctx):
    """``(early_row, late_row, well_uid)`` for a representative treated well."""
    key = f"f8_example|{ARM}|{METRIC}"

    def build():
        wuid = pick_example_well(ctx.tidy, ARM, metric=METRIC)
        if wuid is None:
            return (None, None, None)
        e, l = pick_early_late(ctx.tidy, wuid)
        return (e, l, wuid)

    e, l, wuid = ctx.cache.get(key, build)
    if e is None:
        raise RuntimeError(f"no representative {ARM} well with {METRIC}")
    return e, l, wuid
