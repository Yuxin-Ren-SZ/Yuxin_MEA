"""Shared spike-raster drawing for the report panels.

One drawing routine so every raster in the report reads the same way: units on
the y-axis sorted by firing rate (quiet at the bottom, most active at the top),
time on the x-axis, and — crucially — an explicit unit-count and time window so
two rasters of the same well at different days are visually comparable rather
than silently rescaled to whatever each recording happened to contain.
"""
from __future__ import annotations

import numpy as np

from . import load as L
from .report_style import INK, MUTED

#: Recordings are 300 s; the full window is the default so bursting structure and
#: inter-burst quiet are both visible (a 60 s crop hides the slow rhythm).
FULL_WINDOW = 300.0


def sorted_units(spikes: dict, t_win: float = FULL_WINDOW) -> list[tuple[int, np.ndarray]]:
    """``[(unit_id, spike_times)]`` inside ``t_win``, ascending by firing rate."""
    out = []
    for uid, st in spikes.items():
        st = np.asarray(st, float)
        st = st[(st >= 0) & (st <= t_win)]
        out.append((uid, st))
    out.sort(key=lambda kv: len(kv[1]))          # low -> high firing rate
    return out


def plot_raster(
    ax, spikes: dict, *,
    t_win: float = FULL_WINDOW,
    max_units: int | None = None,
    n_units: int | None = None,
    color: str = INK,
    linewidth: float = 0.25,
    label: str = "",
) -> int:
    """Draw a firing-rate-sorted raster; return the number of units drawn.

    ``max_units`` keeps the densest units (the tail of the sorted list) when a
    well has more than the panel can show. ``n_units`` instead pins the y-axis to
    a fixed height — pass the same value to two panels so an "immature" and a
    "mature" raster of the same well are drawn on identical axes and the
    difference the reader sees is the data, not the layout.
    """
    units = sorted_units(spikes, t_win)
    if max_units is not None and len(units) > max_units:
        units = units[-max_units:]               # keep the most active
    for i, (_, st) in enumerate(units):
        if len(st):
            ax.eventplot(st, lineoffsets=i, linelengths=0.85,
                         linewidths=linewidth, colors=color)
    ax.set_xlim(0, t_win)
    ax.set_ylim(-1, (n_units if n_units is not None else max(len(units), 1)))
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("unit (sorted by firing rate)", fontsize=7)
    ax.tick_params(labelsize=6)
    if label:
        # Above the axes, never inside: a dense raster leaves no clear space and
        # an overlaid label lands on the data.
        ax.set_title(label, loc="left", fontsize=6.5, color=MUTED, pad=3)
    return len(units)


def load_spikes(analysis_root, row) -> dict:
    """``{unit_id: spike_times}`` for one well-recording (empty dict if absent)."""
    try:
        return L.load_curated_spikes(analysis_root, row)
    except (FileNotFoundError, OSError):
        return {}
