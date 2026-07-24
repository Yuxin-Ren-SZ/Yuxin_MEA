"""F2a — Culture and recording schedule."""
from __future__ import annotations

import numpy as np

from ..panel_base import panel_main
from ..report_style import GRID, INK, MUTED, group_color

NAME = "F2a_schedule"
FIGSIZE = (7.2, 2.8)
CAPTION = (
    "Experimental schedule, on two aligned axes: days in vitro along the top and "
    "days relative to treatment (τ) along the bottom. Treatment is applied on τ0 "
    "and the planned experiment window runs to τ+21 (tinted). Ticks on the "
    "recording row mark every HD-MEA network scan actually performed, read from "
    "the analysis tree rather than from the protocol, so the row shows the "
    "recordings the figures are built on; each chip contributes its own row "
    "because the chips were plated on different dates and therefore reach the "
    "same treatment day at different culture ages — which is why every figure "
    "aligns on τ, not on DIV. Wet-lab events (plating, medium changes, sampling "
    "for qPCR and immunofluorescence) come from the protocol; they are not "
    "recorded in the pipeline and are set as editable constants at the top of "
    "this script."
)

# --------------------------------------------------------------------------- #
# EDIT ME — protocol events. The pipeline stores recording dates but no wet-lab
# log, so these cannot be derived from the data; they are stated here so the
# figure and the protocol can be kept in step by editing one place.
# --------------------------------------------------------------------------- #
#: ``(tau_day, label)`` — culture-side events, in days relative to treatment.
PROTOCOL_EVENTS = [
    (-7.0, "plating"),
    (0.0, "treatment"),
    (21.0, "end of window"),
]
#: Medium changes: ``(first_tau, every_n_days)``. Set to ``None`` to hide the row.
MEDIUM_CHANGE = (-5.0, 3.0)
#: Days on which cultures were sampled for qPCR / immunofluorescence.
SAMPLING_DAYS = [0.0, 7.0, 14.0, 21.0]


def _recording_rows(ctx):
    """``[(chip, taus, div_offset)]`` — every network scan, per chip."""
    t = ctx.main
    rows = []
    for chip, sub in t.groupby("sample_id"):
        taus = np.sort(sub.tau.unique().astype(float))
        # DIV at tau 0 — the culture age at treatment, which differs per chip.
        at0 = sub.loc[sub.tau == 0, "DIV"]
        offset = float(at0.iloc[0]) if len(at0) else float(
            sub.DIV.min() - sub.tau.min())
        rows.append((str(chip), taus, offset))
    return sorted(rows)


def build(ax, ctx):
    rows = _recording_rows(ctx)
    lo = min(min(taus.min() for _, taus, _ in rows), PROTOCOL_EVENTS[0][0]) - 1
    hi = max(max(taus.max() for _, taus, _ in rows), 22.0) + 1

    ax.axvspan(0, 21, color=GRID, alpha=0.45, lw=0, zorder=0)
    for x in (0, 21):
        ax.axvline(x, ls="--", color=MUTED, lw=0.9, zorder=1)

    y = 0.0
    yticks, ylabels = [], []

    # Protocol events + sampling + medium changes, on one "culture" row.
    if MEDIUM_CHANGE:
        first, every = MEDIUM_CHANGE
        mc = np.arange(first, hi, every)
        ax.scatter(mc, np.full(len(mc), y), marker="|", s=45, color=MUTED,
                   lw=0.9, zorder=3)
    for tau, label in PROTOCOL_EVENTS:
        ax.scatter([tau], [y], marker="v", s=26, color=INK, zorder=4)
        ax.annotate(label, xy=(tau, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, color=INK)
    yticks.append(y); ylabels.append("culture\n(| medium change)")

    y -= 1.0
    ax.scatter(SAMPLING_DAYS, np.full(len(SAMPLING_DAYS), y), marker="o", s=22,
               facecolor="white", edgecolor=INK, lw=0.9, zorder=4)
    yticks.append(y); ylabels.append("qPCR / IF\nsampling")

    for chip, taus, offset in rows:
        y -= 1.0
        ax.scatter(taus, np.full(len(taus), y), marker="|", s=70,
                   color=group_color("Control"), lw=1.1, zorder=3)
        ax.annotate(f"treated at DIV {offset:.0f}", xy=(taus.max(), y),
                    xytext=(4, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=5.5, color=MUTED)
        yticks.append(y); ylabels.append(f"{chip}\nHD-MEA scans")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_ylim(y - 0.7, 0.9)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("treatment day (tau)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)

    # Second x-axis in DIV. Chips differ, so it is labelled with the modal
    # offset and called out as approximate rather than implying one true DIV.
    offsets = [o for _, _, o in rows]
    med = float(np.median(offsets)) if offsets else 0.0
    top = ax.secondary_xaxis("top", functions=(lambda x: x + med,
                                               lambda d: d - med))
    top.set_xlabel(f"days in vitro (chips treated at DIV {min(offsets):.0f}"
                   f"–{max(offsets):.0f}; axis uses {med:.0f})", fontsize=7)
    top.tick_params(labelsize=6.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
