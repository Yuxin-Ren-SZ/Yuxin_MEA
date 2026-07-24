"""S6 — per-gene qPCR trajectories, both reference frames.

Supplement to F3, which compresses the same numbers into one heatmap. Here each
gene gets its own panel, twice, so the trajectory shape is visible:

* **Block A — vs ``CT DIV 7``.** What was actually measured: every arm's ΔΔCq
  against the shared baseline taken on the treatment day. These curves mix
  *development* with *treatment* — the Control line is the maturation, and it is
  the largest term for the synaptic genes (SLC17A7 rises +4.5 log2 by D7 with no
  treatment at all).
* **Block B — vs the time-matched Control.** The same data with that maturation
  removed (see :func:`~scripts.report.fig3_qpcr.treatment_effects`). Zero means
  "indistinguishable from an untreated culture of the same age". Control is not
  drawn: it is identically 0 by construction.

Genes wrap at four columns inside each block, so the layout is unchanged whether
four genes are done or all thirteen.

No error bars: at n = 1 chip the only available spread is technical-duplicate
SEM, which is not biological replication. Those SEMs are kept in
``F3_qpcr_effects.csv``, and the builders draw across-chip mean ± SEM
automatically once more than one chip is loaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fig3_qpcr import (celltype_ordered_genes, control_reference, gene_label,
                        treatment_effects)
from .report_style import caption, group_color, ordered_groups, save_fig

#: A QC-failed point is still drawn in place if it sits within this fraction of
#: the QC-pass span beyond the axis; further out it would flatten the panel, so
#: it is pinned to the margin as text instead.
_QC_OVERSHOOT = 0.5

#: Panels per row inside each block.
NCOL = 4


# --------------------------------------------------------------------------- #
# Block A — vs the CT DIV 7 calibrator
# --------------------------------------------------------------------------- #
def _panel(ax, sub, days):
    """Draw one gene's QC-pass traces; return the y-limits they imply."""
    for grp in ordered_groups(sub.group.dropna().unique()):
        good = sub[(sub.group == grp) & (sub.qc == "ok")].sort_values("day")
        # Anchor every arm at the shared calibrator (day 0, log2 = 0).
        line = good if good.day.min() == 0 else pd.concat(
            [pd.DataFrame({"day": [0.0], "log2_fold": [0.0]}), good],
            ignore_index=True)
        ax.plot(line.day, line.log2_fold, marker="o", ms=3.5, lw=1.2,
                color=group_color(grp), label=grp, zorder=3)

    ax.axhline(0.0, ls="--", color="0.6", lw=0.7, zorder=1)
    ax.set_xticks(days)
    ok = sub[sub.qc == "ok"]
    lo, hi = (ok.log2_fold.min(), ok.log2_fold.max()) if len(ok) else (-1.0, 1.0)
    pad = max(0.35, 0.15 * (hi - lo))
    return lo - pad, hi + pad


def _draw_qc_fails(ax, sub, ylo, yhi):
    """Open markers for in-range QC failures, margin text for off-scale ones.

    Called after the limits are known so a clipped marker can never render as a
    stray arc at the axis edge, and so one artefact (MAP2 ``CT D14`` = +6.9 log2
    / 119x) cannot flatten the panel.
    """
    bad = sub[sub.qc == "ref_cq_outlier"]
    slack = _QC_OVERSHOOT * (yhi - ylo)
    inside = bad[bad.log2_fold.between(ylo - slack, yhi + slack)]
    for _, r in inside.iterrows():
        ax.plot([r.day], [r.log2_fold], "o", ms=4.5, mfc="none",
                mec=group_color(r.group), mew=1.0, ls="none", zorder=4)
    if len(inside):
        ylo = min(ylo, inside.log2_fold.min() - 0.2)
        yhi = max(yhi, inside.log2_fold.max() + 0.2)
    ax.set_ylim(ylo, yhi)

    for _, r in bad.drop(inside.index).iterrows():
        up = r.log2_fold > yhi
        ax.annotate(f"{r.fold:.3g}x off-scale, QC-fail",
                    xy=(r.day, 0.99 if up else 0.01),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="top" if up else "bottom",
                    fontsize=5.5, color="0.35",
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
    for _, r in sub[sub.qc == "no_amplification"].iterrows():
        ax.annotate("n.a.", xy=(r.day, 0.02), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=5.5, color="0.45",
                    bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))


# --------------------------------------------------------------------------- #
# Block B — vs the time-matched control
# --------------------------------------------------------------------------- #
def _effect_panel(ax, sub, days, n_chips):
    """One gene's treatment-effect trajectories, maturation removed.

    Interpolated-control points are drawn open on a dashed segment, so a value
    that rests on a reconstructed denominator never looks like a measured one.
    """
    for grp in ordered_groups(sub.group.dropna().unique()):
        g = sub[(sub.group == grp) & sub.log2_effect.notna()].sort_values("day")
        if g.empty:
            continue
        colour = group_color(grp)
        rows = list(g.itertuples())
        # A segment touching an interpolated point is dashed, so a value resting
        # on a reconstructed denominator never reads as a measured one.
        for a, b in zip(rows, rows[1:]):
            dashed = "interpolated" in (a.control_source, b.control_source)
            ax.plot([a.day, b.day], [a.log2_effect, b.log2_effect],
                    ls="--" if dashed else "-", lw=1.2, color=colour, zorder=3)
        if n_chips > 1 and g.sem_effect.notna().any():
            ax.errorbar(g.day, g.log2_effect, yerr=g.sem_effect, fmt="none",
                        ecolor=colour, capsize=1.8, elinewidth=0.7, zorder=3)
        measured = g[g.control_source == "measured"]
        interp = g[g.control_source == "interpolated"]
        ax.plot(measured.day, measured.log2_effect, "o", ms=3.5, color=colour,
                ls="none", label=grp, zorder=4)
        # open *square* — distinguishable from the open circle block A uses
        # for a QC failure, which is a different kind of caveat.
        ax.plot(interp.day, interp.log2_effect, "s", ms=4.0, mfc="white",
                mec=colour, mew=1.1, ls="none", zorder=4)

    ax.axhline(0.0, ls="--", color="0.5", lw=0.9, zorder=1)
    ax.set_xticks(days)
    for _, r in sub[sub.qc == "no_amplification"].iterrows():
        ax.annotate("n.a.", xy=(r.day, 0.02), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=5.5, color="0.45")


# --------------------------------------------------------------------------- #
def build_s6(tidy: pd.DataFrame, genes: list[str] | None = None):
    import matplotlib.pyplot as plt

    genes = genes or celltype_ordered_genes(tidy.gene.dropna().unique())
    days = sorted(tidy.day.dropna().unique())
    eff_days = [d for d in days if d > 0]
    effects = treatment_effects(tidy)
    n_chips = int(tidy.sample_id.nunique(dropna=True) or 1)
    interp = sorted(set(control_reference(tidy)
                        .query("control_source == 'interpolated'").day))

    nrow = int(np.ceil(len(genes) / NCOL))
    fig, axes = plt.subplots(2 * nrow, NCOL, squeeze=False,
                             figsize=(2.55 * NCOL, 2.35 * 2 * nrow))

    for i, gene in enumerate(genes):
        r, c = divmod(i, NCOL)
        # --- block A: vs the calibrator ---
        ax = axes[r, c]
        sub = tidy[tidy.gene == gene]
        _draw_qc_fails(ax, sub, *_panel(ax, sub, days))
        ax.set_title(gene_label(gene), loc="left", fontweight="bold", fontsize=7.5)
        if c == 0:
            ax.set_ylabel("log$_2$ fold vs CT DIV 7")
        ax.set_xlabel("days post-treatment")

        # --- block B: vs the time-matched control ---
        ax = axes[nrow + r, c]
        _effect_panel(ax, effects[effects.gene == gene], eff_days, n_chips)
        ax.set_title(gene_label(gene), loc="left", fontweight="bold", fontsize=7.5)
        if c == 0:
            ax.set_ylabel("log$_2$ fold vs\ntime-matched control")
        ax.set_xlabel("days post-treatment")

    for i in range(len(genes), nrow * NCOL):
        r, c = divmod(i, NCOL)
        axes[r, c].axis("off")
        axes[nrow + r, c].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    extra = [plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="0.35",
                        ms=4.5, label="failed QC (ref. Cq shift)"),
             plt.Line2D([], [], marker="s", ls="--", mfc="white", mec="0.35",
                        color="0.35", ms=4.0, lw=1.0,
                        label="interpolated control (B)")]
    fig.legend(handles + extra, labels + [h.get_label() for h in extra],
               loc="lower center", ncol=min(6, len(labels) + 2), fontsize=6.5,
               bbox_to_anchor=(0.5, 0.052))

    fig.suptitle("Figure S6 — qPCR trajectories: measured (A) and with "
                 "development removed (B)", fontsize=9, x=0.02, ha="left",
                 fontweight="bold")
    tag = "; ".join(f"D{int(d)} control interpolated" for d in interp) or \
        "all controls measured"
    fig.text(0.02, 0.038,
             "A: ΔΔCq vs the CT DIV 7 baseline — development and treatment "
             "combined; the Control curve is the maturation term.\n"
             "B: each arm minus the time-matched Control, so the calibrator "
             "cancels and 0 = no difference from an untreated culture of the "
             f"same age ({tag}); dashed segments touch an interpolated point.\n"
             f"n = {n_chips} chip, technical duplicates only — no error bars "
             "and no statistics.",
             fontsize=5.5, color="0.35", ha="left", va="top")
    fig.tight_layout(rect=(0, 0.105, 1, 0.86))
    # Explicit A/B row-framing headers (development-removed framing made visible,
    # not buried in a footnote), placed clear above each block's gene titles.
    a_top = axes[0, 0].get_position().y1
    b_top = axes[nrow, 0].get_position().y1
    fig.text(0.02, a_top + 0.048,
             "A  ΔΔCq vs CT DIV 7 (development + treatment)",
             fontweight="bold", fontsize=8.5, va="bottom")
    fig.text(0.02, b_top + 0.032,
             "B  vs time-matched Control (development removed → 0 = no difference)",
             fontweight="bold", fontsize=8.5, va="bottom")
    caption(fig,
        "Per-gene qPCR expression trajectories over days post-treatment, one "
        "line per condition (see the in-panel note for how A and B differ: A = "
        "ΔΔCq vs the CT DIV 7 baseline, B = each arm minus the time-matched "
        "Control). Descriptive, n=1 chip (CX169), technical duplicates only — no "
        "error bars or statistics.")
    return fig


def render(figure_root, qpcr_dir, genes=None):
    from .fig3_qpcr import load_qpcr_dir
    fig = build_s6(load_qpcr_dir(qpcr_dir), genes)
    return save_fig(fig, "S6_qpcr_trajectories", figure_root, subdir="supp")
