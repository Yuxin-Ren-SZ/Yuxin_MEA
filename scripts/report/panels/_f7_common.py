"""Shared selection and drawing for the F7 connectivity example panels.

One place decides which recording represents each group, so the CCG, STTC-graph
and activity-field panels all describe comparable wells rather than each picking
its own favourite.
"""
from __future__ import annotations

import numpy as np

from .. import load as L
from ..report_style import GRID, INK, MUTED, group_color

#: Example window in treatment days — late enough that the network is mature and
#: any treatment effect has had time to appear, and shared by every F7 example.
TAU_WINDOW = (14, 21)
#: The MaxWell chip's electrode plane. Activity fields are rendered on this fixed
#: rectangle so every field panel covers the same physical area (see f7e).
CHIP_EXTENT = (0.0, 3850.0, 0.0, 2100.0)


def pick_recording(ctx, group: str, min_sig: int = 4, metric: str = "mean_sttc"):
    """A *representative* recording of ``group`` inside the example window.

    Representative, not maximal: the recording whose connectivity strength is
    nearest the group's median, among those with enough significant
    cross-correlogram edges to show. Picking the well with the most significant
    edges would put an outlier in the figure.
    """
    key = f"f7_rec|{group}|{min_sig}|{metric}"

    def build():
        lo, hi = TAU_WINDOW
        sub = ctx.main
        sub = sub[(sub.canonical_group == group) & (sub.tau >= lo) & (sub.tau <= hi)
                  & sub[metric].notna()]
        if sub.empty:
            return None
        med = sub[metric].median()
        order = (sub[metric] - med).abs().to_numpy().argsort()
        fallback = None
        for i in order:
            r = sub.iloc[i]
            n = n_readable_edges(ctx, r)
            if n >= min_sig:
                return r
            if n > 0 and fallback is None:
                fallback = r
        return fallback if fallback is not None else sub.iloc[order[0]]

    row = ctx.cache.get(key, build)
    if row is None:
        raise RuntimeError(f"no {group} recording in tau {TAU_WINDOW} with connectivity")
    return row


#: A correlogram is only readable when both units fired enough to fill 200 bins.
#: 300 spikes over a 300 s recording is 1 Hz — below that a single coincidence
#: dominates the peak z-score and the plot is noise with one tall bar.
MIN_SPIKES = 300


def top_ccg_edges(ctx, row, n: int = 4, min_spikes: int = MIN_SPIKES):
    """``(ccg dict, [(row_index, edge_row)])`` for the n clearest causal edges.

    Significant edges are ranked by peak z **after** dropping sparsely-sampled
    pairs: with only a handful of spikes a lone coincidence produces a large z
    against a near-zero baseline, which is statistically real but visually
    indistinguishable from noise. Rows of the CCG arrays are keyed by ``pairs``,
    never by position in ``edges.parquet`` (the edge table can be longer when the
    pair cap trips), so the join is explicit.
    """
    ccg = L.load_ccg(ctx.analysis_root, row)
    edges = L.load_edges(ctx.analysis_root, row)
    index = {(int(u), int(v)): i for i, (u, v) in enumerate(ccg["pairs"])}
    n_ref = np.asarray(ccg["n_ref_spikes"], float)
    n_tgt = np.asarray(ccg["n_tgt_spikes"], float)

    sig = edges[edges.ccg_sig] if "ccg_sig" in edges.columns else edges
    if sig.empty:
        sig = edges
    cand = []
    for _, e in sig.iterrows():
        i = index.get((int(e.u), int(e.v)))
        if i is None:
            continue
        cand.append((i, e, min(n_ref[i], n_tgt[i])))
    dense = [c for c in cand if c[2] >= min_spikes]
    dense = dense or cand                     # never return nothing at all
    dense.sort(key=lambda c: float(c[1].ccg_peak_z), reverse=True)
    return ccg, [(i, e) for i, e, _ in dense[:n]]


def n_readable_edges(ctx, row, min_spikes: int = MIN_SPIKES) -> int:
    """Count of significant edges whose correlogram is well enough sampled to plot."""
    try:
        ccg = L.load_ccg(ctx.analysis_root, row)
        edges = L.load_edges(ctx.analysis_root, row)
    except (FileNotFoundError, OSError, KeyError):
        return 0
    if "ccg_sig" not in edges.columns or edges.empty:
        return 0
    index = {(int(u), int(v)): i for i, (u, v) in enumerate(ccg["pairs"])}
    n_ref = np.asarray(ccg["n_ref_spikes"], float)
    n_tgt = np.asarray(ccg["n_tgt_spikes"], float)
    n = 0
    for _, e in edges[edges.ccg_sig].iterrows():
        i = index.get((int(e.u), int(e.v)))
        if i is not None and min(n_ref[i], n_tgt[i]) >= min_spikes:
            n += 1
    return n


def plot_ccg(ax, ccg, i, edge, colour, *, show_ylabel=True, show_xlabel=True):
    """One cross-correlogram: counts, hollow-Gaussian baseline, causal windows."""
    lags = np.asarray(ccg["lags_ms"], float)
    counts = np.asarray(ccg["counts"][i], float)
    base = np.asarray(ccg["baseline"][i], float)
    lo, hi = float(ccg["syn_lo_ms"]), float(ccg["syn_hi_ms"])

    for sign in (+1, -1):
        ax.axvspan(sign * lo, sign * hi, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.6, ls=":", zorder=1)
    ax.bar(lags, counts, width=float(ccg["bin_ms"]), color=colour, lw=0, zorder=2)
    ax.plot(lags, base, color=INK, lw=0.8, zorder=3)
    ax.set_xlim(lags[0], lags[-1])
    ax.tick_params(labelsize=5.5)
    ax.set_title(f"{int(edge.u)}→{int(edge.v)}  ·  STTC {edge.sttc:.2f}  ·  "
                 f"lag {edge.ccg_peak_lag_ms:+.0f} ms",
                 loc="left", fontsize=5.5, color=MUTED, pad=2)
    if show_ylabel:
        ax.set_ylabel("coincidences", fontsize=6)
    if show_xlabel:
        ax.set_xlabel("lag (ms)", fontsize=6)


def ccg_grid(fig, ctx, group: str, n: int = 4, ncols: int = 2):
    """Small multiples of one group's strongest cross-correlograms."""
    row = pick_recording(ctx, group)
    ccg, picks = top_ccg_edges(ctx, row, n)
    if not picks:
        raise RuntimeError(f"{group}: no cross-correlograms on disk")
    nrows = -(-len(picks) // ncols)
    gs = fig.add_gridspec(nrows, ncols, hspace=0.62, wspace=0.32)
    colour = group_color(group)
    for k, (i, edge) in enumerate(picks):
        ax = fig.add_subplot(gs[k // ncols, k % ncols])
        plot_ccg(ax, ccg, i, edge, colour,
                 show_ylabel=(k % ncols == 0),
                 show_xlabel=(k // ncols == nrows - 1))
    fig.suptitle(f"{group}  ·  {row.well_uid.split('|')[0]}  ·  "
                 f"DIV {int(row.DIV)} · τ{int(row.tau):+d}",
                 fontsize=6.5, color=MUTED, x=0.02, ha="left")
    return None


def plot_sttc_graph(ax, ctx, row, *, max_edges: int = 4000):
    """Significant STTC edges drawn on the electrode plane, equal aspect."""
    from ..fig7_spatial_connectivity import _positions

    pos = _positions(ctx.analysis_root, row)
    edges = L.load_edges(ctx.analysis_root, row)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if not edges.empty:
        e = edges.nlargest(min(len(edges), max_edges), "sttc")
        smax = float(e.sttc.max()) or 1.0
        for _, row_e in e.iterrows():
            if row_e.u in pos and row_e.v in pos:
                x0, y0 = pos[row_e.u]
                x1, y1 = pos[row_e.v]
                a = max(0.04, min(0.9, float(row_e.sttc) / smax))
                ax.plot([x0, x1], [y0, y1], color="#3B6FB0", alpha=a, lw=0.4, zorder=2)
    ax.scatter(xs, ys, s=3, color="0.35", lw=0, zorder=3)
    ax.set_xlim(CHIP_EXTENT[0], CHIP_EXTENT[1])
    ax.set_ylim(CHIP_EXTENT[2], CHIP_EXTENT[3])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return len(edges), len(pos)


def plot_activity_field(ax, ctx, row, *, cmap: str = "magma"):
    """Firing-rate activity field placed on the **fixed** chip rectangle.

    The pipeline derives each field's extent from that well's unit positions, so a
    sparse well produces a tiny or near-square extent (observed range: 2 µm to
    3852 µm wide). Drawing each field to fill its own axes therefore rendered
    wells at wildly different physical scales — two panels side by side were not
    the same area. Here the image keeps its true extent but the axes are pinned to
    the chip rectangle with equal aspect, so every field panel covers the same
    physical area and a sparse well honestly shows a small patch of activity.
    """
    from ..fig7_spatial_connectivity import _activity_field

    field, extent = _activity_field(ctx.analysis_root, row)
    ax.imshow(field, origin="lower", cmap=cmap, aspect="equal",
              extent=extent if extent else CHIP_EXTENT, interpolation="nearest")
    ax.set_xlim(CHIP_EXTENT[0], CHIP_EXTENT[1])
    ax.set_ylim(CHIP_EXTENT[2], CHIP_EXTENT[3])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(GRID)
        sp.set_linewidth(0.6)


def group_strip(fig, ctx, groups, draw, *, subtitle=None):
    """One representative recording per group, side by side on a shared layout."""
    gs = fig.add_gridspec(1, len(groups), wspace=0.12)
    for i, g in enumerate(groups):
        ax = fig.add_subplot(gs[0, i])
        row = pick_recording(ctx, g)
        draw(ax, ctx, row)
        head = f"{g.replace('_', ' ')}"
        tail = f"{row.well_uid.split('|')[0]} · τ{int(row.tau):+d}"
        ax.set_title(f"{head}\n{tail}", loc="left", fontsize=6,
                     color=group_color(g), fontweight="bold", pad=3)
        if subtitle:
            ax.set_xlabel(subtitle, fontsize=5.5, color=MUTED)
    return None


CCG_CAPTION = (
    "Cross-correlograms of the strongest significantly coupled unit pairs in one "
    "representative {group} well, {win} days after treatment. Each panel counts "
    "target-unit spikes relative to reference-unit spikes (bars, 1 ms bins, ±100 "
    "ms); the black line is the partially-hollow-Gaussian baseline that predicts "
    "the counts expected from each unit's own slow rate fluctuations, so a peak "
    "above it is genuine short-latency coupling rather than shared burst drive "
    "(Stark & Abeles 2009). The shaded strips either side of zero are the causal "
    "window (0.8–8 ms) the significance test uses; a peak on the right means the "
    "reference unit leads. The pair's STTC and peak lag are given above each "
    "panel. The representative well is the one nearest the group median coupling "
    "strength, not the strongest — illustrative, not a group statistic."
)
