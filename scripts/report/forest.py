"""Shared chip-level difference-in-differences forest.

Biological replicate = the **chip** (``sample_id``: CX118/CX138/CX169); the well
is the technical replicate. So every DiD forest in the report draws, per arm:

* the **per-chip mean responses** as the bold points (the ~3 biological
  replicates — this is what "all data points" now shows),
* the wells faint behind them (the technical layer, for context),
* a t-CI over the chip means (df = n_chips − 1), and
* a two-tier significance glyph read **only** from the numbers:
  ``*`` when the BH-FDR q clears 0.05, ``△`` when the uncorrected p clears 0.05
  but q does not (see :func:`report_style.sig_glyph`). Nothing is highlighted
  unless it is FDR-significant — a tinted panel is a claim, so it has to be
  earned rather than chosen.

:func:`did_column_panel` is what the standalone panel scripts call (one metric,
groups side-by-side, at the pre-specified endpoint). :func:`plot_did_forest` and
:func:`group_columns_grid` remain for the older composed figures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S
from .report_style import (
    BIO_SIZE, FAINT, HI_BORDER, HI_FILL, INK, METRIC_LABELS, MUTED, SIG_STAR,
    SIG_SUGGEST, TECH_ALPHA, TECH_GREY, TECH_SIZE, group_color, highlight_row,
    metric_label, ordered_groups, preliminary_tag,
)

# Short x-axis labels for the grouped-column small multiples.
_SHORT_GROUP = {
    "Control": "Ctrl", "IVH_Early": "IVH-E", "IVH_Late": "IVH-L",
    "H2O2_20uM": "H₂O₂-20", "H2O2_10uM": "H₂O₂-10", "NPH": "NPH", "AraC": "AraC",
}


def _short_group(g: str) -> str:
    return _SHORT_GROUP.get(g, g)


# --------------------------------------------------------------------------- #
# Single-metric DiD column panel (the unit every F5/F7/F8 panel script draws)
# --------------------------------------------------------------------------- #
def _robust_ylim(ax, wells, chips, summ, groups, xpos) -> None:
    """Scale to the data, not to the widest confidence interval.

    An arm measured on only two chips gets a t-CI with df=1, which can be an
    order of magnitude taller than every data point and flattens the whole panel
    into a line. Fit the axis to the observed responses and the mean bars, then
    mark any interval that runs off-scale with a caret so it is not hidden.
    """
    vals = [wells.response.to_numpy(float), chips.response.to_numpy(float)]
    if len(summ):
        vals.append(summ["mean"].to_numpy(float))
    v = np.concatenate([a[np.isfinite(a)] for a in vals if len(a)]) if vals else np.array([])
    if not len(v):
        return
    lo, hi = float(np.min(v)), float(np.max(v))
    lo, hi = min(lo, 0.0), max(hi, 0.0)          # the no-change line stays visible
    pad = 0.22 * (hi - lo) + 1e-3
    ax.set_ylim(lo - pad, hi + pad)
    ylo, yhi = ax.get_ylim()
    for _, rr in summ.iterrows():
        if rr.arm not in xpos:
            continue
        for bound, marker, va in ((rr.ci_high, "⌃", "top"), (rr.ci_low, "⌄", "bottom")):
            if not np.isfinite(bound):
                continue
            if (marker == "⌃" and bound > yhi) or (marker == "⌄" and bound < ylo):
                ax.annotate(marker, xy=(xpos[rr.arm], yhi if marker == "⌃" else ylo),
                            ha="center", va=va, fontsize=7,
                            color=group_color(rr.arm), annotation_clip=False)



def did_column_panel(
    ax, metric: str, resp: pd.DataFrame, did: pd.DataFrame, *,
    endpoint: float | None = None, groups: list[str] | None = None,
    ylabel: str = "log₂(post / pre)", show_ylabel: bool = True,
    effect_label: bool = True, title: str | None = None,
) -> pd.DataFrame:
    """One metric, groups side-by-side, at the pre-specified endpoint.

    ``resp`` is :func:`stats.well_response_by_tau` output; ``did`` is
    :func:`stats.did_at_endpoint` for the **whole figure's metric family** (so its
    ``q_bh`` is corrected over the right family — never pass one metric at a time).

    Draws, per group: wells faint (technical replicates), per-chip means bold
    (biological replicates), a mean bar with a 95% t-CI (df = n_chips − 1), and a
    dashed no-change line. A treated group is annotated with ``*`` when q<0.05 and
    ``△`` when p<0.05≤q — **derived from the numbers**, never from a hand-picked
    headline metric. The panel is tinted only when it actually carries an FDR-
    significant effect, so an un-tinted panel means exactly that: nothing survived.

    Returns the ``did`` rows for this metric (the panel's stats sidecar).
    """
    from .report_style import sig_glyph

    sub_did = did[did.metric == metric] if did is not None and len(did) else pd.DataFrame()
    if endpoint is None:
        endpoint = (float(sub_did.endpoint_tau.iloc[0]) if len(sub_did)
                    else S.primary_endpoint(resp))
    r = resp[(resp.metric == metric) & (resp.tauc == endpoint)]
    summ = S.arm_summary_at_endpoint(resp[resp.metric == metric], endpoint)
    cr = S.chip_response_by_tau(resp[resp.metric == metric])
    cr = cr[cr.tauc == endpoint]

    groups = groups or [g for g in ordered_groups(r.arm.unique())]
    xpos = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(1)

    for g in groups:
        x, col = xpos[g], group_color(g)
        # Technical replicates (wells): small, grey, behind — context only.
        w = r.loc[r.arm == g, "response"].to_numpy(float)
        w = w[np.isfinite(w)]
        if len(w):
            jit = (rng.random(len(w)) - 0.5) * 0.30
            ax.scatter(np.full(len(w), x) + jit, w, s=TECH_SIZE, color=TECH_GREY,
                       alpha=TECH_ALPHA, lw=0, zorder=2)
        # Biological replicates (per-chip means): large, coloured, in front —
        # these are the points the test is actually computed over.
        cp = cr.loc[cr.arm == g, "response"].to_numpy(float)
        cp = cp[np.isfinite(cp)]
        if len(cp):
            jit = (rng.random(len(cp)) - 0.5) * 0.16
            ax.scatter(np.full(len(cp), x) + jit, cp, s=BIO_SIZE, facecolor=col,
                       edgecolor="white", linewidth=0.6, zorder=6)
        rr = summ[summ.arm == g]
        if not rr.empty:
            rr = rr.iloc[0]
            ax.plot([x - 0.30, x + 0.30], [rr["mean"]] * 2, color=col, lw=1.8,
                    solid_capstyle="round", zorder=5)
            ax.errorbar(x, rr["mean"],
                        yerr=[[max(rr["mean"] - rr.ci_low, 0)],
                              [max(rr.ci_high - rr["mean"], 0)]],
                        fmt="none", ecolor=col, capsize=2.5, elinewidth=0.9, zorder=5)

    ax.axhline(0, ls="--", color="0.5", lw=0.8, zorder=1)
    ax.set_xticks(range(len(groups)))
    # Chip count under each label: coverage differs by metric and by arm (a chip
    # that stopped recording early, or a well whose metric failed, drops out), and
    # a t-test on two chips has df=1. That has to be visible next to the glyph.
    n_chips = {g: int(summ.loc[summ.arm == g, "n_chips"].iloc[0])
               for g in groups if (summ.arm == g).any()}
    ax.set_xticklabels([f"{_short_group(g)}\nn={n_chips.get(g, 0)}" for g in groups],
                       fontsize=6.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    _robust_ylim(ax, r, cr, summ, groups, xpos)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    # Unitless title: the axis is a log2 ratio / Δ, not the metric's own unit.
    ax.set_title(title or metric_label(metric, unitless=True), loc="left",
                 fontweight="bold", fontsize=8, color=INK)

    # Significance glyphs — read straight off p_vs_control / q_bh.
    # A one-metric family makes BH-FDR a no-op (q == p), so a ★ there would be an
    # uncorrected p wearing the FDR tier's mark. Cap such panels at △.
    single_metric = did is not None and len(did) and did.metric.nunique() <= 1
    fdr_hit = False
    for _, row in sub_did.iterrows():
        if row.arm not in xpos:
            continue
        glyph, colour = sig_glyph(row.p_vs_control,
                                  np.nan if single_metric else row.q_bh)
        if not glyph:
            continue
        fdr_hit = fdr_hit or glyph.startswith("*")
        txt = glyph
        if effect_label and np.isfinite(row.did):
            # Significant digits, not fixed decimals: metrics span betweenness
            # (~1e-3) to spikes-per-burst (~1e0), and "Δ-0.00" reads as no effect.
            txt = f"{glyph} Δ{row.did:+.3g}"
        rr = summ[summ.arm == row.arm]
        ytop = float(rr.iloc[0].ci_high) if not rr.empty else 0.0
        ax.annotate(txt, xy=(xpos[row.arm], ytop), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6, color=colour, fontweight="bold", zorder=6)

    if fdr_hit:                 # tint ONLY when the stats earned it
        ax.set_facecolor(HI_FILL)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(HI_BORDER)
            sp.set_linewidth(1.1)

    note = f"endpoint τ+{endpoint:g}"
    if single_metric:
        note += " · uncorrected (single metric)"
    ax.annotate(note, xy=(0.99, 0.01), xycoords="axes fraction", ha="right",
                va="bottom", fontsize=5.5, color=MUTED)
    return sub_did.reset_index(drop=True)


def stars(q: float) -> str:
    if q is None or (isinstance(q, float) and np.isnan(q)):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def plot_did_forest(
    ax, resp: pd.DataFrame, metrics: list[str], focus_arms: list[str], *,
    xlabel: str, title: str, clip_to_ci: bool = True, footnote: bool = True,
    legend_loc: str = "upper left", highlight_metric: str | None = None,
    effect_labels: bool = False, prelim_tag: bool = True,
    robust_xlim: bool = False,
) -> dict | None:
    """Draw a chip-level DiD forest into ``ax``.

    ``resp`` is :func:`stats.well_response` output already filtered to
    ``Control`` + ``focus_arms``. ``metrics`` is drawn top-to-bottom, so pass it
    story-ordered (headline first). ``highlight_metric`` shades that row's band
    (the effect we foreground); ``effect_labels`` annotates its per-arm DiD Δ.
    Returns ``{summary, vs_control}`` (or ``None`` when no paired wells).
    """
    if resp is None or resp.empty:
        ax.text(0.5, 0.5, "no paired wells", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title, loc="left", fontweight="bold")
        return None

    summ = S.arm_response_summary_chip(resp)
    vc = S.arm_response_vs_control_lmm(resp, focus_arms=focus_arms)
    cr = S.chip_response(resp)
    vc_idx = vc.set_index(["arm", "metric"]) if len(vc) else None

    rng = np.random.default_rng(0)
    treated = [a for a in ordered_groups(resp.arm.unique()) if a in focus_arms]
    arms = ["Control"] + treated
    arm_off = np.linspace(0.32, -0.32, len(arms))

    # Highlight band — only for a row that is actually FDR-significant. A band on
    # a non-significant metric reads as a claim the statistics do not support.
    if highlight_metric in metrics and vc_idx is not None:
        hit = vc[(vc.metric == highlight_metric) & (vc.q_bh < 0.05)]
        if len(hit):
            highlight_row(ax, -metrics.index(highlight_metric), half=0.44, zorder=0)

    yticks, ylabels = [], []
    for mi, metric in enumerate(metrics):
        y0 = -mi
        yticks.append(y0)
        ylabels.append(METRIC_LABELS.get(metric, metric))
        for ai, arm in enumerate(arms):
            r = summ[(summ.metric == metric) & (summ.arm == arm)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = y0 + arm_off[ai]
            is_ctrl = arm == "Control"
            # wells faint behind (technical replicates, for context)
            w = resp.loc[(resp.metric == metric) & (resp.arm == arm),
                         "response"].to_numpy(float)
            w = w[np.isfinite(w)]
            if len(w):
                jit = (rng.random(len(w)) - 0.5) * 0.14
                ax.scatter(w, np.full(len(w), y) + jit, s=TECH_SIZE * 0.7,
                           color=TECH_GREY, alpha=TECH_ALPHA, lw=0, zorder=2)
            # per-chip means bold (the biological replicates)
            cp = cr.loc[(cr.metric == metric) & (cr.arm == arm),
                        "response"].to_numpy(float)
            cp = cp[np.isfinite(cp)]
            if len(cp):
                jit = (rng.random(len(cp)) - 0.5) * 0.10
                ax.scatter(cp, np.full(len(cp), y) + jit, s=BIO_SIZE * 0.8,
                           facecolor=group_color(arm), edgecolor="white",
                           linewidth=0.5, zorder=6)
            xerr = np.array([[max(r.median_response - r.ci_low, 0)],
                             [max(r.ci_high - r.median_response, 0)]])
            ax.errorbar(
                r.median_response, y, xerr=xerr,
                fmt=("D" if is_ctrl else "o"), ms=(5 if is_ctrl else 4.5),
                color=group_color(arm), capsize=2, elinewidth=0.9,
                markerfacecolor=group_color(arm),
                markeredgecolor=("k" if not is_ctrl else group_color(arm)),
                markeredgewidth=0.5, zorder=5,
            )
            # mark: filled ★ = FDR-significant (cluster test); △ = suggestive
            # (consistent 3/3 chips, uncorrected p<0.05, does not survive FDR).
            # On the highlight row (effect_labels) append the DiD effect size Δ.
            if not is_ctrl and vc_idx is not None and (arm, metric) in vc_idx.index:
                from .report_style import sig_glyph

                row = vc_idx.loc[(arm, metric)]
                glyph, gcolour = sig_glyph(row.get("p_vs_control"), row.q_bh)
                s_stars = glyph if glyph.startswith("*") else ""
                label = glyph
                if effect_labels and metric == highlight_metric:
                    # effect size = mean of per-chip DiDs (matches the △/t-CI basis)
                    delta = float(row.get("did", np.nan))
                    if np.isfinite(delta):
                        label = f"{glyph}  Δ{delta:+.2f}" if glyph else f"Δ{delta:+.2f}"
                if label:
                    ax.text(r.ci_high + 0.06, y, label, va="center", ha="left",
                            fontsize=6.5, color=gcolour, fontweight="bold", zorder=6)

    ax.axvline(0, ls="--", color="0.5", lw=0.8, zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_ylim(min(yticks) - 0.6, max(yticks) + 0.6)
    if robust_xlim and len(summ):
        # One metric's huge CI must not set the scale for all the others: use a
        # trimmed range and mark any point/CI that runs off-scale with ⟩ / ⟨.
        lo = float(np.nanpercentile(summ.ci_low, 12))
        hi = float(np.nanpercentile(summ.ci_high, 88))
        lo, hi = min(lo, -0.05), max(hi, 0.05)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.2 * (hi - lo) + 0.05
            ax.set_xlim(lo - pad, hi + pad)
            xlo, xhi = ax.get_xlim()
            for mi, metric in enumerate(metrics):
                for ai, arm in enumerate(arms):
                    r = summ[(summ.metric == metric) & (summ.arm == arm)]
                    if r.empty:
                        continue
                    v = float(r.iloc[0].median_response)
                    yy = -mi + arm_off[ai]
                    if v > xhi:
                        ax.annotate("⟩", (xhi, yy), ha="right", va="center",
                                    fontsize=8, color=group_color(arm),
                                    fontweight="bold", annotation_clip=False)
                    elif v < xlo:
                        ax.annotate("⟨", (xlo, yy), ha="left", va="center",
                                    fontsize=8, color=group_color(arm),
                                    fontweight="bold", annotation_clip=False)
    elif clip_to_ci and len(summ):
        lo = float(np.nanmin(summ.ci_low))
        hi = float(np.nanmax(summ.ci_high))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.3 * (hi - lo) + 0.1
            rpad = pad + (0.28 * (hi - lo) if effect_labels else 0)  # room for Δ
            ax.set_xlim(lo - pad, hi + rpad)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_title(title, loc="left", fontweight="bold")

    handles = [ax.plot([], [], marker="D", ls="", color="k",
                       label="Control (maturation)")[0]]
    handles += [ax.plot([], [], marker="o", ls="", color=group_color(a), label=a)[0]
                for a in treated]
    handles.append(ax.scatter([], [], s=BIO_SIZE * 0.8, facecolor="0.4",
                              edgecolor="white", linewidth=0.5,
                              label="per-chip means (biological, n=3)"))
    handles.append(ax.scatter([], [], s=TECH_SIZE, facecolor=TECH_GREY, lw=0,
                              label="wells (technical)"))
    ax.legend(handles=handles, loc=legend_loc, fontsize=5.5, ncol=1, frameon=False)
    if footnote:
        # Compact shared significance footer (replaces the old wrapped wall).
        from .report_style import SIG_LEGEND

        ax.text(
            0.0, -0.16,
            "Chip-level DiD (t-test of per-chip treated−Control, df=2).  "
            f"{SIG_LEGEND}  ·  Preliminary — n=3 chips/arm.",
            transform=ax.transAxes, fontsize=5, color=MUTED, va="top")
    elif prelim_tag:
        # Sub-forests without a footer still carry the persistent reminder.
        preliminary_tag(ax)
    return {"summary": summ, "vs_control": vc}


# --------------------------------------------------------------------------- #
# Grouped-column small multiples (one panel per metric)
# --------------------------------------------------------------------------- #
# An easier-to-read alternative to the stacked forest: each metric gets its own
# small panel with the groups side-by-side on the x-axis (Control = maturation
# baseline, treated arms read against it). Per-chip means are bold, wells faint,
# a mean bar + 95% t-CI (df=2) per group, a dashed zero line, and △/★ + effect
# size Δ over any suggestive/FDR treated arm. The headline metric panel is tinted.


def _group_column_panel(ax, metric, resp, summ, cr, vc_idx, groups, *,
                        highlight=False, effect_labels=False, show_ylabel=True,
                        ylabel="log₂(post / pre)"):
    rng = np.random.default_rng(1)
    xpos = {g: i for i, g in enumerate(groups)}
    for g in groups:
        x = xpos[g]
        col = group_color(g)
        # wells faint (technical layer)
        w = resp.loc[(resp.metric == metric) & (resp.arm == g),
                     "response"].to_numpy(float)
        w = w[np.isfinite(w)]
        if len(w):
            jit = (rng.random(len(w)) - 0.5) * 0.30
            ax.scatter(np.full(len(w), x) + jit, w, s=TECH_SIZE, color=TECH_GREY,
                       alpha=TECH_ALPHA, lw=0, zorder=2)
        # per-chip means bold (biological replicates)
        cp = cr.loc[(cr.metric == metric) & (cr.arm == g),
                    "response"].to_numpy(float)
        cp = cp[np.isfinite(cp)]
        if len(cp):
            jit = (rng.random(len(cp)) - 0.5) * 0.16
            ax.scatter(np.full(len(cp), x) + jit, cp, s=BIO_SIZE, facecolor=col,
                       edgecolor="white", linewidth=0.6, zorder=6)
        # mean bar + 95% t-CI over the chip means
        rr = summ[(summ.metric == metric) & (summ.arm == g)]
        if not rr.empty:
            rr = rr.iloc[0]
            ax.plot([x - 0.30, x + 0.30], [rr.median_response] * 2,
                    color=col, lw=1.8, solid_capstyle="round", zorder=5)
            ax.errorbar(
                x, rr.median_response,
                yerr=[[max(rr.median_response - rr.ci_low, 0)],
                      [max(rr.ci_high - rr.median_response, 0)]],
                fmt="none", ecolor=col, capsize=2.5, elinewidth=0.9, zorder=5)
    ax.axhline(0, ls="--", color="0.5", lw=0.8, zorder=1)  # no-change reference
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_short_group(g) for g in groups], fontsize=6.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=7)

    ax.set_title(METRIC_LABELS.get(metric, metric), loc="left",
                 fontweight="bold", fontsize=8, color=INK)

    # △/★ + effect size Δ placed just above each suggestive/FDR treated column's
    # upper CI, so it stays with the data and clear of the panel title.
    if vc_idx is not None:
        for g in groups:
            if g == "Control" or (g, metric) not in vc_idx.index:
                continue
            from .report_style import sig_glyph

            row = vc_idx.loc[(g, metric)]
            glyph, gcolour = sig_glyph(row.get("p_vs_control"), row.q_bh)
            if not glyph:
                continue
            txt = glyph
            if effect_labels:
                d = float(row.get("did", np.nan))   # per-chip DiD mean
                if np.isfinite(d):
                    txt = f"{glyph} Δ{d:+.2f}"
            rr = summ[(summ.metric == metric) & (summ.arm == g)]
            ytop = float(rr.iloc[0].ci_high) if not rr.empty else 0.0
            ax.annotate(txt, xy=(xpos[g], ytop), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6, color=gcolour, fontweight="bold", zorder=6)

    if highlight:                       # headline metric — tint the whole panel
        ax.set_facecolor(HI_FILL)
        for name, sp in ax.spines.items():
            sp.set_visible(True)
            sp.set_color(HI_BORDER)
            sp.set_linewidth(1.1)


def group_columns_grid(axes, resp, metrics, focus_arms, *,
                       highlight_metric=None, effect_labels=True,
                       ylabel_axes=None, ylabel="log₂(post / pre)") -> dict | None:
    """Draw grouped-column DiD small multiples, one axes per metric.

    ``axes`` and ``metrics`` are parallel (draw order). ``ylabel_axes`` is the
    set of axes that should show the y-label (typically the left column); all
    stats are computed once and shared across panels.
    """
    if resp is None or resp.empty:
        for ax, m in zip(axes, metrics):
            ax.text(0.5, 0.5, "no paired wells", ha="center", va="center",
                    transform=ax.transAxes)
        return None
    summ = S.arm_response_summary_chip(resp)
    cr = S.chip_response(resp)
    vc = S.arm_response_vs_control_lmm(resp, focus_arms=focus_arms)
    vc_idx = vc.set_index(["arm", "metric"]) if len(vc) else None
    groups = [g for g in ordered_groups(resp.arm.unique())
              if g == "Control" or g in focus_arms]
    ylabel_axes = set(id(a) for a in (ylabel_axes if ylabel_axes is not None else axes))

    def _earned_highlight(metric: str) -> bool:
        """Tint a panel only when the stats earned it — never a hand-picked metric.

        ``highlight_metric`` is kept in the signature so old call sites still
        import cleanly, but it can no longer force a highlight: the band means
        "FDR-significant", so awarding it to a p≈0.08 effect would misreport.
        """
        if vc_idx is None or (metric not in set(vc.metric)):
            return False
        rows = vc[vc.metric == metric]
        return bool((rows.q_bh < 0.05).any())

    for ax, metric in zip(axes, metrics):
        _group_column_panel(
            ax, metric, resp, summ, cr, vc_idx, groups,
            highlight=_earned_highlight(metric), effect_labels=effect_labels,
            show_ylabel=id(ax) in ylabel_axes, ylabel=ylabel)
    return {"summary": summ, "vs_control": vc}
