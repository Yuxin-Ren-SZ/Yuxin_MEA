"""Builder factory for the single-metric difference-in-differences panels.

Every DiD panel in F5/F7/F8 draws the same thing for a different metric, so the
panel scripts declare *what* they show and share the *how* from here. The family
argument matters: ``q_bh`` must be corrected over the whole set of metrics its
figure reports, so each script passes the figure's family and the panel selects
its own row from the result.
"""
from __future__ import annotations

from ..forest import did_column_panel

#: Ratio metrics are log2(post/pre); bounded/near-zero ones are a plain difference
#: (see ``stats.DIFF_METRICS``), so the axis label has to cover both.
MIXED_YLABEL = "response  (log₂ post/pre, or Δ)"
RATIO_YLABEL = "log₂(post / pre)"


#: The shared half of every DiD caption — the method, stated once so it can never
#: drift between panels. Panel scripts prepend one sentence naming their metric.
DID_METHOD = (
    "Difference-in-differences against Control maturation, at the pre-specified "
    "endpoint (the latest treatment-day bin all three chips still contribute to; "
    "printed in the panel corner). Each well is first compared to its own "
    "pre-treatment baseline, the wells of a chip are averaged into one value "
    "(wells are technical replicates, the chip is the biological replicate), and "
    "each chip's treated minus its own Control value is that chip's DiD. Small "
    "grey points = wells (technical replicates), large coloured points = the "
    "per-chip means (biological replicates, the unit the test runs on), bar = "
    "mean of the chip means, whisker = 95% t-CI (df = n_chips − 1), dashed line "
    "= no change. The chip count is printed under each group label, because "
    "coverage differs by metric and a two-chip test has one degree of freedom. "
    "Significance is a one-sample t-test of the per-chip DiDs against zero: "
    "* = q<0.05 after BH-FDR across this figure's metrics; △ = p<0.05 uncorrected "
    "but not FDR-significant; no mark = neither. Panels are tinted only when "
    "FDR-significant, so an untinted panel means nothing survived correction. "
    "Δ is the effect size (mean per-chip DiD). Preliminary: n = 3 chips per arm."
)


def did_caption(lead: str) -> str:
    """``lead`` names the metric and what it means; the method text is shared."""
    return f"{lead} {DID_METHOD}"


def did_build(metric: str, family: list[str], *,
              ylabel: str = MIXED_YLABEL, arms: list[str] | None = None):
    """Return a ``build(ax, ctx)`` drawing ``metric``'s DiD at the endpoint."""

    def build(ax, ctx):
        resp = ctx.response_by_tau(family, arms)
        did = ctx.did(family, arms)
        return did_column_panel(ax, metric, resp, did, ylabel=ylabel)

    build.__doc__ = f"DiD panel for {metric} (FDR family: {', '.join(family)})"
    return build


def did_grid_build(family: list[str], *, ncols: int = 3,
                   ylabel: str = MIXED_YLABEL, arms: list[str] | None = None):
    """Return a ``build_fig`` drawing the whole family as DiD small multiples.

    Used where a family is secondary to the figure's story (the node-level and
    directed metrics) and deserves one compact panel rather than one panel each.
    """
    def build_fig(fig, ctx):
        resp = ctx.response_by_tau(family, arms)
        did = ctx.did(family, arms)
        metrics = [m for m in family if m in set(resp.metric)]
        nrows = -(-len(metrics) // ncols)
        gs = fig.add_gridspec(nrows, ncols, hspace=0.85, wspace=0.45)
        for i, m in enumerate(metrics):
            ax = fig.add_subplot(gs[i // ncols, i % ncols])
            did_column_panel(ax, m, resp, did, ylabel=ylabel,
                             show_ylabel=(i % ncols == 0))
        return did[did.metric.isin(metrics)].reset_index(drop=True)

    return build_fig


# --------------------------------------------------------------------------- #
# DiD time course — the companion to the endpoint test
# --------------------------------------------------------------------------- #
TIMECOURSE_METHOD = (
    "One small panel per metric: the difference-in-differences against Control "
    "over time rather than at a single day. Each chip's treated-minus-own-Control "
    "difference is computed per treatment-day bin; the line is the mean across "
    "chips and the band its 95% t-CI (df = n_chips − 1). Zero means the arm is "
    "changing exactly like Control, i.e. pure maturation. The solid vertical "
    "marks the pre-specified endpoint the formal test uses — the other bins are "
    "shown so a late-emerging or transient effect is visible, but they are not "
    "separately tested (that would multiply the correction family and, at n = 3 "
    "chips, leave nothing detectable). Dashed verticals mark the experiment "
    "window, τ0 to τ+21. Preliminary: n = 3 chips per arm."
)


def _timecourse_axes(ax, metric, dt, endpoint, *, show_ylabel, ylabel):
    import numpy as np
    from scipy.stats import t as tdist

    from ..report_style import INK, MUTED, group_color, metric_label
    from ..trajectory import tau_band

    tau_band(ax)
    sub = dt[dt.metric == metric]
    seen = []
    for arm, g in sub.groupby("arm"):
        xs, ys, los, his = [], [], [], []
        for tc, grp in g.groupby("tauc"):
            v = grp.did.to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) < 2:
                continue
            m = float(np.mean(v))
            h = float(tdist.ppf(0.975, len(v) - 1)) * float(
                np.std(v, ddof=1) / np.sqrt(len(v)))
            xs.append(float(tc)); ys.append(m); los.append(m - h); his.append(m + h)
        if not xs:
            continue
        o = np.argsort(xs)
        xs = np.array(xs)[o]; ys = np.array(ys)[o]
        los = np.array(los)[o]; his = np.array(his)[o]
        # Arms track each other closely; a dash on the second keeps both readable
        # where the lines coincide instead of one hiding the other.
        style = {"dashes": (3, 1.5)} if seen else {}
        ax.plot(xs, ys, "-o", ms=3, color=group_color(arm), label=arm, zorder=3,
                **style)
        ax.fill_between(xs, los, his, color=group_color(arm), alpha=0.13, lw=0,
                        zorder=2)
        seen.append(arm)
        ax._tc_span = (min(getattr(ax, "_tc_span", (np.inf, -np.inf))[0], float(ys.min())),
                       max(getattr(ax, "_tc_span", (np.inf, -np.inf))[1], float(ys.max())))
    ax.axhline(0, ls="--", color="0.5", lw=0.8, zorder=1)
    # Scale to the DiD lines, not to a df=1 confidence band an order of magnitude
    # taller (late bins drop to two chips as CX138 stops recording).
    span = getattr(ax, "_tc_span", None)
    if span and np.isfinite(span[0]):
        lo, hi = min(span[0], 0.0), max(span[1], 0.0)
        pad = 0.35 * (hi - lo) + 1e-3
        ax.set_ylim(lo - pad, hi + pad)
    if endpoint == endpoint:                       # not NaN
        ax.axvline(endpoint, color=INK, lw=1.0, zorder=4)
        ax.annotate("endpoint", xy=(endpoint, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(2, -2), textcoords="offset points", ha="left", va="top",
                    fontsize=5, color=MUTED)
    ax.set_title(metric_label(metric, unitless=True), loc="left", fontsize=7,
                 fontweight="bold", color=INK)
    ax.set_xlabel("treatment day (tau)", fontsize=6.5)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=6.5)
    ax.tick_params(labelsize=6)


def did_timecourse_build(family: list[str], *, ncols: int = 3,
                         ylabel: str = "DiD vs Control", arms: list[str] | None = None):
    """Return a ``build_fig(fig, ctx)`` drawing the family's DiD vs treatment day."""
    from .. import stats as S

    def build_fig(fig, ctx):
        resp = ctx.response_by_tau(family, arms)
        dt = S.did_by_tau(resp)
        endpoint = float(ctx.did(family, arms).endpoint_tau.iloc[0])
        metrics = [m for m in family if m in set(dt.metric)]
        nrows = -(-len(metrics) // ncols)
        gs = fig.add_gridspec(nrows, ncols, hspace=0.75, wspace=0.42)
        axes = []
        for i, m in enumerate(metrics):
            ax = fig.add_subplot(gs[i // ncols, i % ncols])
            _timecourse_axes(ax, m, dt, endpoint, show_ylabel=(i % ncols == 0),
                             ylabel=ylabel)
            axes.append(ax)
        if axes:
            axes[0].legend(fontsize=5.5, frameon=False, loc="best", ncol=1)
        return dt[dt.metric.isin(metrics)].reset_index(drop=True)

    return build_fig
