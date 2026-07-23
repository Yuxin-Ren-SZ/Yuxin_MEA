"""Shared chip-level difference-in-differences forest.

Biological replicate = the **chip** (``sample_id``: CX118/CX138/CX169); the well
is the technical replicate. So every DiD forest in the report draws, per arm:

* the **per-chip mean responses** as the bold points (the ~3 biological
  replicates — this is what "all data points" now shows),
* the wells faint behind them (the technical layer, for context),
* a t-CI over the chip means (df = n_chips − 1), and
* a significance star **only when** the well-nested-in-chip linear mixed model
  gives q<0.05 (BH-FDR) **and** the effect is the same direction across all
  chips. The second clause encodes the biological-replicate logic: a real effect
  must reproduce across independent chips, not ride on within-chip pseudo-
  replication. ``†`` marks a singular chip-variance fit (LMM collapsed to OLS;
  the per-chip consistency is then what supports the call, not the Wald p).

One helper (:func:`plot_did_forest`) replaces the four near-identical
``_panel_forest`` copies in F5/F7/F7b/F8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S
from .report_style import METRIC_LABELS, group_color, ordered_groups


def stars(q: float) -> str:
    if q is None or (isinstance(q, float) and np.isnan(q)):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def plot_did_forest(
    ax, resp: pd.DataFrame, metrics: list[str], focus_arms: list[str], *,
    xlabel: str, title: str, clip_to_ci: bool = True, footnote: bool = True,
    legend_loc: str = "upper left",
) -> dict | None:
    """Draw a chip-level DiD forest into ``ax``.

    ``resp`` is :func:`stats.well_response` output already filtered to
    ``Control`` + ``focus_arms``. Returns ``{summary, vs_control}`` (or ``None``
    when there are no paired wells) so callers can stash the tables.
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
                ax.scatter(w, np.full(len(w), y) + jit, s=3,
                           color=group_color(arm), alpha=0.16, lw=0, zorder=2)
            # per-chip means bold (the biological replicates)
            cp = cr.loc[(cr.metric == metric) & (cr.arm == arm),
                        "response"].to_numpy(float)
            cp = cp[np.isfinite(cp)]
            if len(cp):
                jit = (rng.random(len(cp)) - 0.5) * 0.10
                ax.scatter(cp, np.full(len(cp), y) + jit, s=16,
                           facecolor=group_color(arm), edgecolor="white",
                           linewidth=0.4, alpha=0.95, zorder=4)
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
            # (consistent 3/3 chips, uncorrected p<0.05, does not survive FDR)
            if not is_ctrl and vc_idx is not None and (arm, metric) in vc_idx.index:
                row = vc_idx.loc[(arm, metric)]
                s = stars(row.q_bh) if bool(row.chip_consistent) else ""
                if not s and bool(row.get("suggestive", False)):
                    s = "△"
                if s:
                    ax.text(r.ci_high + 0.04, y, s, va="center", ha="left",
                            fontsize=7, color=group_color(arm))

    ax.axvline(0, ls="--", color="0.5", lw=0.8, zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_ylim(min(yticks) - 0.6, max(yticks) + 0.6)
    if clip_to_ci and len(summ):
        lo = float(np.nanmin(summ.ci_low))
        hi = float(np.nanmax(summ.ci_high))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.3 * (hi - lo) + 0.1
            ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_title(title, loc="left", fontweight="bold")

    handles = [ax.plot([], [], marker="D", ls="", color="k",
                       label="Control (maturation)")[0]]
    handles += [ax.plot([], [], marker="o", ls="", color=group_color(a), label=a)[0]
                for a in treated]
    handles.append(ax.scatter([], [], s=16, facecolor="0.4", edgecolor="white",
                              linewidth=0.4, label="per-chip means (n=3)"))
    ax.legend(handles=handles, loc=legend_loc, fontsize=5.5, ncol=1, frameon=False)
    if footnote:
        ax.text(
            0.0, -0.17,
            "Cluster-summary DiD (t-test of per-chip treated−Control, df=n_chips−1). "
            "★ = q<0.05 (BH-FDR); △ = suggestive (same direction on all 3 chips, "
            "uncorrected p<0.05, does NOT survive FDR at n=3). Bold = per-chip "
            "means (biological replicates); faint = wells.",
            transform=ax.transAxes, fontsize=5, color="0.35", va="top")
    return {"summary": summ, "vs_control": vc}
