"""S2 — Single-unit quality control & waveforms.

Draws on the per-unit ``quality_metrics.pkl`` (pooled over a spread of wells) and
the analyzer templates for one representative well.

Panels: the four curation QC distributions with their gating thresholds, a
peak-channel waveform overlay (narrow vs broad-spiking), and a spike-width vs
firing-rate scatter (a putative cell-type axis).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load as L
from .report_style import OKABE_ITO, save_fig

# curation thresholds (auto_curation defaults).
_THRESH = {
    "presence_ratio": (0.75, ">"),
    "rp_contamination": (0.15, "<"),
    "firing_rate": (0.05, ">"),
    "amplitude_median": (-20.0, "<"),
}
_PT_NARROW_MS = 0.5  # narrow/broad spiking split on peak-to-trough duration


def _sample_wells(tidy: pd.DataFrame, n: int = 48, seed: int = 0) -> pd.DataFrame:
    """A spread of wells across groups for pooling unit metrics."""
    per = max(2, n // max(tidy.canonical_group.nunique(), 1))
    parts = [sub.sample(min(len(sub), per), random_state=seed)
             for _, sub in tidy.groupby("canonical_group")]
    return pd.concat(parts).reset_index(drop=True)


def _hist_qc(ax, units, col, log=False) -> None:
    x = units[col].dropna().to_numpy(float)
    if log:
        x = x[x > 0]
        bins = np.logspace(np.log10(max(x.min(), 1e-3)), np.log10(x.max()), 30)
    else:
        bins = 30
    ax.hist(x, bins=bins, color=OKABE_ITO["sky"], alpha=0.8)
    if log:
        ax.set_xscale("log")
    thr, direction = _THRESH[col]
    ax.axvline(thr, ls="--", color=OKABE_ITO["vermillion"], lw=1.0,
               label=f"gate {direction}{thr:g}")
    ax.set_xlabel(col)
    ax.set_ylabel("units")
    ax.legend(fontsize=6)


def _panel_waveforms(ax, analysis_root, row) -> None:
    try:
        tmpl = L.load_templates(analysis_root, row)   # (units, samples, chans)
        qm = pd.read_pickle(L.quality_metrics_path(analysis_root, row))
    except FileNotFoundError:
        ax.text(0.5, 0.5, "no templates", ha="center", va="center",
                transform=ax.transAxes)
        return
    n_samp = tmpl.shape[1]
    t_ms = np.linspace(-1.0, 2.0, n_samp)  # ms_before=1, ms_after=2
    # peak_to_trough_duration is stored in seconds -> ms for the narrow/broad split
    pt = (qm["peak_to_trough_duration"].to_numpy(float) * 1000.0
          if "peak_to_trough_duration" in qm else None)
    n = min(tmpl.shape[0], 40)
    for u in range(n):
        wf = tmpl[u]
        pk_ch = np.argmax(wf.max(0) - wf.min(0))  # widest peak-to-peak channel
        trace = wf[:, pk_ch]
        trace = trace / (np.abs(trace).max() + 1e-9)
        broad = pt is not None and u < len(pt) and pt[u] >= _PT_NARROW_MS
        ax.plot(t_ms, trace, lw=0.5, alpha=0.6,
                color=(OKABE_ITO["orange"] if broad else OKABE_ITO["blue"]))
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("norm. amplitude")
    ax.plot([], [], color=OKABE_ITO["blue"], label=f"narrow (<{_PT_NARROW_MS}ms)")
    ax.plot([], [], color=OKABE_ITO["orange"], label=f"broad (≥{_PT_NARROW_MS}ms)")
    ax.legend(fontsize=6)


def _panel_celltype(ax, units) -> None:
    if "peak_to_trough_duration" not in units:
        return
    x = units["peak_to_trough_duration"].to_numpy(float) * 1000.0  # s -> ms
    y = units["firing_rate"].to_numpy(float)
    ok = ~np.isnan(x) & ~np.isnan(y) & (y > 0)
    ax.scatter(x[ok], y[ok], s=4, alpha=0.3, lw=0, color="0.3")
    ax.axvline(_PT_NARROW_MS, ls="--", color=OKABE_ITO["vermillion"], lw=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("peak-to-trough duration (ms)")
    ax.set_ylabel("firing rate (Hz)")


def build_s2(tidy, analysis_root):
    import matplotlib.pyplot as plt

    sample = _sample_wells(tidy)
    units = L.load_units(analysis_root, sample)
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.4))
    _hist_qc(axes[0, 0], units, "presence_ratio")
    _hist_qc(axes[0, 1], units, "rp_contamination")
    _hist_qc(axes[0, 2], units, "amplitude_median")
    _hist_qc(axes[1, 0], units, "firing_rate", log=True)
    # representative well = the one with most curated units in the sample
    best = tidy.sort_values("n_curated", ascending=False).iloc[0]
    _panel_waveforms(axes[1, 1], analysis_root, best)
    _panel_celltype(axes[1, 2], units)
    for lab, ax in zip("ABCDEF", axes.ravel()):
        ax.set_title(lab, loc="left", fontweight="bold")
    fig.suptitle(f"Figure S2 — Single-unit QC & waveforms "
                 f"(n={units.attrs.get('n_wells', '?')} wells pooled)",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    return fig, units


def render(tidy, analysis_root, figure_root):
    fig, units = build_s2(tidy, analysis_root)
    return save_fig(fig, "S2_unit_qc_waveforms", figure_root, subdir="supp"), units
