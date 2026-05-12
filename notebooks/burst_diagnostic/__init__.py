"""Burst detector diagnostic toolkit.

Re-exports the public API of :mod:`viz` so callers can write either
``from burst_diagnostic import viz`` or ``from burst_diagnostic import run_batch``.
"""
from . import viz
from .viz import (
    BatchResults,
    build_dashboard_app,
    discover_real_spike_sources,
    fig_cross_stage_flow,
    fig_kill_attribution,
    fig_section_c_lda_pca,
    fig_section_d_boundary_shift,
    fig_section_e_3d_pca,
    fig_section_f_gmm_bic_sweep,
    fig_section_g_time_strip,
    fig_stage1_composite_slider,
    fig_stage2_participation,
    fig_stage3_bmi,
    fig_stage4_gmm_pca,
    load_kilosort_spike_times,
    run_batch,
    save_all_section_htmls,
    save_html,
)

__all__ = [
    "BatchResults",
    "build_dashboard_app",
    "discover_real_spike_sources",
    "fig_cross_stage_flow",
    "fig_kill_attribution",
    "fig_section_c_lda_pca",
    "fig_section_d_boundary_shift",
    "fig_section_e_3d_pca",
    "fig_section_f_gmm_bic_sweep",
    "fig_section_g_time_strip",
    "fig_stage1_composite_slider",
    "fig_stage2_participation",
    "fig_stage3_bmi",
    "fig_stage4_gmm_pca",
    "load_kilosort_spike_times",
    "run_batch",
    "save_all_section_htmls",
    "save_html",
    "viz",
]
