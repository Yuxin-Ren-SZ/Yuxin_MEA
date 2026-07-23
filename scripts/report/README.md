# Journal report figures (`scripts/report/`)

Publication-grade, vector (PDF/SVG) + PNG figures for the HD-MEA network-scan
study, in one consistent style. Reads the pipeline's rolled-up tables + per-well
artifacts; writes to `<figure_root>/report/{main,supp}/`.

## Run

```bash
python -m scripts.report.make_report --config pipeline_config_local.json          # all
python -m scripts.report.make_report --figures f5 f6 s4                            # subset
python -m scripts.report.make_report --figures f3 s5 s6 \
    --qpcr-dir /mnt/Vol20tb2/SadeghLab/yuxin/analysis/qPCR                         # qPCR
python -m scripts.report.make_report --figures f2 \
    --f2-images gfap.tif map2.tif dapi.tif --f2-labels GFAP MAP2 DAPI \
    --f2-scalebar-um 50 --f2-px-per-um 1.5                                         # ICC

# Editable PowerPoint decks (one figure per slide; title + caption as text
# boxes, panels as a >=300 dpi image — 600 for detail-heavy figures):
python -m scripts.report.build_pptx --config pipeline_config_local.json \
    --qpcr-dir /mnt/Vol20tb2/SadeghLab/yuxin/analysis/qPCR                         # -> report/pptx/*.pptx
python -m scripts.report.build_pptx --config pipeline_config_local.json \
    --assemble-only                                                               # re-lay-out only (fast)
```

Every figure carries an on-image **caption** (below the panels) explaining what
each panel shows and how it is plotted; `report_style.caption()` draws it for the
standalone PNG/PDF/SVG and stores the text so `build_pptx` can place it — with the
title — in **editable text boxes** instead of baking it in. `build_pptx` caches a
caption-less PNG per figure under `report/pptx/_assets/` (`manifest.json`); layout
edits to the deck constants reassemble via `--assemble-only` without re-rendering.

## Figures

| id | file | source | needs input |
|----|------|--------|-------------|
| F1 | `F1_graphical_abstract` | user schematic assets | `--f1-assets` |
| F2 | `F2_icc` | user ICC micrographs | `--f2-images` (+labels/scale) |
| F3 | `F3_qpcr` gene x condition effect heatmap (cells in **fold-change**) (+ `F3_qpcr_tidy.csv`, `F3_qpcr_effects.csv`) | qPCR plate dirs (`*_ddcq.csv`, else raw AriaMx export + `*_labels.csv`) | `--qpcr-dir` (+ optional `--qpcr-genes`; legacy `--qpcr-table`; else SYNTHETIC placeholder) |
| F4 | `F4_activity_development` | `tidy_long.csv` + curated spikes | — |
| F5 | `F5_burst_phenotype` | `tidy_long.csv` (DiD stats); IVH_E/L + H2O2_20uM vs Control | `--rosglo-table` (panel B; else placeholder) |
| F6 | `F6_ml_characterization` (A HMM posteriors + B feature UMAP; representative well) + `F6d_burst_modulation` | `debug_trace.pkl`, `tidy_long.csv` | — |
| F6b | `F6b_umap_migration` | per-well pooled `debug_trace.pkl` (hand-picked chip-matched CX138 trio, shared UMAP; 4 columns = auto-spaced developmental **stages** Day 0→latest, not fixed days) | — |
| F6c | `F6c_burst_archetypes` | pooled `network_bursts.pkl` across focus groups (global burst-type clustering + per-group composition) | — |
| F7 | `F7_spatial_connectivity` | spatial activity maps + STTC graph-level metrics | — |
| F7b | `F7b_node_directed` | node cartography (hubs/leaves/participation) + directed transfer-entropy graph + DiD forests | — |
| F8 | `F8_criticality` | neuronal avalanches: power-laws (τ/α), crackling γ, branching ratio (MR), DCC + DiD | — |
| S1 | `S1_method_comparison` | `burst_method_comparison/per_well.csv` | — |
| S2 | `S2_unit_qc_waveforms` | `quality_metrics.pkl`, templates | — |
| S3 | `S3_representative_rasters` | curated spikes | — |
| S4 | `S4_crossgroup_matched_div` | `tidy_long.csv` (matched DIV) | — |
| S5 | `S5_reference_genes` | qPCR reference-gene screen plate (ACTB/GAPDH/HPRT1) | `--qpcr-dir` |
| S6 | `S6_qpcr_trajectories` | per-gene curves, vs `CT DIV 7` (A) and vs time-matched control (B) | `--qpcr-dir` |

## Key analysis decisions (read before trusting the stats)

* **Replicate hierarchy: chip = biological replicate, well = technical
  replicate.** The three chips (`sample_id`: CX118/CX138/CX169) are the
  independent biological samples; the wells within a chip share one plating /
  CSF-application batch, so treating them as independent is pseudo-replication
  (Lazic 2010, *BMC Neurosci*). Recordings within a well are aggregated first
  (median), so they never count as replicates either. Two consequences run
  through every treatment-response figure (`stats.py`, `forest.py`):
  - **"All data points" = per-chip means** (`stats.chip_response`): the bold
    scatter points are the ~3 biological replicates per arm; individual wells are
    drawn faint behind them for context.
  - **Inference = cluster-summary DiD** (`stats.arm_response_vs_control_lmm`).
    Treatment is applied *within* chip (every chip carries Control + treated
    wells), so a random-**intercept** model `response ~ arm + (1|chip)` does NOT
    protect the arm effect: the contrast is within-chip and its SE comes from the
    **well-level residual df (~13), not the 3 chips** — pseudo-replication in a
    mixed-model coat (its Wald p ran 40-400× too small here). Generalising across
    chips needs a random *slope* `(arm|chip)`, unidentifiable at 3 groups; its
    finite-sample equivalent is a **one-sample t-test of the per-chip DiDs vs 0**
    (`stats.within_chip_did`, df = n_chips − 1). That drives `p_vs_control` /
    `q_bh`. The mixed model is still fit but only for the point estimate
    (`lmm_coef`, ≈ `did_chipmean`); its `lmm_p` is kept for transparency, never
    for significance. A forest **★ needs q<0.05 (BH-FDR)**; **△** flags a
    *suggestive* effect (same direction on all 3 chips, uncorrected p<0.05, but
    does NOT survive FDR at n=3).
* **Design is within-well pre/post.** A treated well is labelled `Control` on its
  baseline recordings (`tau < 0`) and its treatment name post (`tau > 0`).
* **The raw pre→post change is confounded with development** — the culture
  matures over the same interval. Every treatment-response headline is therefore
  a **difference-in-differences**: each treated arm's `log2(post/pre)` (or Δ)
  response vs **Control wells'** response (maturation baseline).
  **Finding on this dataset — an n=3 pilot: NOTHING survives FDR.** At the
  biological-replicate (chip) level every effect sits at chip-level p≈0.03-0.08,
  so after BH-FDR within each figure's metric family **no arm × metric is
  confirmed**. What there *is* is a set of **suggestive, consistent-direction
  signals** (marked △: same sign on all 3 chips, uncorrected p<0.05):
  - **Criticality (F8/F6d), all in IVH_Early:** avalanche size-exponent α flatter
    (chips [−0.085,−0.134,−0.063], p≈0.046), burst modulation ↑ (F6d, p≈0.043),
    DCC ↑ but weaker ([+0.18,+0.11,+0.06], p≈0.077 → not even △). Direction is
    "drifting off criticality," but it is a hint, not a result.
  - **Burst phenotype (F5):** **median firing rate ↓ in both IVH arms**
    (p≈0.030 / 0.032, △); everything else tracks Control maturation (nb_rate ↑,
    duration ↓ are developmental — Control shows nb_rate +0.93 log2 too).
  - **Connectivity / node / directed (F7, F7b):** null. (A stray △ on a single
    node/directed metric, e.g. assortativity, is expected by chance across the
    ~18 uncorrected node+directed tests — △ is uncorrected p<0.05; nothing
    survives FDR.)
  Report this as a pilot: the biology is coherent and worth following up with
  more chips, but **do not claim significance**. Statistical note: an earlier
  intercept-only mixed model (`response ~ arm + (1|chip)`) reported these as
  FDR-significant (q≈0.002-0.05) — that was **anti-conservative** (the arm
  contrast is within-chip, so its Wald SE came from ~13 wells, not 3 chips). The
  cluster-summary t-test above is the honest replicate-level test. S4
  (matched-DIV cross-group) agrees on the nulls.
* **We do NOT reuse `treatment_comparison/stats.csv`.** That table paired on
  `role`, which mis-files every baseline as a control well and collapses paired
  n to ~3/arm. `report/stats.py` recomputes on `tau` sign (paired n = 7-13 per
  arm; 19 Control). (The bug is in `scripts/compare_treatment_groups.py`; patch
  separately if the pipeline table is needed elsewhere.)
* **F3 removes the developmental confound the same way F5 does.** Every arm was
  split from one culture on the treatment day, so they share a single baseline
  sample (`CT DIV 7`) and there is no per-arm day 0. The difference-in-differences
  therefore collapses to a **time-matched-control contrast**, and the calibrator
  cancels exactly:
  `effect(arm,d) = ΔΔCq(arm,d) − ΔΔCq(Control,d) = ΔCq(arm,d) − ΔCq(Control,d)`.
  Same estimand as the MEA DiD above, so F3 and F5-F8 rest on one logic.
  `fig3_qpcr.treatment_effects()` computes it; `tests/test_qpcr_report.py` locks
  the calibrator-independence. Without this the curves are dominated by
  maturation — Control alone rises +4.5 log2 for SLC17A7 by D7.
  **Finding on this dataset (n = 1, descriptive):** both IVH arms move together
  and progressively — GFAP up (4.5-4.8x by D21), SLC17A7/SLC32A1 down (to
  0.10-0.25x) — while H2O2_20uM shows a different profile.
  F3 reports **fold-change** (2^-ΔΔCq); its colour scale is spaced
  logarithmically about 1x, because fold change is asymmetric (2x up = 2.0,
  2x down = 0.5) and a linear colour ramp would compress every downregulated
  cell against zero. S6 keeps a log2 y-axis, where a symmetric linear axis is
  the readable choice for trajectories.
* **F3/S5/S6 are descriptive, at n = 1 chip.** One CX169 sample per condition,
  technical duplicates only, so there are **no statistics and no error bars** —
  the only available spread is technical, which is not biological replication.
  Propagated SEMs stay in `F3_qpcr_effects.csv`. The **data layer and CSVs are
  multi-chip ready** — per-chip calibration (`_calibrate_per_chip`) plus
  across-chip mean ± SEM (`aggregate_chips`) — and S6 starts drawing those
  error bars once more than one chip is loaded. **Figure rendering at n ≥ 2 is
  provisional**: the F3 heatmap shows across-chip means with no spread (it is a
  heatmap), and only a synthetic duplicate chip has been rendered so far.
  Revisit whether a plain across-chip SEM is the right summary, and whether the
  `stats.py` Wilcoxon/BH-FDR machinery should be wired in, when real replicate
  chips exist — an n = 2 SEM is not an inferential result. ΔΔCq assumes ~100 % amplification efficiency (no standard curves were
  run). Reference gene = GAPDH, calibrator = `CT DIV 7` (tau 0, the treatment
  day) — see S5 for the screen.
* **qPCR panel coverage: 4 of 13 planned target genes** (plates 1-2 of 7 in
  `qpcr-layout-neuronal-8plate.csv`). `GENE_TIERS` holds the full panel —
  priority 1 MAP2/GFAP/SLC17A7/SLC32A1/PAX6/STX1A, priority 2
  OLIG2/KCNMA1(BK)/KCNN2(SK)/HCN1(HCN)/SOX2/PVALB/SST. New **gene** plates are
  picked up by dropping their export directory under `--qpcr-dir`; the heatmap
  grows rows and S6 wraps at 4 columns, no code change. A second **chip** is
  calibrated separately (`_calibrate_per_chip`) rather than pooled against
  CX169's baseline.
* **Two qPCR conditions failed technically and are shown flagged, never silently
  dropped.** `CT D14` (all three reference genes shift ~+10 Cq on every plate →
  cDNA/RNA loading failure; its MAP2 ratio reads 119x) and `20uM H2O2 D7` (no
  amplification anywhere). QC-failed cells are **excluded from the F3 colour
  scale and its vmax** — otherwise the +6.89 log2 artefact sets the limits and
  washes every real cell out, the 2D form of the axis clamp S6 uses. Because
  `CT D14` failed, D14 has no measured time-matched control; Control ΔCq is
  near-flat D7→D21 so it is **linearly interpolated and marked `†`** (never
  extrapolated past the last measured day — those come out NaN).
* **AraC (3 wells, CX118 only) and NPH (4 wells, CX138 only)** have no chip
  replication → shown as *exploratory* (open markers, no p-value) in main
  figures, never annotated significant.

## Modules

`report_style.py` (palette `GROUP_PALETTE`, `apply_style`, `save_fig` vector
export) · `load.py` (tables + per-well artifact loaders) · `stats.py`
(within-well response; per-chip aggregation `chip_response`; within-chip DiD
companion `within_chip_did`; chip-level DiD via well-nested-in-chip `MixedLM`
`arm_response_vs_control_lmm`; matched-DIV) · `forest.py` (shared chip-level DiD
forest, used by F5/F7/F7b/F8) · `build_pptx.py` (editable multi-page PPTX export:
caption-less high-dpi image + title/caption text boxes per slide, render/assemble
passes) · `figN_*.py` builders · `fig3_qpcr.py` (qPCR plate loader, ΔΔCq fallback,
`treatment_effects` / `control_reference` / `aggregate_chips`; reused by
`fig_s5_refgene.py` and `fig_s6_qpcr_trajectories.py`) · `make_report.py` driver.
