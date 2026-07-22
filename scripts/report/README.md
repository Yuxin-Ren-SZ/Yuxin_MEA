# Journal report figures (`scripts/report/`)

Publication-grade, vector (PDF/SVG) + PNG figures for the HD-MEA network-scan
study, in one consistent style. Reads the pipeline's rolled-up tables + per-well
artifacts; writes to `<figure_root>/report/{main,supp}/`.

## Run

```bash
python -m scripts.report.make_report --config pipeline_config_local.json          # all
python -m scripts.report.make_report --figures f5 f6 s4                            # subset
python -m scripts.report.make_report --figures f3 --qpcr-table qpcr.csv            # qPCR
python -m scripts.report.make_report --figures f2 \
    --f2-images gfap.tif map2.tif dapi.tif --f2-labels GFAP MAP2 DAPI \
    --f2-scalebar-um 50 --f2-px-per-um 1.5                                         # ICC
```

## Interactive composer

`make_report` builds a *fixed* set of figures — the panel list, the grid and the
treatment arms are hard-coded in each `build_fX`. When the layout is still being
decided, use the composer instead:

```bash
python -m scripts.report.composer --config pipeline_config_local.json   # http://127.0.0.1:8060
```

Every panel is rendered and shown on one page. Drag panels between figures,
resize them on a 12-column grid, add or remove treatment arms, split a four-panel
figure in two or merge two into one, then **Export** — which writes, into
`<figure_root>/report/composed/<name>/`:

* `figures.pdf` — one page per figure,
* `<FigureID>.{pdf,png,svg}` per figure,
* `figure_spec.json` — **the resume file**: load it next session (top-bar *Load*,
  or `--` pass it to the CLI below) to get the layout, groups and every panel
  parameter back,
* `manifest.json` — outputs, data fingerprint and per-panel warnings.

Headless, for scripting or to re-render a spec unchanged:

```bash
python -m scripts.report.composer.compose figure_spec.json --out /tmp/figs
python -m scripts.report.composer.compose --name report        # the built-in default
```

Notes:

* **Preview equals export.** Panel thumbnails come from the same matplotlib code
  the PDF uses, and a panel's `(x, y, w, h)` on the canvas are the same integers
  the exporter feeds to `GridSpec`. Per-figure *Preview* renders the whole figure
  through the export path when you want to be certain.
* **Group coverage is shown.** Node-graph, criticality and directed-TE metrics
  were only computed for the focus arms, so the group list marks
  `NPH — no node/criticality/directed` and the affected panels say so rather than
  drawing an empty row.
* Panels live in `panels.py` (registry + `RenderContext`), `panels_stats.py`
  (forests, matched-DIV), `panels_artifacts.py` (rasters, maps, graphs,
  avalanches, unit QC) and `panels_cohort.py` (burst archetypes, state
  manifolds). Adding a panel is one `register(PanelSpec(...))` call; it then
  appears in the palette with its parameters as widgets.
* **Expensive shared work is fit once.** F6c's four views share one cohort-wide
  UMAP + KMeans, and an F6b row of timepoints shares one pooled per-well
  embedding, via `RenderContext.dataset`. Placing four archetype panels costs
  one fit, not four.

## Figures

| id | file | source | needs input |
|----|------|--------|-------------|
| F1 | `F1_graphical_abstract` | user schematic assets | `--f1-assets` |
| F2 | `F2_icc` | user ICC micrographs | `--f2-images` (+labels/scale) |
| F3 | `F3_qpcr` | user qPCR table | `--qpcr-table` (else SYNTHETIC placeholder) |
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

## Key analysis decisions (read before trusting the stats)

* **Unit of analysis = physical well (`well_uid`), never the recording.** Every
  box/point/p-value aggregates a well's recordings first (median). The chip
  (`sample_id`) is the biological replicate.
* **Design is within-well pre/post.** A treated well is labelled `Control` on its
  baseline recordings (`tau < 0`) and its treatment name post (`tau > 0`).
* **The raw pre→post change is confounded with development** — the culture
  matures over the same interval. F5's headline is therefore a
  **difference-in-differences**: each treated arm's `log2(post/pre)` response vs
  **Control wells'** response (maturation baseline; 19 paired Control wells),
  Mann-Whitney + BH-FDR (`stats.arm_response_vs_control`). **Finding on this
  dataset: treated arms track Control maturation; no arm differs from Control
  after FDR.** The large raw pre→post changes (nb_rate ↑, duration ↓) are
  developmental, not treatment effects — Control shows nb_rate +0.93 log2 too.
  S4 (matched-DIV cross-group) independently agrees. Report the null honestly.
* **We do NOT reuse `treatment_comparison/stats.csv`.** That table paired on
  `role`, which mis-files every baseline as a control well and collapses paired
  n to ~3/arm. `report/stats.py` recomputes on `tau` sign (paired n = 7-13 per
  arm; 19 Control). (The bug is in `scripts/compare_treatment_groups.py`; patch
  separately if the pipeline table is needed elsewhere.)
* **AraC (3 wells, CX118 only) and NPH (4 wells, CX138 only)** have no chip
  replication → shown as *exploratory* (open markers, no p-value) in main
  figures, never annotated significant.
* **Ratio vs difference is decided by whether a metric is ever zero**, not by
  whether it is non-negative (`stats.RATIO_METRICS` / `DIFF_METRICS`). A log2
  ratio needs a strictly positive baseline *and* post; with either at zero the
  `_EPS` guard fabricates a response of ~±30 log2 units that then drives the
  forest medians and CIs. Twelve bounded graph descriptors are frequently
  exactly zero — `hub_fraction` in 1364/1516 wells, `leaf_fraction` 454/1516,
  `edge_density`/`global_efficiency` 86/1843, `clustering_coeff` 123/1843 — so
  they are difference metrics, and **the F7/F7b/F8 forest x-axes are `post − pre`
  for those rows**, not log2. Ratio is kept only where zero is rare and
  meaningful (rates, counts, durations) or impossible (MLE avalanche exponents,
  small-worldness σ). `well_response` flags any remaining `eps_floored` row; on
  the current cohort there are none.

## Modules

`report_style.py` (palette `GROUP_PALETTE`, `apply_style`, `save_fig` vector
export) · `load.py` (tables + per-well artifact loaders) · `stats.py`
(within-well response, matched-DIV, well-level Wilcoxon/MWU + BH-FDR) ·
`figN_*.py` builders · `make_report.py` driver.
