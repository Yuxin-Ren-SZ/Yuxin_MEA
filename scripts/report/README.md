# Journal report figures (`scripts/report/`)

Publication-grade figures for the HD-MEA network-scan study. **The deliverable is
one image per subplot**, not one image per figure: each panel is a standalone
script producing a clean, caption-free PNG/PDF/SVG that can be dropped into a
manuscript, slide or poster unchanged, with its caption in a sibling `.txt`.

## Run

```bash
# one panel, on its own
python -m scripts.report.panels.f4c_firing_rate
python -m scripts.report.panels.f7e_activity_fields --formats png --dpi 600

# many panels in one process (shares the cache; much faster than looping)
python -m scripts.report.render_panels --all
python -m scripts.report.render_panels --figures f4 f7
python -m scripts.report.render_panels --panels f5a_hmm_posteriors f5b_did_nb_rate

# arrange the rendered panels into per-figure review sheets (no re-render)
python -m scripts.report.assemble_figure --figures f4 f5 f7

# portrait US-Letter PowerPoint: one figure per page, panels as images and
# everything else (figure number, panel letters, caption) as editable text boxes
python -m scripts.report.panel_pptx

# the immunofluorescence panel takes image files (blank frames until supplied)
python -m scripts.report.panels.f3c_icc --images map2_d0.tif ... --px-per-um 1.5

# the supplement (S1-S5) is still one figure per builder, not panels
python -m scripts.report.make_report --qpcr-dir /path/to/qPCR
python -m scripts.report.build_pptx --qpcr-dir /path/to/qPCR
```

Output layout under `<figure_root>/report/`:

```
panels/     F4c_firing_rate.{png,pdf,svg}       the panel itself, no caption
panels/     F5b_did_nb_rate_stats.csv           the p and q behind its glyphs
captions/   F4c_firing_rate.txt                 the caption, as text
figures/    F4_sheet.png, F4_captions.txt       assembled review sheet
supp/       S2_unit_qc_waveforms.{png,pdf,svg}  the supplement, captions baked in
pptx/       report_panels.pptx                  portrait Letter deck, editable text
pptx/       report_supp.pptx                    the supplement deck
_cache/                                          shared derived tables (safe to delete)
```

Superseded output from earlier generations of this code lives beside the tree, in
`<figures_base>/report_archive/<date>/`, with a README saying what each directory
was and what replaced it. Nothing under `report/` is stale.

## Writing a panel

```python
"""F4d — Network-burst rate vs treatment day."""
from .. import panel_data as D
from ..panel_base import panel_main
from ..trajectory import plot_trajectory

NAME    = "F4d_nb_rate"
FIGSIZE = (3.4, 2.6)
CAPTION = "Network-burst rate across maturation, aligned on treatment day (τ)..."

def build(ax, ctx):          # or build_fig(fig, ctx) / make_fig(ctx)
    plot_trajectory(ax, ctx.main, "nb_rate", D.MAIN_ARMS, legend=True)

if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
```

`build(ax, ctx)` for one axes, `build_fig(fig, ctx)` for a composite,
`make_fig(ctx)` when the grid size depends on the data. Returning a DataFrame
writes `<NAME>_stats.csv` beside the image. `panel_data.context()` gives every
panel the same disk-cached data — running panels separately never re-fits the
minute-long burst-archetype clustering or the pooled UMAP.

## Figure map

| id | panels | what it shows |
|----|--------|---------------|
| F1 | — | graphical abstract (prepared externally) |
| F2 | `f2a_schedule` | culture + recording schedule, DIV and τ axes |
| F3 | `f3a_qpcr_heatmap` `f3b_qpcr_trajectories` `f3c_icc` | cell characterisation: qPCR at days 0/7/14/21 + immunofluorescence (blank frames until images are supplied) |
| F4 | `f4a/f4b` rasters, `f4c–f4f` trajectories | recording schema works and the cultures mature: 300 s rasters pre-treatment vs τ+14, then firing rate, burst rate, burst-interval CV and burst duration vs τ |
| F5 | `f5a` HMM posteriors, `f5b–f5f` DiD, `f5g` time course | how the detector sees a recording, then the burst-phenotype treatment effect |
| F6 | `f6a` `f6b` UMAP, `f6c–f6e` archetypes | feature-space and developmental embeddings; cohort-wide burst archetypes with composition, feature profile and ≥3 example rasters each |
| F7 | `f7a–f7c` CCG, `f7d` STTC graphs, `f7e` activity fields, `f7f` distance, `f7g–f7m` graph DiD, `f7n/f7o` burst modulation, `f7p–f7s` node + directed | connectivity, from single-pair correlograms up to graph, node and directed metrics |
| F8 | `f8a/f8b` trajectories, `f8c–f8g` DiD, `f8h/f8i` avalanches | neuronal-avalanche criticality |
| S1–S5 | `fig_s*.py` (composed, built by `make_report`) | method comparison, unit QC, rasters, cross-group at matched DIV, reference genes |

Main figures show **Control, IVH_Early, IVH_Late** only. H2O2, NPH and AraC stay
in the supplement (S4) — NPH and AraC exist on one chip each, so they cannot
support a chip-level test at all.

## Statistics — read this before trusting a glyph

* **Chip = biological replicate, well = technical replicate.** The three chips
  (CX118/CX138/CX169) are the independent samples; wells within a chip share one
  plating and one CSF application, so counting them as independent is
  pseudo-replication (Lazic 2010). Recordings are aggregated to the well first,
  so they never count either. Raw points encode this: **small grey = wells,
  large coloured = per-chip means**, and the test runs on the chip means.
* **Every treatment read-out is a difference-in-differences.** The raw pre→post
  change is confounded with development, because the culture matures over the
  same interval. Each treated arm is read against the Control arm's change.
* **Time points are not pooled.** `stats.well_response_by_tau` compares each well
  to *its own* pre-treatment baseline separately in each post-treatment day bin
  — the ΔΔCq logic applied to a time series. Wells collapse to a chip mean, each
  chip's treated-minus-own-Control value is its DiD, and the test is a
  one-sample t-test of those against zero (df = n_chips − 1).
* **One pre-specified endpoint, one FDR family.** The formal test runs at the
  latest τ bin all three chips still reach (τ+14 on this dataset — CX138 stops
  recording at τ15 while the others run to τ27). BH-FDR corrects across **that
  figure's metrics at that endpoint**; the metric families are the `F5_METRICS`
  /`F7_METRICS`/`F8_METRICS` constants in `panel_data.py`, and a panel must pass
  its whole family or `q` silently collapses to `p`. Every other bin is plotted
  in the `*_did_timecourse` panels and written to the stats CSV, but is not
  separately tested — that would multiply the family and, at n=3, leave nothing
  detectable.
* **Two tiers, and nothing else may promote a mark.**
  `*`/`**`/`***` when **q** < 0.05/0.01/0.001 (BH-FDR); `△` when the uncorrected
  **p** < 0.05 but q does not clear. No direction-consistency clause, no
  hand-picked headline metric. A panel is tinted **only** when it carries an
  FDR-significant effect.
* **What that yields on this dataset: no highlights and no stars.** Across the 52
  endpoint tests, **zero** clear FDR (lowest q ≈ 0.13) and five carry △:

  | arm | metric | chips | Δ | p | q |
  |---|---|---|---|---|---|
  | IVH_Late | burst duration | 2 | −1.60 | 0.046 | 0.23 |
  | IVH_Early | spikes per burst | 3 | −1.94 | 0.042 | 0.23 |
  | IVH_Early | mean betweenness | 3 | −0.005 | 0.034 | 0.20 |
  | IVH_Late | rich-club coeff. | 3 | −0.344 | 0.011 | 0.13 |
  | IVH_Early | branching ratio (MR) | 3 | −0.109 | 0.049 | 0.25 |

  Note the chip count per row: coverage differs by metric and arm, and it is
  printed under each group's x-tick, because a t-test on two chips has df = 1.
  Report this as a pilot with consistent-direction signals, not as a result —
  **do not loosen the threshold to make highlights reappear**. Regenerate this
  table from `report/panels/*_stats.csv` after any recompute.
* **`participation_mean` and `rich_club` are difference metrics, not ratios.**
  Both are bounded in [0, 1] and genuinely reach 0 in sparse wells, where a log2
  ratio produced −30 "responses" that swamped every other chip.
* **We do not reuse `treatment_comparison/stats.csv`.** It pairs on `role`, which
  mis-files every treated well's baseline recordings as control wells (a treated
  well is labelled `Control` before treatment and its treatment name after).
  `report/stats.py` recomputes on the sign of `tau`.
* **F3 is descriptive at n = 1 chip** — technical duplicates only, so no error
  bars and no statistics. It removes the developmental confound the same way:
  every arm was split from one culture on the treatment day and shares a single
  baseline sample, so the DiD collapses to a time-matched-control contrast and
  the calibrator cancels exactly. Reference gene GAPDH (see S5); ΔΔCq assumes
  ~100 % amplification efficiency. Two conditions failed technically (`CT D14`,
  all reference genes shifted ~+10 Cq; `20uM H2O2 D7`, no amplification) and are
  shown flagged and excluded from the colour scale, never silently dropped.

## Modules

`report_style.py` — palette, tokens, replicate-encoding sizes, `sig_glyph`,
`apply_style`, `save_fig` · `load.py` — tables and per-well artifacts
(`load_ccg`, `load_edges`, `load_curated_spikes`, …) · `stats.py` — well/chip
responses, `well_response_by_tau`, `did_at_endpoint`, `primary_endpoint` ·
`forest.py` — `did_column_panel` (the unit every DiD panel draws) ·
`trajectory.py` — τ trajectories and the shared `tau_band` experiment-window
marker · `raster.py` — firing-rate-sorted rasters · `panel_data.py` — cached data
context and the FDR families · `panel_base.py` — the panel runner ·
`panels/` — one module per subplot · `render_panels.py`, `assemble_figure.py`,
`panel_pptx.py` — the main-figure drivers · `make_report.py`, `build_pptx.py` —
the supplement (S1–S5), which is still composed one figure per builder. The
`fig*.py` modules are those builders plus the per-axes helpers the panels import
out of them; their composed `render()` entry points are gone, so each figure has
exactly one producer.
