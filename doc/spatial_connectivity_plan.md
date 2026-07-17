# Spatial activity maps & functional connectivity — implementation plan

Status: **design / not yet built.** Reserved as report figure **F7** (placeholder
in the current draft). Build immediately after the draft is frozen.

## Motivation

Two figures that reviewers expect from an HD-MEA paper are absent from the
current pipeline (confirmed by exhaustive grep — no STTC, cross-correlation,
functional-connectivity, or electrode-level activity-map code exists):

1. **Spatial activity map** — where on the electrode plane the network is active,
   and how coordinated activity propagates in space during a network burst.
2. **Functional connectivity** — pairwise coupling between sorted units (Spike
   Time Tiling Coefficient), summarised as a graph with network topology metrics.

Both are *new computation*, not new plotting. This document specifies the data
contracts, module/function signatures, task-DAG integration, compute cost, and
the F7 figure spec so the work can start without re-discovery.

Unit of analysis stays the **physical well** (`well_uid`); per-recording scalars
roll up the same way `nb_rate` etc. already do (see
[`treatment_comparison_metrics.md`](treatment_comparison_metrics.md)), so F7's
group comparisons reuse `scripts/report/stats.py` (well-level, chip = replicate).

## Inputs available on disk (already produced)

Per well-recording, under `analysis_root/<stage>/<recording_key>/<rec>/<well>/`:

| Input | Path (subdir/file) | Contents |
|---|---|---|
| Curated spike trains | `curation_data/…/auto_curation/curated_spike_times.npy` | 0-d object array → `{unit_id: spike_times_s}` (the burst detectors' input contract) |
| Unit locations | `curation_data/…/auto_curation/quality_metrics.pkl` cols `loc_x, loc_y`; or `analyzer_data/…/extensions/unit_locations/unit_locations.npy` `(n_units, 3)` | monopolar-triangulation x,y (µm) on the electrode plane |
| Per-unit firing rate | `quality_metrics.pkl` col `firing_rate` | for the activity map weighting |
| Detected network bursts | `burst_detection_data/…/network_bursts.pkl` (or `ml_burst_data_umap/…`) | `start,end` per burst → burst windows for propagation |

`recording_dur_s` is derivable from spike-time max or from the analyzer.

---

## (A) Spatial activity map — moderate effort

### Module `src/yuxin_mea/analysis/spatial_map.py`

```python
def unit_positions(quality_metrics: pd.DataFrame) -> np.ndarray:  # (n,2) µm
def activity_field(pos, firing_rate, *, bins=64, sigma_um=50.0) -> np.ndarray:
    """Gaussian-KDE / 2-D histogram of firing rate over the electrode plane."""
def burst_propagation(spikes: dict[int, np.ndarray], pos: np.ndarray,
                      bursts: pd.DataFrame) -> pd.DataFrame:
    """Per burst: each unit's first-spike latency from burst onset →
    spatial propagation gradient. Returns per-burst dict of
    {origin_xy, speed_um_per_ms, r2_planar_fit, n_participating}."""
def spatial_summary(field, prop) -> dict:
    """Well-level scalars: active_area_frac, activity_centroid,
    activity_gini (spatial concentration), mean_prop_speed_um_ms,
    prop_planarity (median r2), mean_prop_delay_ms."""
```

* **Activity field**: weight each unit position by `firing_rate`, KDE with
  `sigma_um≈50` (matches the analyzer sparsity radius), normalise. Gives the
  per-well 2-D map + a spatial-concentration scalar (Gini of the field).
* **Burst propagation**: for each detected network burst, take each
  participating unit's first spike after `start`; fit a plane
  `latency ≈ a·x + b·y + c`; propagation speed = 1/‖∇‖, direction from ∇,
  origin = arg-min latency. Aggregate median speed / planarity per well.

### Output contract `spatial_map_data/…/spatial_map/`

* `activity_field.npy` — `(bins, bins)` normalised field.
* `burst_propagation.parquet` — one row per burst (origin_xy, speed, r2, n).
* `spatial_metrics.json` — well-level scalars (schema = `spatial_summary` keys).
* `diagnostics.json` — `n_units, bins, sigma_um, n_bursts_used`.

### Compute cost
Cheap: KDE is O(n_units·bins²); propagation is O(n_bursts·n_units). Seconds per
well; a full 1900-well pass is I/O-bound, single-process fine (`n_jobs` optional).

---

## (B) Functional connectivity (STTC) — significant effort

### Module `src/yuxin_mea/analysis/connectivity.py`

Spike Time Tiling Coefficient (Cutts & Eglen, *J Neurosci* 2014) — the standard
MEA connectivity measure, insensitive to firing-rate differences.

```python
def sttc_pair(a, b, dt, T) -> float:
    """STTC for two spike trains, tiling window dt (s), recording length T."""
def sttc_matrix(spikes: dict[int, np.ndarray], dt=0.02, T=None) -> np.ndarray:
    """Symmetric (n_units, n_units) STTC. Vectorised tiling-fraction calc."""
def graph_metrics(W: np.ndarray, thresh: str | float = "shuffle") -> dict:
    """Threshold to adjacency, then networkx metrics:
    mean_sttc, edge_density, mean_degree, clustering_coeff, modularity
    (greedy), global_efficiency, small_worldness (vs degree-preserving null)."""
def sttc_significance(spikes, dt, T, n_shuffle=100) -> np.ndarray:
    """Per-edge p via jittered/circular-shuffled trains → significance mask."""
```

* **dt sweep**: compute at `dt ∈ {5, 10, 20, 50} ms`; report the full sweep in
  diagnostics, use `dt=20 ms` as the primary edge weight.
* **Thresholding**: keep edges above the 95th percentile of a shuffled null
  (circular jitter) so density is comparable across wells.
* **Spatial embedding**: pair STTC with `loc_x/loc_y` to render the graph on the
  electrode plane and to compute a distance-vs-connectivity decay curve.

### Output contract `connectivity_data/…/connectivity/`

* `sttc_matrix.npy` — `(n_units, n_units)` at primary dt.
* `sttc_sweep.npz` — matrices for each dt.
* `graph_metrics.json` — well-level scalars (schema = `graph_metrics` keys).
* `edges.parquet` — significant edges: `u, v, sttc, dist_um`.
* `diagnostics.json` — `dt_primary, n_units, n_edges, thresh_method, n_shuffle`.

### Compute cost — the one to watch
`sttc_matrix` is **O(n_units² · median_spikes)** per well; with the shuffle null
(×`n_shuffle`) this is the expensive step. n_units ≤ ~250, spikes ≤ ~10⁵.
Mitigations: vectorise the tiling-fraction over unit pairs (numpy), cap
`n_shuffle` (100), optionally Numba-JIT `sttc_pair`, and run two-phase like the
existing GPU/CPU split — **CPU `n_jobs`≈11**, one well per worker. Budget a full
1900-well pass at hours, not minutes; make it resumable via the pipeline cache.

---

## Pipeline-DAG integration

Add two tasks mirroring the `burst_detection` I/O contract
(`src/yuxin_mea/tasks/burst_detection.py` is the template — reads
`curated_spike_times.npy`, writes a per-well output dir, records status in
`pipeline_cache.json`):

* `src/yuxin_mea/tasks/spatial_map.py` → stage `spatial_map_data`.
* `src/yuxin_mea/tasks/connectivity.py` → stage `connectivity_data`.

Both depend on `auto_curation` (need curated spikes + `loc_x/loc_y`);
`connectivity` optionally on `burst_detection` if restricting STTC to burst
epochs. Register in the scheduler task list and add config blocks to
`pipeline_config_local.json` (`dt`, `sigma_um`, `n_shuffle`, `n_jobs`).

### Roll-up into the report tables
Extend `scripts/compare_treatment_groups.py::METRIC_SPECS` with the new scalars
(`mean_sttc`, `edge_density`, `modularity`, `small_worldness`,
`mean_prop_speed_um_ms`, `activity_gini`) so they land in `tidy_long.csv`.
Then F7 group stats come **for free** from `scripts/report/stats.py`
(well_uid unit, within-well pre/post + matched-DIV), identical to F5/S4.

---

## F7 figure spec (report)

New builder `scripts/report/fig7_spatial_connectivity.py`, style via
`report_style` (shared palette, vector export), same well_uid unit:

* **A** Activity field — example well's 2-D firing-rate map (electrode plane),
  control vs treated at matched DIV.
* **B** Burst propagation — units coloured by first-spike latency for one burst,
  arrow = fitted propagation gradient.
* **C** Functional-connectivity graph — STTC edges drawn on the electrode plane
  (edge alpha ∝ STTC), one control + one treated example.
* **D** Group comparison — within-well response (log2 post/pre) forest of
  `mean_sttc, edge_density, modularity, small_worldness`, reusing
  `fig5_burst_phenotype`'s forest machinery (confirmatory vs exploratory tiers,
  AraC/NPH single-chip open markers).

## Build order

1. `analysis/spatial_map.py` + unit test on one well; eyeball A/B.
2. `analysis/connectivity.py` (`sttc_pair`/`sttc_matrix` first, validate vs a
   known small case; then graph metrics + shuffle null).
3. Wrap both as `tasks/*`, wire into scheduler + config, small batch run.
4. Extend `METRIC_SPECS`, regenerate `tidy_long.csv`.
5. `fig7_spatial_connectivity.py`, add `f7` to `make_report.py`.
