# ML Burst Detection — Algorithm Reference

End-to-end description of the machine-learning burst detector, from spike trains
to the final three-level burst hierarchy. Covers feature computation, HDBSCAN
clustering, burst-cluster selection, temporal merging, and event materialization.

**Source modules**
- Orchestrator: `src/yuxin_mea/analysis/ml_burst_detector.py` (`compute_ml_bursts`)
- Features: `src/yuxin_mea/analysis/ml_burst_features.py` (`build_feature_matrix`)
- Per-unit HMM: `src/yuxin_mea/analysis/ml_burst_hmm.py` (`fit_all_units`)
- Clustering + merge: `src/yuxin_mea/analysis/ml_burst_cluster.py`
- Shared merge helpers: `src/yuxin_mea/analysis/burst_common.py`
- Config: `MLBurstConfig` (`ml_burst_detector.py`), task defaults in
  `src/yuxin_mea/tasks/ml_burst_detection.py`

**Input.** A dict `spike_times[unit_id] -> np.ndarray` of spike timestamps (seconds)
for one well, plus the recording start/end. **Output.** Three pandas DataFrames —
`burstlets`, `network_bursts`, `superbursts` — written as pickles, plus
`metrics.json`, `diagnostics.json`, `plot_signals.npy`, and (when `debug=True`)
`debug_trace.pkl`.

---

## Pipeline overview

```
spike_times
  │
  ├─ 1. Binning            → bin_size, bins, t_centers
  ├─ 2. Per-unit HMM       → fits, posteriors (burst-state prob per unit per bin)
  ├─ 3. Feature matrix     → X  (n_bins × 26), feature_names
  ├─ 4. Clustering         → z-norm → (UMAP) → HDBSCAN → labels per bin
  │      Burst selection   → burst_labels  (which clusters = burst)
  │      Burst mask        → boolean per-bin mask
  ├─ 5. Temporal merge     → morphological closing + valley merge → candidates
  ├─ 6. Burstlets          → materialize candidates + soft llr gate   [Level 1]
  └─ 7. Hierarchy merge    → network_bursts [Level 2] → superbursts   [Level 3]
```

---

## 1. Binning

`compute_ml_bursts` lines ~287-304.

- **Adaptive mode** (default, `bin_size_mode="adaptive"`): `bin_size` =
  `median(log10 ISI)` over all units, converted to ms and clamped to **[10, 30] ms**
  (`_adaptive_bin_size_s`). Also yields `biological_isi_s`, the raw median ISI,
  which scales every downstream gap.
- **Fixed mode**: `bin_size = fixed_bin_size_s` (default 0.02 s); `biological_isi_s`
  is set equal to the bin size.

Derived quantities:
- `bins` = edges, `t_centers` = bin centers, `n_bins` = count.
- `sigma_fast = clip(isi_bins, 1, 2)`, `sigma_slow = clip(5·isi_bins, 3, 8)` —
  Gaussian smoothing widths for the population signals.
- `burstlet_merge_gap_s = 3.0 · biological_isi_s`
- `network_merge_gap_s = max(10.0 · biological_isi_s, network_merge_gap_min_s)`
  (`network_merge_gap_min_s` default 0.75 s).

A spike-count matrix `spike_matrix` (n_units × n_bins) is built once and reused.

---

## 2. Per-unit HMM

`fit_all_units` (`ml_burst_hmm.py`). For each unit, fit a **2-state Hidden Markov
Model** (background vs burst) on its binned spike counts via Baum-Welch EM.

- States have Poisson emission rates `λ_bg` and `λ_burst`.
- A fit is **rejected** if `λ_burst / λ_bg < hmm_min_rate_ratio` (default 1.5), or
  if the unit has fewer than `hmm_min_spikes` (default 50) spikes.
- Output per unit: `posteriors[u]` = P(burst state | counts) per bin, plus the
  fitted rates used later for the LLR features.

Population signals computed here and carried to the output events:
- `pfr = spike_counts_total / bin_size` (population firing rate)
- `participation_raw = active_unit_fraction`
- `ws_sharp = gaussian_filter1d(participation_raw, sigma_fast)` — sharp synchrony
- `ws_smooth = gaussian_filter1d(rate_per_unit, sigma_slow)` — smooth rate

At least one unit must fit, else `MLBurstError`.

---

## 3. Feature matrix (26-D)

`build_feature_matrix` (`ml_burst_features.py`). Produces `X` of shape
`(n_bins, 26)` and `feature_names`. Each row describes one time bin. Per-unit
quantities are aggregated across fit units with **mean / std / top-quantile**
(`unit_agg_quantile`, default 0.9), NaN-aware.

| Group | Count | Columns | Meaning |
|-------|-------|---------|---------|
| HMM posteriors | 5 | `post_frac_gt_0_5`, `post_mean`, `post_std`, `post_q90`, `post_entropy` | Aggregates of per-unit burst-state posterior. `post_frac_gt_0_5` = fraction of units with posterior > 0.5. |
| Population | 2 | `PFR`, `participation` | Population firing rate; active-unit fraction. |
| Multi-scale Fano | 4 | `FF0..FF3` | Variance/mean of counts at 4 temporal scales (`ff_scale_multipliers` × bin, clamped 5-100 ms). |
| Per-unit LLR | 3 | `llr_hmm_mean`, `llr_hmm_std`, `llr_hmm_q90` | Two-rate Poisson log-likelihood ratio (λ_burst vs λ_bg) from the HMM fit. |
| ISI shape | 4 | `inv_isi_mean`, `inv_isi_std`, `cv_isi_mean`, `lv_mean` | Inverse-ISI (firing speed) + Shinomoto LV in a sliding window. |
| ΔFR vs baseline | 2 | `dfr_unit_mean`, `dfr_unit_std` | `(count/Δt − λ_bg)/λ_bg` per unit. |
| Temporal derivatives | 6 | `dPFR_short/long`, `dParticipation_short/long`, `dLLR_short/long` | First differences at short (`deriv_sigma_short_bins`=1.5) and long (`deriv_sigma_long_bins`=8.0) scales. Positive `dPFR_short` = rising edge. |

The column **`cluster_ranking_feature`** (default `post_frac_gt_0_5`) becomes the
**ranking signal** that drives every "is this a burst" and "is this valley deep
enough" decision downstream.

---

## 4. Clustering and burst-cluster selection

`cluster_bins` (`ml_burst_cluster.py:164`).

### 4a. Z-normalization

- `bg_mask` = bins in the bottom `background_quantile` (default 0.5) of the
  ranking feature.
- `X_norm = (X − μ_bg) / σ_bg`, where μ, σ are computed over the background bins
  only. This centers the feature space on resting activity.

### 4b. Embedding (`cluster_embedding_mode`)

- `"none"`: cluster `X_norm` directly (optionally PCA-reduced).
- `"umap"` (used in current runs): embed `X_norm` with UMAP
  (`umap_n_neighbors`=30, `umap_min_dist`=0.0, `umap_n_components`=5). Rationale:
  in raw space the burst is a low-density filament HDBSCAN dumps into noise; UMAP
  collapses it into a dense region HDBSCAN can recover. Falls back to `X_norm` if
  `umap-learn` is unavailable.

### 4c. HDBSCAN

`hdbscan.HDBSCAN` with `min_cluster_size`=30, `min_samples`=5,
`cluster_selection_epsilon`=0.0, `cluster_selection_method`="eom",
`metric`="euclidean". Output: integer `labels` per bin (`-1` = noise).

> Note: if `import hdbscan` fails, the code silently falls back to a hard
> threshold on the ranking feature at `fallback_posterior_threshold` (0.3), and
> `diagnostics["cluster_decision"] == "fallback_threshold"` for every well. Check
> this flag before trusting cluster structure.

### 4d. Cluster ranking

Each cluster is ranked by the **mean of its ranking feature** (in raw, not
z-normed, space). Produces `cluster_rank = {cluster_id: mean_score}`, descending.

### 4e. Burst-cluster selection — `_select_burst_clusters`

Burst is **multi-cluster** (the burst trajectory fragments across the manifold):

```
top, top_score = highest-ranked cluster and its score
base = ranking_values[ranking_values <= median(all)]          # background half
med  = median(base);  mad = median(|base − med|)
thr  = med + burst_mad_scale · max(mad, 1e-6)                  # burst_mad_scale = 3.0
thr  = max(thr, med + 0.25·(top_score − med))                 # MAD-floor
burst_labels = {top} ∪ {c : cluster_rank[c] > thr}
```

The MAD-floor prevents over-selection when the background MAD collapses to ~0.
**This selection step is the sole determinant of burst-vs-rest at the bin level**
— and the main lever for over/under-detection.

### 4f. Burst mask — `burst_bin_mask`

```
mask_pre_merge = np.isin(labels, burst_labels)
```
Boolean per-bin mask. At this point only cluster membership matters; temporal
continuity has not been considered yet.

---

## 5. Temporal merge — `temporal_merge`

Turns the per-bin mask into time-extent candidates. Two steps.

### 5a. Morphological closing — `_binary_closing_1d(mask, closing_bins)`

1D closing (dilate then erode), `closing_bins` default 3. **Fills gaps shorter
than `closing_bins` bins (~90 ms)**; longer gaps survive. Each remaining run of
`True` becomes a candidate `{start, end, start_idx, end_idx}` via
`_mask_to_candidates`. Output mask: `closed_mask`.

### 5b. Valley merge — `_iter_merge`

First set the merge threshold from the background (bins outside `closed_mask`):

```
bg = ranking_signal[~closed_mask]
threshold = median(bg) + merge_mad_scale · max(MAD(bg), 1e-6)   # merge_mad_scale = 0.75
```

Then walk candidates left to right; merge `cur` and `nxt` when
`gap ≤ gap_s` **AND** the valley between them passes one of:

1. No bins in the valley (truly adjacent) → fire if `gap ≤ bin_size`.
2. `gap ≤ gap_tolerance_bins · bin_size` (default 0; disabled) → fire regardless.
3. Valley minimum `vm ≥ merge_floor_frac · threshold` (`merge_floor_frac`=0.70) —
   the relaxed "still in burst regime" criterion.

Here `gap_s = burstlet_merge_gap_s = 3 · biological_isi_s`. Returns the merged
candidate list, `closed_mask`, and the effective `threshold`.

---

## 6. Burstlets (Level 1) + soft gate

`compute_ml_bursts` lines ~392-452.

Each candidate is materialized into a burstlet record with these columns:

```
start, end, duration_s, peak_synchrony, peak_time, synchrony_energy,
participation, total_spikes, burst_peak, fragment_count,
llr_aggregate, posterior_peak, posterior_mean, ff_peak
```

- `peak_synchrony`/`peak_time` from `ws_sharp`; `burst_peak` from `pfr`.
- `posterior_peak`/`posterior_mean` from the ranking signal over the event.
- `llr_aggregate` = mean of `llr_hmm_mean` over the event.

**Soft modulation gate** (`min_burst_modulation`, default 0.1): drop any burstlet
whose `llr_aggregate < min_burst_modulation`. This is the one place real
population modulation (LLR) is required, but the threshold is low.
`burst_modulation_index` = max `llr_aggregate` across burstlets (reported in
diagnostics).

Result: `burstlets_raw` = **Level 1**.

---

## 7. Hierarchy merge — network_bursts (L2) and superbursts (L3)

Two further passes with increasing gap and stricter valley criteria. Both finalize
merged groups via `_ml_finalize_event`, which aggregates sub-events and recomputes
quality columns (`n_sub_events`, `fragment_count`, posterior/llr/ff peaks).

### 7a. network_bursts — `_merge_strict_local`

- Gap: `nxt.start − cur.end ≤ burstlet_merge_gap_s` (= 3 · isi).
- Valley: `vm ≥ threshold` (stricter than `_iter_merge`'s `floor_frac · threshold`).
- Adjacent burstlets passing both are merged into one network burst.

Result: **Level 2**. (When burstlets are already well separated, L2 == L1.)

### 7b. superbursts — `_merge_clustered_local`

- Gap: `≤ network_merge_gap_s` (= max(10 · isi, 0.75 s)).
- Valley: `baseline < vm < threshold` — a shallow dip between two network bursts.
- **Keep only merged events with `n_sub_events ≥ 2`** (must genuinely combine ≥2
  network bursts).

Result: **Level 3** (often empty when bursts are isolated).

---

## 8. Outputs

Written by `PickleBurstOutputWriter` to
`{output_root}/{recording_key}/{rec}/{well}/ml_burst_detection/`:

| File | Contents |
|------|----------|
| `burstlets.pkl`, `network_bursts.pkl`, `superbursts.pkl` | The three DataFrames (schema above). |
| `metrics.json` | Per-level `{n_events, total_duration, ...}`. |
| `diagnostics.json` | `adaptive_bin_ms`, `biological_isi_s`, `cluster_decision`, `cluster_embedding_mode`, `cluster_n_clusters`, `cluster_burst_label(s)`, `cluster_ranking`, `merge_threshold`, `burstlet/network_merge_gap_s`, `ranking_feature`, `burst_modulation_index`, `feature_names`, … |
| `plot_signals.npy` | `t`, `participation_signal` (ws_sharp), `rate_signal` (ws_smooth), `ranking_signal`, `ff_signal`, `llr_signal`, `posterior_matrix_mean`, burst peak times/values, `merge_threshold`. |
| `debug_trace.pkl` *(debug)* | `MLBurstTrace`: `feature_matrix`, `posterior_matrix`, `hdbscan_labels`, `hdbscan_probabilities`, `cluster_ranking`, `burst_labels`, `burst_mask_pre_merge`, `burst_mask_post_closing`, `candidates_pre_hierarchy`, scaler stats, etc. |

---

## 9. Key parameters

| Parameter | Default | Stage | Effect |
|-----------|---------|-------|--------|
| `bin_size_mode` / `fixed_bin_size_s` | adaptive / 0.02 | 1 | Time resolution. |
| `hmm_min_rate_ratio` | 1.5 | 2 | Reject weak HMM fits. |
| `cluster_ranking_feature` | `post_frac_gt_0_5` | 3-7 | Signal driving all burst/valley decisions. |
| `cluster_embedding_mode` | `umap` | 4b | Raw vs UMAP clustering space. |
| `hdbscan_min_cluster_size` / `min_samples` | 30 / 5 | 4c | Cluster granularity. |
| `burst_mad_scale` | 3.0 | 4e | Burst-cluster selection threshold (× MAD). |
| `closing_bins` | 3 | 5a | Gap filled inside burst mask (over-merge risk). |
| `merge_mad_scale` | 0.75 | 5b | Valley merge threshold (× MAD). |
| `merge_floor_frac` | 0.70 | 5b | Relaxed valley floor for burstlet merge. |
| `network_merge_gap_min_s` | 0.75 | 7b | Floor on superburst gap. |
| `min_burst_modulation` | 0.1 | 6 | Soft LLR gate on burstlets. |

---

## 10. Failure modes to watch

The selection threshold (§4e) and merge threshold (§5b) are both derived from the
**background MAD** of the ranking signal. In weakly-modulated wells the MAD
collapses toward zero, both thresholds sink near baseline, and two problems follow:

- **Over-counting** — marginal clusters (real partial recruitment, or HMM-posterior
  artifacts with near-zero population activity) clear the selection floor and add
  spurious bins; isolated bins survive as very short (~1-bin) "bursts".
- **Over-merging** — a low-activity cluster paints a near-continuous mask, closing
  (§5a) welds it into a long run, and the collapsed merge threshold never lets a
  real internal valley split it.

The ranking feature `post_frac_gt_0_5` is posterior-only — it can rank a cluster as
"burst" with no corroborating `PFR` / `participation` / `llr_hmm_mean`. Requiring
population corroboration at selection time is the natural guard.
