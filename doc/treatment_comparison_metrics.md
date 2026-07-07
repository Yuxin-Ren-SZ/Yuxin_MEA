# Treatment-comparison metrics

The 10 per-well readouts compared by `scripts/compare_treatment_groups.py`
(`METRIC_SPECS` in that file). Each is a single scalar **per well per recording**;
the script collapses a well's time series to one value (see *Response* below) and
compares treatment arms against Control.

Unit of analysis = **well**. All bursting metrics are read from `scan_type == Network`
recordings only.

---

## How the metrics become a "response"

Every well is normalised to **its own pre-treatment baseline** (paired mode):

- **`ratio` metrics** (rates, counts, durations — strictly positive):
  `response = log2( post_median / pre_median )`. 0 = no change; +1 = doubled; −1 = halved.
- **`diff` metrics** (indices / small integer counts):
  `response = post_median − pre_median`.

In `cross_sectional` mode the response is just the absolute post-treatment median
(no baseline needed — use it for arms with no pre-treatment recordings, e.g. NPH, AraC).

Cliff's delta then compares each arm's per-well responses against Control's.

---

## 1. Network bursting
Synchronised, network-wide burst events (the `network_bursts` tier of the detector).
Source: `burst_detection_data/<key>/<rec>/<well>/burst_detection/metrics.json → network_bursts.*`
(traditional detector; switch to the ML detector with `--burst-dirname ml_burst_data_umap`).
Computed in `analysis/burst_detector.py::_level_metrics`. Kind = `ratio`.

| metric | key | definition | ↑ means |
|---|---|---|---|
| `nb_rate` | `network_bursts.rate` | network bursts per second (`count / recording_duration`) | more frequent network bursting |
| `nb_count` | `network_bursts.count` | number of network bursts in the recording (length-dependent — `nb_rate` is the normalised form) | more bursts (or longer recording) |
| `nb_duration_mean` | `network_bursts.duration.mean` | mean duration of a network burst (seconds) | longer synchronised events |
| `nb_spikes_per_burst_mean` | `network_bursts.spikes_per_burst.mean` | mean `total_spikes` per burst (spikes summed over all units inside the burst) | larger / more intense bursts |
| `nb_ibi_mean` | `network_bursts.inter_event_interval.mean` | mean gap between consecutive burst onsets (seconds) — the inverse of rate | sparser bursting (longer quiet gaps) |

Note: `nb_rate`, `nb_count`, and `nb_ibi_mean` are related (rate ≈ 1/IBI; count = rate × duration),
so they tend to move together. `nb_duration_mean` and `nb_spikes_per_burst_mean` describe burst *shape/size*.

---

## 2. Firing / activity
Single-unit activity after automatic curation.
Source: `curation_data/<key>/<rec>/<well>/auto_curation/quality_metrics.pkl`
(aggregated like `analysis/curation_summary.py::aggregate_curation_summaries`). Kind = `ratio`.

| metric | definition | ↑ means |
|---|---|---|
| `median_firing_rate` | median firing rate (Hz) across **curated** units (units passing all QC thresholds) | higher overall spiking |
| `n_curated` | number of units that passed auto-curation | more active, well-isolated units (viability / yield proxy) |

`n_curated` counts only QC-passing units, so it drops if activity dies, units de-isolate,
or the well degrades — a coarse health readout alongside `median_firing_rate`.

---

## 3. Burst-type composition
Burst structure and population diversity from the ML detector.
Source: `ml_burst_data_umap/<key>/<rec>/<well>/ml_burst_detection/diagnostics.json`. Kind = `diff`.

| metric | key | definition | ↑ means |
|---|---|---|---|
| `burst_modulation_index` | `burst_modulation_index` | strength of burst modulation above Poisson background = max `llr_aggregate` over burstlets (`ml_burst_detector.py:406`) | more sharply bursty (vs tonic) firing |
| `burst_type_k` | `burst_typing.k` | number of distinct burst *types* found by the per-well clustering (kmeans/gmm) | more diverse burst shapes |
| `cluster_n_clusters` | `cluster_n_clusters` | number of functional **unit** clusters detected in the well (`assignment.n_clusters`) | more heterogeneous unit population |

> **Cross-well caveat:** burst typing is a **per-well** clustering, so the burst-type *identities*
> (type 0, type 1, …) are arbitrary and **not comparable across wells** — that is why the raw
> per-type *proportions* are deliberately **not** used. Only the cross-well-comparable scalars
> above (`burst_modulation_index`, the *count* `burst_type_k`, `cluster_n_clusters`) are compared.

---

## Choosing metrics for a run
- Default: all 10.
- Fewer metrics → lighter multiple-testing (BH-FDR) burden. With small per-arm n, restricting to
  1–2 pre-chosen primaries (e.g. `--metrics nb_rate median_firing_rate`) makes real effects far
  more likely to clear `q < 0.05`.
- `nb_rate` (network bursting) + `median_firing_rate` (single-unit activity) together give a
  compact, largely non-redundant summary of network excitability.

See `scripts/compare_treatment_groups.py` for the extraction (`METRIC_SPECS`) and the tutorial
in the script docstring for the full workflow.
