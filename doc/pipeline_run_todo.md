# Pipeline Run — Findings & TODO

_Run: CX118 / CX138 / CX169, Network scans only. Output:
`/mnt/Vol20tb2/SadeghLab/yuxin/analysis`. Config: `pipeline_config_run.json`.
Finished 2026-07-01._

## Final state (1596 wells)

| task | complete | failed | not_run |
|---|---|---|---|
| preprocessing | 1583 | — | 13 |
| sorting | 1568 | — | 28 |
| analyzer | 1568 | — | 28 |
| auto_curation | 1568 | — | 28 |
| auto_merge | 1568 | — | 28 |
| burst_detection | 1534 | 34 | 28 |
| ml_burst_detection | 1532 | 36 | 28 |

## Failure diagnosis

### 1. "spike_times contains no units" — 34 burst + 34 ml_burst (same wells)

**Not dead wells.** Sorting found real units; `auto_curation` removed **all** of
them → curated_spike_times = 0 units → downstream burst/ml_burst error out.

Example `CX118/260215/T003346/Network/000033/rec0001/well007`: 34 units, 2421
spikes at sorting; **0 survive curation.** AND-gate breakdown:

| gate (config) | units passing (of 34) |
|---|---|
| presence_ratio ≥ 0.75 | 13 |
| rp_contamination ≤ 0.15 | 21 |
| firing_rate ≥ 0.05 | 21 |
| **amplitude_median ≤ -20** | **5**  ← dominant killer |
| ALL four (AND) | **0** |

`amplitude_median` for this well: min -34.6, median **-12.6 µV** — most units sit
between -20 and 0, so the -20 amplitude bar + 0.75 presence bar intersect to zero.
CX118 is a lower-amplitude prep; thresholds tuned on other data wipe it out.

### 2. "no HMM fits succeeded — sparse recording" — 2 ml_burst wells

Genuinely sparse (too few spikes for HMM). Distinct from the curation issue.

### 3. not_run — 28 wells

Unsorted (13 preproc never ran + 15 sorting fails, never `--retry-failed`) →
whole downstream chain stayed not_run. Independent of curation.

## Decision (2026-07-01)

**Leave the 34 curation-zeroed wells excluded** for now. Accepted as low-quality.
No curation change this run.

## TODO (future)

- [ ] Revisit `auto_curation` thresholds for low-amplitude preps (CX118 esp.).
      Candidate loosening: `amplitude_median_max -20 → -10`,
      `presence_ratio_min 0.75 → 0.5`. Config: `pipeline_config_run.json`
      `tasks.auto_curation`. Would recover most of the 34 wells.
      Re-run `auto_curation → burst_detection → ml_burst_detection` for affected
      wells afterward (curation is deterministic; plain `--retry-failed` re-fails).
- [ ] Consider making the 4 curation gates configurable per-sample, or relaxing
      the amplitude gate globally — it removed the most units (only 5/34 passed).
- [ ] For the 2 sparse "no HMM fits" wells: lower `hmm_min_spikes` /
      `hmm_min_rate_ratio`, or accept as excluded.
- [ ] Recover the 28 not_run wells: staged retry with `--retry-failed`
      (preprocessing -j8 → sorting -j1 → downstream -j4). Some sorting fails may
      be hard (dead wells) and re-fail.
- [ ] Verify whether well007 is representative of all 34 (spot-checked 1;
      mechanism is config-global so likely, but not exhaustively confirmed).
