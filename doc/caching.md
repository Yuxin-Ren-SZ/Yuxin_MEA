# Caching & freshness model

How the system keeps the **dashboard** and the **pipeline run** consistent with what is on
disk, and — the question this doc exists to answer honestly — **what happens after the raw
data is modified**.

All line references are into `src/yuxin_mea/…` at the time of writing; treat them as anchors,
not guarantees.

---

## TL;DR

- **The dashboard is always current with respect to on-disk state.** Recording and pipeline
  *listings* are re-read from JSON on every callback (no memoization). The only cached thing is
  the expensive per-recording plate data (24 wells) and its raster PNGs, and those are
  **busted by a file-stat signature** — when the pipeline rewrites an output, the next Load
  re-reads it automatically.
- **Nothing automatically detects an in-place change to the raw data or the config file.**
  There is no file-watcher, no mtime/size/hash comparison in the dataset scan, and the
  pipeline's config-snapshot check is present but never called. Overwriting `data.raw.h5` in
  place, or editing a task's params, leaves every cache and every task status untouched.
- **Freshness after an upstream change is achieved by three manual steps**, only the last of
  which is automatic:
  1. **Scan disk** — rebuild `experiment_cache.json` (dataset layer).
  2. **Refresh / re-queue the affected task(s)** — reset pipeline status so they re-run.
  3. Tasks re-run and rewrite their outputs → **the dashboard auto-busts by signature** on the
     next Load.

If you only remember one thing: *"Scan disk" and a task "refresh" are not optional after you
touch raw data or config — the system will otherwise keep serving the old result forever.*

---

## The three persisted stores

| Store | Path | Written by | Read by |
|---|---|---|---|
| **Dataset cache** | `<analysis_root>/experiment_cache.json` | `DatasetManager` → `JsonCacheStore.save` (`dataset/cache.py`) | dashboard listings, CLI, plate-data signature |
| **Pipeline cache** | `<analysis_root>/pipeline_cache.json` | `PipelineManager` → `JsonPipelineCacheStore.save` (`pipeline/cache.py`) | dashboard status grid, `yuxin-mea-run` |
| **Dashboard scratch** | `<analysis_root>/../dashboard_cache/` | dashboard render callbacks | dashboard only |

The dashboard scratch dir (local SSD, never the NAS — `dashboard/app.py:_resolve_cache_root`)
holds the flask `FileSystemCache` (`flask_cache/`), the raster PNGs (`raster_png/`), and — if
generated — the viewer bundles.

### Which dashboard reads are cached vs. always-fresh

| Dashboard read | Cached? | Source |
|---|---|---|
| Recording list / metadata (`load_recordings_detail`, `load_recordings_df`) | **No — fresh every callback** | `experiment_cache.json` |
| Pipeline status grid (`load_pipeline_df`) | **No — fresh every callback** | `pipeline_cache.json` |
| Well→group map (`well_group_map`) | **No — fresh every callback** | `experiment_cache.json` |
| Plate viewer 24-well payload (`load_plate_data_cached`) | **Yes — signature-keyed** | task outputs |
| Plate viewer raster PNGs (`render_overview_pngs`) | **Yes — signature-keyed** | task outputs |

The always-fresh loaders live in `dashboard/data.py:36-282`. By design they build a **fresh**
`JsonCacheStore`/`JsonPipelineCacheStore` and re-read the file on each call rather than going
through `DatasetManager`/`PipelineManager` — the managers mutate state on construction, which
would violate the dashboard's read-only contract (`dashboard/data.py:1-8`). Consequence: a new
recording or a changed task status appears on the next page render immediately, with no cache
to invalidate.

---

## Dashboard freshness

### Listings — always live

Covered above: `load_recordings_detail` / `load_pipeline_df` / `load_recordings_df` /
`well_group_map` re-read the JSON stores every callback (`dashboard/data.py:36-282`). No TTL, no
memoize, nothing to bust.

### Derived plate data — bust by signature

The heavy path (parsing 24 wells of spike/burst arrays) is cached across three tiers and keyed
by a **staleness signature** — a `stat`-only tuple over the exact files it reads, never opening
them:

```python
# dashboard/data_cache.py:35-50
def _stat(path):
    try: st = path.stat()
    except OSError: return (-1, -1)          # missing file → sentinel, still busts the key
    return (int(st.st_mtime_ns), int(st.st_size))

def data_sig(paths):
    return tuple((str(p), *_stat(Path(p))) for p in paths)
```

`_manifest_sig` (`data_cache.py:53-75`) builds that signature over, per well,
`plot_signals.npy` and `curated_spike_times.npy`, plus `experiment_cache.json`. (Event `.pkl`
tables are deliberately *not* stat'd — they are rewritten in the same task run as
`plot_signals.npy`, so that file's mtime already covers them.)

Lookup order in `load_plate_data_cached` (`data_cache.py:87-161`), key
`(recording_key, source, burst_terminal, sig)`:

1. **L1** — in-process LRU, 12 entries (`_CACHE`, `data_cache.py:31-32`).
2. **L2** — flask `FileSystemCache` on scratch, `CACHE_DEFAULT_TIMEOUT: 0` → **never expires**
   (`dashboard/cache.py:31-54`). Same signature is folded into its sha1 key (`make_key`,
   `cache.py:61-64`).
3. **Tier 3** — prebuilt viewer bundle (`analysis/viewer_bundle.py`), accepted only if its
   embedded signature matches the live one (`read_viewer_bundle(..., expected_signature)`,
   `viewer_bundle.py:67-89`); otherwise treated as a miss.
4. **Cold** — `load_plate_data` reads the NAS, then back-fills L1 + L2.

Raster PNGs use the same idea per well: the filename token is a sha1 over
`(recording_key, source, well_id, per-well data_sig)` (`analysis/raster_image.py:99-137`), so a
re-run of one well re-renders only that PNG.

**How a pipeline re-run refreshes the dashboard:** the task rewrites `plot_signals.npy` /
`curated_spike_times.npy` → `st_mtime_ns`/`st_size` change → `data_sig` yields a new tuple → the
key changes at **all** tiers → the old entries are simply never looked up again (they age out of
the LRU; the L2 entry is orphaned but harmless). This is *bust by signature, never by timer*
(`data_cache.py:6-8`, `cache.py:47`). No manual dashboard action is needed once the output
files change.

---

## Pipeline freshness

A per-(recording, well) task is stored as a `TaskRecord` (`pipeline/task_record.py:23-30`):
`status`, `dependencies`, `output_path`, `last_updated`, `error`, and a `config` **snapshot**
frozen when the task transitions to RUNNING (`pipeline/manager.py:195-198`). There is **no**
input signature, no raw-file hash, no output hash.

### What makes a task run

Eligibility is **status-driven only** (`pipeline/manager.py:115-165`):

```python
# manager.py:135
eligible = {TaskStatus.NOT_RUN, TaskStatus.FAILED} if retry_failed else {TaskStatus.NOT_RUN}
```

plus every immediate dependency COMPLETE (`_deps_complete`, `manager.py:377-382`). Config, raw
inputs, and upstream *output freshness* are never inspected here. A task whose status is
COMPLETE is skipped, full stop.

### What makes a completed task re-run — all manual/explicit

- **`refresh(task)`** (`manager.py:255-304`) — resets that task **and all transitive
  dependents** to `NOT_RUN`, walking `_reverse_deps` via `_cascade_tasks`
  (`manager.py:392-410`). This is the cascade. Triggered from the dashboard: the Pipeline page
  and the Recordings "refresh()" button (`dashboard/pages/pipeline.py:802`,
  `dashboard/pages/recordings.py:607`). `bulk_refresh` (`manager.py:306`) does it across wells.
- **`recover_from_crash()`** (`manager.py:337-366`) — at CLI startup (`cli/run.py`), resets
  every **non-COMPLETE** task to `NOT_RUN` (clears stuck RUNNING). Does not touch COMPLETE.
- **`--retry-failed`** — includes FAILED tasks in eligibility for one run.
- **A brand-new `recording_key`** queued via `add_well` — a genuinely new run directory.

### The config-snapshot check exists but is not wired in

`is_task_complete` compares the frozen snapshot to the current config:

```python
# manager.py:210-225
current = self._config_provider.get_config(task_name, recording_key, well_id)
return record.config == current      # False if status≠COMPLETE OR config changed
```

**However, `is_task_complete` has no callers** — the runner (`get_next_task` /
`is_all_complete`) keys only off `status == COMPLETE`. So **editing a task's params in
`pipeline_config*.json` does not, by itself, mark anything stale.** See the caveat at the end;
several notebooks claim otherwise.

---

## Dataset scan freshness

The dataset cache maps `cache_key = Sample/Date/Plate/ScanType/RunID` → `RecordingEntry`
(`dataset/entries.py`). `RecordingEntry` records `mtime` and `file_size` of `data.raw.h5` — but
**those fields are write-only; no code ever compares them.** Keying is purely by the
path-derived `cache_key`.

### Startup (`_initialise`) — incremental, new-date-only

On every `DatasetManager` construction (`dataset/manager.py:284-319`), the scan diffs only
`(sample_id, date)` **directory sets**:

```python
# manager.py:297-303
missing  = cached_date_keys - disk_date_keys   # → warning only; entries kept
new_keys = disk_date_keys  - cached_date_keys   # → deep-scanned and added
```

So startup picks up a **brand-new Date directory** automatically, but:
- a modified/overwritten `data.raw.h5` under an already-cached date is **invisible**;
- a new run added inside an already-cached date is **invisible**;
- a deleted recording is only **warned about, not removed**.

### Full reconciliation (`refresh()`) — blind rebuild

`refresh()` (`manager.py:272-278`) does `clear()` → `_scan_all()` → `save`: it re-stats and
re-reads every recording and rebuilds all entries from scratch. This is the **only** way an
in-place file change, an added run, or a removed recording is reconciled — and it works by
rebuilding, not by detecting the change (the new mtime/size/metadata land as a side effect).
It is triggered by the **Scan disk** button on the Recordings page
(`dashboard/pages/recordings.py:121-127, 603-609`).

`refresh_groupnames()` (`manager.py:210-243`) is a narrower operation: it re-reads only the
`mxassay.metadata` sidecar per cached recording and updates `well.metadata["groupname"]` in
place, without walking `data.raw.h5`. Triggered by the "Refresh groups" button.

---

## Scenario → propagation matrix

What each layer does after a given change. **auto** = happens with no user action;
**manual** = requires the named action; **N/A** = not applicable.

| Change on disk | Dataset cache (`experiment_cache.json`) | Pipeline status (`pipeline_cache.json`) | Dashboard |
|---|---|---|---|
| **Raw `data.raw.h5` overwritten in place** (same path/key) | **manual** — *Scan disk* (`refresh()`); startup diff won't notice (`manager.py:297-303`). *Detectable:* `check_cache.py --verify-provenance` → `RAW-CHANGED` | **manual** — *refresh(task)* / re-queue; status stays COMPLETE (`manager.py:135`) | listings **auto** after Scan disk; plate data **auto** only after tasks re-run & rewrite outputs |
| **New run added inside an existing Date dir** | **manual** — *Scan disk*; startup diffs only Date dirs (`manager.py:297-303`) | **manual** — queue the new well(s) (`add_well`) | **auto** once the well is in the caches |
| **Brand-new Date directory** | **auto** — next `DatasetManager` build deep-scans it (`manager.py:303-314`) | **manual** — queue its wells | **auto** (listings are always fresh) |
| **A task/well re-run rewrites its output** | N/A | **auto** — the runner wrote the new status | plate data **auto** — signature busts on next Load (`data_cache.py:53-75`) |
| **Config param edited** (Settings Save or file edit) | N/A | **manual** — *refresh(task)*; `is_task_complete` is never called (`manager.py:210-225`). *Detectable:* `--verify-provenance` → `CONFIG-CHANGED` | N/A (config isn't cached) |
| **`mxassay.metadata` overwritten** (groupname/labels) | **manual** — *Refresh groups* (`refresh_groupnames`); content-hashed every scan | N/A — no task reads metadata (labels only) | listings **auto** after refresh. *Detectable:* `--verify-provenance` → `METADATA-CHANGED` |

---

## End-to-end: "I modified the raw data — how do I make everything current?"

Only the last step is automatic. The chain:

1. **Scan disk** (Recordings page) → `DatasetManager.refresh()` clears + rebuilds
   `experiment_cache.json` with the new file's mtime/size/structure.
2. **Refresh or re-queue the affected task(s):**
   - *Refresh* (Pipeline/Recordings button) → `PipelineManager.refresh(task)` resets that task
     **and all downstream dependents** to `NOT_RUN` (`manager.py:255-304`), or
   - re-queue the well(s) so they are picked up on the next `yuxin-mea-run`.
3. **Run the pipeline** → eligible tasks re-run and rewrite their outputs
   (`plot_signals.npy`, `curated_spike_times.npy`, …).
4. **Dashboard auto-refreshes** → on the next plate-viewer Load, `_manifest_sig` re-stats the
   changed outputs, the signature changes, and every cache tier misses and cold-reloads
   (`data_cache.py:87-161`).

**Interlock nuance — don't stop at step 1.** Because `experiment_cache.json` is itself part of
`_manifest_sig` (`data_cache.py:53-75`), *Scan disk alone busts the dashboard plate cache*.
But that cold reload re-reads the **same** task outputs — so without step 2–3 (an actual task
re-run) the dashboard just re-parses identical data and shows the same result. Scan disk makes
the *listing* current; only a task re-run makes the *derived data* current.

---

## Caveat: config-snapshot invalidation is documented but not wired in

`TaskRecord` freezes a config snapshot at RUNNING, and `is_task_complete`
(`pipeline/manager.py:210-225`) would return `False` when a task's current config differs from
that snapshot. **But `is_task_complete` has no callers** (only its definition and a docstring in
`pipeline/config_provider.py`); the runner's eligibility (`get_next_task`, `is_all_complete`)
keys only off `status == COMPLETE`.

Several notebooks state the opposite — e.g. *"the config-snapshot check will detect the change
and mark all wells stale"* (`notebooks/v2/03_auto_merge.ipynb`, `…/04_analyzer.ipynb`,
`…/05_auto_curation.ipynb`, `…/01_si_preprocessing.ipynb`) and *"a config change … invalidates
that task and its downstream dependents"* (`notebooks/v2/00_full_pipeline.ipynb`). **These
claims are not borne out by the code.** In practice a config edit requires a manual
`refresh(task)` (which cascades to dependents) to take effect. This is a documentation-vs-code
discrepancy to be aware of, flagged here — not resolved.

---

## Provenance & reproducibility (fingerprints + stamps + verify)

The caching layers above answer "is what I'm looking at current?". Provenance answers the
stronger question: **"was this pipeline output produced from the same raw data + config that
are on disk now?"** — so a result can always be trusted or flagged as stale. Implemented in
`src/yuxin_mea/provenance/`.

### What is fingerprinted

- **`data.raw.h5`** — a **structure-aware, size-partitioned** sha256 (`fingerprint.py:h5_fingerprint`).
  A MaxWell recording is 30–100 GB, almost all of it the raw voltage array
  `recordings/<rec>/<well>/groups/routed/raw` `(n_ch, n_frames) uint16`. Hashing the whole file
  over the NAS is prohibitive, so datasets are split by size (default 16 MB):
  - **small datasets are hashed in full** — the analysis-critical part
    (`settings/{gain,lsb,hpf,sampling,spike_threshold,mapping}`, `channels`, `spikes`, `events`,
    top-level `version`/… and every attr), so a gain/mapping/sampling change is caught exactly;
  - **large datasets are sampled** — fixed windows at deterministic *fractional* offsets, with
    shape/dtype folded in (so a change in recording length or electrode count changes the hash).
  The `method` id (`h5struct-v1`) is stored so the method is itself versioned. `--full-hash`
  streams the large datasets in full (`h5full-v1`). Everything streams in fixed blocks — constant
  memory regardless of file size.
- **`mxassay.metadata`** — always a full sha256 (`fingerprint.py:file_hash`); tiny and frequently
  overwritten.
- **config** — sha256 of the resolved task params (`fingerprint.py:params_hash`), i.e. exactly
  what the task ran with (matches `TaskRecord.config`).

### Where the record lives (durable + fast)

- **Dataset cache** — `RecordingEntry.raw_fingerprint = {"h5": …, "metadata": …}` in
  `experiment_cache.json`, computed at scan time (`dataset/manager.py:_populate_fingerprint`).
  This is the *current* raw state as last scanned.
- **Pipeline result** — at task completion the runner stamps `TaskRecord.provenance`
  (`{"h5", "metadata", "config_hash", "config_file", "stamped_at"}`) **and** writes a durable
  **`yuxin_provenance.json`** sidecar in the task's output dir (`provenance/sidecar.py`). The
  sidecar travels with the artifact and survives a pipeline-cache reset/rebuild/crash-recovery —
  the cache copy is just the fast-access mirror. Both are written parent-side / worker-side around
  `pipeline/manager.py:update_status` and `cli/run.py`. (Filename is *not* `provenance.json` —
  spikeinterface already writes that in sorter/extractor dirs; a `_schema` marker guards against
  any foreign file being misread as a stamp.)

### Cost — why fingerprinting is tiered, not automatic

Hashing a 30–100 GB h5 over the NAS is **I/O-latency bound**, not bandwidth bound, and stalls
under contention: on a real 53 GB file the *sampled* fingerprint did not finish in ~16 min at
loadavg ≈ 29 (process stuck in `D` state). Note the sampled tier reads only ~0.5 GB — the cost is
the tree walk (~400–500 metadata round-trips) plus ~1,150 scattered gzip-chunk seeks, not volume.

So raw h5 hashing is **tiered and entirely opt-in**:

| Mode | What it hashes | Cost |
|---|---|---|
| **`stat`** (default) | nothing — size/mtime only (already on the entry) | free |
| **`struct`** | small analysis-critical datasets (gain/lsb/mapping/sampling/channels/spikes) + attrs + raw **shapes**; **no bulk reads** of the raw array | walk + ~tens of MB |
| `content` | + **samples** the raw array (32 windows/well) | + ~0.5 GB, ~1,150 seeks |
| `full` | + reads the raw array entirely | + 30–100 GB |

`struct` is the pragmatic middle: it still catches any settings/gain/mapping change and a
re-acquisition (shape change), and — because file size is folded in — usually a raw rewrite too;
it only misses a raw edit that preserves the exact file size. `mxassay.metadata` is always
content-hashed (tiny) in every mode.

Hashes are **reused when the h5 stat is unchanged** (stat-drift trigger), so any tier is computed
at most once per file. Run a content/full pass off-peak.

Entry points: `DatasetManager(fingerprint_mode=…)` / `.compute_content_fingerprints(mode=…)`,
`yuxin-mea-run --hash-raw {stat,struct,content,full}`,
`check_cache.py --verify-provenance --hash-raw {struct,content,full}`.
- At run time the stamp copies whatever fingerprint the entry has (content if a pass ran, else
  stat) and additionally re-`stat`s the file actually read; a drift from the scanned stat is
  flagged on the stamp (`drift_from_scan`) and logged (warn, non-blocking).

### Detecting drift — `check_cache.py --verify-provenance`

Compares every COMPLETE task's stamp against the current raw fingerprint + config
(`provenance/verify.py:verify_provenance`). Per task it reports:

| Status | Meaning | Remedy |
|---|---|---|
| `OK` | stamp matches current raw + config | — |
| `RAW-CHANGED` | the h5 differs from what produced the output | re-run analysis |
| `CONFIG-CHANGED` | the task's params changed since it ran | re-run analysis |
| `METADATA-CHANGED` | `mxassay.metadata` changed | labels only — `refresh_groupnames`, no re-run |
| `UNKNOWN` | no stamp (output predates provenance) | unverifiable, **not** a mismatch |

Exit `1` on any *computational* drift (RAW/CONFIG); `--hash-raw {struct,content,full}`
re-fingerprints from disk at that tier instead of trusting the cached fingerprint (omit = free);
`--mark-stale` resets the RAW/CONFIG-drifted tasks
(and their dependents, via `PipelineManager.refresh`) to `NOT_RUN` so the next run recomputes them
(metadata drift is left alone — labels only). Read-only unless `--mark-stale`. `params_hash` here
is the first real caller of the previously-dead `is_task_complete` comparison.

**Dashboard** surfaces the same status as a badge (`dashboard/components/badges.py`) on the
Recordings detail card and the plate-viewer modal, via `dashboard.data.recording_provenance`
(cache-only — no NAS re-hash, no NAS sidecar reads, so it's cheap enough to compute on render).

### Run flags (`yuxin-mea-run`)

- `--rescan` — force a full dataset rescan (fresh fingerprints) before draining. Off by default
  (NAS-costly).
- `--hash-raw {stat,struct,content,full}` — how much of the raw h5 to fingerprint before draining,
  so stamps carry more than size/mtime (see the tier table above). Default `stat` (no hashing);
  `struct` is the cheap useful tier. Reused when the h5 stat is unchanged. (`--full-hash` is a
  deprecated alias for `--hash-raw full`.)

---

## Possible follow-ups (not implemented; listed for the record)

- Wire `is_task_complete` (config-snapshot compare) into `get_next_task` eligibility so a config
  edit auto-invalidates the affected task (and, via `_cascade_tasks`, its dependents) — making
  the notebook claims true. (Provenance now *detects* this; wiring it would *act* on it.)
- Compare the stored `mtime`/`file_size` on `RecordingEntry` during the startup scan to detect
  in-place raw-file changes instead of only diffing Date directories.
- Have `_initialise` optionally reconcile deletions (drop entries whose raw file is gone) rather
  than only warning.
