# MEA Pipeline — Architecture

## Overview

The MEA analysis pipeline has two independent modules:

```
┌─────────────────────┐         ┌─────────────────────┐
│  DatasetManager     │         │   PipelineManager   │
│  ─────────────────  │         │   ────────────────  │
│  Discovers raw      │         │  Tracks per-well    │
│  recordings on the  │         │  analysis progress  │
│  NAS, parses        │         │  across stages      │
│  metadata, caches   │         │  (preprocessing →   │
│  recording + well   │         │  sorting → … +      │
│  info to JSON.      │         │  custom analyses).  │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           ▼                               ▼
   experiment_cache.json          pipeline_cache.json
       (analysis_dir)                (analysis_dir)
```

The two managers do **not** import each other. They are linked by the caller:
`RecordingEntry.cache_key + well_id == PipelineEntry.pipeline_key`.

Both caches live in `analysis_dir`. Both modules are stdlib-only.

---

## Module 1 — DatasetManager

Discovers MEA recordings under a `data_root` directory, parses the canonical path
hierarchy, reads `mxassay.metadata` for recording- and well-level info, and
caches everything as JSON.

### Path hierarchy

```
<SampleID>/<Date(6d)>/<PlateID>/<ScanType>/<RunID>/data.raw.h5
                                                  /mxassay.metadata
```

`data_root` may be supplied at root level (contains SampleID dirs) or sample
level (IS a single SampleID dir). Detection is automatic from the first-layer
children — anything matching `^\d{6}$` means sample level.

### Components

| File | Purpose |
|---|---|
| `recording_entry.py` | `RecordingEntry` (frozen dataclass, recording-level) + `WellEntry` (mutable, well-level) |
| `metadata_extractor.py` | `BaseMetadataExtractor` + `MxassayMetadataExtractor` + `DummyMetadataExtractor` |
| `_mxassay_decoder.py` | Private. Decodes Qt-INI mxassay.metadata files to dicts |
| `cache_store.py` | `BaseCacheStore` + `JsonCacheStore` (atomic write) |
| `manager.py` | `DatasetManager` (scan, diff, cache, query) |

### Data model

```python
@dataclass(frozen=True)
class RecordingEntry:
    sample_id, date, plate_id, scan_type, run_id: str
    data_path:     Path     # absolute path to data.raw.h5
    file_size:     int
    mtime:         float
    discovered_at: float
    metadata:      dict[str, Any]            # recording-level fields from mxassay.metadata
    wells:         dict[str, WellEntry]      # populated lazily from mxassay.metadata

    cache_key: str  # "{sample_id}/{date}/{plate_id}/{scan_type}/{run_id}"

@dataclass
class WellEntry:
    well_id:  str               # e.g., "well000"
    metadata: dict[str, Any]    # per-well fields (groupname, density, annotations, …)
```

`metadata` and `wells` are **dynamic dicts** — keys vary by MaxWell software version
and by user-defined annotations. No field names are hardcoded.

### `MetadataExtractor` API

```python
@dataclass
class RecordingMetadata:
    fields: dict[str, Any]      # merged [properties] + [runtime] sections
    wells:  list[WellMetadata]  # selected wells only

@dataclass
class WellMetadata:
    well_id: str
    fields:  dict[str, Any]     # MaxWell slot fields + user annotations, flattened

class BaseMetadataExtractor(ABC):
    def get(self, metadata_path: Path) -> RecordingMetadata: ...

class MxassayMetadataExtractor(BaseMetadataExtractor):
    """Real implementation. Returns empty RecordingMetadata for missing files."""

class DummyMetadataExtractor(BaseMetadataExtractor):
    """Offline-dev placeholder. Ignores the path; returns 6 dummy wells."""
```

### `DatasetManager` API

```python
DatasetManager(
    data_root:          Path,
    analysis_dir:       Path,
    max_workers:        int | None = None,
    cache_store:        BaseCacheStore | None = None,
    metadata_extractor: BaseMetadataExtractor | None = None,
)

mgr.recordings                    # list[RecordingEntry]
mgr.get_by(key, method, value)    # filter by field; method: ==, !=, <, <=, >, >=, contain, not contain
mgr.get_wells(recording_key)      # dict[str, WellEntry]
mgr.register_well(rec_key, well_id, metadata)   # add/merge a well manually
mgr.update_well_metadata(rec_key, well_id, metadata)
mgr.refresh()                     # clear cache, full rescan, rewrite cache
```

### Behaviour

**Startup**: Load cache; diff Date-level dirs only (cheap on slow NAS). New Date
keys → scan in parallel via `ThreadPoolExecutor`, populate metadata via the
extractor, merge into cache. Missing Date dirs → log a warning, keep the entry.

**`refresh()`**: Clear cache, re-scan everything, re-extract all metadata,
overwrite the cache file.

**Cache file**: `analysis_dir/experiment_cache.json`, atomic write (tempfile +
`os.replace`). One top-level dict keyed by `cache_key`; `wells` nested under
each entry.

---

## Module 2 — PipelineManager

Tracks per-well analysis progress across an open set of stages. Built-in chain
is `preprocessing → sorting → (curation) → analyzer`; any custom stage name is
accepted and gets its dependency list at registration time.

### Components

| File | Purpose |
|---|---|
| `stage_record.py` | `StageStatus`, `StageRecord` dataclass, `BUILTIN_DEPS`, stage name constants |
| `pipeline_entry.py` | `PipelineEntry` dataclass (one per (recording, well)) |
| `config_provider.py` | `BaseConfigProvider` + `DummyConfigProvider` |
| `well_metadata.py` | `BaseWellMetadataProvider` + `DummyWellMetadataProvider` |
| `cache_store.py` | `BasePipelineCacheStore` + `JsonPipelineCacheStore` |
| `manager.py` | `PipelineManager` |

### Data model

```python
@dataclass
class PipelineEntry:
    recording_key: str                     # = RecordingEntry.cache_key
    well_id:       str                     # e.g., "well000"
    created_at:    float
    stages:        dict[str, StageRecord]  # open dict — custom stages allowed

    pipeline_key: str  # "{recording_key}/{well_id}"

@dataclass
class StageRecord:
    status:       str            # NOT_RUN | RUNNING | COMPLETE | FAILED
    dependencies: list[str]      # immediate upstream stage names
    output_path:  Path | None
    last_updated: float | None
    config:       dict[str, Any] # snapshot frozen at mark_stage_running() time
    error:        str | None
```

### `PipelineManager` API

```python
PipelineManager(
    analysis_dir:           Path,
    config_provider:        BaseConfigProvider | None = None,
    well_metadata_provider: BaseWellMetadataProvider | None = None,
    cache_store:            BasePipelineCacheStore | None = None,
)

# Stage lifecycle
mgr.mark_stage_running(rkey, well, stage, depends_on=None)  # snapshots config
mgr.mark_stage_complete(rkey, well, stage, output_path)
mgr.mark_stage_failed(rkey, well, stage, error)
mgr.reset_stage(rkey, well, stage)

# Queries
mgr.is_stage_complete(rkey, well, stage)   # status==COMPLETE AND cached config == current config
mgr.is_stage_ready(rkey, well, stage)      # all immediate deps pass is_stage_complete()
mgr.get_stage(rkey, well, stage)
mgr.get_entry(rkey, well)
mgr.get_or_create_entry(rkey, well)
mgr.get_entries_for_recording(rkey)
mgr.get_wells(rkey)
mgr.get_well_metadata(rkey, well)          # delegates to well_metadata_provider
mgr.entries                                # list[PipelineEntry]
mgr.refresh()                              # reload cache; warn on missing output_paths
```

### Key design points

**Dependency transitivity is free.** Each stage stores only its *immediate*
deps. `is_stage_ready()` calls `is_stage_complete()` on each dep, and that
function recurses into the dep's own config check. A stale ancestor → all
descendants return `is_stage_ready() == False` automatically. No graph code.

**Config staleness is the canonical "should we skip?" test.**
`is_stage_complete()` returns True iff status is `COMPLETE` **and** the cached
config (frozen at `mark_stage_running()` time) equals the current config from
the injected `ConfigProvider`. Changing a stage's config invalidates that
stage and everything downstream of it.

**Custom stages are plain strings.** Pass any name to `mark_stage_running()`
with `depends_on=[...]`. Built-in stages just have entries in `BUILTIN_DEPS`
that get used when `depends_on` is `None`.

**Injectable providers.** `ConfigProvider` and `WellMetadataProvider` are
both swappable; the manager has no opinion on their backing implementation.
Defaults are `DummyConfigProvider` (returns `{}`) and
`DummyWellMetadataProvider` (returns `{}`).

### Cache file

`analysis_dir/pipeline_cache.json`, atomic write. Keyed by `pipeline_key`;
stages nested under each entry. Same structure as the experiment cache.

---

## Linking the two managers

```python
em = DatasetManager(data_root, analysis_dir,
                       metadata_extractor=MxassayMetadataExtractor())
pm = PipelineManager(analysis_dir)

for rec in em.recordings:
    for well_id in rec.wells:
        if pm.is_stage_ready(rec.cache_key, well_id, STAGE_PREPROCESSING):
            pm.mark_stage_running(rec.cache_key, well_id, STAGE_PREPROCESSING)
            run_preprocessing(rec.data_path, well_id, ...)
            pm.mark_stage_complete(rec.cache_key, well_id, STAGE_PREPROCESSING, out_path)
```

The caller is the only place that knows about both managers.

---

## Constraints

- Stdlib only — no third-party deps in either module.
- `RecordingEntry` is `frozen=True`; mutable dict fields (`metadata`, `wells`)
  are populated in-place but cannot be reassigned.
- All cache writes are atomic (`tempfile.mkstemp` + `os.replace`) to survive
  partial writes on the NAS.
- No backward compatibility for old cache formats: a missing required key
  in a cached entry causes that entry to be dropped silently on load.
