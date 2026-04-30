# Agent Notes

Concise codebase map for future coding agents. Production code only.

## Repo Map

- `dataset_manager/` raw MEA discovery, metadata parse, recording/well cache.
- `pipeline_manager/` per-well task registry, queue, status cache.
- `config_manager/` JSON config loader/provider.
- `pipeline_tasks/` concrete analysis tasks.
- `pipeline_tasks/analysis/` burst detector algorithm and output writer.
- `config/` example/default pipeline config JSON.
- `notebooks/` manual pipeline workflows.
- `tests/` unit/integration tests for managers and tasks.
- `scripts/` helper scripts.
- `doc/architecture.md` older architecture note; partly stale naming.

## Main Flow

raw data scan -> recording/well cache -> per-well task queue -> preprocessing -> sorting -> auto_merge -> analyzer -> auto_curation -> burst_detection

## Relations And Restrictions

- `DatasetManager` and `PipelineManager` independent; no imports between managers.
- Link key: `RecordingEntry.cache_key + "/" + compound well_id`.
- `RecordingEntry.cache_key`: `sample_id/date/plate_id/scan_type/run_id`.
- Pipeline `well_id` must be compound: `rec0000/well000`.
- `PreprocessingTask.split_compound_well_id()` enforces compound well id.
- Register task deps before dependent tasks.
- `PipelineManager.register_task()` uses `task_name` and `dependencies`.
- `TaskRecord.config` snapshot set when status becomes `running`.
- `is_task_complete()` true only when status complete and current config equals snapshot.
- On startup, non-complete cached tasks reset to `not_run`.
- `ConfigManager` does not merge task defaults; task `run()` must call `resolve_params()`.
- JSON cache writes are atomic via tempfile + `os.replace`.
- `DatasetManager.refresh()` full rescan; startup scan only new date dirs.
- Missing cached date dirs logged; cached entries kept.
- Metadata dict keys dynamic; do not hardcode MaxWell annotation keys.
- `auto_curation` writes `curated_spike_times.npy`; hard input contract for `burst_detection`.

## Config Manager

`config_manager/config_manager.py`

- `ConfigManager`: central JSON config provider; implements `BaseConfigProvider`.
- `__init__`: init template seeds and loaded config stores.
- `register_task(task_class)`: store task defaults for template generation only.
- `set_global(key, value)`: store global seed for template generation.
- `get_global(key, default)`: loaded global wins over seed.
- `get_task_params(task_name)`: task params from loaded JSON; `{}` if absent.
- `get_config(task_name, recording_key, well_id)`: pipeline config snapshot source.
- `generate_template(path)`: write new JSON template; never overwrite existing file.
- `load(path)`: load `"global"` and `"tasks"` sections from JSON.
- `save(path)`: atomic write of loaded config only.

## Dataset Manager

`dataset_manager/recording_entry.py`

- `WellEntry`: mutable well metadata holder.
- `RecordingEntry`: frozen recording fields; mutable metadata/wells/h5 maps.
- `RecordingEntry.cache_key`: recording identity string.
- `RecordingEntry.from_path(...)`: parse `data.raw.h5` path into entry; root or sample layout.

`dataset_manager/metadata_extractor.py`

- `WellMetadata`: parsed well id plus fields dict.
- `RecordingMetadata`: recording fields plus selected wells.
- `BaseMetadataExtractor`: parser interface.
- `BaseMetadataExtractor.get(metadata_path)`: return `RecordingMetadata`.
- `_well_to_fields(well)`: flatten decoded well fields and annotations.
- `MxassayMetadataExtractor`: real `mxassay.metadata` parser.
- `MxassayMetadataExtractor.get(metadata_path)`: missing file -> empty metadata.
- `DummyMetadataExtractor`: offline fixed metadata.
- `DummyMetadataExtractor.get(metadata_path)`: ignore path; return dummy recording/wells.

`dataset_manager/_mxassay_decoder.py`

- `_qt_escaped_to_bytes(s)`: decode Qt escaped byte text.
- `_decode_qt_variant(value)`: decode supported Qt `@Variant(...)` values.
- `_coerce_scalar(value)`: string -> bool/int/float/variant/string.
- `_unix_to_iso(value)`: Unix timestamp -> UTC ISO string when plausible.
- `_read_ini_like(path)`: minimal case-preserving INI reader.
- `_well_name_from_id(well_id, columns)`: numeric well id -> plate name.
- `_parse_wells(wells_section)`: parse wells, annotations, selected wells.
- `decode_mxassay_metadata(path, add_iso_times)`: public metadata decoder.
- `_json_default(obj)`: CLI JSON fallback.
- `main()`: CLI decode command.

`dataset_manager/cache_store.py`

- `BaseCacheStore`: recording cache interface.
- `BaseCacheStore.load()`: return cache dict.
- `BaseCacheStore.save(entries)`: persist cache dict.
- `_RecordingEntryEncoder`: JSON `Path` encoder.
- `_RecordingEntryEncoder.default(obj)`: `Path` -> string.
- `_recording_entry_decoder(d)`: JSON object hook -> entries/wells.
- `JsonCacheStore`: `experiment_cache.json` store.
- `JsonCacheStore.__init__(analysis_dir)`: set cache path.
- `JsonCacheStore.load()`: load typed entries; missing file -> `{}`.
- `JsonCacheStore.save(entries)`: atomic JSON save.
- `_entry_to_dict(entry)`: typed entry -> JSON dict.

`dataset_manager/dataset_manager.py`

- `DatasetManager`: discover/cache/query raw recordings.
- `__init__(...)`: set roots/stores/extractor; initialize cache.
- `recordings`: cached entries list.
- `get_recording_by(filters)`: AND filter recording fields and same-well metadata.
- `get_by(key, method, value)`: deprecated wrapper; supports old arg order.
- `get_wells(recording_key)`: wells dict or `{}`.
- `get_path(entry)`: absolute data path.
- `register_well(recording_key, well_id, metadata)`: add/merge well; save cache.
- `update_well_metadata(recording_key, well_id, metadata)`: merge existing well metadata.
- `_matches_recording_filters(entry, filters)`: recording-level filter check.
- `_matches_well_filters(entry, filters)`: all well filters must match same well.
- `refresh()`: clear cache; full rescan; save.
- `_initialise()`: load cache; scan new date dirs.
- `_scan_all()`: full scan helper.
- `_detect_root_level()`: return `"sample"` or `"root"`.
- `_collect_disk_date_keys(root_level)`: visible `(sample_id, date)` pairs.
- `_scan_date_keys(date_keys, root_level)`: threaded date scanning.
- `_scan_date_dir(sample_id, date_dir, root_level)`: find valid runs.
- `_populate_h5_structure(entry, data_path)`: read H5 recordings/wells.
- `_populate_metadata(entry, run_dir)`: merge parsed metadata.
- `_iter_dirs(parent)`: yield child dirs; ignore/list warnings on errors.

## Pipeline Manager

`pipeline_manager/base_task.py`

- `BaseAnalysisTask`: abstract pipeline task contract.
- `__init_subclass__(...)`: validate concrete subclasses have `task_name`, `dependencies`.
- `default_params()`: task fallback params.
- `resolve_params(config_params)`: shallow merge defaults with config; config wins.
- `run(recording_key, well_id, data_path, params)`: execute task; return output path.

`pipeline_manager/config_provider.py`

- `BaseConfigProvider`: config snapshot interface.
- `BaseConfigProvider.get_config(...)`: return current task config.
- `DummyConfigProvider`: no-op config provider.
- `DummyConfigProvider.get_config(...)`: always `{}`.

`pipeline_manager/task_record.py`

- `TaskStatus`: string constants `not_run`, `running`, `complete`, `failed`.
- `TaskStatus.validate(value)`: reject unknown status.
- `TaskRecord`: per-task mutable status/deps/output/error/config.

`pipeline_manager/work_item.py`

- `WorkItem`: immutable queued work triple.

`pipeline_manager/pipeline_entry.py`

- `PipelineEntry`: one recording/well plus task records.
- `PipelineEntry.pipeline_key`: `recording_key/well_id`.

`pipeline_manager/well_metadata.py`

- `BaseWellMetadataProvider`: future per-well metadata interface.
- `BaseWellMetadataProvider.get_metadata(...)`: return well metadata dict.
- `DummyWellMetadataProvider`: no-op metadata provider.
- `DummyWellMetadataProvider.get_metadata(...)`: always `{}`.

`pipeline_manager/cache_store.py`

- `BasePipelineCacheStore`: pipeline cache interface.
- `BasePipelineCacheStore.load()`: return entries by pipeline key.
- `BasePipelineCacheStore.save(entries)`: persist entries.
- `_Encoder`: JSON `Path` encoder.
- `_Encoder.default(obj)`: `Path` -> string.
- `_decode(d)`: JSON object hook -> `TaskRecord`/`PipelineEntry`.
- `_entry_to_dict(entry)`: pipeline entry -> JSON dict.
- `JsonPipelineCacheStore`: `pipeline_cache.json` store.
- `JsonPipelineCacheStore.__init__(analysis_dir)`: set cache path.
- `JsonPipelineCacheStore.load()`: load typed entries; missing file -> `{}`.
- `JsonPipelineCacheStore.save(entries)`: atomic JSON save.

`pipeline_manager/manager.py`

- `PipelineManager`: task DAG registry and per-well scheduler.
- `__init__(analysis_dir, config_provider, cache_store)`: load cache; reset stale tasks.
- `register_computation_task(name, dependencies)`: add task type; patch existing entries.
- `register_task(task_class)`: register `BaseAnalysisTask` subclass.
- `add_well(recording_key, well_id)`: create pipeline entry with all registered tasks.
- `get_next_task(n, type, retry_failed, recording_keys)`: ready work items.
- `update_status(work_item, status, output_path, error)`: mutate task state; snapshot config on running.
- `is_task_complete(work_item)`: complete plus config still current.
- `is_all_complete()`: every registered task complete for every entry.
- `entries`: cached pipeline entries list.
- `get_entry(recording_key, well_id)`: one entry or `None`.
- `get_entries_for_recording(recording_key)`: entries with recording prefix.
- `refresh(task_name, recording_key, well_id)`: reset task plus dependents.
- `_reset_stale_tasks()`: startup reset non-complete tasks.
- `_make_task_record(task_name)`: new `TaskRecord`.
- `_deps_complete(entry, task_name)`: immediate deps complete.
- `_cascade_tasks(task_name)`: task plus transitive dependents.
- `_reset_task_record(record)`: reset status/output/error.
- `_require_entry(recording_key, well_id)`: entry or `KeyError`.
- `_require_task(entry, task_name)`: task or `KeyError`.

## Pipeline Tasks

`pipeline_tasks/preprocessing.py`

- `PreprocessingTask`: SpikeInterface preprocessing for one Maxwell stream.
- `default_params()`: output/filter/reference/save defaults.
- `split_compound_well_id(well_id)`: `rec_name`, `well_id`; require slash.
- `build_output_path(output_root, recording_key, rec_name, well_id)`: preprocessed zarr path.
- `_apply_common_reference(rec, spre, params)`: local/global common reference; local fallback to global.
- `run(...)`: read Maxwell, convert unsigned, bandpass, reference, cast, save zarr.

`pipeline_tasks/sorting.py`

- `SortingTask`: Kilosort sorting for preprocessed stream.
- `default_params()`: sorter/output/VRAM preset defaults.
- `split_compound_well_id(well_id)`: delegate to preprocessing parser.
- `build_output_paths(output_root, recording_key, rec_name, well_id)`: sorter/cleaned output paths.
- `_suppress_kilosort_console()`: cap Kilosort logger noise.
- `_detect_total_vram_gb(torch_module)`: CUDA VRAM GB or `0.0`.
- `_build_kilosort_params(...)`: choose high/low VRAM kwargs; convert seconds to batch size.
- `_resolve_sorting_params(params)`: shallow merge plus nested preset merge.
- `run(...)`: load preprocessed recording, run sorter, clean spikes/empty units, save sorting.

`pipeline_tasks/auto_merge.py`

- `AutoMergeTask`: optional SpikeInterface unit merge.
- `default_params()`: merge/output defaults.
- `split_compound_well_id(well_id)`: delegate to preprocessing parser.
- `build_output_path(output_root, recording_key, rec_name, well_id)`: auto-merge path.
- `run(...)`: pass-through save when disabled; else temp analyzer, auto-merge, cleanup.

`pipeline_tasks/analyzer.py`

- `AnalyzerTask`: build SortingAnalyzer and compute extensions.
- `default_params()`: analyzer/output/extension defaults.
- `split_compound_well_id(well_id)`: delegate to preprocessing parser.
- `build_output_path(output_root, recording_key, rec_name, well_id)`: analyzer path.
- `run(...)`: load preprocessed + merged sorting; estimate sparsity; compute extensions.

`pipeline_tasks/auto_curation.py`

- `AutoCurationTask`: metric threshold curation.
- `default_params()`: curation thresholds and paths.
- `split_compound_well_id(well_id)`: delegate to preprocessing parser.
- `build_output_path(curation_output_root, recording_key, rec_name, well_id)`: curation dir.
- `_apply_thresholds(metrics, p)`: keep flags and rejection reasons.
- `run(...)`: load analyzer metrics, mark curated, write parquet/log/spike times.

`pipeline_tasks/burst_detection.py`

- `BurstDetectionTask`: network burst detection for curated units.
- `default_params()`: burst detector and path defaults.
- `split_compound_well_id(well_id)`: delegate to preprocessing parser.
- `build_curation_output_path(...)`: auto-curation input dir.
- `build_output_path(output_root, recording_key, rec_name, well_id)`: burst output dir.
- `run(...)`: load curated spike times, run detector, write outputs.

## Burst Analysis

`pipeline_tasks/analysis/burst_detector.py`

- `BurstDetectorError`: insufficient spike data error.
- `BurstDetectorConfig`: frozen detector parameters; some filters kept but disabled.
- `BurstResults`: event DataFrames, metrics, diagnostics, plot data.
- `_stats(x)`: mean/std/cv.
- `_level_metrics(events, total_dur)`: aggregate event stats.
- `_get_valley_min(prev, nxt, ws_sharp, t_centers)`: valley between events.
- `_finalize(evs, s, e, units, spike_times, n_units)`: merged event summary.
- `_merge_strict(...)`: merge close events only if valley stays above floor.
- `_merge_clustered(...)`: merge clustered network bursts into superbursts.
- `compute_network_bursts(spike_times, config)`: detect burstlets/network bursts/superbursts.
- `compute_network_bursts._to_df(events)`: nested list-to-DataFrame helper.

`pipeline_tasks/analysis/burst_output.py`

- `BurstOutputWriter`: result persistence interface.
- `BurstOutputWriter.write(results, output_dir)`: persist results.
- `BurstOutputWriter.read(output_dir)`: reload results.
- `ParquetBurstOutputWriter`: parquet + JSON + npy implementation.
- `ParquetBurstOutputWriter.write(results, output_dir)`: write event parquet, metrics, diagnostics, plot signals.
- `ParquetBurstOutputWriter.read(output_dir)`: reconstruct `BurstResults`.
- `_atomic_json_write(data, dest)`: atomic JSON write.

## Scripts

`scripts/strip_notebook_outputs.py`

- `iter_notebooks(argv)`: supplied notebooks or all repo notebooks.
- `strip_notebook(path)`: clear code outputs, execution counts, widget metadata.
- `main()`: CLI entrypoint; print changed notebooks.

## Tests

- `tests/test_experiment_manager.py`: dataset entries, cache store, scanning, filtering, metadata/well behavior.
- `tests/test_metadata_extractor.py`: metadata extraction manual-review output.
- `tests/test_preprocessing_task.py`: compound ids, output paths, preprocessing order, reference fallback/errors.
- `tests/test_sorting_task.py`: Kilosort params, VRAM presets, output paths, pipeline scheduling.
- `tests/test_burst_detection_task.py`: task paths, defaults, detector config handoff, missing curation file.
- `tests/test_burst_detector.py`: detector schema, accuracy, parquet roundtrip, reference equivalence.
- `tests/test_doc.md`: test/doc note.
