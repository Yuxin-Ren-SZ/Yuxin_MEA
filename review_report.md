# MEA Codebase Audit & Improvement Backlog

**Branch:** `refactor` · **Audit date:** 2026-05-14 · **HEAD at audit:** `80b42d9`

## Context

The `refactor` branch has just landed a 5-phase restructure (commits `deb4072 → 80b42d9`, 97 files, ~15k LOC) that produced the current `src/yuxin_mea/` layout: `dataset → pipeline → tasks + analysis + config + dashboard + cli`, 236 tests, plus `AGENTS.md` and `doc/architecture.md`. A multi-agent code review earlier today surfaced several issue clusters (test coverage, type design, error handling) but only at the headline level. `TODO.md` lists 9 pending items. This document is a full re-audit producing a prioritized backlog the user can pick a slice from.

**Deliverable shape**: this is a **triage backlog**, not a sequenced roadmap. Items are grouped by category and ranked by severity within each. The "Recommended top items" section at the end highlights high-impact picks but does **not** propose phases — the next conversation should pick a slice.

**Sources merged**: (1) the earlier multi-agent review headlines, (2) `TODO.md`, (3) a fresh breadth-first pass by 2 parallel Explore agents (skeleton + surface/tests) using a uniform finding schema. Seven high-severity claims were spot-checked against source — five confirmed verbatim, two slightly overstated by the surface agent (B4 / B6) and have been re-graded to medium with corrected framing. Remaining 52 findings rely on agent reports + file:line citations; users should re-verify at implementation time.

**Scale**: 53 production `.py` files (~11k LOC), 18 test files (~236 tests). Audit covered every production file plus tests, notebooks/v2, and the doc set.

## Scoring rubric

- **Severity**: `critical` = data loss / crash on common path / broken invariant with no test · `high` = silent wrong results / blocks user workflow · `medium` = edge-case bug / awkward API / plausible-input validation gap · `low` = cosmetic / minor cleanup.
- **Effort**: `XS` <1h · `S` 1–4h · `M` 4–12h · `L` 1–3d · `XL` 3+d.
- **Status**: `prior` = previously known (multi-agent review) · `todo` = from `TODO.md` · `new` = first surfaced in this audit.

---

## 1. Correctness (bugs and fragile logic)

### B1. `type(x) != si.BaseRecording` instead of `isinstance` *(severity: high, effort: XS, status: new)*
`src/yuxin_mea/tasks/sorting.py:309, 337` — also adjacent TODO typo "excetipn". Direct type-equality breaks for any SpikeInterface subclass; `si.load(...)` returns subclasses, so this guard is one library refactor away from a false positive. Replace with `isinstance(...)`; while there, replace the TODO comments with real messages.

### B2. JSON cache stores crash on corrupt cache files *(severity: high, effort: S, status: prior)*
`src/yuxin_mea/dataset/cache.py:104` and `src/yuxin_mea/pipeline/cache.py:85` call `json.load(...)` with no `JSONDecodeError` handling. A truncated or partially written cache (NAS hiccup, crash mid-save) takes the whole app down on next start instead of recovering. Fix: wrap in try/except, log + back up the corrupt file to `*.corrupt-<ts>`, return `{}`. Same fix shape in both files — extract a `_safe_json_load(path)` helper into a small `src/yuxin_mea/_jsonio.py` and reuse.

### B3. CMR silently falls back to global with no log + missing logger import *(severity: high, effort: XS, status: prior)*
`src/yuxin_mea/tasks/preprocessing.py:130–157` — when local common-median-reference fails, code falls back to global CMR and returns. The TODO at line 156 (`# TODO(logger): warn that local CMR failed and global CMR fallback succeeded.`) acknowledges this but is unimplementable today because the module has **no `logger` defined**. Fix: add `import logging; logger = logging.getLogger(__name__)` at the top, then implement the warning with the local exception message inline. Data-quality decision the user must see.

### B4. WellRecord `_make_well_record` workaround is brittle *(severity: medium, effort: S, status: prior)*
`src/yuxin_mea/analysis/plate_raster_synchrony.py:893–901` catches `TypeError`, checks `"event_intervals" not in str(exc)` and re-raises otherwise. So the masking IS bounded to that one field name — but as soon as another field is added/removed, a parallel branch is needed. Replace the string-matching hack with `dataclasses.fields(WellRecord)` filtering: drop kwargs that aren't field names of the current `WellRecord` and log them once. Ties into T6 (freeze `WellRecord`) and T5 (`status` literal type).

### B5. Bare `except Exception` swallows MemoryError/KeyboardInterrupt in plate loader *(severity: high, effort: S, status: new)*
`src/yuxin_mea/analysis/plate_raster_synchrony.py:779, 861, 874, 930` — four `except Exception  # noqa: BLE001` blocks. Best-effort loading is fine; suppressing `MemoryError`, `RecursionError`, etc. is not. Replace with `(OSError, ValueError, EOFError, pickle.UnpicklingError, pd.errors.ParserError)`. Include the offending file path in the error status string (currently bare "spike_times error").

### B6. `_save_any_form` is hard to follow and untested *(severity: medium, effort: M, status: new)*
`src/yuxin_mea/dashboard/pages/settings.py:218–298`. The `target_index` derivation at lines 257–261 walks `ctx.outputs_list[0]` and does check `form == target_form_id` via `_id_match_for_save`, so it's not directly buggy — Dash's pattern-matched parallel-array contract holds. But the logic is dense (three output lists, status-out built positionally, dirty-out/errors-out built differently) and it has zero callback-level test coverage (G2). Either keep the code and add tests, or simplify by emitting one Output per form. Recommend the test-first path — the callback is the heart of the settings page.

### B7. VRAM detection swallows every error as "0 GB" *(severity: medium, effort: S, status: prior)*
`src/yuxin_mea/tasks/sorting.py:232–239` — `_detect_total_vram_gb` returns `0.0` for both "no CUDA" (normal) and "CUDA driver broken" (deserves a warning). Today, a broken CUDA stack silently routes through low-VRAM presets and the user blames bad sort quality. Fix: only swallow `AttributeError` / "cuda not available"; log a warning before returning 0.0 on any other exception.

### B8. Dashboard cache loaders crash on schema-shifted JSON *(severity: medium, effort: S, status: new)*
`src/yuxin_mea/dashboard/data.py:35–88` — `load_recordings_df` / `load_pipeline_df` call `store.load()` then deref into `.wells`, `.tasks` without guards. If a cache predates a schema field, the page crashes hard with a `KeyError`. Fix: try/except around the build loop, log + skip the offending row, return the partial DataFrame.

### B9. `register_computation_task` mutates `entry.tasks` mid-iteration without snapshot *(severity: medium, effort: S, status: new)*
`src/yuxin_mea/pipeline/manager.py:79–81` mutates each `entry.tasks` dict while iterating cache entries. If save then fails, in-memory state diverges from disk. Fix: build a new dict per entry and swap atomically; save first, mutate in-memory only on success.

### B10. `set_task_params` / `set_globals` use shallow `dict()` copy *(severity: medium, effort: S, status: new)*
`src/yuxin_mea/config/manager.py:140, 144` — nested dicts (e.g., `sorter_kwargs`) share references across callers. Mutating a nested value in one place leaks to others. Fix: `copy.deepcopy` on assignment, or document the contract clearly and rely on tests to guard.

---

## 2. Test Coverage (gaps with concrete fix shape)

### G1. `IterativeBurstDetectionTask.run()` has zero coverage *(severity: critical, effort: M, status: prior)*
`src/yuxin_mea/tasks/iterative_burst_detection.py:196–259` — the entire 64-line `run()` (config build, spike load, output write) is uncovered. Mirror `tests/test_burst_detection_task.py` shape: synthetic spike-time .npy + tmp output dir → run → assert output paths exist + quality columns present. This is the single highest-leverage test add in the codebase.

### G2. Dashboard `pages/settings.py` callbacks fully untested *(severity: high, effort: M, status: prior)*
Five callbacks (`_populate_globals`, `_populate_task_tabs`, `_mark_forms_dirty`, `_toggle_save_buttons`, `_save_any_form`) — `tests/test_config_builder.py` only covers the form builder helpers. End-to-end fix: use the Dash callback test helper to fire the callbacks with simulated states, then assert on outputs. Without this, B6 will silently regress.

### G3. `pages/plate_viewer.py` callbacks untested *(severity: high, effort: M, status: new)*
`src/yuxin_mea/dashboard/pages/plate_viewer.py:164–300` — only registration is smoke-tested in `tests/test_plate_viewer_page.py`. The three callbacks (`_populate_recordings`, `_on_load`, `_on_export`) have no coverage. Add direct callback tests with mocked `current_app.config`.

### G4. `pages/burst_diagnostic.py` callbacks untested *(severity: high, effort: M, status: new)*
Five callbacks plus the module-global `_LOADED_BATCHES` state machine are uncovered. Add `tests/test_burst_diagnostic_page.py` exercising load / recompute / dropdown population / error cases.

### G5. Nested-dict round-trip only tested for one task *(severity: medium, effort: S, status: prior)*
`tests/test_config_builder.py::test_collect_values_reconstructs_nested_dict_from_dotted_keys` is hardcoded to one task. Two tasks have nested dict params today (`SortingTask`, `AnalyzerTask`); a third would silently slip through. Parametrize the test over all tasks whose `params_schema()` contains a `ParamSpec(type="dict", nested_schema=...)`.

### G6. Cache corruption / schema-drift recovery untested *(severity: medium, effort: S, status: prior)*
Combines with B2 + B8: tests should write a truncated JSON cache and a schema-shifted JSON cache, and assert (a) dataset/pipeline stores return `{}` with a warning, (b) dashboard loaders return empty DataFrame with a warning. Without tests these regressions ship.

### G7. Burst-detector "insufficient data" error paths untested *(severity: medium, effort: S, status: new)*
`BurstDetectorError` and `IterativeBurstError` are raised in `analysis/burst_detector.py` and `analysis/iterative_burst_detector.py`. No test asserts they fire on empty/sparse input. Add `pytest.raises(...)` cases — cheap insurance.

### G8. Kilosort label-file parsing returns None silently on malformed TSVs *(severity: medium, effort: S, status: prior)*
`src/yuxin_mea/analysis/burst_diagnostic.py:113–134` (`_read_kilosort_keep_clusters`). Missing/renamed columns silently return `None`, causing the detector to load **all** clusters including noise. Fix: validate columns up front, raise `ValueError` with a clear message; add a malformed-TSV test.

### G9. Three task `run()` paths besides G1 are also uncovered *(severity: medium, effort: M each, status: new)*
The audit's coverage matrix shows `tasks/auto_merge.py`, `tasks/analyzer.py`, `tasks/auto_curation.py` each have only param-schema parity tests — their `run()` methods are not exercised. Severity is below G1 because preprocessing/sorting/burst_detection ARE covered and these are simpler glue, but a regression here would only be caught by the user running the notebook.

### G10. `WellRecord` missing-field validation untested *(severity: medium, effort: S, status: new)*
B4 fix needs a test: write a `WellRecord(**kwargs)` call with a deliberately wrong key, assert the error message lists the offending field.

---

## 3. Error Handling (silent failures, broad excepts)

### EH1. `JsonPipelineCacheStore.save` silently eats temp-file cleanup failures *(severity: medium, effort: S, status: new)*
`src/yuxin_mea/pipeline/cache.py:98–102` — after a save failure, the `os.unlink(tmp)` is wrapped in a broad except; orphan temp files accumulate on NAS. Log at WARNING before swallowing.

### EH2. `dataset/manager.py:_populate_h5_structure` catches `Exception` broadly *(severity: low, effort: S, status: new)*
`src/yuxin_mea/dataset/manager.py:441` — narrow to `(OSError, h5py.Error)`; also clarify whether h5py is required or optional (the `ImportError` path at 426–429 suggests optional, but no graceful degradation downstream).

### EH3. Dashboard callbacks don't validate state types *(severity: medium, effort: S, status: new)*
`pages/plate_viewer.py:241–256, 277–300` — `marker_size`, `n_recent_spikes`, etc. arrive from browser State as `Any`. Today a tampered State raises `ValueError` in `_build_config` and surfaces as a bare "❌ {exc}" toast. Use `Literal`/`Annotated` type hints + a small validator helper that returns user-facing messages.

---

## 4. Types & Type Design

### T1. `RecordingEntry(frozen=True)` is a half-promise *(severity: medium, effort: M, status: new)*
`src/yuxin_mea/dataset/entries.py:47–68` — `frozen=True` prevents reassigning `entry.metadata`, but `entry.metadata["key"] = ...` and `entry.wells[id].metadata = ...` are still legal. The architecture doc claims immutability. Either (a) wrap nested mutable containers in `MappingProxyType` at construction, or (b) downgrade the doc claim to "top-level fields only". (a) is safer; (b) is honest. Pick one — don't leave the gap.

### T2. `TaskStatus.validate()` exists but is never called *(severity: medium, effort: XS, status: new)*
`src/yuxin_mea/pipeline/task_record.py:15–20` defines `validate()`; `PipelineManager.update_status()` (`pipeline/manager.py:176`) does its own inline check against `_VALID_UPDATE_STATUSES`. Two validation paths, neither used at construction time. Pick one — delete the other. Recommend keeping `TaskStatus.validate()` and calling it from `update_status` and `_make_task_record`.

### T3. `TaskRecord` is mutable despite being documented as a "snapshot" *(severity: low, effort: S, status: new)*
`src/yuxin_mea/pipeline/task_record.py:23–30`. Either freeze it and reconstruct on mutation, or update `AGENTS.md`'s "snapshotted" language. Mutating-in-place is what the manager actually does, so the doc is the likely thing to soften.

### T4. `ParamSpec(type="dict", nested_schema=None)` silently accepts anything *(severity: low, effort: S, status: new)*
`src/yuxin_mea/config/schema.py:165–176`. A `dict`-typed spec with no `nested_schema` short-circuits validation. Either enforce at `ParamSpec.__post_init__` (raise if `type="dict"` without `nested_schema`), or document explicitly. Adding `__post_init__` validation is the lower-effort path and catches the bug at task-class load time.

### T5. `PlateViewerConfig.display_mode` and `WellRecord.status` are bare `str` *(severity: medium / low, effort: S, status: new)*
`src/yuxin_mea/analysis/plate_raster_synchrony.py:54, 72`. Use `Literal["raster","synchrony","both"]` and `Literal["ok","missing","plot_signals error","spike_times error"]` respectively. Mypy/pyright catches typos at edit time. Adding a runtime check in `build_plate_figure` for the display_mode value is the belt-and-suspenders fix.

### T6. `WellRecord` is mutable; pages could corrupt it across callbacks *(severity: low, effort: S, status: new)*
`src/yuxin_mea/analysis/plate_raster_synchrony.py:62–73`. Add `frozen=True`; spike_times and plot_signals arrays should be immutable post-load anyway.

---

## 5. Architecture Drift / Docs

### D1. README + AGENTS.md still list `plate_viewer` as a task *(severity: high, effort: S, status: new)*
Phase 5 (`4c0c664`) moved plate viewer from `tasks/` to `dashboard/pages/`. But:
- `AGENTS.md:33` still says `tasks/` includes plate_viewer (the file list at lines 211–283 doesn't, but the prose is stale).
- `README.md:31–37` lists it in the pipeline.
- `README.md:69–75` doesn't mention the plate-viewer dashboard page.
Fix all three to match the actual layout described in `architecture.md:115–119` (which is correct).

### D2. `doc/architecture.md` mixes "stage" / "task" terminology *(severity: medium, effort: S, status: new)*
The doc was written pre-Phase-3 when "stage" was the canonical term. Some sections still use it. The doc is technically accurate but reads as drifted. Either (a) do a sweep and rename consistently, or (b) prepend a banner pointing readers to `AGENTS.md`. (b) is 5 minutes; (a) is 2–4h.

### D3. `AGENTS.md` doesn't mention `TaskStatus.validate()` *(severity: low, effort: XS, status: new)*
`AGENTS.md:154–155`. One-line add.

### D4. `_LOADED_BATCHES` module-global is not flagged in AGENTS.md as single-user *(severity: low, effort: XS, status: new)*
`src/yuxin_mea/dashboard/pages/burst_diagnostic.py:45`. AGENTS.md notes it exists but not the concurrency assumption. One-line clarification.

---

## 6. API Design

### API1. `BasePlateLevelTask._run_template` is a workaround, not a clean override *(severity: low, effort: S, status: new)*
`src/yuxin_mea/pipeline/base_plate_level_task.py:32–50, 95–122`. Subclasses must override `run()` and delegate to `_run_template()`; the comment admits this is to dodge `__init_subclass__` validation. Cleaner: define a concrete `run()` in the base that calls `_run_template(...)`, and have subclasses override `aggregate_records` / `write_output` only.

### API2. `SortingTask` re-implements param merging with custom `_resolve_sorting_params` *(severity: medium, effort: S, status: new)*
`src/yuxin_mea/tasks/sorting.py:266–277`. Because `BaseAnalysisTask.resolve_params` is shallow, sorting needs a custom deep merge for `high_vram_sorter_kwargs` / `low_vram_sorter_kwargs`. Either (a) add an optional `deep_keys: list[str]` arg to `resolve_params`, or (b) keep the custom resolver but document the pattern in `BaseAnalysisTask` so future tasks know when to use which.

### API3. `ConfigManager` has parallel "seed" and "loaded" state with implicit precedence *(severity: low, effort: M, status: new)*
`src/yuxin_mea/config/manager.py:66–110`. Four dicts: `_global_set`, `_global_loaded`, `_task_defaults`, `_task_loaded`. The "loaded" wins in getters but the seeds win in `generate_template`. Refactor to a single "current" view with explicit `register_default(...)` semantics, or document the precedence in the class docstring (it isn't today).

### API4. Page callbacks read state from `current_app.config["YUXIN_MEA"]` *(severity: medium, effort: M, status: new)*
Every page does `yuxin_ctx = current_app.config.get("YUXIN_MEA", {})` at callback time. Makes tests need a Flask context (one reason G2–G4 stayed unwritten). Inject via a small `DashContext` class set in `build_app`, or pass via `dcc.Store` and `clientside_callback`. Bigger refactor — only do if the callback test sprint reveals the friction.

---

## 7. Dead Code

### DC1. `DatasetManager.get_by()` deprecated but still exported *(severity: low, effort: S, status: new)*
`src/yuxin_mea/dataset/manager.py:117–136` is marked deprecated. If callers have all moved to `get_recording_by()`, remove it. If not, give it a `DeprecationWarning`.

### DC2. `_DEFAULTS` in plate_viewer page duplicates `PlateViewerConfig` defaults *(severity: low, effort: S, status: new)*
`src/yuxin_mea/dashboard/pages/plate_viewer.py:55–62`. Replace with `dataclasses.asdict(PlateViewerConfig())` — keeps them in lockstep.

### DC3. `BurstDetectorConfig` has "disabled filter" fields that pollute the schema *(severity: low, effort: S, status: new)*
`src/yuxin_mea/analysis/burst_detector.py:15–31`. Several fields (`gamma`, `min_burstlet_participation`, `min_absolute_rate_hz` …) are documented as "not yet wired". They still appear in `BurstDetectionTask.params_schema()` so users see them in the Settings UI without effect. Either wire them up or move them out of `params_schema`.

### DC4. Unused imports in some dashboard pages *(severity: low, effort: XS, status: new)*
`pages/home.py`, `pages/recordings.py`, `pages/pipeline.py`. Run `ruff check --select F401` and clean. Trivial.

---

## 8. Features (TODO.md items) — verified pending, with framing

### F1. Optional-task auto-skip *(severity: medium, effort: L, status: todo)*
`TODO.md:3–5` — today `auto_merge` is enabled=False by default but `analyzer` hard-depends on it, so users must hand-edit dependencies. Right shape: `BaseAnalysisTask.is_disabled(params) -> bool` (default False) + `PipelineManager` rewrite of `_deps_complete` to treat disabled deps as satisfied. Requires deciding what "disabled" means — `enabled=False` param convention vs. a class-level flag. Recommend the param-based version: it's already how `auto_merge` works.

### F2. `get_next_task` DFS *(severity: low, effort: M, status: todo — verify intent first)*
`TODO.md:6`. Current `pipeline/manager.py:138–161` iterates entries first then tasks per entry — which is effectively per-well DFS already. **Read the code with the user before "fixing" this.** Possibly the TODO is stale, or it means "process all tasks for one well before moving to next" which is what already happens.

### F3. mxassay.metadata per-run parse merging into `RecordingEntry` *(severity: medium, effort: M, status: todo)*
`TODO.md:18`. Extractor exists (`dataset/metadata.py`) and `_mxassay_decoder.py` handles Qt format. The remaining work is pulling per-run fields (sampling rate, chip ID, assay type) and merging into `RecordingEntry`. Likely composes with T1 (frozen-vs-mutable) — decide that first.

### F4. `RecordingEntry.is_valid()` *(severity: low, effort: XS, status: todo)*
`TODO.md:19`. One method on `entries.py` — `return self.data_path.exists()`. Trivial.

### F5. `DatasetManager.summary()` *(severity: low, effort: S, status: todo)*
`TODO.md:20`. Print a per-sample/date count table. Useful for notebook orientation.

### F6. `spikeinterfacePreprocessor` class storing a dict of spre jobs *(severity: medium, effort: M, status: todo)*
`TODO.md:8`. Today `PreprocessingTask.run` hardcodes the bandpass + CMR sequence. Generalizing to a configurable pipeline of `spre.*` jobs is genuinely useful but unbounded — define which jobs the user actually wants to swap before designing. Defer until the requirements firm up.

### F7. Retrievable config with hash-based version DB *(severity: low, effort: XL, status: todo)*
`TODO.md:10–12`. Significant scope — git-like content-addressed config + code-hash binding. Out of scope for a single sprint; capture as a separate design discussion.

### F8. Extract utility functions *(severity: low, effort: S, status: todo)*
`TODO.md:7`. Concrete candidates surfaced by this audit:
- `_safe_json_load(path)` (used in B2 fix, dataset/cache + pipeline/cache + dashboard/data)
- `split_compound_well_id` is currently re-exported five times across tasks via `PreprocessingTask.split_compound_well_id` — could live in `pipeline/well_id.py`
- `_atomic_json_write` exists in `analysis/burst_output.py` and a near-duplicate pattern in both cache stores — consolidate.

---

## Cross-cutting themes

Several findings are not isolated bugs but instances of the same pattern. Acting on the pattern is more efficient than patching individual sites.

1. **Silent fallbacks** — B3 (CMR), B7 (VRAM), G8 (label-file), B5 (plate raster `except Exception`), EH1 (temp-file cleanup), EH2 (h5py). Pattern: catch broad exception, return a degraded value, no log. Fix shape is uniform: narrow the except clause and add a warning. Worth a single audit-and-fix sweep.
2. **Invariant claimed in docs, not enforced in code** — T1 (frozen RecordingEntry with mutable nested), T2 (TaskStatus.validate unused), T3 (TaskRecord doc says snapshot but is mutable), D1+D2+D3 (docs drift after Phase 5). Pattern: either tighten the code or soften the doc, but don't leave the gap.
3. **Untested glue at the layer boundaries** — G1 (task run), G2–G4 (dashboard callbacks), G6 (cache corruption recovery), G9 (other task run methods). The internals of each layer are well-tested; the wires between them aren't.
4. **JSON cache fragility** — B2, B8, EH1, G6 all touch the same family. A small `_jsonio` module with `safe_load` + `atomic_write` would absorb most of them.

---

## Recommended top items (impact-ranked, no sequencing)

Among 59 findings, these stand out as highest return per unit effort. Pick a slice that fits the available time; the items don't depend on each other and can be picked in any order.

| Rank | ID | Title | Severity | Effort |
|------|------|--------------------------------------------------|----------|--------|
| 1 | G1 | Test `IterativeBurstDetectionTask.run()` | critical | M |
| 2 | B3 | CMR fallback warning + logger import | high | XS |
| 3 | B2/G6 | JSON cache corruption recovery + tests | high | S |
| 4 | D1 | Update README/AGENTS.md re Phase 5 plate viewer | high | S |
| 5 | B1 | Replace `type() !=` with `isinstance` in sorting | high | XS |
| 6 | B4+T5+T6 | WellRecord field-rename safety + literal types + freeze | medium | S |
| 7 | G2/G3/G4 | Dashboard callback tests (bundle) | high | M each |
| 8 | T2 | Unify `TaskStatus.validate()` + remove dead check | medium | XS |
| 9 | B6 | Settings `_save_any_form` simplification + tests | medium | M |
| 10 | F1 | Optional-task auto-skip | medium | L |

A reasonable "tight sprint" would be items 1–6 (most are XS/S and address the highest-leverage problems). A "test-coverage focused" sprint would be items 1, 7, and G5/G7/G8. F1 is the single biggest feature value but stands alone.

---

## Critical files referenced

- `src/yuxin_mea/dataset/cache.py` — B2, F8 (utility extraction)
- `src/yuxin_mea/dataset/entries.py` — T1, F4
- `src/yuxin_mea/dataset/manager.py` — DC1, EH2, F5
- `src/yuxin_mea/pipeline/cache.py` — B2, EH1
- `src/yuxin_mea/pipeline/manager.py` — B9, F1, F2, T2
- `src/yuxin_mea/pipeline/task_record.py` — T2, T3
- `src/yuxin_mea/pipeline/base_plate_level_task.py` — API1
- `src/yuxin_mea/config/manager.py` — B10, API3
- `src/yuxin_mea/config/schema.py` — T4
- `src/yuxin_mea/tasks/preprocessing.py` — B3
- `src/yuxin_mea/tasks/sorting.py` — B1, B7, API2
- `src/yuxin_mea/tasks/iterative_burst_detection.py` — G1
- `src/yuxin_mea/tasks/{auto_merge,analyzer,auto_curation}.py` — G9
- `src/yuxin_mea/analysis/plate_raster_synchrony.py` — B4, B5, T5, T6
- `src/yuxin_mea/analysis/burst_detector.py` — DC3, G7
- `src/yuxin_mea/analysis/burst_diagnostic.py` — G8, D4
- `src/yuxin_mea/dashboard/data.py` — B8, G6
- `src/yuxin_mea/dashboard/pages/settings.py` — B6, G2
- `src/yuxin_mea/dashboard/pages/plate_viewer.py` — G3, EH3, DC2
- `src/yuxin_mea/dashboard/pages/burst_diagnostic.py` — G4
- `tests/test_config_builder.py` — G5
- `tests/test_burst_detection_task.py` — template for G1 / G9
- `AGENTS.md`, `README.md`, `doc/architecture.md` — D1, D2, D3, D4

## Existing utilities to reuse

- `_atomic_json_write` in `src/yuxin_mea/analysis/burst_output.py` — pattern for B2/B8/EH1.
- `JsonCacheStore.save` (`dataset/cache.py:107–122`) — atomic-write reference; use this shape when consolidating into `_jsonio`.
- `ParamSpec.__post_init__` (already there for `frozen=True`) — place to add the T4 type-vs-nested_schema check.
- `tests/test_burst_detection_task.py` — directly mirror its shape for G1 and the G9 task tests.
- `tests/test_config_builder.py::test_collect_values_reconstructs_nested_dict_from_dotted_keys` — parametrize as the G5 fix.

## Verification (how to validate a chosen slice once executed)

When picking a slice for the next sprint, validate via:

1. **Per-item assertion**: each item in the backlog includes a "Suggested fix" — implementation should match it or document why it deviated.
2. **Test suite**: `conda run -n yuxin_mea pytest -x` should remain green; new items G1, G2–G4, G5–G8 should add tests that fail-before-fix and pass-after.
3. **Lint sweep**: `ruff check src/yuxin_mea` should be no-noisier than before (DC4 specifically lowers noise).
4. **Docs sync**: after any item from §5 lands, AGENTS.md and README.md should match the actual code layout (grep test: no `plate_viewer` under "tasks" prose).
5. **Spot-check on real data**: run notebooks/v2/00_full_pipeline.ipynb end-to-end after merging — items B1, B3, B7, B8 are the ones most likely to surface during a real run.
6. **Dashboard manual test**: `yuxin-mea-dashboard --config pipeline_config.json` and click through Settings save / Plate viewer load — covers anything B6/G2 touches.

---

## Open questions for the execution conversation

1. **F2 intent**: is the existing per-entry-then-per-task iteration in `get_next_task` already what you wanted, or is "DFS" something more specific?
2. **T1 direction**: tighten `RecordingEntry` to truly immutable, or soften the doc to "top-level immutable"? Both are fine — the choice affects F3 (mxassay merge) design.
3. **Test coverage scope**: should G2–G4 be tackled together (they're the same shape and would share a Dash callback test helper) or one at a time?
4. **F6/F7 scoping**: defer the `spikeinterfacePreprocessor` and hash-based config DB to a separate design conversation? Both feel premature without firmer requirements.
