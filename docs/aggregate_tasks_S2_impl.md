# Aggregate Tasks — S2 Implementation Guide (phased)

Companion to [`aggregate_tasks_S2_design.md`](./aggregate_tasks_S2_design.md).
S2 is the unified-scheduler rewrite, to be done in a **separate worktree** off the
merged S1. Each phase keeps the suite green; S1 code is deleted only in the last
phase, after a parity harness passes.

## Starting point (S1 inventory)

New in S1 (branch `feat/aggregate-tasks`):
- `src/yuxin_mea/pipeline/scope.py`, `well_dims.py`, `groupname.py`,
  `aggregate_task.py`, `aggregate_cache.py`, `aggregate_scheduler.py`
- `src/yuxin_mea/analysis/cluster_overlay.py`, `ml_trace_io.py`
- `src/yuxin_mea/tasks/cluster_overlay_tasks.py` (+ `AGGREGATE_TASK_CLASSES`)
- CLI: `--aggregate/--no-aggregate/--aggregate-only`, `_build_scheduler`,
  `_drain_aggregate` in `cli/run.py`
- tests: `test_scope.py`, `test_aggregate_scheduler.py`,
  `test_cluster_overlay_task.py`, `test_aggregate_task_classes_registry.py`

Reused verbatim by S2: `scope.py`, `well_dims.py`, `groupname.py`, and the
barrier classification. Replaced by S2: `aggregate_scheduler.py`,
`aggregate_cache.py`, the aggregate drain.

## Phase 0 — lift shared abstractions (no behaviour change)

- Confirm `Scope`/`WellDims`/`Member`/`groupname` have no dependency on the
  per-well manager (they don't). Add `ScopeRelation` enum: `SAME`, `CONTAINS`,
  `FINEST`, plus `Scope.relation_to(other) -> ScopeRelation` and
  `Scope.contains(self_key, other_key, other_scope) -> bool` (the containment
  primitive from the design). Unit-test containment exhaustively (well⊂recording,
  well⊂sample, recording⊂sample, sample_group⊂sample, disjoint→reject).
- Deliverable: new predicates + tests; nothing else changes.

## Phase 1 — `ScopedInstance` + unified store (parallel to old caches)

- `ScopedInstance(scope, scope_key, tasks: dict[str, TaskRecord])`. A `well`
  instance's `scope_key` = `{recording_key, rec_name, well_id}`.
- `InstanceStore` (one JSON) with an `object_hook` that decodes both the legacy
  `PipelineEntry`/`TaskRecord` shape and the new `ScopedInstance` shape. Loader
  maps old `pipeline_cache.json` entries → well `ScopedInstance`s and old
  `aggregate_cache.json` records → aggregate `ScopedInstance`s. Write the unified
  file; keep reading the old two during migration.
- Test: round-trip; loading a real `pipeline_cache.json` + `aggregate_cache.json`
  yields the expected instance set.

## Phase 2 — `UnifiedScheduler.get_next(scope?, task?)`

- One scheduler replacing both `PipelineManager.get_next_task` and
  `AggregateScheduler.plan`. Instances live in `InstanceStore`. Dependencies
  resolve via `ScopeRelation`:
  - `SAME` → same-key lookup (the per-well fast path).
  - `CONTAINS` → enumerate dependency instances by containment.
- Barrier = the single predicate from the design (COMPLETE/FAILED/blocker/
  cascade-hole), computed once per pass. `eligible-now` derives from the queue,
  never by re-deriving deps.
- Topological order over the scope-relation DAG (well tasks are leaves; aggregate
  tasks above them; sample-of-groups above sample-groups, etc.).
- Test parity: for a frozen cache, `UnifiedScheduler` must reproduce the per-well
  `get_next_task` order AND the S1 `AggregateScheduler.plan` decisions.

## Phase 3 — register instances; single drain

- Replace `PipelineManager.add_well(rk, well)` with
  `register_instance(finest_scope, scope_key)`; auto-queue (`_auto_queue_wells`)
  registers the finest instances. Aggregate instances are materialized lazily by
  the scheduler (bucketed from the finest instances), so nothing pre-registers
  them.
- `cli/run.py`: one drain loop over `UnifiedScheduler.get_next(...)`; retire
  `_drain_aggregate` and the separate aggregate pass. `--scheduler=legacy|unified`
  selects old vs new; default `legacy` until parity is proven, then flip.
- `_run_one` dispatches on instance scope: finest → `BaseAnalysisTask.run(...)`
  adapter; coarser → `BaseAggregateTask.run(scope_key, members, params)`.
- Test: full-pipeline smoke (preprocessing→…→ml→overlay) under `--scheduler=unified`
  on a staged fixture recording.

## Phase 4 — real groupname resolver; delete S1 duplication

- Implement `canonicalize_groupname` for real (reuse
  `compare_treatment_groups.py`'s `(sample_id, plate_id, raw_groupname) →
  canonical_group` CSV). Add `groupname_canonical` to `WellDims`; point group
  presets at it.
- After parity holds: delete `aggregate_scheduler.py`, `aggregate_cache.py`, the
  legacy `PipelineManager` scheduling methods (or keep as a thin legacy adapter),
  and the `--scheduler=legacy` path. Migrate configs; drop `aggregate_cache.json`
  once the unified store is authoritative.

## Test strategy

- **Golden parity harness** (build in Phase 1, run every phase): freeze S1's
  `plan()` decisions and a couple of rendered overlay HTMLs (hash, not bytes —
  Plotly output is stable given `random_state`) as fixtures; assert S2 matches.
- Keep every S1 test green until Phase 4; then port
  `test_aggregate_scheduler.py` → `test_unified_scheduler.py`.
- Do NOT delete `test_task_classes_registry.py` invariants — they still guard the
  finest-scope task set.

## Rollback

`--scheduler=legacy|unified` reads the same on-disk caches, so a bad unified run
is reverted by flipping the flag. Keep the legacy path until Phase 4 sign-off.
