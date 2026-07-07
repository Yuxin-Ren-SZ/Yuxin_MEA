# Aggregate Tasks — S2 Design Guide (unified scheduler)

Status: **future work.** S1 (the additive aggregate layer) is live on branch
`feat/aggregate-tasks`. S2 is the planned rewrite that unifies per-well and
aggregate scheduling into one scope-keyed model, to be done in a separate
worktree. This document is the design; the companion
[`aggregate_tasks_S2_impl.md`](./aggregate_tasks_S2_impl.md) is the phased plan.

## Why unify

S1 deliberately runs a **parallel** scheduler (`AggregateScheduler`) alongside
the per-well `PipelineManager` to avoid touching working code. The cost is
duplication: two schedulers, two caches (`pipeline_cache.json` +
`aggregate_cache.json`), two barrier implementations, two re-run rules, two drain
loops in `cli/run.py`. S2 collapses these so there is **one** notion of a task
instance, one dependency graph, one barrier predicate, one cache.

The key realization (already encoded in S1's `Scope`): **a per-well task is just
the finest scope.** `well` = `Scope(group_by=("recording_key","rec_name","well_id"))`.
Everything the per-well manager does is a special case of scope scheduling.

## Core generalization

| S1 (today) | S2 (target) |
|---|---|
| `PipelineEntry(recording_key, well_id, tasks)` | `ScopedInstance(scope, scope_key, tasks)` |
| cache key `f"{recording_key}/{well_id}"` | cache key `f"{scope.name}::{encode_path(scope_key)}"` |
| `add_well(rk, well)` | `register_instance(scope, scope_key)` (well = finest scope) |
| `_forward_deps` (intra-entry) | scope-relation dependency DAG |
| `PipelineManager.get_next_task` + `AggregateScheduler.plan` | one `UnifiedScheduler.get_next(scope?, task?)` |
| `pipeline_cache.json` + `aggregate_cache.json` | one unified store |

`ScopedInstance` reuses `TaskRecord` unchanged. A well instance's `scope_key` is
`{"recording_key": ..., "rec_name": ..., "well_id": ...}`; its tasks are the
per-well tasks. An aggregate instance's tasks are the aggregate tasks.

## Unified dependency graph

Dependencies today are task-name → task-name and assumed to resolve **within one
entry**. S2 makes the resolution explicit via a **scope relation** between the
downstream instance and its dependency's instances:

- **same** — well→well (e.g. `burst_detection` depends on `auto_curation` for the
  *same* well). Downstream and dependency share `scope_key`.
- **containment (finer)** — aggregate→well or aggregate→coarser-aggregate. The
  dependency's instances whose members are a subset of the downstream instance's
  members. E.g. a `sample` instance (`group_by=(sample_id,)`) depends on all
  `well` instances with that `sample_id`; a `sample` instance depends on all
  `recording` instances of that sample.

Formally: instance *B* (scope *S_B*, key *k_B*) depends on task *D* (scope *S_D*).
The dependency instances are those *D*-instances whose key agrees with *k_B* on the
dimensions *S_B* and *S_D* share, **and** whose group_by is a superset-or-equal of
*S_B*'s on the shared dims (i.e. *D* is at least as fine). This is the **scope
containment** relation — the single most important primitive S2 must get right,
and the primary seam S1 leaves (`AggregateScheduler._blockers` raises
`NotImplementedError` for aggregate→aggregate deps).

## Barrier as one predicate

S1's barrier (in `aggregate_scheduler._evaluate`) already has the correct shape;
S2 generalizes it. For a downstream instance and dependency *D*, resolve the
dependency instances via the scope relation, then classify each:

```
COMPLETE                              -> usable
FAILED                               -> hole (tolerate)
eligible-now(D) or RUNNING           -> BLOCKER  -> instance WAITs
NOT_RUN & not eligible-now(D)        -> cascade hole (tolerate)
```

`eligible-now(D)` = *D*-instances whose own upstream is complete and are about to
run — computed once per pass from the queue, never by re-deriving deps. This is
literally S1's algorithm with "member well" replaced by "dependency instance."
The per-well "all deps complete for this entry" rule is the degenerate case where
the relation is `same` and there is exactly one dependency instance.

## Cache unification

One store, keyed by `(scope, scope_key, task)`. `member_hash` and `config`
snapshots become first-class for **every** level:

- Well instances today only carry a `config` snapshot (the config-mismatch
  re-run). S2 adds `member_hash` uniformly; for a well it hashes the single
  member (itself), so it's a no-op there but unifies the re-run rule:
  **re-run when member_hash changes OR config changes**, at every scope.

Back-compat: the S2 loader reads both `pipeline_cache.json` and
`aggregate_cache.json`, maps each into `ScopedInstance` records, and writes the
unified file. Keep the old files readable during migration.

## Groupname resolution (promote the S1 hook)

S1 only warns (`groupname.detect_groupname_issues`) and
`canonicalize_groupname` is the identity. S2 replaces it with a real resolver:
reuse the `canonical_group` template CSV pattern already in
`scripts/compare_treatment_groups.py` (which maps `(sample_id, plate_id,
raw_groupname) → canonical_group`). Add a `groupname_canonical` dimension to
`WellDims` derived through the resolver, so `group_by=("sample_id",
"groupname_canonical")` merges spelling variants.

## Seams S1 leaves for S2 (reuse verbatim)

These S1 modules are designed to survive into S2 unchanged:

- `pipeline/scope.py` — `Scope`, `DIMENSION_NAMES`, `encode_path`, presets.
- `pipeline/well_dims.py` — `WellDims`, `Member`.
- `pipeline/groupname.py` — the resolver hook + issue detection.
- `pipeline/aggregate_task.py` — `BaseAggregateTask.run(scope_key, members, params)`
  is already scope-generic; per-well `BaseAnalysisTask.run` becomes a thin adapter
  (or both converge on one signature).
- The barrier classification in `aggregate_scheduler._evaluate`.

What S2 replaces/deletes: `AggregateScheduler` (folded into `UnifiedScheduler`),
`aggregate_cache.py` (folded into the unified store), the separate aggregate drain
in `cli/run.py`. Because S1 **never** couples aggregate logic into
`PipelineManager._forward_deps` or `add_well`, S2 can rewrite the scheduler
without unwinding hacks in the per-well code.

## Migration & compatibility

- Feature flag `--scheduler=legacy|unified`; both read the same on-disk caches.
- Parity harness: freeze S1's `plan()` decisions + rendered outputs for a fixed
  cache as golden fixtures; assert S2 produces the same instance set + outputs.
- Roll out unified behind the flag; delete S1 duplication only after parity holds.

## Risks / open questions

- **Scope-containment ambiguity** — coarse-on-coarse deps, or scopes that share no
  dimensions, need a well-defined (or explicitly rejected) relation.
- **Re-run storms** — a `sample` instance re-runs whenever *any* member well's ml
  re-runs; batch/debounce, or make re-run opt-in per task.
- **Parallel aggregate execution** — S1 runs aggregate instances serially; S2
  should schedule independent instances concurrently (they write disjoint
  outputs), which needs the same parent-owns-cache discipline as `_drain_parallel`.
- **Instance lifecycle** — when a recording is deleted/re-scanned, stale
  `ScopedInstance`s (and their outputs) need garbage collection.
