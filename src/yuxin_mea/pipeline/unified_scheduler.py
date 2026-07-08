"""UnifiedScheduler: one scope-keyed scheduler for the whole pipeline (S2).

Subsumes both legacy schedulers over a single :class:`InstanceStore`:

* the per-well ``PipelineManager.get_next_task`` — the FINEST scope
  (``SCOPE_WELL``). Eligibility is the legacy status rule verbatim: a task is
  returned iff its own status is NOT_RUN (or FAILED with ``retry_failed``) AND
  every dependency is COMPLETE. A COMPLETE task is never re-returned by the queue
  (config-mismatch re-runs still go through an explicit refresh, as before).
* the S1 ``AggregateScheduler.plan`` — every COARSER scope. Eligibility is the S1
  barrier (blocker→WAIT, holes tolerated) with the S1 member-hash+config re-run
  rule, generalized so a dependency may itself be an aggregate instance
  (scope-containment, the aggregate→aggregate case S1 raised on).

The two eligibility rules stay distinct on purpose — collapsing them into one
predicate would make the finest queue return COMPLETE tasks the legacy queue
never returned, breaking parity. What is unified is everything else: one instance
model, one store, one dependency DAG resolved by :class:`ScopeRelation`, and one
pass that computes cross-scope "eligible-now" once, bottom-up.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field

from .scope import SCOPE_WELL, Scope, ScopeRelation
from .scoped_instance import ScopedInstance, scoped_instance_key
from .task_record import TaskRecord, TaskStatus
from .well_dims import Member, WellDims

# Aggregate decisions (mirror aggregate_scheduler constants).
RUN = "run"
RERUN = "rerun"
SKIP_EMPTY = "skip_empty"
WAIT = "wait"
UPTODATE = "uptodate"
RUNNING = "running"    # instance whose own task is mid-run — blocks its dependents


@dataclass
class WorkUnit:
    """A (instance, task) unit that is eligible to run this pass."""
    scope_name: str
    scope_key: dict[str, str]
    task_name: str
    members: list[Member] = field(default_factory=list)   # empty for finest tasks

    @property
    def instance_key(self) -> str:
        return scoped_instance_key(self.scope_name, self.scope_key)


@dataclass
class InstanceDecision:
    """The planned decision for one aggregate (instance, task)."""
    scope_name: str
    scope_key: dict[str, str]
    task_name: str
    decision: str
    members_complete: list[Member] = field(default_factory=list)
    n_total: int = 0
    member_hash: str = ""

    @property
    def instance_key(self) -> str:
        return scoped_instance_key(self.scope_name, self.scope_key)


def _member_identity(m: Member) -> str:
    """Stable identity string for a member.

    The well branch is byte-identical to ``aggregate_scheduler._member_fingerprint``
    so per-well aggregate fingerprints match S1 exactly. Aggregate-instance members
    (scope_key set) get their own branch keyed by scope identity.
    """
    if m.scope_key is not None:
        return (
            f"{m.scope_name}|{Scope.encode_path(m.scope_key)}"
            f"|{m.upstream_output_path}|{m.upstream_last_updated}"
        )
    return (
        f"{m.recording_key}|{m.compound_well_id}"
        f"|{m.upstream_output_path}|{m.upstream_last_updated}"
    )


def _member_fingerprint(members_complete: list[Member]) -> str:
    items = sorted(_member_identity(m) for m in members_complete)
    return hashlib.sha1("\n".join(items).encode("utf-8")).hexdigest()


class ScopeRegistry:
    """Maps a scope *name* to its :class:`Scope` (with its ``where`` lambda).

    Seeded from the finest scope plus every aggregate task's ``scope``. Also maps
    each task name to the scope it lives at (finest tasks → ``SCOPE_WELL``).
    """

    def __init__(self, finest_scope: Scope, agg_tasks: list) -> None:
        self.by_name: dict[str, Scope] = {finest_scope.name: finest_scope}
        self.task_scope: dict[str, Scope] = {}
        self.finest = finest_scope
        self._agg_names: set[str] = set()
        for t in agg_tasks:
            self.by_name[t.scope.name] = t.scope
            self.task_scope[t.task_name] = t.scope
            self._agg_names.add(t.task_name)

    def scope_of_task(self, task_name: str) -> Scope:
        """Scope a task lives at — finest unless it is a registered aggregate task."""
        return self.task_scope.get(task_name, self.finest)

    def is_aggregate_task(self, task_name: str) -> bool:
        return task_name in self._agg_names


class UnifiedScheduler:
    def __init__(
        self,
        store,
        agg_task_classes,
        config_mgr,
        experiment_cache: dict | None = None,
        *,
        finest_scope: Scope = SCOPE_WELL,
    ) -> None:
        self.store = store
        self.instances: dict[str, ScopedInstance] = store.load()
        self.cm = config_mgr
        self.exp = experiment_cache or {}
        self.finest_scope = finest_scope
        self.agg_tasks = [c() for c in agg_task_classes]
        self.registry = ScopeRegistry(finest_scope, self.agg_tasks)
        self._agg_order = self._topo_order_aggregates()
        self._dims: list[WellDims] | None = None

    # ------------------------------------------------------------------ #
    # Well universe + bucketing
    # ------------------------------------------------------------------ #
    def _groupname_lut(self) -> dict[tuple[str, str], str]:
        lut: dict[tuple[str, str], str] = {}
        for rk, rec in self.exp.items():
            for wid, w in (rec.get("wells") or {}).items():
                md = w.get("metadata") or {}
                g = str(md.get("groupname", "?") or "?")
                lut[(rk, str(wid))] = g
                own = w.get("well_id")
                if own is not None:
                    lut[(rk, str(own))] = g
        return lut

    def _finest_instances(self) -> list[ScopedInstance]:
        """Finest instances in store insertion order (parity: legacy cache order)."""
        return [
            i for i in self.instances.values()
            if i.scope_name == self.finest_scope.name
        ]

    def build_well_dims(self) -> list[WellDims]:
        if self._dims is not None:
            return self._dims
        gl = self._groupname_lut()
        dims: list[WellDims] = []
        for inst in self._finest_instances():
            rk = inst.scope_key["recording_key"]
            rec_name = inst.scope_key.get("rec_name", "")
            well_id = inst.scope_key.get("well_id", "")
            compound = f"{rec_name}/{well_id}" if rec_name else well_id
            groupname = gl.get((rk, well_id), "?")
            dims.append(WellDims.from_entry(rk, compound, groupname))
        self._dims = dims
        return dims

    def instances_for(self, scope: Scope) -> dict[tuple, list[WellDims]]:
        buckets: dict[tuple, list[WellDims]] = defaultdict(list)
        for d in self.build_well_dims():
            if scope.matches(d):
                buckets[scope.well_key(d)].append(d)
        return buckets

    # ------------------------------------------------------------------ #
    # Dependency-instance resolution (containment)
    # ------------------------------------------------------------------ #
    def resolve_dependency_keys(
        self, down_scope: Scope, dims_list: list[WellDims], dep_scope: Scope
    ) -> list[dict[str, str]]:
        """The dep_scope instance keys a downstream instance's members fall under.

        Validated by ``relation_to`` (raises on disjoint/coarse-on-coarse), then
        resolved by well-set membership — never by parsing keys.
        """
        down_scope.relation_to(dep_scope)   # raises if no containment relation
        seen: dict[str, dict[str, str]] = {}
        for d in dims_list:
            if not dep_scope.matches(d):
                continue
            key = dep_scope.scope_key_dict(d)
            seen.setdefault(Scope.encode_path(key), key)
        return list(seen.values())

    # ------------------------------------------------------------------ #
    # Topological order of aggregate tasks (deps that are agg tasks first)
    # ------------------------------------------------------------------ #
    def _topo_order_aggregates(self) -> list:
        by_name = {t.task_name: t for t in self.agg_tasks}
        order: list = []
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(name: str) -> None:
            if name in done or name not in by_name:
                return
            if name in visiting:
                raise ValueError(f"Cycle in aggregate task dependencies at {name!r}.")
            visiting.add(name)
            for dep in by_name[name].dependencies:
                if dep in by_name:
                    visit(dep)
            visiting.discard(name)
            done.add(name)
            order.append(by_name[name])

        for t in self.agg_tasks:
            visit(t.task_name)
        return order

    # ------------------------------------------------------------------ #
    # The single pass
    # ------------------------------------------------------------------ #
    def _run_pass(self) -> tuple[list, list[InstanceDecision]]:
        """One scheduling pass. Returns (ordered finest rows, aggregate decisions).

        ``about_to_run[task]`` = instance keys where ``task`` is eligible-now OR
        RUNNING — the cross-scope "eligible-now" set, computed finest-first so a
        coarse instance can read it without re-deriving deps.
        """
        about_to_run: dict[str, set[str]] = defaultdict(set)

        # --- Stage A: finest sweep (entry-major, mirrors manager.get_next_task) ---
        finest_rows: list[tuple[ScopedInstance, str, str, bool]] = []
        for inst in self._finest_instances():
            for task_name, rec in inst.tasks.items():
                deps_ok = self._finest_deps_complete(inst, rec)
                finest_rows.append((inst, task_name, rec.status, deps_ok))
                # eligible-now (NOT_RUN + deps complete) OR running → blocks dependents
                if rec.status == TaskStatus.RUNNING or (
                    rec.status == TaskStatus.NOT_RUN and deps_ok
                ):
                    about_to_run[task_name].add(inst.instance_key)

        # --- Stage B: aggregate sweep (topological; deps' about_to_run ready) ---
        decisions: list[InstanceDecision] = []
        for task in self._agg_order:
            for scope_key, dims_list in self._buckets_with_keys(task.scope):
                inst_key = scoped_instance_key(task.scope.name, scope_key)
                existing = self.instances.get(inst_key)
                rec = existing.tasks.get(task.task_name) if existing else None
                if rec is not None and rec.status == TaskStatus.RUNNING:
                    # a mid-run instance is not re-planned, but it blocks dependents
                    about_to_run[task.task_name].add(inst_key)
                    decisions.append(InstanceDecision(
                        task.scope.name, scope_key, task.task_name, RUNNING,
                        n_total=len(dims_list),
                    ))
                    continue
                dec = self._evaluate_aggregate(task, scope_key, dims_list, about_to_run)
                decisions.append(dec)
                if dec.decision in (RUN, RERUN):
                    about_to_run[task.task_name].add(dec.instance_key)
        return finest_rows, decisions

    def _buckets_with_keys(self, scope: Scope) -> list[tuple[dict[str, str], list[WellDims]]]:
        out = []
        for dims_list in self.instances_for(scope).values():
            out.append((scope.scope_key_dict(dims_list[0]), dims_list))
        return out

    def _finest_deps_complete(self, inst: ScopedInstance, rec: TaskRecord) -> bool:
        for dep in rec.dependencies:
            dep_rec = inst.tasks.get(dep)
            if dep_rec is None or dep_rec.status != TaskStatus.COMPLETE:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Aggregate barrier (ports aggregate_scheduler._evaluate + containment)
    # ------------------------------------------------------------------ #
    def _evaluate_aggregate(
        self, task, scope_key, dims_list, about_to_run: dict[str, set[str]]
    ) -> InstanceDecision:
        deps = list(task.dependencies)
        primary = deps[0] if deps else None
        per_well_deps = all(not self.registry.is_aggregate_task(dep) for dep in deps)

        has_blocker = False
        complete: list[Member] = []

        if per_well_deps:
            # S1 path: members are the downstream instance's WELLS; a well is usable
            # iff EVERY dependency is COMPLETE-with-output for that well.
            for d in dims_list:
                well_key = self.finest_scope.scope_key_dict(d)
                well_inst_key = scoped_instance_key(self.finest_scope.name, well_key)
                well_inst = self.instances.get(well_inst_key)
                if any(well_inst_key in about_to_run.get(dep, set()) for dep in deps):
                    has_blocker = True
                    continue
                if self._all_deps_usable(well_inst, deps):
                    prec = well_inst.tasks.get(primary) if well_inst else None
                    complete.append(Member(
                        dims=d,
                        upstream_status=TaskStatus.COMPLETE,
                        upstream_output_path=(prec.output_path if prec else None),
                        upstream_last_updated=(prec.last_updated if prec else None),
                    ))
                # else FAILED / cascade NOT_RUN / COMPLETE-no-output → hole
            n_total = len(dims_list)
        else:
            # Aggregate→aggregate: members are the PRIMARY dependency's instances.
            if len(deps) != 1:
                raise ValueError(
                    f"Aggregate task {task.task_name!r} with an aggregate dependency "
                    "must declare exactly one dependency."
                )
            dep_scope = self.registry.scope_of_task(primary)
            dep_keys = self.resolve_dependency_keys(task.scope, dims_list, dep_scope)
            for dep_key in dep_keys:
                dep_inst_key = scoped_instance_key(dep_scope.name, dep_key)
                dep_inst = self.instances.get(dep_inst_key)
                if dep_inst_key in about_to_run.get(primary, set()):
                    has_blocker = True
                    continue
                drec = dep_inst.tasks.get(primary) if dep_inst else None
                if drec is not None and drec.status == TaskStatus.COMPLETE \
                        and drec.output_path is not None:
                    complete.append(Member(
                        dims=self._representative_dims(dep_scope, dep_key, dims_list),
                        upstream_status=TaskStatus.COMPLETE,
                        upstream_output_path=drec.output_path,
                        upstream_last_updated=drec.last_updated,
                        scope_key=dep_key,
                        scope_name=dep_scope.name,
                    ))
                # else RUNNING already caught as blocker; FAILED / NOT_RUN /
                # COMPLETE-without-output (e.g. SKIP_EMPTY upstream) → hole
            n_total = len(dep_keys)

        if has_blocker:
            return InstanceDecision(
                task.scope.name, scope_key, task.task_name, WAIT, n_total=n_total
            )

        fp = _member_fingerprint(complete)
        cfg = self.cm.get_task_params(task.task_name)
        existing = self.instances.get(
            scoped_instance_key(task.scope.name, scope_key)
        )
        existing_rec = existing.tasks.get(task.task_name) if existing else None
        up_to_date = (
            existing_rec is not None
            and existing_rec.status == TaskStatus.COMPLETE
            and existing_rec.member_hash == fp
            and existing_rec.config == cfg
        )
        if not complete:
            decision = UPTODATE if up_to_date else SKIP_EMPTY
        elif up_to_date:
            decision = UPTODATE
        else:
            decision = RERUN if (
                existing_rec and existing_rec.status == TaskStatus.COMPLETE
            ) else RUN
        return InstanceDecision(
            task.scope.name, scope_key, task.task_name, decision,
            members_complete=complete, n_total=n_total, member_hash=fp,
        )

    def _all_deps_usable(self, well_inst: ScopedInstance | None, deps: list[str]) -> bool:
        if well_inst is None:
            return False
        for dep in deps:
            rec = well_inst.tasks.get(dep)
            if rec is None or rec.status != TaskStatus.COMPLETE or rec.output_path is None:
                return False
        return True

    def _representative_dims(
        self, dep_scope: Scope, dep_key: dict[str, str], dims_list: list[WellDims]
    ) -> WellDims:
        for d in dims_list:
            if dep_scope.matches(d) and dep_scope.scope_key_dict(d) == dep_key:
                return d
        return dims_list[0]

    # ------------------------------------------------------------------ #
    # Public views over one pass
    # ------------------------------------------------------------------ #
    def finest_queue(
        self,
        n: int = 1,
        retry_failed: bool = False,
        recording_keys=None,
        task_names=None,
    ) -> list[WorkUnit]:
        """Eligible finest units in legacy ``get_next_task`` order (entry-major)."""
        eligible = (
            {TaskStatus.NOT_RUN, TaskStatus.FAILED} if retry_failed
            else {TaskStatus.NOT_RUN}
        )
        rk_filter = set(recording_keys) if recording_keys is not None else None
        tn_filter = set(task_names) if task_names is not None else None
        finest_rows, _ = self._run_pass()
        out: list[WorkUnit] = []
        for inst, task_name, status, deps_ok in finest_rows:
            if len(out) >= n:
                break
            if rk_filter is not None and inst.scope_key["recording_key"] not in rk_filter:
                continue
            if tn_filter is not None and task_name not in tn_filter:
                continue
            if status not in eligible or not deps_ok:
                continue
            out.append(WorkUnit(inst.scope_name, dict(inst.scope_key), task_name))
        return out

    def aggregate_plan(self, task_names=None) -> list[InstanceDecision]:
        allow = set(task_names) if task_names else None
        _, decisions = self._run_pass()
        if allow is None:
            return decisions
        return [d for d in decisions if d.task_name in allow]

    # ------------------------------------------------------------------ #
    # Registration + status writes (unified store)
    # ------------------------------------------------------------------ #
    def _ensure_instance(self, scope_name: str, scope_key: dict[str, str]) -> ScopedInstance:
        key = scoped_instance_key(scope_name, scope_key)
        inst = self.instances.get(key)
        if inst is None:
            inst = ScopedInstance(scope_name, dict(scope_key), {})
            self.instances[key] = inst
        return inst

    def _save(self) -> None:
        self.store.save(self.instances)
        self._dims = None   # membership may have changed

    def register_finest(self, recording_key, compound_well_id, well_task_specs) -> None:
        """Register a finest (per-well) instance — the ``add_well`` replacement.

        ``well_task_specs`` is an iterable of ``(task_name, dependencies)`` in
        pipeline order (all initialized NOT_RUN). Idempotent: existing task
        records are left untouched.
        """
        if "/" in compound_well_id:
            rec_name, well_id = compound_well_id.split("/", 1)
        else:
            rec_name, well_id = "", compound_well_id
        scope_key = {"recording_key": recording_key, "rec_name": rec_name, "well_id": well_id}
        key = scoped_instance_key(self.finest_scope.name, scope_key)
        inst = self.instances.get(key)
        if inst is None:
            # stamp created_at so the pipeline_cache.json projection round-trips
            # (matches legacy add_well, which sets created_at at queue time).
            inst = ScopedInstance(
                self.finest_scope.name, dict(scope_key), {}, created_at=time.time()
            )
            self.instances[key] = inst
        changed = False
        for task_name, deps in well_task_specs:
            if task_name not in inst.tasks:
                inst.tasks[task_name] = TaskRecord(
                    TaskStatus.NOT_RUN, list(deps), None, None, None
                )
                changed = True
        if changed:
            self._save()

    def update_status(
        self, scope_name, scope_key, task_name, status, *,
        output_path=None, error=None, config=None,
        member_hash=None, n_members_total=None, n_members_complete=None,
        dependencies=None,
    ) -> None:
        """Mutate one task record in place (mirrors ``PipelineManager.update_status``).

        Preserves the record's ``dependencies`` unless overridden, so finest
        dependency resolution keeps working after a status flip. Persists the store.
        """
        inst = self._ensure_instance(scope_name, scope_key)
        rec = inst.tasks.get(task_name)
        if rec is None:
            rec = TaskRecord(status, list(dependencies or []), None, None, None)
            inst.tasks[task_name] = rec
        rec.status = status
        rec.last_updated = time.time()   # mirrors PipelineManager.update_status;
        # load-bearing: the aggregate re-run fingerprint hashes upstream last_updated,
        # and the dashboard inspector shows it. Leaving it None disables re-runs.
        rec.error = error if status == TaskStatus.FAILED else None
        if status == TaskStatus.COMPLETE and output_path is not None:
            rec.output_path = output_path
        if config is not None:
            rec.config = config
        if member_hash is not None:
            rec.member_hash = member_hash
        if n_members_total is not None:
            rec.n_members_total = n_members_total
        if n_members_complete is not None:
            rec.n_members_complete = n_members_complete
        if dependencies is not None:
            rec.dependencies = list(dependencies)
        self._save()

    def recover_from_crash(self) -> int:
        """Reset every RUNNING task back to NOT_RUN (a prior run died mid-task)."""
        n = 0
        for inst in self.instances.values():
            for rec in inst.tasks.values():
                if rec.status == TaskStatus.RUNNING:
                    rec.status = TaskStatus.NOT_RUN
                    rec.error = None
                    n += 1
        if n:
            self._save()
        return n
