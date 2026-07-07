"""AggregateScheduler: schedules scope-level aggregate tasks (S1, additive).

Reads the per-well :class:`PipelineManager` cache to evaluate barriers; never
mutates it. Owns its own ``aggregate_cache.json`` (via ``JsonAggregateCacheStore``).

Barrier (per instance / bucket, for dependency D):
    * classify each member well's D status:
        COMPLETE                          -> usable member
        FAILED                            -> hole (tolerate)
        eligible-now or RUNNING (blocker) -> the whole instance WAITs
        NOT_RUN & not eligible-now        -> cascade hole (tolerate)
    * any blocker            -> WAIT (re-evaluate next pass)
    * else no COMPLETE member -> SKIP_EMPTY (record complete, 0 members)
    * else                    -> RUN on the COMPLETE subset

"eligible-now" is computed via the public ``get_next_task`` API (no
reimplementation of the intra-entry ``_deps_complete``): a well appears there iff
D's own upstream is complete and D is about to run. Because the CLI runs the
aggregate pass AFTER the per-well drain reaches steady state, blockers are
normally empty and every bucket fires immediately on its complete subset.

Re-run: an already-COMPLETE instance re-runs when its member-set fingerprint
changes (a new recording completed, or a member's dependency re-ran with new
output/timestamp) OR its config changed — mirrors the per-well config-mismatch
re-run.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .aggregate_cache import (
    AggregateInstanceRecord,
    JsonAggregateCacheStore,
    instance_key,
)
from .aggregate_task import BaseAggregateTask
from .groupname import detect_groupname_issues
from .task_record import TaskStatus
from .well_dims import Member, WellDims

logger = logging.getLogger(__name__)

# Plan decisions
RUN = "run"
RERUN = "rerun"
SKIP_EMPTY = "skip_empty"
WAIT = "wait"
UPTODATE = "uptodate"


@dataclass
class PlannedInstance:
    task: BaseAggregateTask
    scope_key: dict[str, str]
    decision: str
    members_complete: list[Member] = field(default_factory=list)
    n_total: int = 0
    member_hash: str = ""

    @property
    def key(self) -> str:
        return instance_key(self.task.task_name, self.scope_key)


def _member_fingerprint(members_complete: list[Member]) -> str:
    items = sorted(
        f"{m.recording_key}|{m.compound_well_id}|{m.upstream_output_path}|{m.upstream_last_updated}"
        for m in members_complete
    )
    return hashlib.sha1("\n".join(items).encode("utf-8")).hexdigest()


class AggregateScheduler:
    def __init__(
        self,
        pipeline_mgr,
        config_mgr,
        experiment_cache: dict,
        agg_task_classes: "tuple[type[BaseAggregateTask], ...]",
        store: JsonAggregateCacheStore,
    ) -> None:
        self.pm = pipeline_mgr
        self.cm = config_mgr
        self.exp = experiment_cache or {}
        self.tasks: list[BaseAggregateTask] = [c() for c in agg_task_classes]
        self._agg_task_names = {t.task_name for t in self.tasks}
        self.store = store
        self._records: dict[str, AggregateInstanceRecord] = store.load()
        self._dims: list[WellDims] | None = None
        self._warned_groupname: set[str] = set()

    # ------------------------------------------------------------------
    # Well universe + dimensions
    # ------------------------------------------------------------------
    def _groupname_lut(self) -> dict[tuple[str, str], str]:
        lut: dict[tuple[str, str], str] = {}
        for rk, rec in self.exp.items():
            for wid, w in (rec.get("wells") or {}).items():
                md = w.get("metadata") or {}
                g = str(md.get("groupname", "?") or "?")
                lut[(rk, str(wid))] = g
                # also key by the well's own well_id field if it differs
                own = w.get("well_id")
                if own is not None:
                    lut[(rk, str(own))] = g
        return lut

    def build_well_dims(self) -> list[WellDims]:
        if self._dims is not None:
            return self._dims
        gl = self._groupname_lut()
        dims: list[WellDims] = []
        for entry in self.pm.entries:
            physical = entry.well_id.split("/")[-1]
            groupname = gl.get((entry.recording_key, physical), "?")
            dims.append(WellDims.from_entry(entry.recording_key, entry.well_id, groupname))
        self._dims = dims
        return dims

    # ------------------------------------------------------------------
    # Bucketing
    # ------------------------------------------------------------------
    def bucket(self, task: BaseAggregateTask) -> dict[tuple, list[WellDims]]:
        buckets: dict[tuple, list[WellDims]] = defaultdict(list)
        for d in self.build_well_dims():
            if task.scope.matches(d):
                buckets[task.scope.well_key(d)].append(d)
        return buckets

    # ------------------------------------------------------------------
    # Barrier
    # ------------------------------------------------------------------
    def _blockers(self, dep_task_name: str) -> set[tuple[str, str]]:
        """Wells for which dependency D is eligible-now or RUNNING (would block)."""
        if dep_task_name in self._agg_task_names:
            raise NotImplementedError(
                f"Aggregate→aggregate dependency ({dep_task_name!r}) is not supported "
                "in S1. Depend on a per-well task, or implement scope-containment "
                "(the primary S2 seam)."
            )
        eligible = self.pm.get_next_task(
            n=10**9, task_names=[dep_task_name], retry_failed=False
        )
        blockers = {(w.recording_key, w.well_id) for w in eligible}
        for e in self.pm.entries:
            rec = e.tasks.get(dep_task_name)
            if rec is not None and rec.status == TaskStatus.RUNNING:
                blockers.add((e.recording_key, e.well_id))
        return blockers

    def _evaluate(
        self,
        task: BaseAggregateTask,
        bucket: list[WellDims],
        blockers: dict[str, set[tuple[str, str]]],
    ) -> PlannedInstance:
        scope_key = task.scope.scope_key_dict(bucket[0])
        primary = task.dependencies[0] if task.dependencies else None

        has_blocker = False
        complete: list[Member] = []
        for d in bucket:
            entry = self.pm.get_entry(d.recording_key, d.compound_well_id)
            key = (d.recording_key, d.compound_well_id)
            # blocker if ANY dependency would still run for this well
            if any(key in blockers.get(dep, set()) for dep in task.dependencies):
                has_blocker = True
                continue
            # member is usable only if EVERY dependency is COMPLETE
            all_complete = True
            for dep in task.dependencies:
                rec = entry.tasks.get(dep) if entry else None
                if rec is None or rec.status != TaskStatus.COMPLETE:
                    all_complete = False
                    break
            if all_complete:
                prec = entry.tasks.get(primary) if (entry and primary) else None
                complete.append(Member(
                    dims=d,
                    upstream_status=TaskStatus.COMPLETE,
                    upstream_output_path=(prec.output_path if prec else None),
                    upstream_last_updated=(prec.last_updated if prec else None),
                ))
            # else: FAILED / cascade NOT_RUN -> hole (tolerated)

        if has_blocker:
            return PlannedInstance(task, scope_key, WAIT, n_total=len(bucket))

        fp = _member_fingerprint(complete)
        cfg = self.cm.get_task_params(task.task_name)
        existing = self._records.get(instance_key(task.task_name, scope_key))
        up_to_date = (
            existing is not None
            and existing.status == TaskStatus.COMPLETE
            and existing.member_hash == fp
            and existing.config == cfg
        )
        if not complete:
            decision = UPTODATE if up_to_date else SKIP_EMPTY
        elif up_to_date:
            decision = UPTODATE
        else:
            decision = RERUN if (existing and existing.status == TaskStatus.COMPLETE) else RUN
        return PlannedInstance(
            task, scope_key, decision,
            members_complete=complete, n_total=len(bucket), member_hash=fp,
        )

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    def plan(self, task_names: list[str] | None = None) -> list[PlannedInstance]:
        allow = set(task_names) if task_names else None
        out: list[PlannedInstance] = []
        for task in self.tasks:
            if allow is not None and task.task_name not in allow:
                continue
            if "groupname" in task.scope.group_by and task.task_name not in self._warned_groupname:
                for line in detect_groupname_issues(self.build_well_dims()):
                    logger.warning("[%s] %s", task.task_name, line)
                self._warned_groupname.add(task.task_name)
            blockers = {dep: self._blockers(dep) for dep in task.dependencies}
            for bucket in self.bucket(task).values():
                out.append(self._evaluate(task, bucket, blockers))
        return out

    def collision_report(self, task: BaseAggregateTask) -> list[str]:
        if "groupname" not in task.scope.group_by:
            return []
        return detect_groupname_issues(self.build_well_dims())

    # ------------------------------------------------------------------
    # Status writes (own cache only)
    # ------------------------------------------------------------------
    def _put(self, rec: AggregateInstanceRecord) -> None:
        rec.last_updated = time.time()
        self._records[rec.instance_key] = rec
        self.store.save(self._records)

    def mark_running(self, task: BaseAggregateTask, scope_key: dict[str, str]) -> None:
        self._put(AggregateInstanceRecord(
            task_name=task.task_name, scope_name=task.scope.name, scope_key=scope_key,
            status=TaskStatus.RUNNING, config=self.cm.get_task_params(task.task_name),
        ))

    def mark_complete(
        self, task, scope_key, output_path, member_hash, n_total, n_complete
    ) -> None:
        self._put(AggregateInstanceRecord(
            task_name=task.task_name, scope_name=task.scope.name, scope_key=scope_key,
            status=TaskStatus.COMPLETE, output_path=output_path, member_hash=member_hash,
            n_members_total=n_total, n_members_complete=n_complete,
            config=self.cm.get_task_params(task.task_name),
        ))

    def mark_failed(self, task, scope_key, error, member_hash, n_total, n_complete) -> None:
        self._put(AggregateInstanceRecord(
            task_name=task.task_name, scope_name=task.scope.name, scope_key=scope_key,
            status=TaskStatus.FAILED, error=error, member_hash=member_hash,
            n_members_total=n_total, n_members_complete=n_complete,
            config=self.cm.get_task_params(task.task_name),
        ))

    def mark_skipped_empty(self, task, scope_key, member_hash) -> None:
        """Record a bucket with zero complete members as COMPLETE/0 so it does not
        churn every pass (re-evaluated only when membership changes)."""
        self._put(AggregateInstanceRecord(
            task_name=task.task_name, scope_name=task.scope.name, scope_key=scope_key,
            status=TaskStatus.COMPLETE, output_path=None, member_hash=member_hash,
            n_members_total=0, n_members_complete=0,
            config=self.cm.get_task_params(task.task_name),
        ))
