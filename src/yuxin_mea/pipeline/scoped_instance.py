"""ScopedInstance: one task-instance at any scope (S2's single instance model).

S2 collapses the per-well ``PipelineEntry`` and the aggregate
``AggregateInstanceRecord`` into ONE shape: an instance is a ``(scope_name,
scope_key)`` identity holding a ``{task_name: TaskRecord}`` dict — exactly the
per-well ``PipelineEntry`` shape, generalized to any scope. A per-well entry is
the finest instance (``scope_name="well"``); its ``scope_key`` is
``{recording_key, rec_name, well_id}`` and its tasks are the per-well tasks.

The instance stores the scope *name* (a string), never a :class:`Scope` object —
a ``Scope``'s ``where`` is a lambda and is not JSON-serializable; the name
rehydrates to a ``Scope`` via a registry at scheduling time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .scope import Scope
from .task_record import TaskRecord


def scoped_instance_key(scope_name: str, scope_key: dict[str, str]) -> str:
    """Stable key for one instance: ``<scope_name>::<encoded_scope_key>``.

    Distinct from :func:`aggregate_cache.instance_key` (which is task-keyed); this
    is the unified store's key — one instance holds many tasks.
    """
    return f"{scope_name}::{Scope.encode_path(scope_key)}"


@dataclass
class ScopedInstance:
    scope_name: str                        # "well" for the finest; "recording"/"sample"/…
    scope_key: dict[str, str]              # {dimension: value} identity
    tasks: dict[str, TaskRecord] = field(default_factory=dict)  # task_name → record
    created_at: float | None = None

    @property
    def instance_key(self) -> str:
        return scoped_instance_key(self.scope_name, self.scope_key)
