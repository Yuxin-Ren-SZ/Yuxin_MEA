"""InstanceStore: the S2 instance store (a ``{task_name: TaskRecord}`` dict per scope).

Mirrors ``pipeline/cache.py``'s atomic-write + object_hook pattern, but persists
:class:`ScopedInstance` records instead of per-well ``PipelineEntry`` records.

Two implementations:

* :class:`JsonInstanceStore` — one JSON file, all scopes. Used for the aggregate
  layer under :class:`LayeredInstanceStore`, and standalone in unit tests.
* :class:`LayeredInstanceStore` — the production store: the FINEST layer lives in
  the dashboard-native ``pipeline_cache.json`` (so the dashboard + scripts read/write
  it unchanged), the aggregate layer in ``instance_store.json``. The unified scheduler
  is oblivious — it still calls ``store.load()`` / ``store.save(instances)``.

The migration loaders (``load_legacy_wells`` / ``load_legacy_aggregates``) and the
projection (``project_wells_to_entries``) map between ``ScopedInstance``s and the
legacy on-disk shapes.
"""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from .aggregate_cache import AggregateInstanceRecord, JsonAggregateCacheStore
from .cache import JsonPipelineCacheStore, _decode as _decode_task  # reuse TaskRecord decoding
from .pipeline_entry import PipelineEntry
from .scoped_instance import ScopedInstance, scoped_instance_key
from .task_record import TaskRecord

FINEST_SCOPE_NAME = "well"

INSTANCE_STORE_FILENAME = "instance_store.json"

_INSTANCE_KEYS = {"scope_name", "scope_key", "tasks"}


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _decode(d: dict):
    # object_hook fires bottom-up: inner task dicts decode to TaskRecord first,
    # then the enclosing instance dict decodes here.
    decoded = _decode_task(d)
    if isinstance(decoded, TaskRecord):
        return decoded
    if _INSTANCE_KEYS <= d.keys():
        return ScopedInstance(
            scope_name=d["scope_name"],
            scope_key=d["scope_key"],
            tasks=d["tasks"],
            created_at=d.get("created_at"),
        )
    return d


def _instance_to_dict(inst: ScopedInstance) -> dict:
    return {
        "scope_name": inst.scope_name,
        "scope_key": inst.scope_key,
        "created_at": inst.created_at,
        "tasks": {
            name: {
                "status":       t.status,
                "dependencies": t.dependencies,
                "output_path":  str(t.output_path) if t.output_path is not None else None,
                "last_updated": t.last_updated,
                "error":        t.error,
                "config":       t.config,
                "member_hash":        t.member_hash,
                "n_members_total":    t.n_members_total,
                "n_members_complete": t.n_members_complete,
            }
            for name, t in inst.tasks.items()
        },
    }


class InstanceStore(ABC):
    @abstractmethod
    def load(self) -> dict[str, ScopedInstance]:
        """Return all instances keyed by instance_key. Empty dict if missing."""

    @abstractmethod
    def save(self, instances: dict[str, ScopedInstance]) -> None:
        """Persist instances, replacing any existing store."""


class JsonInstanceStore(InstanceStore):
    """Stores the unified instance set as JSON with atomic writes."""

    def __init__(self, analysis_dir: Path) -> None:
        self._path = Path(analysis_dir) / INSTANCE_STORE_FILENAME

    def load(self) -> dict[str, ScopedInstance]:
        if not self._path.exists():
            return {}
        with self._path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh, object_hook=_decode)
        return {k: v for k, v in raw.items() if isinstance(v, ScopedInstance)}

    def save(self, instances: dict[str, ScopedInstance]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: _instance_to_dict(i) for key, i in instances.items()}
        fd, tmp = tempfile.mkstemp(
            dir=self._path.parent, prefix=".instance_tmp_", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, cls=_Encoder)
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


# --------------------------------------------------------------------------- #
# Migration loaders — build ScopedInstances from the two legacy on-disk files.
# --------------------------------------------------------------------------- #

def _split_compound_well_id(compound_well_id: str) -> tuple[str, str]:
    if "/" in compound_well_id:
        rec_name, well_id = compound_well_id.split("/", 1)
        return rec_name, well_id
    return "", compound_well_id


def load_legacy_wells(entries: dict[str, PipelineEntry]) -> dict[str, ScopedInstance]:
    """Map legacy per-well ``PipelineEntry``s → finest ``ScopedInstance``s.

    Iterates ``entries`` in natural (insertion) order so the finest-instance
    ordering matches ``pipeline_cache.json`` — a parity constraint for the
    unified scheduler's ``get_next`` order.
    """
    out: dict[str, ScopedInstance] = {}
    for entry in entries.values():
        rec_name, well_id = _split_compound_well_id(entry.well_id)
        scope_key = {
            "recording_key": entry.recording_key,
            "rec_name": rec_name,
            "well_id": well_id,
        }
        inst = ScopedInstance("well", scope_key, entry.tasks, entry.created_at)
        out[inst.instance_key] = inst
    return out


def load_legacy_aggregates(
    records: dict[str, AggregateInstanceRecord],
) -> dict[str, ScopedInstance]:
    """Map legacy ``AggregateInstanceRecord``s → aggregate ``ScopedInstance``s.

    A post-load *reduction*, not an object_hook: siblings that share
    ``(scope_name, scope_key)`` but differ in ``task_name`` must be grouped into
    one instance's ``tasks`` dict, which an object_hook (one record at a time)
    cannot do.
    """
    out: dict[str, ScopedInstance] = {}
    for r in records.values():
        key = scoped_instance_key(r.scope_name, r.scope_key)
        inst = out.get(key)
        if inst is None:
            inst = ScopedInstance(r.scope_name, dict(r.scope_key), {}, r.last_updated)
            out[key] = inst
        inst.tasks[r.task_name] = TaskRecord(
            status=r.status,
            dependencies=[],
            output_path=r.output_path,
            last_updated=r.last_updated,
            error=r.error,
            config=r.config,
            member_hash=r.member_hash,
            n_members_total=r.n_members_total,
            n_members_complete=r.n_members_complete,
        )
    return out


def project_wells_to_entries(
    instances: dict[str, ScopedInstance],
) -> dict[str, PipelineEntry]:
    """Inverse of :func:`load_legacy_wells`: finest ``ScopedInstance``s → the
    ``pipeline_cache.json`` shape. Only ``scope_name == "well"`` instances project;
    others are ignored. ``TaskRecord``s (incl. the Phase-0 ``member_hash``/
    ``n_members_*`` fields) pass through unchanged and round-trip via ``cache.py``.
    """
    out: dict[str, PipelineEntry] = {}
    for inst in instances.values():
        if inst.scope_name != FINEST_SCOPE_NAME:
            continue
        rec_name = inst.scope_key.get("rec_name", "")
        well_id = inst.scope_key.get("well_id", "")
        compound = f"{rec_name}/{well_id}" if rec_name else well_id
        entry = PipelineEntry(
            recording_key=inst.scope_key["recording_key"],
            well_id=compound,
            created_at=inst.created_at if inst.created_at is not None else 0.0,
            tasks=inst.tasks,
        )
        out[entry.pipeline_key] = entry
    return out


class LayeredInstanceStore(InstanceStore):
    """Production store: FINEST layer ↔ ``pipeline_cache.json`` (dashboard-native),
    AGGREGATE layer ↔ ``instance_store.json``.

    The scheduler is oblivious: ``load()`` merges both layers into one
    ``{instance_key: ScopedInstance}`` dict; ``save()`` splits it back by scope. The
    dashboard + scripts read/write ``pipeline_cache.json`` exactly as before, and the
    unified scheduler sees their edits because it reloads that file each run.
    """

    def __init__(self, analysis_dir: Path) -> None:
        self.dir = Path(analysis_dir)
        self._pipe = JsonPipelineCacheStore(self.dir)
        self._agg = JsonInstanceStore(self.dir)

    def load(self) -> dict[str, ScopedInstance]:
        finest = load_legacy_wells(self._pipe.load())
        agg = {
            k: i for k, i in self._agg.load().items()
            if i.scope_name != FINEST_SCOPE_NAME
        }
        if not agg:
            # first unified run: migrate S1 aggregate state (if any) once.
            legacy = JsonAggregateCacheStore(self.dir).load()
            if legacy:
                agg = load_legacy_aggregates(legacy)
        return {**finest, **agg}

    def save(self, instances: dict[str, ScopedInstance]) -> None:
        finest = {
            k: i for k, i in instances.items() if i.scope_name == FINEST_SCOPE_NAME
        }
        agg = {
            k: i for k, i in instances.items() if i.scope_name != FINEST_SCOPE_NAME
        }
        self._pipe.save(project_wells_to_entries(finest))
        self._agg.save(agg)
