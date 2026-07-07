"""Persistent status store for aggregate task instances.

Separate file (``aggregate_cache.json``) and separate schema from the per-well
``pipeline_cache.json`` — the additive S1 rule is that the per-well cache is
never touched. Mirrors ``pipeline/cache.py``'s atomic-write + object_hook
decode pattern.
"""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from .scope import Scope

AGGREGATE_CACHE_FILENAME = "aggregate_cache.json"

_AGG_KEYS = {
    "task_name", "scope_name", "scope_key", "status", "output_path",
    "member_hash", "n_members_total", "n_members_complete", "config",
    "error", "last_updated",
}


@dataclass
class AggregateInstanceRecord:
    task_name: str
    scope_name: str
    scope_key: dict[str, str]
    status: str                       # TaskStatus constant
    output_path: Path | None = None
    member_hash: str = ""             # fingerprint of the COMPLETE subset consumed
    n_members_total: int = 0          # bucket size (terminal members)
    n_members_complete: int = 0
    config: dict = field(default_factory=dict)
    error: str | None = None
    last_updated: float | None = None

    @property
    def instance_key(self) -> str:
        return instance_key(self.task_name, self.scope_key)


def instance_key(task_name: str, scope_key: dict[str, str]) -> str:
    """Stable key for one aggregate instance: ``<task>::<encoded_scope_key>``."""
    return f"{task_name}::{Scope.encode_path(scope_key)}"


class BaseAggregateCacheStore(ABC):
    @abstractmethod
    def load(self) -> dict[str, AggregateInstanceRecord]:
        """Return all instance records keyed by instance_key. Empty if missing."""

    @abstractmethod
    def save(self, records: dict[str, AggregateInstanceRecord]) -> None:
        """Persist records, replacing any existing cache."""


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _decode(d: dict):
    if _AGG_KEYS <= d.keys():
        op = d["output_path"]
        return AggregateInstanceRecord(
            task_name=d["task_name"],
            scope_name=d["scope_name"],
            scope_key=d["scope_key"],
            status=d["status"],
            output_path=Path(op) if op is not None else None,
            member_hash=d.get("member_hash", ""),
            n_members_total=d.get("n_members_total", 0),
            n_members_complete=d.get("n_members_complete", 0),
            config=d.get("config", {}),
            error=d.get("error"),
            last_updated=d.get("last_updated"),
        )
    return d


def _record_to_dict(r: AggregateInstanceRecord) -> dict:
    return {
        "task_name": r.task_name,
        "scope_name": r.scope_name,
        "scope_key": r.scope_key,
        "status": r.status,
        "output_path": str(r.output_path) if r.output_path is not None else None,
        "member_hash": r.member_hash,
        "n_members_total": r.n_members_total,
        "n_members_complete": r.n_members_complete,
        "config": r.config,
        "error": r.error,
        "last_updated": r.last_updated,
    }


class JsonAggregateCacheStore(BaseAggregateCacheStore):
    """Stores aggregate-instance records as JSON with atomic writes."""

    def __init__(self, analysis_dir: Path) -> None:
        self._path = Path(analysis_dir) / AGGREGATE_CACHE_FILENAME

    def load(self) -> dict[str, AggregateInstanceRecord]:
        if not self._path.exists():
            return {}
        with self._path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh, object_hook=_decode)
        return {k: v for k, v in raw.items() if isinstance(v, AggregateInstanceRecord)}

    def save(self, records: dict[str, AggregateInstanceRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: _record_to_dict(r) for key, r in records.items()}
        fd, tmp = tempfile.mkstemp(
            dir=self._path.parent, prefix=".aggregate_tmp_", suffix=".json"
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
