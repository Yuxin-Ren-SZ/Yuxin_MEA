from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from .pipeline_entry import PipelineEntry
from .task_record import TaskRecord, TaskStatus

PIPELINE_CACHE_FILENAME = "pipeline_cache.json"

_TASK_KEYS  = {"status", "dependencies", "output_path", "last_updated", "error", "config"}
_ENTRY_KEYS = {"recording_key", "well_id", "created_at", "tasks"}


class BasePipelineCacheStore(ABC):
    @abstractmethod
    def load(self) -> dict[str, PipelineEntry]:
        """Return all cached entries keyed by pipeline_key. Empty dict if missing."""

    @abstractmethod
    def save(self, entries: dict[str, PipelineEntry]) -> None:
        """Persist entries, replacing any existing cache."""


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _decode(d: dict):
    if _TASK_KEYS <= d.keys():
        op = d["output_path"]
        return TaskRecord(
            status=d["status"],
            dependencies=d["dependencies"],
            output_path=Path(op) if op is not None else None,
            last_updated=d["last_updated"],
            error=d["error"],
            config=d.get("config", {}),
        )
    if _ENTRY_KEYS <= d.keys():
        return PipelineEntry(
            recording_key=d["recording_key"],
            well_id=d["well_id"],
            created_at=float(d["created_at"]),
            tasks=d["tasks"],
        )
    return d


def _entry_to_dict(entry: PipelineEntry) -> dict:
    return {
        "recording_key": entry.recording_key,
        "well_id":       entry.well_id,
        "created_at":    entry.created_at,
        "tasks": {
            name: {
                "status":       t.status,
                "dependencies": t.dependencies,
                "output_path":  str(t.output_path) if t.output_path is not None else None,
                "last_updated": t.last_updated,
                "error":        t.error,
                "config":       t.config,
            }
            for name, t in entry.tasks.items()
        },
    }


class JsonPipelineCacheStore(BasePipelineCacheStore):
    """Stores the pipeline cache as JSON with atomic writes (tempfile + os.replace)."""

    def __init__(self, analysis_dir: Path) -> None:
        self._path = Path(analysis_dir) / PIPELINE_CACHE_FILENAME

    def load(self) -> dict[str, PipelineEntry]:
        if not self._path.exists():
            return {}
        with self._path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh, object_hook=_decode)
        return {k: v for k, v in raw.items() if isinstance(v, PipelineEntry)}

    def save(self, entries: dict[str, PipelineEntry]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: _entry_to_dict(e) for key, e in entries.items()}
        fd, tmp = tempfile.mkstemp(
            dir=self._path.parent, prefix=".pipeline_tmp_", suffix=".json"
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
