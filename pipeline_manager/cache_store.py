from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from .pipeline_entry import PipelineEntry
from .stage_record import StageRecord, StageStatus

PIPELINE_CACHE_FILENAME = "pipeline_cache.json"

# Keys that identify a serialised StageRecord dict
_STAGE_KEYS = {"status", "dependencies", "output_path", "last_updated", "config", "error"}

# Keys that identify a serialised PipelineEntry dict
_ENTRY_KEYS = {"recording_key", "well_id", "created_at", "stages"}


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BasePipelineCacheStore(ABC):
    @abstractmethod
    def load(self) -> dict[str, PipelineEntry]:
        """Return all cached entries keyed by pipeline_key. Empty dict if no cache."""

    @abstractmethod
    def save(self, entries: dict[str, PipelineEntry]) -> None:
        """Persist entries, replacing any existing cache."""


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------

class _PipelineEntryEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# JSON decoder (object_hook — called bottom-up, inner objects first)
# ---------------------------------------------------------------------------

def _pipeline_decoder(d: dict):
    if _STAGE_KEYS <= d.keys():
        op = d["output_path"]
        return StageRecord(
            status=d["status"],
            dependencies=d["dependencies"],
            output_path=Path(op) if op is not None else None,
            last_updated=d["last_updated"],
            config=d["config"],
            error=d["error"],
        )

    if _ENTRY_KEYS <= d.keys():
        # d["stages"] is already dict[str, StageRecord] — decoded by the hook above
        return PipelineEntry(
            recording_key=d["recording_key"],
            well_id=d["well_id"],
            created_at=float(d["created_at"]),
            stages=d["stages"],
        )

    return d


# ---------------------------------------------------------------------------
# JSON implementation
# ---------------------------------------------------------------------------

def _entry_to_dict(entry: PipelineEntry) -> dict:
    return {
        "recording_key": entry.recording_key,
        "well_id":       entry.well_id,
        "created_at":    entry.created_at,
        "stages": {
            name: {
                "status":       s.status,
                "dependencies": s.dependencies,
                "output_path":  str(s.output_path) if s.output_path is not None else None,
                "last_updated": s.last_updated,
                "config":       s.config,
                "error":        s.error,
            }
            for name, s in entry.stages.items()
        },
    }


class JsonPipelineCacheStore(BasePipelineCacheStore):
    """Stores the pipeline cache as JSON at analysis_dir/pipeline_cache.json.

    Writes are atomic (temp file + os.replace) to prevent partial-write corruption.
    """

    def __init__(self, analysis_dir: Path) -> None:
        self._path = analysis_dir / PIPELINE_CACHE_FILENAME

    def load(self) -> dict[str, PipelineEntry]:
        if not self._path.exists():
            return {}
        with self._path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh, object_hook=_pipeline_decoder)
        return {k: v for k, v in raw.items() if isinstance(v, PipelineEntry)}

    def save(self, entries: dict[str, PipelineEntry]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: _entry_to_dict(entry) for key, entry in entries.items()}
        fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent, prefix=".pipeline_tmp_", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, cls=_PipelineEntryEncoder)
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
