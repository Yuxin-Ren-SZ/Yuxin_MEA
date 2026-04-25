from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from .recording_entry import RecordingEntry

CACHE_FILENAME = "experiment_cache.json"


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseCacheStore(ABC):
    @abstractmethod
    def load(self) -> dict[str, RecordingEntry]:
        """Return all cached entries keyed by cache_key. Empty dict if no cache."""

    @abstractmethod
    def save(self, entries: dict[str, RecordingEntry]) -> None:
        """Persist entries, replacing any existing cache."""


# ---------------------------------------------------------------------------
# JSON encoder / decoder
# ---------------------------------------------------------------------------

class _RecordingEntryEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _recording_entry_decoder(d: dict) -> RecordingEntry | dict:
    """object_hook: rebuild a RecordingEntry when all expected keys are present."""
    _ENTRY_KEYS = {
        "sample_id", "date", "plate_id", "scan_type", "run_id",
        "data_path", "file_size", "mtime", "discovered_at",
    }
    if _ENTRY_KEYS <= d.keys():
        return RecordingEntry(
            sample_id=d["sample_id"],
            date=d["date"],
            plate_id=d["plate_id"],
            scan_type=d["scan_type"],
            run_id=d["run_id"],
            data_path=Path(d["data_path"]),
            file_size=int(d["file_size"]),
            mtime=float(d["mtime"]),
            discovered_at=float(d["discovered_at"]),
        )
    return d


# ---------------------------------------------------------------------------
# JSON implementation
# ---------------------------------------------------------------------------

class JsonCacheStore(BaseCacheStore):
    """Stores the recording cache as a JSON file at analysis_dir/experiment_cache.json.

    Writes are atomic (temp file + os.replace) to prevent partial-write corruption,
    which matters when the analysis dir lives on a NAS.
    """

    def __init__(self, analysis_dir: Path) -> None:
        self._path = analysis_dir / CACHE_FILENAME

    def load(self) -> dict[str, RecordingEntry]:
        if not self._path.exists():
            return {}
        with self._path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh, object_hook=_recording_entry_decoder)
        # raw is a dict[str, RecordingEntry] after the object_hook runs on values
        return {k: v for k, v in raw.items() if isinstance(v, RecordingEntry)}

    def save(self, entries: dict[str, RecordingEntry]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: _entry_to_dict(entry) for key, entry in entries.items()}
        # Atomic write: write to temp file in same dir, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent, prefix=".cache_tmp_", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, cls=_RecordingEntryEncoder)
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def _entry_to_dict(entry: RecordingEntry) -> dict:
    return {
        "sample_id":     entry.sample_id,
        "date":          entry.date,
        "plate_id":      entry.plate_id,
        "scan_type":     entry.scan_type,
        "run_id":        entry.run_id,
        "data_path":     str(entry.data_path),
        "file_size":     entry.file_size,
        "mtime":         entry.mtime,
        "discovered_at": entry.discovered_at,
    }
