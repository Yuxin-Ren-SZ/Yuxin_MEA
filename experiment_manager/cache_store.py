from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from .recording_entry import RecordingEntry, WellEntry

CACHE_FILENAME = "experiment_cache.json"

# Sentinel key sets used by the object_hook to identify each record type.
# WellEntry is checked first because its keys are a strict subset; checking
# RecordingEntry first could accidentally match a future WellEntry extension.
_WELL_KEYS  = {"well_id", "metadata"}
_ENTRY_KEYS = {
    "sample_id", "date", "plate_id", "scan_type", "run_id",
    "data_path", "file_size", "mtime", "discovered_at", "metadata", "wells",
}


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


def _recording_entry_decoder(d: dict) -> RecordingEntry | WellEntry | dict:
    """object_hook: reconstruct typed objects from their serialised dicts.

    object_hook fires bottom-up, so WellEntry dicts inside a recording's
    'wells' field are decoded before the RecordingEntry dict that contains them.
    The outer wells dict (keyed by well IDs, not sentinel keys) passes through
    as a plain dict and is received by the RecordingEntry branch already typed.
    """
    # WellEntry: check before RecordingEntry (fewer keys, unambiguous)
    if _WELL_KEYS <= d.keys() and "sample_id" not in d:
        return WellEntry(
            well_id=d["well_id"],
            metadata=d["metadata"],
        )

    if _ENTRY_KEYS <= d.keys():
        # d["wells"] values are already WellEntry objects, decoded by the hook above
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
            metadata=d["metadata"],
            wells=d["wells"],
        )

    return d


# ---------------------------------------------------------------------------
# JSON implementation
# ---------------------------------------------------------------------------

class JsonCacheStore(BaseCacheStore):
    """Stores the recording cache as a JSON file at analysis_dir/experiment_cache.json.

    Cache hierarchy (recording-level fields stored once; wells nested underneath):
      recording_key → { sample_id, date, …, wells: { well_id → { well_id, metadata } } }

    Writes are atomic (temp file + os.replace) to prevent partial-write corruption.
    """

    def __init__(self, analysis_dir: Path) -> None:
        self._path = analysis_dir / CACHE_FILENAME

    def load(self) -> dict[str, RecordingEntry]:
        if not self._path.exists():
            return {}
        with self._path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh, object_hook=_recording_entry_decoder)
        return {k: v for k, v in raw.items() if isinstance(v, RecordingEntry)}

    def save(self, entries: dict[str, RecordingEntry]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: _entry_to_dict(entry) for key, entry in entries.items()}
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
        "metadata":      entry.metadata,
        "wells": {
            wid: {"well_id": we.well_id, "metadata": we.metadata}
            for wid, we in entry.wells.items()
        },
    }
