from __future__ import annotations

from dataclasses import dataclass, field

from .stage_record import StageRecord


@dataclass
class PipelineEntry:
    recording_key: str                      # matches RecordingEntry.cache_key (h5 file UID)
    well_id:       str                      # e.g. "well000"
    created_at:    float                    # POSIX timestamp of first creation
    stages:        dict[str, StageRecord]   # open dict; grows as stages are registered

    @property
    def pipeline_key(self) -> str:
        """Unique cache key for this (recording, well) pair."""
        return f"{self.recording_key}/{self.well_id}"
