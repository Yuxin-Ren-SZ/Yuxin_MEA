from __future__ import annotations

from dataclasses import dataclass

from .task_record import TaskRecord


@dataclass
class PipelineEntry:
    recording_key: str                     # matches RecordingEntry.cache_key
    well_id:       str                     # e.g. "well000"
    created_at:    float                   # POSIX timestamp of first creation
    tasks:         dict[str, TaskRecord]   # task_name → TaskRecord

    @property
    def pipeline_key(self) -> str:
        return f"{self.recording_key}/{self.well_id}"
