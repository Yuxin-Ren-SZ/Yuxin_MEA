from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkItem:
    """Returned by PipelineManager.get_next_task().

    Use recording_key with DatasetManager to retrieve the RecordingEntry
    (data path, metadata) needed to actually run the computation.
    """

    recording_key: str  # matches RecordingEntry.cache_key
    well_id:       str
    task_name:     str
