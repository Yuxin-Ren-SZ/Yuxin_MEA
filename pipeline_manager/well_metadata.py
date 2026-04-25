from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseWellMetadataProvider(ABC):
    """Returns per-well metadata for a given (recording, well) pair.

    Future implementation will read mxassay.metadata from the run directory.
    Swap in the real provider to populate metadata without changing PipelineManager.
    """

    @abstractmethod
    def get_metadata(self, recording_key: str, well_id: str) -> dict[str, Any]:
        """Return metadata dict for this well (chip ID, assay type, etc.)."""


class DummyWellMetadataProvider(BaseWellMetadataProvider):
    """Placeholder — returns {} until mxassay.metadata parsing is implemented."""

    def get_metadata(self, recording_key: str, well_id: str) -> dict[str, Any]:
        return {}
