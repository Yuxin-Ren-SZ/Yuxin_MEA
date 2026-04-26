from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._mxassay_decoder import decode_mxassay_metadata

# Keys added by the decoder that are not metadata fields from the file itself.
_DECODER_INTERNAL_KEYS = frozenset({
    "id",               # used as well_id identifier, not a metadata field
    "annotations",      # dict — flattened into fields below
    "annotations_raw",  # list form of annotations — redundant
    "annotation_size",  # internal counter
    "groupcolor",       # QColor binary blob — not analysis metadata
})


@dataclass
class WellMetadata:
    """All per-well fields from a single mxassay.metadata entry.

    `fields` contains every key/value from the decoded well record:
    MaxWell slot fields (groupname, control, well_name, concentration, …)
    plus all user-defined annotation properties (Plating Date, Density, …).
    Keys vary per project — no assumption is made about which fields exist.
    """
    well_id: str
    fields:  dict[str, Any] = field(default_factory=dict)


class BaseMetadataExtractor(ABC):
    """Parses mxassay.metadata and returns per-well metadata.

    Swap in a subclass without touching ExperimentManager.
    """

    @abstractmethod
    def get(self, metadata_path: Path) -> list[WellMetadata]:
        """Return metadata for selected wells in metadata_path.

        Args:
            metadata_path: Absolute path to the mxassay.metadata file.
                           Implementations should return [] for missing files.
        """


def _well_to_fields(well: dict[str, Any]) -> dict[str, Any]:
    """Flatten a decoded well dict into a single metadata fields dict."""
    fields = {k: v for k, v in well.items() if k not in _DECODER_INTERNAL_KEYS}
    fields.update(well.get("annotations", {}))
    return fields


class MxassayMetadataExtractor(BaseMetadataExtractor):
    """Reads a MaxWell/MaxLab mxassay.metadata file and returns per-well metadata.

    Only selected wells (those actually recorded) are returned. Returns []
    when the file does not exist so scans continue in dev environments.
    """

    def get(self, metadata_path: Path) -> list[WellMetadata]:
        if not metadata_path.exists():
            return []

        meta = decode_mxassay_metadata(metadata_path, add_iso_times=False)
        wells_section = meta.get("wells")
        if not isinstance(wells_section, dict):
            return []

        wells_data: dict[int, Any] = wells_section.get("wells", {})
        selected_ids: list[int] = wells_section.get("selected_wells") or list(wells_data)

        return [
            WellMetadata(
                well_id=f"well{int(wid):03d}",
                fields=_well_to_fields(well),
            )
            for wid in selected_ids
            if (well := wells_data.get(int(wid))) is not None
        ]


class DummyMetadataExtractor(BaseMetadataExtractor):
    """Placeholder — ignores metadata_path and returns fixed dummy wells.

    Provides two groups at three density levels for offline development.
    """

    _DUMMY_WELLS: list[WellMetadata] = [
        WellMetadata("well000", {"groupname": "control",   "density": 10_000.0}),
        WellMetadata("well001", {"groupname": "treatment", "density": 10_000.0}),
        WellMetadata("well002", {"groupname": "control",   "density": 15_000.0}),
        WellMetadata("well003", {"groupname": "treatment", "density": 15_000.0}),
        WellMetadata("well004", {"groupname": "control",   "density": 20_000.0}),
        WellMetadata("well005", {"groupname": "treatment", "density": 20_000.0}),
    ]

    def get(self, metadata_path: Path) -> list[WellMetadata]:
        return list(self._DUMMY_WELLS)
