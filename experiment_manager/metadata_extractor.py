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
    """All per-well fields from a single mxassay.metadata well entry.

    `fields` contains every key/value from the decoded well record:
    MaxWell slot fields (groupname, control, well_name, concentration, …)
    plus all user-defined annotation properties (Plating Date, Density, …).
    Keys vary per project — no assumption is made about which fields exist.
    """
    well_id: str
    fields:  dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordingMetadata:
    """All metadata from a single mxassay.metadata file.

    `fields` is a flat merge of the [properties] and [runtime] sections —
    recording-level fields that describe the whole run (project_title, chipid,
    started, runid, tag, …). Keys vary by MaxWell software version.

    `wells` is the per-well breakdown for selected wells only.
    """
    fields: dict[str, Any]     = field(default_factory=dict)
    wells:  list[WellMetadata] = field(default_factory=list)


class BaseMetadataExtractor(ABC):
    """Parses mxassay.metadata and returns both recording- and well-level metadata.

    Swap in a subclass without touching ExperimentManager.
    """

    @abstractmethod
    def get(self, metadata_path: Path) -> RecordingMetadata:
        """Parse metadata_path and return a RecordingMetadata.

        Args:
            metadata_path: Absolute path to the mxassay.metadata file.
                           Return RecordingMetadata(fields={}, wells=[]) for
                           missing files rather than raising.
        """


def _well_to_fields(well: dict[str, Any]) -> dict[str, Any]:
    """Flatten a decoded well dict into a single metadata fields dict."""
    fields = {k: v for k, v in well.items() if k not in _DECODER_INTERNAL_KEYS}
    fields.update(well.get("annotations", {}))
    return fields


class MxassayMetadataExtractor(BaseMetadataExtractor):
    """Reads a MaxWell/MaxLab mxassay.metadata file.

    Recording-level fields: merged [properties] + [runtime] sections.
    Well-level fields: per selected well, slot fields + annotation properties.
    Returns RecordingMetadata(fields={}, wells=[]) when the file does not exist.
    """

    def get(self, metadata_path: Path) -> RecordingMetadata:
        if not metadata_path.exists():
            return RecordingMetadata()

        meta = decode_mxassay_metadata(metadata_path, add_iso_times=True)

        # Recording-level: merge properties + runtime (ISO fields and
        # actual_runtime_seconds included — computed from file data).
        recording_fields: dict[str, Any] = {
            **meta.get("properties", {}),
            **meta.get("runtime", {}),
        }

        # Well-level: selected wells only.
        wells_section = meta.get("wells")
        wells: list[WellMetadata] = []
        if isinstance(wells_section, dict):
            wells_data: dict[int, Any] = wells_section.get("wells", {})
            selected_ids: list[int] = wells_section.get("selected_wells") or list(wells_data)
            wells = [
                WellMetadata(
                    well_id=f"well{int(wid):03d}",
                    fields=_well_to_fields(well),
                )
                for wid in selected_ids
                if (well := wells_data.get(int(wid))) is not None
            ]

        return RecordingMetadata(fields=recording_fields, wells=wells)


class DummyMetadataExtractor(BaseMetadataExtractor):
    """Placeholder — ignores metadata_path and returns fixed dummy data.

    Provides plausible recording- and well-level fields for offline development.
    """

    _DUMMY_RECORDING_FIELDS: dict[str, Any] = {
        "runid":   "000000",
        "tag":     "dummy run",
        "chipid":  "DUMMY",
        "progress": 100,
    }

    _DUMMY_WELLS: list[WellMetadata] = [
        WellMetadata("well000", {"groupname": "control",   "density": 10_000.0}),
        WellMetadata("well001", {"groupname": "treatment", "density": 10_000.0}),
        WellMetadata("well002", {"groupname": "control",   "density": 15_000.0}),
        WellMetadata("well003", {"groupname": "treatment", "density": 15_000.0}),
        WellMetadata("well004", {"groupname": "control",   "density": 20_000.0}),
        WellMetadata("well005", {"groupname": "treatment", "density": 20_000.0}),
    ]

    def get(self, metadata_path: Path) -> RecordingMetadata:
        return RecordingMetadata(
            fields=dict(self._DUMMY_RECORDING_FIELDS),
            wells=list(self._DUMMY_WELLS),
        )
