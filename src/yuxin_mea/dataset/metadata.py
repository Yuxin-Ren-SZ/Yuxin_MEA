from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._mxassay_decoder import decode_mxassay_metadata

# --------------------------------------------------------------------------- #
# Assay tag -> hours since the last media change
# --------------------------------------------------------------------------- #
#: ``Run #000019`` — stripped before any number is read, or a bare-number tag
#: like ``"Run #000022 24"`` would yield the run number instead of the interval.
#: The ``#`` is what identifies a run id, so ``Run`` is optional and a lone
#: ``"#000048"`` is still recognised as a run id and not read as 48 hours; a tag
#: that is just ``"24"``, with no ``#``, keeps its number.
_RUN_PREFIX = re.compile(r"^\s*(?:run\s*)?#\s*\d+\s*", re.I)
#: ``24H`` / ``0.5 h`` / ``15H``. The ``H`` is **mandatory** — an interval is only
#: an interval when it is labelled as one, so a bare number anywhere in the tag is
#: not read as hours. The unit may sit mid-tag (``"24H Post Drug"``), and the
#: negative lookahead keeps ``H`` a unit rather than the first letter of a word.
_HOURS = re.compile(r"(\d+(?:\.\d+)?)\s*[Hh](?![A-Za-z])")
_MINUTES = re.compile(r"(\d+(?:\.\d+)?)\s*min", re.I)
_IMMEDIATE = re.compile(r"\bimm\b", re.I)


def parse_hours_since_media(tag: str | None) -> float | None:
    """Hours between the last media change and this recording, from its assay tag.

    The tag is free text a human typed at the scanner, so the real inventory
    looks like ``"Run #000007 24H"``, ``"Run #000009 Imm 0.5H"``,
    ``"Run #000019 Using 015 as base 15H"``, ``"Run #000064 24H Post Drug"``,
    ``"Run #000054 5 min post drug. using 052 conf"`` and
    ``"Run #000017 Old_config"``.

    An hour is only read where it is **labelled** ``H``; the label may sit
    anywhere in the tag, not just at the end. A bare number is never an interval
    — ``"Run #000022 24"`` records no unit, and guessing hours from it would put
    a number in the column that nobody wrote down.

    Returns ``None`` — never ``0.0`` — when the tag records no interval at all.
    "Nobody wrote down when the media was changed" and "recorded at the media
    change" are different facts, and collapsing them would silently admit
    untimed recordings into an acute-timepoint window.
    """
    body = _RUN_PREFIX.sub("", str(tag or "")).strip()
    if not body:
        return None
    m = _MINUTES.search(body)
    if m:
        return float(m.group(1)) / 60.0
    hits = _HOURS.findall(body)
    if hits:
        # Last match wins: in "NU Imm 1.5H" the explicit number has to beat the
        # bare "Imm" below, and "Using 015 as base 15H" must not read the 015.
        return float(hits[-1])
    return 0.0 if _IMMEDIATE.search(body) else None

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

    Swap in a subclass without touching DatasetManager.
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
