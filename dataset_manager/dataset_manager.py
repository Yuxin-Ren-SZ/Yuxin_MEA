from __future__ import annotations

import dataclasses
import logging
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .cache_store import BaseCacheStore, JsonCacheStore
from .metadata_extractor import BaseMetadataExtractor, MxassayMetadataExtractor, RecordingMetadata
from .recording_entry import RecordingEntry, WellEntry

logger = logging.getLogger(__name__)

_DATE_PATTERN = re.compile(r"^\d{6}$")

# Operator dispatch for get_by()
_OPS: dict[str, Any] = {
    "==":          lambda a, b: a == b,
    "!=":          lambda a, b: a != b,
    "<":           lambda a, b: a < b,
    "<=":          lambda a, b: a <= b,
    ">":           lambda a, b: a > b,
    ">=":          lambda a, b: a >= b,
    "contain":     lambda a, b: b in a,
    "not contain": lambda a, b: b not in a,
}

_VALID_KEYS = {f.name for f in dataclasses.fields(RecordingEntry)}


class DatasetManager:
    """Discovers and caches MEA recordings under a data root directory.

    Supports two data_root layouts:
      - Root level:   data_root / SampleID / Date / PlateID / ScanType / RunID
      - Sample level: data_root / Date / PlateID / ScanType / RunID
                      (SampleID is inferred from data_root.name)

    On each startup only the Date-level directories are checked against the cache;
    deeper scanning only runs for newly discovered (SampleID, Date) pairs.
    """

    def __init__(
        self,
        data_root:          Path,
        analysis_dir:       Path,
        max_workers:        int | None = None,
        cache_store:        BaseCacheStore | None = None,
        metadata_extractor: BaseMetadataExtractor | None = None,
    ) -> None:
        self._data_root          = Path(data_root)
        self._analysis_dir       = Path(analysis_dir)
        self._max_workers        = max_workers
        self._store              = cache_store or JsonCacheStore(self._analysis_dir)
        self._metadata_extractor = metadata_extractor or MxassayMetadataExtractor()
        self._cache: dict[str, RecordingEntry] = {}

        self._initialise()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def recordings(self) -> list[RecordingEntry]:
        return list(self._cache.values())

    def get_recording_by(
        self,
        filters: list[tuple[str, str, Any]],
    ) -> list[RecordingEntry]:
        """Filter recordings by recording fields and well metadata.

        Args:
            filters: Conditions in (key, method, value) order. Recording keys
                must be RecordingEntry fields. Well metadata keys use
                'wells.<metadata_key>' and match when the same well satisfies
                all well filters.

        Raises:
            ValueError: If a recording key is not valid or method is unknown.
        """
        recording_filters: list[tuple[str, str, Any]] = []
        well_filters: list[tuple[str, str, Any]] = []

        for key, method, value in filters:
            if method not in _OPS:
                raise ValueError(
                    f"Unknown method {method!r}. Valid methods: {sorted(_OPS)}"
                )

            if key.startswith("wells."):
                metadata_key = key.removeprefix("wells.")
                if not metadata_key:
                    raise ValueError(
                        "Unknown key 'wells.'. Expected 'wells.<metadata_key>'."
                    )
                well_filters.append((metadata_key, method, value))
            else:
                if key not in _VALID_KEYS:
                    raise ValueError(
                        f"Unknown key {key!r}. Valid keys: {sorted(_VALID_KEYS)}"
                    )
                recording_filters.append((key, method, value))

        return [
            entry
            for entry in self._cache.values()
            if self._matches_recording_filters(entry, recording_filters)
            and self._matches_well_filters(entry, well_filters)
        ]

    def get_by(self, key: str, method: Any, value: Any) -> list[RecordingEntry]:
        """Deprecated compatibility wrapper for get_recording_by().
        Args:
            key:    A field name of RecordingEntry (e.g. 'scan_type', 'file_size').
            method: One of '==', '!=', '<', '<=', '>', '>=', 'contain', 'not contain'.
            value:  The value to compare against.

        Raises:
            ValueError: If key is not a valid RecordingEntry field or method is unknown.
        """
        warnings.warn(
            "DatasetManager.get_by() is deprecated; use get_recording_by() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(method, str) and method in _OPS:
            return self.get_recording_by([(key, method, value)])
        if isinstance(value, str) and value in _OPS:
            return self.get_recording_by([(key, value, method)])
        return self.get_recording_by([(key, method, value)])

    # ------------------------------------------------------------------
    # Well management
    # ------------------------------------------------------------------

    def get_wells(self, recording_key: str) -> dict[str, WellEntry]:
        """Return the wells dict for a recording, or {} if the key is unknown."""
        entry = self._cache.get(recording_key)
        return entry.wells if entry is not None else {}

    def get_path(self, entry: RecordingEntry) -> Path:
        """Return the absolute path to entry's data file under the current data root."""
        return self._data_root / entry.data_path

    def register_well(
        self,
        recording_key: str,
        well_id: str,
        metadata: dict | None = None,
    ) -> WellEntry:
        """Add or update a well under a recording entry and persist the cache.

        If the well already exists its metadata is merged (new keys override
        existing ones; existing keys not in metadata are preserved).

        Args:
            recording_key: Must match an existing RecordingEntry.cache_key.
            well_id:        e.g. "well000".
            metadata:       Per-well data (groups, density, assay type, …).
                            Defaults to {}.

        Raises:
            KeyError: If recording_key is not found in the cache.
        """
        entry = self._cache.get(recording_key)
        if entry is None:
            raise KeyError(
                f"Recording {recording_key!r} not found in cache. "
                "Ensure DatasetManager has scanned the data root first."
            )
        incoming = metadata or {}
        if well_id in entry.wells:
            entry.wells[well_id].metadata.update(incoming)
        else:
            entry.wells[well_id] = WellEntry(well_id=well_id, metadata=dict(incoming))

        self._store.save(self._cache)
        logger.debug("Registered well %s under %s.", well_id, recording_key)
        return entry.wells[well_id]

    def update_well_metadata(
        self,
        recording_key: str,
        well_id: str,
        metadata: dict,
    ) -> None:
        """Merge new metadata into an existing well and persist.

        Raises:
            KeyError: If recording_key or well_id is not found in the cache.
        """
        entry = self._cache.get(recording_key)
        if entry is None:
            raise KeyError(f"Recording {recording_key!r} not found in cache.")
        if well_id not in entry.wells:
            raise KeyError(
                f"Well {well_id!r} not registered under {recording_key!r}. "
                "Call register_well() first."
            )
        entry.wells[well_id].metadata.update(metadata)
        self._store.save(self._cache)
        logger.debug("Updated metadata for well %s/%s.", recording_key, well_id)

    def _matches_recording_filters(
        self,
        entry: RecordingEntry,
        filters: list[tuple[str, str, Any]],
    ) -> bool:
        return all(
            _OPS[method](getattr(entry, key), value)
            for key, method, value in filters
        )

    def _matches_well_filters(
        self,
        entry: RecordingEntry,
        filters: list[tuple[str, str, Any]],
    ) -> bool:
        if not filters:
            return True

        for well in entry.wells.values():
            if all(
                metadata_key in well.metadata
                and _OPS[method](well.metadata[metadata_key], value)
                for metadata_key, method, value in filters
            ):
                return True
        return False

    def refresh(self) -> None:
        """Clear the cache and re-scan all directories from scratch."""
        logger.info("Refreshing cache — full rescan of %s", self._data_root)
        self._cache.clear()
        self._scan_all()
        self._store.save(self._cache)
        logger.info("Refresh complete. %d recordings cached.", len(self._cache))

    # ------------------------------------------------------------------
    # Internal initialisation
    # ------------------------------------------------------------------

    def _initialise(self) -> None:
        self._cache = self._store.load()
        logger.info(
            "Loaded %d entries from cache. Checking for new Date directories...",
            len(self._cache),
        )

        root_level = self._detect_root_level()
        disk_date_keys = self._collect_disk_date_keys(root_level)
        cached_date_keys = {
            (e.sample_id, e.date) for e in self._cache.values()
        }

        missing = cached_date_keys - disk_date_keys
        for sample_id, date in sorted(missing):
            logger.warning(
                "Cached Date directory no longer on disk: %s/%s", sample_id, date
            )

        new_keys = disk_date_keys - cached_date_keys
        if not new_keys:
            logger.info("No new Date directories found.")
            return

        logger.info("Found %d new Date director(ies). Scanning...", len(new_keys))
        new_entries = self._scan_date_keys(new_keys, root_level)

        for entry in new_entries:
            self._cache[entry.cache_key] = entry

        self._store.save(self._cache)
        logger.info(
            "Scan complete. Added %d new recordings (%d total).",
            len(new_entries),
            len(self._cache),
        )

    def _scan_all(self) -> None:
        """Full rescan used by refresh()."""
        root_level = self._detect_root_level()
        all_date_keys = self._collect_disk_date_keys(root_level)
        entries = self._scan_date_keys(all_date_keys, root_level)
        for entry in entries:
            self._cache[entry.cache_key] = entry

    # ------------------------------------------------------------------
    # Root-level detection
    # ------------------------------------------------------------------

    def _detect_root_level(self) -> str:
        """Return 'sample' if data_root's children are Date dirs, else 'root'."""
        try:
            children = [p.name for p in self._data_root.iterdir() if p.is_dir()]
        except OSError as exc:
            logger.error("Cannot list data_root %s: %s", self._data_root, exc)
            return "root"

        for name in children[:10]:  # sample first few to avoid scanning entire NAS
            if _DATE_PATTERN.match(name):
                logger.debug(
                    "Detected sample-level data_root (SampleID = %s)",
                    self._data_root.name,
                )
                return "sample"
        return "root"

    # ------------------------------------------------------------------
    # Date-key collection
    # ------------------------------------------------------------------

    def _collect_disk_date_keys(self, root_level: str) -> set[tuple[str, str]]:
        """Return (sample_id, date) pairs visible on disk."""
        keys: set[tuple[str, str]] = set()

        if root_level == "sample":
            sample_id = self._data_root.name
            for date_dir in self._iter_dirs(self._data_root):
                if _DATE_PATTERN.match(date_dir.name):
                    keys.add((sample_id, date_dir.name))
        else:
            for sample_dir in self._iter_dirs(self._data_root):
                for date_dir in self._iter_dirs(sample_dir):
                    if _DATE_PATTERN.match(date_dir.name):
                        keys.add((sample_dir.name, date_dir.name))

        return keys

    # ------------------------------------------------------------------
    # Threaded scanning
    # ------------------------------------------------------------------

    def _scan_date_keys(
        self,
        date_keys: set[tuple[str, str]],
        root_level: str,
    ) -> list[RecordingEntry]:
        """Scan each (sample_id, date) pair concurrently and return all entries."""
        results: list[RecordingEntry] = []

        def make_date_dir(sample_id: str, date: str) -> Path:
            if root_level == "sample":
                return self._data_root / date
            return self._data_root / sample_id / date

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(
                    self._scan_date_dir,
                    sample_id,
                    make_date_dir(sample_id, date),
                    root_level,
                ): (sample_id, date)
                for sample_id, date in date_keys
            }
            for future in as_completed(futures):
                sample_id, date = futures[future]
                try:
                    entries = future.result()
                    results.extend(entries)
                    logger.debug(
                        "Scanned %s/%s — %d recording(s) found.",
                        sample_id, date, len(entries),
                    )
                except Exception as exc:
                    logger.error(
                        "Error scanning %s/%s: %s", sample_id, date, exc, exc_info=True
                    )

        return results

    def _scan_date_dir(
        self,
        sample_id: str,
        date_dir: Path,
        root_level: str,
    ) -> list[RecordingEntry]:
        """Walk a Date directory and return one RecordingEntry per valid run."""
        entries: list[RecordingEntry] = []
        discovered_at = time.time()
        sample_id_override = sample_id if root_level == "sample" else None

        for plate_dir in self._iter_dirs(date_dir):
            for scan_dir in self._iter_dirs(plate_dir):
                for run_dir in self._iter_dirs(scan_dir):
                    data_file = run_dir / "data.raw.h5"
                    if not data_file.is_file():
                        continue
                    try:
                        entry = RecordingEntry.from_path(
                            data_path=data_file,
                            data_root=self._data_root,
                            sample_id_override=sample_id_override,
                            discovered_at=discovered_at,
                        )
                        self._populate_h5_structure(entry, data_file)
                        self._populate_metadata(entry, run_dir)
                        entries.append(entry)
                    except (ValueError, OSError) as exc:
                        logger.warning(
                            "Skipping %s: %s", data_file, exc
                        )

        return entries

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _populate_h5_structure(self, entry: RecordingEntry, data_path: Path) -> None:
        """Open data.raw.h5 with h5py and populate entry.h5_recordings and entry.wells.

        h5_recordings is set to {rec_name: [well_ids]} — the authoritative source of
        which (rec_name, well_id) pairs exist in the file.  Any well_id found here but
        absent from entry.wells is added with an empty metadata dict so downstream code
        can always use h5_recordings as the iteration source.
        """
        try:
            import h5py
        except ImportError:
            logger.warning("h5py not available; skipping h5 structure discovery for %s", data_path)
            return

        try:
            with h5py.File(data_path, "r") as h5f:
                recs_group = h5f.get("recordings")
                if recs_group is None:
                    logger.warning("No 'recordings' group in %s", data_path)
                    return
                structure = {
                    rec_name: sorted(recs_group[rec_name].keys())
                    for rec_name in sorted(recs_group.keys())
                }
        except Exception as exc:
            logger.warning("Could not read h5 structure from %s: %s", data_path, exc)
            return

        entry.h5_recordings.update(structure)

        # Ensure every well_id that appears in the h5 file has a WellEntry so
        # callers can always iterate entry.wells for biological metadata.
        for well_ids in structure.values():
            for well_id in well_ids:
                if well_id not in entry.wells:
                    entry.wells[well_id] = WellEntry(well_id=well_id)

    def _populate_metadata(self, entry: RecordingEntry, run_dir: Path) -> None:
        """Call the metadata extractor and merge results into entry.metadata and entry.wells.

        Recording-level fields go into entry.metadata; per-well fields into entry.wells.
        Logs a warning if the extractor raises, but does not abort the scan.
        """
        metadata_path = run_dir / "mxassay.metadata"
        try:
            rec_meta: RecordingMetadata = self._metadata_extractor.get(metadata_path)
        except Exception as exc:
            logger.warning("Metadata extraction failed for %s: %s", metadata_path, exc)
            return

        entry.metadata.update(rec_meta.fields)

        for wm in rec_meta.wells:
            if wm.well_id in entry.wells:
                entry.wells[wm.well_id].metadata.update(wm.fields)
            else:
                entry.wells[wm.well_id] = WellEntry(well_id=wm.well_id, metadata=dict(wm.fields))

    @staticmethod
    def _iter_dirs(parent: Path):
        """Yield immediate subdirectories of parent, ignoring permission errors."""
        try:
            for p in parent.iterdir():
                if p.is_dir():
                    yield p
        except OSError as exc:
            logger.warning("Cannot list directory %s: %s", parent, exc)
