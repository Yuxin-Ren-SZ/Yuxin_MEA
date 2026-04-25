from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

# Relative to a root-level data_root: SampleID/Date/PlateID/ScanType/RunID
_ROOT_REGEX = re.compile(
    r"^(?P<SampleID>[^/]+)"
    r"/(?P<Date>\d{6})"
    r"/(?P<PlateID>[^/]+)"
    r"/(?P<ScanType>[^/]+)"
    r"/(?P<RunID>\d+)$"
)

# Relative to a sample-level data_root: Date/PlateID/ScanType/RunID
_SAMPLE_REGEX = re.compile(
    r"^(?P<Date>\d{6})"
    r"/(?P<PlateID>[^/]+)"
    r"/(?P<ScanType>[^/]+)"
    r"/(?P<RunID>\d+)$"
)


@dataclass(frozen=True)
class RecordingEntry:
    sample_id:     str
    date:          str    # 6-digit string, e.g. "240415"
    plate_id:      str
    scan_type:     str
    run_id:        str    # kept as str to preserve leading zeros
    data_path:     Path   # absolute path to data.raw.h5
    file_size:     int    # bytes
    mtime:         float  # POSIX timestamp of data.raw.h5
    discovered_at: float  # POSIX timestamp when first scanned by the manager

    @property
    def cache_key(self) -> str:
        return f"{self.sample_id}/{self.date}/{self.plate_id}/{self.scan_type}/{self.run_id}"

    @classmethod
    def from_path(
        cls,
        data_path: Path,
        data_root: Path,
        sample_id_override: str | None = None,
        discovered_at: float | None = None,
    ) -> "RecordingEntry":
        """Build a RecordingEntry by parsing data_path relative to data_root.

        Args:
            data_path: Absolute path to data.raw.h5.
            data_root: Absolute path to the data root (root or sample level).
            sample_id_override: When data_root is at sample level, pass the
                directory name here so the SampleID field is populated correctly.
            discovered_at: Override discovery timestamp (defaults to now).

        Raises:
            ValueError: If the relative path does not match the expected structure.
        """
        run_dir = data_path.parent
        rel = run_dir.relative_to(data_root).as_posix()

        if sample_id_override is not None:
            m = _SAMPLE_REGEX.match(rel)
            if m is None:
                raise ValueError(
                    f"Path does not match sample-level pattern: {rel!r}"
                )
            g = m.groupdict()
            sample_id = sample_id_override
        else:
            m = _ROOT_REGEX.match(rel)
            if m is None:
                raise ValueError(
                    f"Path does not match root-level pattern: {rel!r}"
                )
            g = m.groupdict()
            sample_id = g["SampleID"]

        stat = os.stat(data_path)
        return cls(
            sample_id=sample_id,
            date=g["Date"],
            plate_id=g["PlateID"],
            scan_type=g["ScanType"],
            run_id=g["RunID"],
            data_path=data_path,
            file_size=stat.st_size,
            mtime=stat.st_mtime,
            discovered_at=discovered_at if discovered_at is not None else time.time(),
        )
