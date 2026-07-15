"""Durable provenance sidecar written next to each task's output.

The stamp lives in two places (see ``doc/caching.md``): a fast copy in
``pipeline_cache.json`` (``TaskRecord.provenance``) and this durable
``provenance.json`` inside the task's output directory. The sidecar travels with
the artifact and survives a pipeline-cache reset / rebuild / crash-recovery,
which is what makes the record a reproducibility guarantee rather than mere
cache bookkeeping.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# NOT "provenance.json" — spikeinterface already writes that in sorter/extractor
# output dirs. Namespaced to avoid colliding with it (and anything else).
PROVENANCE_FILENAME = "yuxin_provenance.json"
_SCHEMA = "yuxin_mea.provenance/1"  # marker so a foreign json is never misread as ours


def sidecar_dir(output_path: Path | str) -> Path:
    """Directory the sidecar belongs in for a task's returned ``output_path``.

    Tasks return either a per-well **directory** (e.g. ``auto_curation`` /
    ``burst_detection`` / ``ml_burst_detection`` / a ``.zarr`` store) or a
    per-well **file** inside such a directory. The sidecar must live in the
    per-well dir either way — using ``.parent`` unconditionally would, for
    directory-returning tasks, place it in the *recording-level* dir shared by
    all wells, so every well would clobber the same ``provenance.json``. Write
    and verify both route through here so they always agree.
    """
    p = Path(output_path)
    return p if p.is_dir() else p.parent


def write_sidecar(output_dir: Path | str, stamp: dict[str, Any]) -> Path:
    """Atomically write ``<output_dir>/provenance.json`` (tmp + ``os.replace``).

    Mirrors ``JsonCacheStore.save``'s atomic-write discipline so a crashed write
    never leaves a half-written sidecar. Returns the path written. Best-effort:
    the caller decides whether a failure here should surface (it should not sink
    a completed task).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / PROVENANCE_FILENAME
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".provenance_tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"_schema": _SCHEMA, **stamp}, fh, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def read_sidecar(output_dir: Path | str) -> dict[str, Any] | None:
    """Read our sidecar; ``None`` if missing/unreadable/not ours.

    Validates the ``_schema`` marker so a foreign ``json`` that happens to share
    the name (or a future format change) is never mistaken for a stamp.
    """
    target = Path(output_dir) / PROVENANCE_FILENAME
    try:
        with open(target, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("_schema") != _SCHEMA:
        return None
    return data
