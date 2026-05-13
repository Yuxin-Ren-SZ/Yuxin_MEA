"""Pure cache loaders for dashboard pages.

These functions use the underlying `JsonCacheStore` and
`JsonPipelineCacheStore` directly rather than `DatasetManager` /
`PipelineManager`. Instantiating the managers triggers `_initialise()` /
`_reset_stale_tasks()` respectively, which mutate the cache state — that
would violate the dashboard's strictly-read-only contract.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from yuxin_mea.dataset.cache import JsonCacheStore
from yuxin_mea.pipeline.cache import JsonPipelineCacheStore


_RECORDING_COLUMNS = [
    "sample_id",
    "date",
    "plate_id",
    "scan_type",
    "run_id",
    "n_wells",
    "n_recs",
    "file_size_mb",
    "mtime",
    "cache_key",
]


def load_recordings_df(analysis_root: Path) -> pd.DataFrame:
    """Read `experiment_cache.json` into a row-per-recording DataFrame.

    Returns an empty DataFrame with the expected columns when the cache is
    missing or empty, so downstream pages can render an honest empty state
    without special-casing.
    """
    store = JsonCacheStore(analysis_root)
    entries = store.load()
    rows = [
        {
            "sample_id": e.sample_id,
            "date": e.date,
            "plate_id": e.plate_id,
            "scan_type": e.scan_type,
            "run_id": e.run_id,
            "n_wells": len(e.wells),
            "n_recs": len(e.h5_recordings),
            "file_size_mb": e.file_size // (1024 * 1024),
            "mtime": datetime.fromtimestamp(e.mtime).isoformat(timespec="seconds"),
            "cache_key": e.cache_key,
        }
        for e in entries.values()
    ]
    if not rows:
        return pd.DataFrame(columns=_RECORDING_COLUMNS)
    return pd.DataFrame(rows, columns=_RECORDING_COLUMNS).sort_values(
        ["sample_id", "date", "run_id"]
    )


def load_pipeline_df(analysis_root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read `pipeline_cache.json` and pivot to a (recording, well) × task matrix.

    Returns `(df, task_names)`. `task_names` is the sorted union of task
    names seen across all entries — pages use it to attach conditional
    formatting to each task column. Cells where a task is not present on a
    given entry are filled with the em-dash `—`.
    """
    store = JsonPipelineCacheStore(analysis_root)
    entries = store.load()
    if not entries:
        return pd.DataFrame(columns=["recording_key", "well_id"]), []

    task_names = sorted({t for e in entries.values() for t in e.tasks})
    rows = []
    for e in entries.values():
        row = {"recording_key": e.recording_key, "well_id": e.well_id}
        for tn in task_names:
            tr = e.tasks.get(tn)
            row[tn] = tr.status if tr else "—"
        rows.append(row)
    df = pd.DataFrame(rows, columns=["recording_key", "well_id", *task_names])
    return df.sort_values(["recording_key", "well_id"]), task_names
