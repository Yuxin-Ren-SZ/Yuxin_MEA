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
from yuxin_mea.pipeline.task_record import TaskStatus


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


def load_recordings_detail(
    analysis_root: Path,
) -> tuple[list[dict], dict[str, dict[str, str]]]:
    """Return (recordings, well_pipeline_status) for the Datasets master/detail view.

    recordings — one dict per RecordingEntry:
        sample_id, date, plate_id, scan_type, run_id, cache_key,
        data_path (str), file_size_mb (int), n_wells (int),
        wells (list[str]): compound well IDs "{rec_name}/{well_id}"

    well_pipeline_status — keyed by pipeline_key "{cache_key}/{rec}/{well_id}":
        {task_name: status_string}
    """
    rec_store = JsonCacheStore(analysis_root)
    entries = rec_store.load()

    recordings: list[dict] = []
    for e in entries.values():
        wells: list[str] = [
            f"{rec_name}/{well_id}"
            for rec_name, well_ids in e.h5_recordings.items()
            for well_id in well_ids
        ]
        recordings.append(
            {
                "sample_id": e.sample_id,
                "date": e.date,
                "plate_id": e.plate_id,
                "scan_type": e.scan_type,
                "run_id": e.run_id,
                "cache_key": e.cache_key,
                "data_path": str(e.data_path),
                "file_size_mb": e.file_size // (1024 * 1024),
                "n_wells": len(wells),
                "wells": wells,
            }
        )
    recordings.sort(key=lambda r: (r["sample_id"], r["date"], r["run_id"]))

    pipe_store = JsonPipelineCacheStore(analysis_root)
    pipe_entries = pipe_store.load()
    well_pipeline_status: dict[str, dict[str, str]] = {
        key: {tn: tr.status for tn, tr in entry.tasks.items()}
        for key, entry in pipe_entries.items()
    }

    return recordings, well_pipeline_status


def filter_recordings(
    recordings: list[dict],
    well_pipeline_status: dict[str, dict],
    *,
    scan_types: list[str] | None = None,
    dates: list[str] | None = None,
    queue_status: str = "all",
) -> list[dict]:
    """Filter recordings for the Datasets page.  Pure function, no Dash deps."""
    out = recordings
    if scan_types:
        out = [r for r in out if r["scan_type"] in scan_types]
    if dates:
        out = [r for r in out if r["date"] in dates]
    if queue_status == "queued":
        out = [r for r in out if any(
            k.startswith(r["cache_key"] + "/") for k in well_pipeline_status
        )]
    elif queue_status == "not_queued":
        out = [r for r in out if not any(
            k.startswith(r["cache_key"] + "/") for k in well_pipeline_status
        )]
    return out


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
