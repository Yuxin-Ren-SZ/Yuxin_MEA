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
        # Distinct biological groups across this recording's wells (a recording
        # spans many wells that may belong to different groups). Read from the
        # per-well `groupname` populated by the metadata extractor at scan time.
        groups = sorted(
            {
                g
                for w in e.wells.values()
                if (g := w.metadata.get("groupname"))
            }
        )
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
                "groups": groups,
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


def recording_provenance(analysis_root: Path, config_path) -> dict[str, str]:
    """Per-recording provenance status for dashboard badges (read-only, cheap).

    Compares each COMPLETE task's stamp against the **cached** raw fingerprint +
    current config (no NAS re-hash). Returns ``{cache_key: status}`` where status
    is OK / RAW-CHANGED / CONFIG-CHANGED / METADATA-CHANGED / UNKNOWN (worst-case
    per recording). Recordings with no verified task are absent. Returns ``{}``
    on any error so a provenance hiccup never breaks the page.
    """
    try:
        from yuxin_mea.config import ConfigManager
        from yuxin_mea.provenance import status_by_recording

        cm = ConfigManager()
        if config_path and Path(config_path).exists():
            cm.load(config_path)
        return status_by_recording(cm, Path(analysis_root))
    except Exception:  # noqa: BLE001 — provenance is advisory, never fatal to the view
        return {}


def well_group_map(analysis_root: Path) -> dict[str, str]:
    """Map each pipeline_key ``"{cache_key}/{rec_name}/{well_id}"`` → groupname.

    Reads the same `experiment_cache.json` as the recordings loaders and keys
    the result to match the Pipeline page's `pipeline_key`
    (``f"{recording_key}/{well_id}"`` where `well_id` already carries the
    `rec_name/` prefix). Wells without a cached groupname are omitted.
    """
    store = JsonCacheStore(analysis_root)
    entries = store.load()
    out: dict[str, str] = {}
    for e in entries.values():
        for rec_name, well_ids in e.h5_recordings.items():
            for wid in well_ids:
                we = e.wells.get(wid)
                if we is None:
                    continue
                group = we.metadata.get("groupname")
                if group is not None:
                    out[f"{e.cache_key}/{rec_name}/{wid}"] = group
    return out


def filter_recordings(
    recordings: list[dict],
    well_pipeline_status: dict[str, dict],
    *,
    sample_ids: list[str] | None = None,
    scan_types: list[str] | None = None,
    dates: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    groups: list[str] | None = None,
    statuses: list[str] | None = None,
    queue_status: str = "all",
) -> list[dict]:
    """Filter recordings for the Datasets page.  Pure function, no Dash deps.

    Recording-level semantics: a recording matches `groups`/`statuses` when
    *any* of its wells qualifies (a recording spans many wells that may belong
    to different groups and be at different pipeline stages). Dates are 6-digit
    ``"YYMMDD"`` strings, so `date_from`/`date_to` compare lexically.
    """
    out = recordings
    if sample_ids:
        out = [r for r in out if r["sample_id"] in sample_ids]
    if scan_types:
        out = [r for r in out if r["scan_type"] in scan_types]
    if dates:
        out = [r for r in out if r["date"] in dates]
    if date_from:
        out = [r for r in out if r["date"] >= date_from]
    if date_to:
        out = [r for r in out if r["date"] <= date_to]
    if groups:
        group_set = set(groups)
        out = [r for r in out if group_set & set(r.get("groups", []))]
    if statuses:
        status_set = set(statuses)
        out = [
            r for r in out
            if any(
                status_set & set(tasks.values())
                for key, tasks in well_pipeline_status.items()
                if key.startswith(r["cache_key"] + "/")
            )
        ]
    if queue_status == "queued":
        out = [r for r in out if any(
            k.startswith(r["cache_key"] + "/") for k in well_pipeline_status
        )]
    elif queue_status == "not_queued":
        out = [r for r in out if not any(
            k.startswith(r["cache_key"] + "/") for k in well_pipeline_status
        )]
    return out


def filter_pipeline_df(
    df: pd.DataFrame,
    *,
    sample_ids: list[str] | None = None,
    scan_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    groups: list[str] | None = None,
    statuses: list[str] | None = None,
    group_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Filter the pipeline matrix DataFrame from `load_pipeline_df`.

    `recording_key` is ``sample_id/date/plate_id/scan_type/run_id`` so the
    sample/date/scan-type facets are derived by splitting it — no extra data
    needed. Group is joined per-row via `group_map` keyed by
    ``f"{recording_key}/{well_id}"``. Status matches when *any* task cell on the
    row is in `statuses`. Returns the filtered DataFrame (unchanged if no
    filters are active).
    """
    if df.empty:
        return df
    task_cols = [c for c in df.columns if c not in ("recording_key", "well_id")]
    keep = pd.Series(True, index=df.index)

    if sample_ids or scan_types or date_from or date_to:
        parts = df["recording_key"].str.split("/", expand=True)
        # parts: 0=sample_id, 1=date, 2=plate_id, 3=scan_type, 4=run_id
        if sample_ids:
            keep &= parts[0].isin(sample_ids)
        if scan_types:
            keep &= parts[3].isin(scan_types)
        if date_from:
            keep &= parts[1] >= date_from
        if date_to:
            keep &= parts[1] <= date_to

    if groups:
        gmap = group_map or {}
        group_set = set(groups)
        row_group = df.apply(
            lambda r: gmap.get(f"{r['recording_key']}/{r['well_id']}"), axis=1
        )
        keep &= row_group.isin(group_set)

    if statuses and task_cols:
        status_set = set(statuses)
        keep &= df[task_cols].apply(
            lambda row: bool(status_set & set(row.values)), axis=1
        )

    return df[keep]


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
