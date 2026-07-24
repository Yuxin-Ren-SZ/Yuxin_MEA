"""Tests for the yuxin_mea.dashboard package.

Phase 2 deliberately avoids end-to-end Dash callback tests (those need
`pytest-dash` / `dash[testing]` browser harness, which is heavy). These tests
cover the CLI, the data loaders, and a smoke-build of the Dash app. No test
calls `app.run()` — that would block.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from yuxin_mea.dashboard import build_app
from yuxin_mea.dashboard.cli import main
from yuxin_mea.dashboard.data import (
    HOURS_UNKNOWN,
    filter_recordings,
    load_pipeline_df,
    load_recordings_df,
)
from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
from yuxin_mea.pipeline.pipeline_entry import PipelineEntry
from yuxin_mea.pipeline.task_record import TaskRecord, TaskStatus


def _write_minimal_config(path: Path, analysis_root: Path) -> None:
    """Write a config that points analysis_root at a real tmpdir."""
    path.write_text(
        json.dumps(
            {
                "global": {
                    "data_root": str(analysis_root / "raw"),
                    "analysis_root": str(analysis_root),
                },
                "tasks": {},
            }
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_warns_on_missing_config_but_proceeds(capsys, monkeypatch):
    """Phase 3: missing config no longer exits — dashboard launches in
    config-only mode after a stderr warning."""
    from unittest.mock import MagicMock

    fake_app = MagicMock()
    monkeypatch.setattr(
        "yuxin_mea.dashboard.app.build_app",
        lambda *a, **kw: fake_app,
    )

    rc = main(["--config", "/definitely/does/not/exist.json", "--port", "0"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "Config file not found" in err
    assert "config-only mode" in err
    fake_app.run.assert_called_once()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def test_build_app_stashes_config_on_server():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = tmp_path / "pipeline_config.json"
        _write_minimal_config(cfg, tmp_path)

        app = build_app(cfg)

        ctx = app.server.config["YUXIN_MEA"]
        assert ctx["config_path"] == cfg
        assert ctx["analysis_root"] == tmp_path
        assert ctx["data_root"] == tmp_path / "raw"


def test_build_app_tolerates_unset_globals():
    """A config with empty globals must still build an app — pages render an empty state."""
    with TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "pipeline_config.json"
        cfg.write_text(json.dumps({"global": {}, "tasks": {}}))

        app = build_app(cfg)
        ctx = app.server.config["YUXIN_MEA"]
        assert ctx["analysis_root"] is None
        assert ctx["data_root"] is None


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------


_RECORDING_COLUMNS = {
    "sample_id", "date", "plate_id", "scan_type", "run_id",
    "n_wells", "n_recs", "file_size_mb", "mtime", "cache_key",
}


def test_load_recordings_df_empty_cache():
    with TemporaryDirectory() as tmp:
        df = load_recordings_df(Path(tmp))
    assert df.empty
    assert set(df.columns) == _RECORDING_COLUMNS


def test_load_pipeline_df_empty_cache():
    with TemporaryDirectory() as tmp:
        df, task_names = load_pipeline_df(Path(tmp))
    assert df.empty
    assert list(df.columns) == ["recording_key", "well_id"]
    assert task_names == []


def test_load_pipeline_df_pivots_status():
    with TemporaryDirectory() as tmp:
        analysis_root = Path(tmp)
        store = JsonPipelineCacheStore(analysis_root)

        entries = {
            "SampleA/240415/PlateX/Network/001/rec0000/well000": PipelineEntry(
                recording_key="SampleA/240415/PlateX/Network/001",
                well_id="rec0000/well000",
                created_at=0.0,
                tasks={
                    "preprocessing": TaskRecord(
                        status=TaskStatus.COMPLETE,
                        dependencies=[],
                        output_path=None,
                        last_updated=None,
                        error=None,
                    ),
                    "sorting": TaskRecord(
                        status=TaskStatus.NOT_RUN,
                        dependencies=["preprocessing"],
                        output_path=None,
                        last_updated=None,
                        error=None,
                    ),
                },
            ),
            "SampleA/240415/PlateX/Network/001/rec0001/well006": PipelineEntry(
                recording_key="SampleA/240415/PlateX/Network/001",
                well_id="rec0001/well006",
                created_at=0.0,
                tasks={
                    "preprocessing": TaskRecord(
                        status=TaskStatus.FAILED,
                        dependencies=[],
                        output_path=None,
                        last_updated=None,
                        error="boom",
                    ),
                    # No sorting task on this entry — cell should render as "—".
                },
            ),
        }
        store.save(entries)

        df, task_names = load_pipeline_df(analysis_root)

    assert task_names == ["preprocessing", "sorting"]
    assert list(df.columns) == ["recording_key", "well_id", "preprocessing", "sorting"]
    assert len(df) == 2
    row0 = df.iloc[0].to_dict()
    row1 = df.iloc[1].to_dict()
    assert row0["well_id"] == "rec0000/well000"
    assert row0["preprocessing"] == "complete"
    assert row0["sorting"] == "not_run"
    assert row1["well_id"] == "rec0001/well006"
    assert row1["preprocessing"] == "failed"
    assert row1["sorting"] == "—"


# ---------------------------------------------------------------------------
# Page modules import cleanly (catches typos before user runs the dashboard)
# ---------------------------------------------------------------------------


def test_page_modules_import():
    from yuxin_mea.dashboard.pages import (  # noqa: F401
        burst_diagnostic, home, pipeline, plate_viewer, recordings,
    )


def test_burst_diagnostic_page_registered():
    """After build_app, dash.page_registry must include /burst-diagnostic."""
    import dash

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = tmp_path / "pipeline_config.json"
        _write_minimal_config(cfg, tmp_path)
        build_app(cfg)  # registers pages as a side-effect of construction

        paths = {p["path"] for p in dash.page_registry.values()}
        assert "/burst-diagnostic" in paths


# ---------------------------------------------------------------------------
# filter_recordings tests
# ---------------------------------------------------------------------------

_SAMPLE_RECS = [
    {"cache_key": "S1/260101/P1/Network/001", "sample_id": "S1",
     "date": "260101", "scan_type": "Network", "wells": []},
    {"cache_key": "S1/260101/P1/ActivityScan/002", "sample_id": "S1",
     "date": "260101", "scan_type": "ActivityScan", "wells": []},
    {"cache_key": "S1/260102/P1/Network/003", "sample_id": "S1",
     "date": "260102", "scan_type": "Network", "wells": []},
]

_SAMPLE_PIPE_STATUS = {
    "S1/260101/P1/Network/001/rec0/well000": {"preprocessing": "complete"},
}


def test_filter_by_scan_type():
    out = filter_recordings(_SAMPLE_RECS, _SAMPLE_PIPE_STATUS, scan_types=["Network"])
    assert len(out) == 2
    assert all(r["scan_type"] == "Network" for r in out)


def test_filter_by_date():
    out = filter_recordings(_SAMPLE_RECS, _SAMPLE_PIPE_STATUS, dates=["260102"])
    assert len(out) == 1
    assert out[0]["cache_key"].endswith("003")


def test_filter_queued():
    out = filter_recordings(_SAMPLE_RECS, _SAMPLE_PIPE_STATUS, queue_status="queued")
    assert len(out) == 1
    assert out[0]["cache_key"].endswith("001")


def test_filter_not_queued():
    out = filter_recordings(_SAMPLE_RECS, _SAMPLE_PIPE_STATUS, queue_status="not_queued")
    assert len(out) == 2
    keys = {r["cache_key"] for r in out}
    assert "S1/260101/P1/Network/001" not in keys


def test_filter_all_returns_everything():
    out = filter_recordings(_SAMPLE_RECS, _SAMPLE_PIPE_STATUS, queue_status="all")
    assert len(out) == 3


# ---------------------------------------------------------------------------
# New filter facets: sample / date range / group / status
# ---------------------------------------------------------------------------

_GROUP_RECS = [
    {"cache_key": "S1/260101/P1/Network/001", "sample_id": "S1",
     "date": "260101", "scan_type": "Network", "wells": [], "groups": ["Control"]},
    {"cache_key": "S2/260103/P1/Network/002", "sample_id": "S2",
     "date": "260103", "scan_type": "Network", "wells": [], "groups": ["IVH", "Control"]},
    {"cache_key": "S2/260105/P1/Network/003", "sample_id": "S2",
     "date": "260105", "scan_type": "Network", "wells": [], "groups": ["NPH"]},
]

_GROUP_STATUS = {
    "S1/260101/P1/Network/001/rec0/well000": {"preprocessing": "complete", "sorting": "running"},
    "S2/260103/P1/Network/002/rec0/well000": {"preprocessing": "failed"},
    "S2/260105/P1/Network/003/rec0/well000": {"preprocessing": "not_run"},
}


def test_filter_by_sample_id():
    out = filter_recordings(_GROUP_RECS, _GROUP_STATUS, sample_ids=["S2"])
    assert {r["cache_key"] for r in out} == {
        "S2/260103/P1/Network/002", "S2/260105/P1/Network/003"
    }


def test_filter_by_date_range():
    out = filter_recordings(_GROUP_RECS, _GROUP_STATUS, date_from="260102", date_to="260104")
    assert len(out) == 1
    assert out[0]["cache_key"].endswith("002")


def test_filter_date_from_only():
    out = filter_recordings(_GROUP_RECS, _GROUP_STATUS, date_from="260104")
    assert {r["date"] for r in out} == {"260105"}


def test_filter_by_group_any_well():
    # A recording matches if ANY of its groups is selected.
    out = filter_recordings(_GROUP_RECS, _GROUP_STATUS, groups=["Control"])
    assert {r["cache_key"] for r in out} == {
        "S1/260101/P1/Network/001", "S2/260103/P1/Network/002"
    }


def test_filter_by_status_any_well():
    out = filter_recordings(_GROUP_RECS, _GROUP_STATUS, statuses=["failed"])
    assert len(out) == 1
    assert out[0]["cache_key"].endswith("002")

    out2 = filter_recordings(_GROUP_RECS, _GROUP_STATUS, statuses=["complete", "running"])
    assert out2[0]["cache_key"].endswith("001") and len(out2) == 1


def test_filter_combines_facets():
    out = filter_recordings(
        _GROUP_RECS, _GROUP_STATUS,
        sample_ids=["S2"], groups=["Control"], date_from="260101", date_to="260110",
    )
    assert len(out) == 1
    assert out[0]["cache_key"].endswith("002")


# ---------------------------------------------------------------------------
# Plate viewer: same-sample-together, run_id-ordered recording sort
# ---------------------------------------------------------------------------


def test_resolve_context_existing_and_missing():
    from yuxin_mea.dashboard.app import resolve_context

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = tmp_path / "pipeline_config.json"
        _write_minimal_config(cfg, tmp_path)

        ctx = resolve_context(cfg)
        assert ctx["config_exists"] is True
        assert ctx["analysis_root"] == tmp_path
        assert ctx["data_root"] == tmp_path / "raw"
        # cache_root defaults to a local sibling of analysis_root.
        assert ctx["cache_root"] == tmp_path.parent / "dashboard_cache"

        missing = resolve_context(tmp_path / "nope.json")
        assert missing["config_exists"] is False
        assert missing["analysis_root"] is None
        assert missing["data_root"] is None


def test_resolve_context_raises_on_malformed():
    from yuxin_mea.dashboard.app import resolve_context

    with TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.json"
        bad.write_text("{ this is not json ")
        with pytest.raises(Exception):
            resolve_context(bad)


def test_list_config_entries_dirs_first_json_only():
    from yuxin_mea.dashboard.pages.settings import _list_config_entries

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sub_b").mkdir()
        (root / "sub_a").mkdir()
        (root / "a.json").write_text("{}")
        (root / "b.json").write_text("{}")
        (root / "notes.txt").write_text("ignore me")  # non-json excluded

        entries = _list_config_entries(str(root))
        names = [e["name"] for e in entries]
        # ".." first, then dirs (name-sorted), then .json files (name-sorted).
        assert names[0] == "../"
        assert names[1:] == ["sub_a/", "sub_b/", "a.json", "b.json"]
        assert "notes.txt" not in names
        kinds = {e["name"]: e["kind"] for e in entries}
        assert kinds["sub_a/"] == "dir" and kinds["a.json"] == "file"


def test_plate_viewer_sort_groups_sample_then_run_id():
    from yuxin_mea.dashboard.pages.plate_viewer import _sort_recordings

    # Deliberately scrambled input across samples, run_ids, and dates.
    recs = [
        {"sample_id": "S2", "run_id": "002", "date": "260105", "cache_key": "S2b"},
        {"sample_id": "S1", "run_id": "010", "date": "260101", "cache_key": "S1-r10"},
        {"sample_id": "S1", "run_id": "002", "date": "260109", "cache_key": "S1-r02"},
        {"sample_id": "S2", "run_id": "001", "date": "260103", "cache_key": "S2a"},
        # Same sample+run across two dates → date is the tiebreak.
        {"sample_id": "S1", "run_id": "002", "date": "260101", "cache_key": "S1-r02-early"},
    ]
    order = [r["cache_key"] for r in _sort_recordings(recs)]
    # S1 before S2; within S1 run_id 002 before 010 (leading-zero lexical); within
    # the two run_id=002 rows the earlier date wins.
    assert order == ["S1-r02-early", "S1-r02", "S1-r10", "S2a", "S2b"]


# ---------------------------------------------------------------------------
# filter_pipeline_df
# ---------------------------------------------------------------------------


def _pipe_df():
    import pandas as pd
    return pd.DataFrame(
        [
            {"recording_key": "S1/260101/P1/Network/001", "well_id": "rec0/well000",
             "preprocessing": "complete", "sorting": "running"},
            {"recording_key": "S2/260103/P1/ActivityScan/002", "well_id": "rec0/well001",
             "preprocessing": "failed", "sorting": "not_run"},
            {"recording_key": "S2/260105/P1/Network/003", "well_id": "rec0/well002",
             "preprocessing": "complete", "sorting": "complete"},
        ],
        columns=["recording_key", "well_id", "preprocessing", "sorting"],
    )


def test_filter_pipeline_df_derives_facets_from_key():
    from yuxin_mea.dashboard.data import filter_pipeline_df
    df = _pipe_df()
    # sample from key part 0
    assert set(filter_pipeline_df(df, sample_ids=["S2"])["recording_key"]) == {
        "S2/260103/P1/ActivityScan/002", "S2/260105/P1/Network/003"
    }
    # scan_type from key part 3
    assert set(filter_pipeline_df(df, scan_types=["Network"])["well_id"]) == {
        "rec0/well000", "rec0/well002"
    }
    # date range from key part 1
    assert list(filter_pipeline_df(df, date_from="260102", date_to="260104")["well_id"]) == [
        "rec0/well001"
    ]


def test_filter_pipeline_df_status_any_cell():
    from yuxin_mea.dashboard.data import filter_pipeline_df
    df = _pipe_df()
    out = filter_pipeline_df(df, statuses=["failed"])
    assert list(out["well_id"]) == ["rec0/well001"]
    # rows with ANY complete cell
    out2 = filter_pipeline_df(df, statuses=["complete"])
    assert set(out2["well_id"]) == {"rec0/well000", "rec0/well002"}


def test_filter_pipeline_df_group_join():
    from yuxin_mea.dashboard.data import filter_pipeline_df
    df = _pipe_df()
    gmap = {
        "S1/260101/P1/Network/001/rec0/well000": "Control",
        "S2/260103/P1/ActivityScan/002/rec0/well001": "IVH",
        "S2/260105/P1/Network/003/rec0/well002": "Control",
    }
    out = filter_pipeline_df(df, groups=["Control"], group_map=gmap)
    assert set(out["well_id"]) == {"rec0/well000", "rec0/well002"}


def test_filter_pipeline_df_empty_and_noop():
    from yuxin_mea.dashboard.data import filter_pipeline_df
    import pandas as pd
    assert filter_pipeline_df(pd.DataFrame()).empty
    df = _pipe_df()
    # no filters → unchanged length
    assert len(filter_pipeline_df(df)) == 3


# ---------------------------------------------------------------------------
# pagination.page_bounds
# ---------------------------------------------------------------------------


def test_page_bounds_basic():
    from yuxin_mea.dashboard.components.pagination import page_bounds
    # 120 items, 50/page → 3 pages
    assert page_bounds(120, 0, 50) == (0, 50, 3, 0)
    assert page_bounds(120, 1, 50) == (50, 100, 3, 1)
    assert page_bounds(120, 2, 50) == (100, 120, 3, 2)


def test_page_bounds_clamps_overshoot():
    from yuxin_mea.dashboard.components.pagination import page_bounds
    # requesting page 9 of a 3-page set clamps to last page
    assert page_bounds(120, 9, 50) == (100, 120, 3, 2)
    # negative clamps to 0
    assert page_bounds(120, -3, 50) == (0, 50, 3, 0)


def test_page_bounds_empty_and_exact_multiple():
    from yuxin_mea.dashboard.components.pagination import page_bounds
    # empty → one empty page
    assert page_bounds(0, 0, 50) == (0, 0, 1, 0)
    # exact multiple → no trailing empty page
    assert page_bounds(100, 1, 50) == (50, 100, 2, 1)


def test_date_converters_roundtrip():
    from yuxin_mea.dashboard.components.filter_bar import iso_to_yymmdd, yymmdd_to_iso
    assert iso_to_yymmdd("2026-02-03") == "260203"
    assert yymmdd_to_iso("260203") == "2026-02-03"
    # round-trip both directions
    assert iso_to_yymmdd(yymmdd_to_iso("240415")) == "240415"
    assert yymmdd_to_iso(iso_to_yymmdd("2024-04-15")) == "2024-04-15"
    # DatePickerRange may append a time suffix — first 10 chars are used
    assert iso_to_yymmdd("2026-02-03T00:00:00") == "260203"


def test_date_converters_empty_passthrough():
    from yuxin_mea.dashboard.components.filter_bar import iso_to_yymmdd, yymmdd_to_iso
    assert iso_to_yymmdd(None) is None
    assert iso_to_yymmdd("") is None
    assert yymmdd_to_iso(None) is None
    assert yymmdd_to_iso("") is None


def test_pager_state_label_and_disabled():
    from yuxin_mea.dashboard.components.pagination import pager_state
    # first page → prev disabled, next enabled
    assert pager_state(0, 3, 120) == ("page 1 / 3 · 120 total", True, False)
    # middle page → both enabled
    assert pager_state(1, 3, 120) == ("page 2 / 3 · 120 total", False, False)
    # last page → next disabled
    assert pager_state(2, 3, 120) == ("page 3 / 3 · 120 total", False, True)
    # single page → both disabled
    assert pager_state(0, 1, 10) == ("page 1 / 1 · 10 total", True, True)


# ---------------------------------------------------------------------------
# hours-since-media facet
# ---------------------------------------------------------------------------
_HOURS_RECS = [
    {"cache_key": "S1/260101/P1/Network/001", "sample_id": "S1", "date": "260101",
     "scan_type": "Network", "wells": [], "groups": ["Control"],
     "hours_since_media": 24.0},
    {"cache_key": "S1/260102/P1/Network/002", "sample_id": "S1", "date": "260102",
     "scan_type": "Network", "wells": [], "groups": ["Control"],
     "hours_since_media": 0.5},
    {"cache_key": "S1/260103/P1/Network/003", "sample_id": "S1", "date": "260103",
     "scan_type": "Network", "wells": [], "groups": ["IVH"],
     "hours_since_media": None},
]


def test_filter_by_hours_since_media():
    out = filter_recordings(_HOURS_RECS, {}, hours_since_media=[24.0])
    assert {r["cache_key"] for r in out} == {"S1/260101/P1/Network/001"}


def test_filter_hours_selects_untimed_recordings_explicitly():
    """Untimed scans are a category you can ask for, not a hidden remainder."""
    out = filter_recordings(_HOURS_RECS, {}, hours_since_media=[HOURS_UNKNOWN])
    assert {r["cache_key"] for r in out} == {"S1/260103/P1/Network/003"}


def test_filter_hours_accepts_several_timepoints():
    out = filter_recordings(_HOURS_RECS, {}, hours_since_media=[24.0, 0.5])
    assert len(out) == 2


def test_filter_hours_combines_with_the_other_facets():
    out = filter_recordings(
        _HOURS_RECS, {}, groups=["Control"], hours_since_media=[24.0, 0.5],
    )
    assert len(out) == 2
    out2 = filter_recordings(
        _HOURS_RECS, {}, groups=["IVH"], hours_since_media=[24.0],
    )
    assert out2 == []


def test_filter_hours_absent_is_a_noop():
    assert len(filter_recordings(_HOURS_RECS, {})) == 3
