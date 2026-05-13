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
from yuxin_mea.dashboard.data import load_pipeline_df, load_recordings_df
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
