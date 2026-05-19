"""Tests for the `yuxin-mea-run` worker CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yuxin_mea.cli.run import build_parser, main


def test_parser_smoke():
    parser = build_parser()
    args = parser.parse_args([
        "--config", "/tmp/c.json",
        "--tasks", "preprocessing,sorting",
        "--recordings", "rec1,rec2",
        "--retry-failed",
        "--max-tasks", "5",
        "--jobs", "4",
        "--dry-run",
    ])
    assert args.config == Path("/tmp/c.json")
    assert args.tasks == "preprocessing,sorting"
    assert args.recordings == "rec1,rec2"
    assert args.retry_failed is True
    assert args.max_tasks == 5
    assert args.jobs == 4
    assert args.dry_run is True


def test_parser_jobs_defaults_to_one():
    parser = build_parser()
    args = parser.parse_args(["--config", "/tmp/c.json"])
    assert args.jobs == 1


def test_parser_requires_config():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_missing_config_returns_2(capsys):
    rc = main(["--config", "/definitely/does/not/exist.json"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()


def test_dry_run_on_empty_cache(tmp_path, capsys):
    """A config that points at an empty analysis_root → dry-run reports 0
    eligible tasks and exits cleanly."""
    analysis = tmp_path / "analysis"
    data = tmp_path / "data"
    analysis.mkdir()
    data.mkdir()

    config = tmp_path / "pipeline_config.json"
    config.write_text(json.dumps({
        "global": {"data_root": str(data), "analysis_root": str(analysis)},
        "tasks": {},
    }))

    rc = main(["--config", str(config), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0 task(s) eligible" in out


def test_unknown_task_filter_rejected(tmp_path, capsys):
    analysis = tmp_path / "analysis"
    data = tmp_path / "data"
    analysis.mkdir()
    data.mkdir()
    config = tmp_path / "c.json"
    config.write_text(json.dumps({
        "global": {"data_root": str(data), "analysis_root": str(analysis)},
        "tasks": {},
    }))

    rc = main([
        "--config", str(config),
        "--tasks", "nonexistent_task",
        "--dry-run",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "nonexistent_task" in err


def test_parallel_drain_empty_queue(tmp_path, capsys):
    """--jobs N with no eligible work must clean up the pool and exit 0."""
    analysis = tmp_path / "analysis"
    data = tmp_path / "data"
    analysis.mkdir()
    data.mkdir()
    config = tmp_path / "pipeline_config.json"
    config.write_text(json.dumps({
        "global": {"data_root": str(data), "analysis_root": str(analysis)},
        "tasks": {},
    }))

    rc = main(["--config", str(config), "--jobs", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Ran 0 task(s)" in out
