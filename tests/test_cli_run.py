"""Tests for the `yuxin-mea-run` worker CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yuxin_mea.cli.run import build_parser, main
from yuxin_mea.dataset.cache import JsonCacheStore as DatasetCacheStore
from yuxin_mea.dataset.entries import RecordingEntry
from yuxin_mea.pipeline.cache import JsonPipelineCacheStore


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


# ---------------------------------------------------------------------------
# Auto-queue tests
# ---------------------------------------------------------------------------

def _seed_dataset_cache(analysis: Path, recs: dict[str, dict]) -> None:
    """Seed experiment_cache.json with recordings. recs maps cache_key to
    {h5_recordings: {rec_name: [well_ids]}}."""
    store = DatasetCacheStore(analysis)
    entries = {}
    for key, info in recs.items():
        parts = key.split("/")
        entry = RecordingEntry(
            sample_id=parts[0], date=parts[1], plate_id=parts[2],
            scan_type=parts[3], run_id=parts[4],
            data_path=Path(f"{key}/data.raw.h5"),
            file_size=1000, mtime=0.0, discovered_at=0.0,
            h5_recordings=info.get("h5_recordings", {}),
        )
        entries[key] = entry
    store.save(entries)


def _make_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    analysis = tmp_path / "analysis"
    data = tmp_path / "data"
    analysis.mkdir()
    data.mkdir()
    config = tmp_path / "pipeline_config.json"
    config.write_text(json.dumps({
        "global": {"data_root": str(data), "analysis_root": str(analysis)},
        "tasks": {},
    }))
    return config, analysis, data


def test_auto_queue_adds_wells(tmp_path, caplog):
    config, analysis, _data = _make_config(tmp_path)
    rec_key = "S1/260101/P1/Network/000001"
    _seed_dataset_cache(analysis, {
        rec_key: {"h5_recordings": {"rec0000": ["well000", "well001", "well002"]}},
    })

    rc = main(["--config", str(config), "--recordings", rec_key, "--dry-run"])
    assert rc == 0

    pipe_store = JsonPipelineCacheStore(analysis)
    pipe = pipe_store.load()
    expected_keys = {f"{rec_key}/rec0000/{w}" for w in ["well000", "well001", "well002"]}
    assert set(pipe.keys()) == expected_keys


def test_auto_queue_skips_already_queued(tmp_path, caplog):
    config, analysis, _data = _make_config(tmp_path)
    rec_key = "S1/260101/P1/Network/000001"
    _seed_dataset_cache(analysis, {
        rec_key: {"h5_recordings": {"rec0000": ["well000"]}},
    })

    # Queue once
    main(["--config", str(config), "--recordings", rec_key])
    pipe = JsonPipelineCacheStore(analysis).load()
    assert len(pipe) == 1

    # Running again should not duplicate
    main(["--config", str(config), "--recordings", rec_key])
    pipe = JsonPipelineCacheStore(analysis).load()
    assert len(pipe) == 1


def test_auto_queue_warns_unknown_recording(tmp_path, caplog):
    config, analysis, _data = _make_config(tmp_path)
    import logging
    with caplog.at_level(logging.WARNING):
        rc = main(["--config", str(config), "--recordings", "NO/SUCH/REC/Net/001"])
    assert rc == 0
    assert "not found in dataset cache" in caplog.text
