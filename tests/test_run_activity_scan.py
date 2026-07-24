"""Driver tests for scripts/run_activity_scan.py.

Covers discovery (ActivityScan only, zero-size skipped), --dry-run, skip-if-fresh,
--force, and the one-bad-well-is-not-fatal contract. Also the guard that the
activity scan is NOT a registered pipeline task — registering it would patch a
record into all 1900+ Network wells and could auto-queue kilosort on the
checkerboard configs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.test_activity_scan import _build_h5  # reuse the real-layout fixture


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_activity_scan.py"
    spec = importlib.util.spec_from_file_location("run_activity_scan", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_activity_scan"] = mod
    spec.loader.exec_module(mod)
    return mod


ras = _load_script()

AS_KEY = "CX999/260101/T000001/ActivityScan/000001"
NET_KEY = "CX999/260101/T000001/Network/000002"


@pytest.fixture
def workspace(tmp_path: Path) -> dict:
    data_root = tmp_path / "raw"
    analysis_root = tmp_path / "analysis"
    analysis_root.mkdir(parents=True)

    # a real (tiny) ActivityScan h5 with two wells, two configs each
    as_dir = data_root / AS_KEY
    as_dir.mkdir(parents=True)
    _build_h5(as_dir / "data.raw.h5", {
        0: [{"electrodes": list(range(100, 130)), "planted": {5: (8, -55.0)},
             "noise_uv": 2.0},
            {"electrodes": list(range(100, 130)), "planted": {5: (8, -55.0)},
             "noise_uv": 2.0}],
        1: [{"electrodes": list(range(200, 230)), "planted": {3: (5, -45.0)},
             "noise_uv": 2.0}],
    })

    entries = {
        AS_KEY: {
            "sample_id": "CX999", "scan_type": "ActivityScan",
            "data_path": f"{AS_KEY}/data.raw.h5",
            "h5_recordings": {"rec0000": ["well000", "well001"]},
        },
        NET_KEY: {   # must be ignored by discovery
            "sample_id": "CX999", "scan_type": "Network",
            "data_path": f"{NET_KEY}/data.raw.h5",
            "h5_recordings": {"rec0000": ["well000"]},
        },
    }
    (analysis_root / "experiment_cache.json").write_text(json.dumps(entries))

    config = tmp_path / "pipeline_config.json"
    config.write_text(json.dumps({
        "global": {"data_root": str(data_root), "analysis_root": str(analysis_root)},
        "tasks": {"activity_scan": {
            "output_root": str(analysis_root / "activity_scan_data")}},
    }))
    return {"data_root": data_root, "analysis_root": analysis_root,
            "config": config, "entries": entries,
            "output_root": analysis_root / "activity_scan_data"}


def test_discovery_activity_only_skips_network(workspace: dict):
    jobs = ras.discover_jobs(workspace["entries"], workspace["data_root"],
                            workspace["output_root"])
    keys = {j.recording_key for j in jobs}
    assert keys == {AS_KEY}                       # Network excluded
    assert {j.well_id for j in jobs} == {"well000", "well001"}


def test_discovery_skips_zero_size_h5(workspace: dict, tmp_path: Path):
    entries = dict(workspace["entries"])
    empty_key = "CX999/260102/T000001/ActivityScan/000009"
    empty_dir = workspace["data_root"] / empty_key
    empty_dir.mkdir(parents=True)
    (empty_dir / "data.raw.h5").touch()           # 0 bytes
    entries[empty_key] = {"sample_id": "CX999", "scan_type": "ActivityScan",
                          "data_path": f"{empty_key}/data.raw.h5",
                          "h5_recordings": {"rec0000": ["well000"]}}
    jobs = ras.discover_jobs(entries, workspace["data_root"], workspace["output_root"])
    assert empty_key not in {j.recording_key for j in jobs}


def test_dry_run_writes_nothing(workspace: dict):
    rc = ras.main(["--config", str(workspace["config"]), "--dry-run"])
    assert rc == 0
    assert not workspace["output_root"].exists()


def test_computes_and_is_idempotent(workspace: dict):
    rc = ras.main(["--config", str(workspace["config"])])
    assert rc == 0
    well0 = workspace["output_root"] / AS_KEY / "well000"
    assert (well0 / "stats.json").exists()
    assert (well0 / "electrodes.parquet").exists()
    assert (well0 / "yuxin_provenance.json").exists()
    assert (workspace["output_root"] / AS_KEY / "manifest.json").exists()

    mtime = (well0 / "stats.json").stat().st_mtime_ns
    # a second run must skip everything (fresh) and not rewrite
    rc2 = ras.main(["--config", str(workspace["config"])])
    assert rc2 == 0
    assert (well0 / "stats.json").stat().st_mtime_ns == mtime


def test_force_recomputes(workspace: dict):
    ras.main(["--config", str(workspace["config"])])
    well0 = workspace["output_root"] / AS_KEY / "well000"
    mtime = (well0 / "stats.json").stat().st_mtime_ns
    rc = ras.main(["--config", str(workspace["config"]), "--force"])
    assert rc == 0
    assert (well0 / "stats.json").stat().st_mtime_ns != mtime


def test_samples_filter(workspace: dict):
    rc = ras.main(["--config", str(workspace["config"]), "--samples", "NOPE",
                   "--dry-run"])
    assert rc == 0
    assert not workspace["output_root"].exists()


def test_one_bad_well_is_counted_not_fatal(workspace: dict, monkeypatch):
    real = ras.compute_well_activity if hasattr(ras, "compute_well_activity") else None

    from yuxin_mea.analysis import activity_scan as core
    orig = core.compute_well_activity

    def _flaky(h5_path, well_index, cfg, **kw):
        if int(well_index) == 0:
            raise RuntimeError("planted failure")
        return orig(h5_path, well_index, cfg, **kw)

    monkeypatch.setattr(core, "compute_well_activity", _flaky)
    rc = ras.main(["--config", str(workspace["config"])])
    assert rc == 1                                 # a failure -> nonzero exit
    # …but the healthy well still got written
    assert (workspace["output_root"] / AS_KEY / "well001" / "stats.json").exists()
    assert not (workspace["output_root"] / AS_KEY / "well000" / "stats.json").exists()


def test_missing_config_is_usage_error(tmp_path: Path):
    assert ras.main(["--config", str(tmp_path / "nope.json")]) == 2


def test_activity_scan_is_not_a_pipeline_task():
    """Guard: activity_scan must never enter TASK_CLASSES.

    register_computation_task patches a NOT_RUN record into EVERY existing
    pipeline entry (1900+ Network wells) and cli/run auto-queues every task on an
    ActivityScan's 56×6 h5_recordings — i.e. preprocessing + kilosort on the
    checkerboard configs. The activity scan is intentionally a standalone script
    for exactly this reason; if it appears here, that safety is gone.
    """
    from yuxin_mea.tasks import TASK_CLASSES

    names = {c.task_name for c in TASK_CLASSES}
    assert "activity_scan" not in names, (
        "activity_scan was registered as a pipeline task — this patches a record "
        "into every Network well and can auto-queue kilosort on checkerboard "
        "configs. Keep it a standalone script (scripts/run_activity_scan.py)."
    )
