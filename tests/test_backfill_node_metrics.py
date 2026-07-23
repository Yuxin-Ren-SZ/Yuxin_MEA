"""Tests for scripts/backfill_node_metrics.py.

The script fills node metrics into connectivity outputs that predate the
producer, recomputing from the edge table rather than re-running STTC.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_script():
    """Import the script by path (scripts/ is not an importable package)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "backfill_node_metrics.py"
    spec = importlib.util.spec_from_file_location("backfill_node_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backfill_node_metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


bnm = _load_script()

RK = "CXTEST/260401/T000001/Network/000001"
REC = "rec0000"


def _write_connectivity_output(d: Path, n: int = 6, *, with_nodes: bool = False) -> None:
    """A connectivity output dir carrying the two inputs the backfill reads."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    W = rng.random((n, n)) * 0.4
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    unit_ids = np.arange(n)
    np.save(d / "sttc_matrix.npy", W)
    np.savez(d / "sttc_sweep.npz", dt_20ms=W, unit_ids=unit_ids)
    rows = [{"u": i, "v": j, "sttc": float(W[i, j]), "dist_um": float(50 * abs(i - j))}
            for i in range(n) for j in range(i + 1, n) if W[i, j] > 0.15]
    pd.DataFrame(rows, columns=["u", "v", "sttc", "dist_um"]).to_parquet(
        d / "edges.parquet")
    (d / "graph_metrics.json").write_text(json.dumps({"n_nodes": n}))
    if with_nodes:
        (d / "node_metrics.parquet").parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"degree": np.zeros(n), "role": ["stale"] * n},
                     index=pd.Index(unit_ids, name="unit_id")).to_parquet(
            d / "node_metrics.parquet")
        (d / "node_scalars.json").write_text(json.dumps({"n_nodes": -1}))


@pytest.fixture
def analysis_root(tmp_path: Path) -> Path:
    root = tmp_path / "analysis"
    root.mkdir()
    wells = {"well001": False, "well002": False, "well003": True}
    cache = {}
    for well, has_nodes in wells.items():
        d = root / "connectivity_data" / RK / REC / well / "connectivity"
        _write_connectivity_output(d, with_nodes=has_nodes)
        cache[f"{RK}/{REC}/{well}"] = {
            "recording_key": RK, "well_id": f"{REC}/{well}",
            "tasks": {"connectivity": {"status": "complete", "output_path": str(d)}},
        }
    # a well whose connectivity never completed — must not be touched
    incomplete = root / "connectivity_data" / RK / REC / "well004" / "connectivity"
    _write_connectivity_output(incomplete)
    cache[f"{RK}/{REC}/well004"] = {
        "recording_key": RK, "well_id": f"{REC}/well004",
        "tasks": {"connectivity": {"status": "not_run", "output_path": str(incomplete)}},
    }
    (root / "pipeline_cache.json").write_text(json.dumps(cache))
    return root


def _config(tmp_path: Path, analysis_root: Path) -> Path:
    cfg = tmp_path / "pipeline_config.json"
    cfg.write_text(json.dumps({
        "global": {"data_root": str(tmp_path / "raw"),
                   "analysis_root": str(analysis_root)},
        "tasks": {},
    }))
    return cfg


def _conn_dir(analysis_root: Path, well: str) -> Path:
    return analysis_root / "connectivity_data" / RK / REC / well / "connectivity"


def test_discovery_uses_manifest_and_skips_incomplete(analysis_root: Path):
    dirs = bnm.connectivity_dirs(analysis_root, None)
    names = {d.parent.name for d in dirs}
    assert names == {"well001", "well002", "well003"}
    assert "well004" not in names


def test_discovery_honours_recording_filter(analysis_root: Path):
    assert bnm.connectivity_dirs(analysis_root, {"OTHER/KEY"}) == []
    assert len(bnm.connectivity_dirs(analysis_root, {RK})) == 3


def test_writes_missing_and_skips_present(tmp_path: Path, analysis_root: Path):
    cfg = _config(tmp_path, analysis_root)
    assert bnm.main(["--config", str(cfg)]) == 0

    for well in ("well001", "well002"):
        d = _conn_dir(analysis_root, well)
        nm = pd.read_parquet(d / "node_metrics.parquet")
        assert len(nm) == 6
        assert {"role", "participation", "z_within"} <= set(nm.columns)
        assert json.loads((d / "node_scalars.json").read_text())["n_nodes"] == 6

    # well003 already had (stale sentinel) node metrics -> left alone
    stale = _conn_dir(analysis_root, "well003")
    assert json.loads((stale / "node_scalars.json").read_text())["n_nodes"] == -1


def test_force_overwrites_existing(tmp_path: Path, analysis_root: Path):
    cfg = _config(tmp_path, analysis_root)
    assert bnm.main(["--config", str(cfg), "--force"]) == 0
    stale = _conn_dir(analysis_root, "well003")
    assert json.loads((stale / "node_scalars.json").read_text())["n_nodes"] == 6


def test_dry_run_writes_nothing(tmp_path: Path, analysis_root: Path):
    cfg = _config(tmp_path, analysis_root)
    assert bnm.main(["--config", str(cfg), "--dry-run"]) == 0
    assert not (_conn_dir(analysis_root, "well001") / "node_metrics.parquet").exists()
    # and the already-present one is untouched
    stale = _conn_dir(analysis_root, "well003")
    assert json.loads((stale / "node_scalars.json").read_text())["n_nodes"] == -1


def test_backfill_matches_the_task_producer(tmp_path: Path, analysis_root: Path):
    """The script and ConnectivityTask must agree — otherwise the dataset ends up
    a mixture of two code paths, which is the defect this whole change fixes."""
    from yuxin_mea.analysis.graph_nodes import compute_node_metrics

    cfg = _config(tmp_path, analysis_root)
    bnm.main(["--config", str(cfg)])

    d = _conn_dir(analysis_root, "well001")
    edges = pd.read_parquet(d / "edges.parquet")
    with np.load(d / "sttc_sweep.npz") as z:
        unit_ids = z["unit_ids"]
    expected = compute_node_metrics(edges, node_ids=unit_ids)
    pd.testing.assert_frame_equal(
        pd.read_parquet(d / "node_metrics.parquet"), expected.node_metrics)


def test_missing_config_is_a_usage_error(tmp_path: Path):
    assert bnm.main(["--config", str(tmp_path / "nope.json")]) == 2


def test_failure_is_counted_not_fatal(tmp_path: Path, analysis_root: Path,
                                      monkeypatch):
    """One unreadable well must not abort the sweep over the rest."""
    bad = _conn_dir(analysis_root, "well001")
    (bad / "sttc_sweep.npz").write_text("not an npz")
    cfg = _config(tmp_path, analysis_root)

    assert bnm.main(["--config", str(cfg)]) == 1          # non-zero: something failed
    # ...but the healthy well was still processed
    assert (_conn_dir(analysis_root, "well002") / "node_metrics.parquet").exists()
