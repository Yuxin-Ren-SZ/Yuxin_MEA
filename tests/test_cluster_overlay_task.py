"""ClusterOverlay aggregate tasks: end-to-end render from staged debug traces."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yuxin_mea.pipeline.task_record import TaskStatus
from yuxin_mea.pipeline.well_dims import Member, WellDims
from yuxin_mea.tasks import ClusterOverlayByRecordingTask, ClusterOverlayByWellTask

pytest.importorskip("umap")   # UMAP fit required by the render


def _stage_trace(ml_dir: Path, n: int = 40, seed: int = 0) -> None:
    """Write a minimal debug_trace.pkl + diagnostics.json into an ml dir."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    labels = np.array([(1 if i % 3 == 1 else (2 if i % 3 == 2 else -1)) for i in range(n)])
    trace = SimpleNamespace(
        feature_matrix=X, hdbscan_labels=labels, t_centers=np.arange(n, dtype=float),
        scaler_mean=X.mean(0), scaler_std=X.std(0) + 1e-6,
        feature_names=[f"f{i}" for i in range(5)],
    )
    ml_dir.mkdir(parents=True, exist_ok=True)
    with (ml_dir / "debug_trace.pkl").open("wb") as fh:
        pickle.dump(trace, fh)
    (ml_dir / "diagnostics.json").write_text(json.dumps({"cluster_burst_labels": [1]}))


def _member(root: Path, rk: str, cw: str, seed: int, stage: bool = True) -> Member:
    ml_dir = root / rk.replace("/", "_") / cw.replace("/", "_") / "ml_burst_detection"
    if stage:
        _stage_trace(ml_dir, seed=seed)
    return Member(
        dims=WellDims.from_entry(rk, cw, "Control"),
        upstream_status=TaskStatus.COMPLETE,
        upstream_output_path=ml_dir,
        upstream_last_updated=1.0,
    )


def test_by_well_renders(tmp_path):
    rk = "CX9/260101/PLATE/Network/000001"
    members = [_member(tmp_path, rk, f"rec0000/well00{i}", seed=i) for i in range(3)]
    task = ClusterOverlayByWellTask()
    out = task.run(
        {"recording_key": rk}, members,
        {"output_root": str(tmp_path / "fig"), "max_bins_per_rec": 30, "n_neighbors": 5},
    )
    assert out.exists() and out.name == "overlay.html"
    assert "__agg__" in str(out) and "cluster_overlay_by_well" in str(out)
    assert out.stat().st_size > 1000


def test_by_recording_renders_animated(tmp_path):
    uid = "CX9|PLATE|well000"
    # three recordings of the same physical well
    members = [
        _member(tmp_path, f"CX9/26010{d}/PLATE/Network/00000{d}", "rec0000/well000", seed=d)
        for d in (1, 2, 3)
    ]
    task = ClusterOverlayByRecordingTask()
    out = task.run(
        {"well_uid": uid}, members,
        {"output_root": str(tmp_path / "fig"), "max_bins_per_rec": 30, "n_neighbors": 5},
    )
    assert out.exists()
    html = out.read_text()
    assert '"name":"cum0"' in html          # animation frames present
    assert "Leave previous" in html


def test_skips_missing_and_raises_when_all_missing(tmp_path):
    rk = "CX9/260101/PLATE/Network/000001"
    good = _member(tmp_path, rk, "rec0000/well000", seed=1)
    missing = _member(tmp_path, rk, "rec0000/well001", seed=2, stage=False)  # no trace on disk
    task = ClusterOverlayByWellTask()
    out = task.run(
        {"recording_key": rk}, [good, missing],
        {"output_root": str(tmp_path / "fig"), "max_bins_per_rec": 30, "n_neighbors": 5},
    )
    assert out.exists()   # rendered from the one good series, missing skipped

    only_missing = _member(tmp_path, rk, "rec0000/well002", seed=3, stage=False)
    with pytest.raises(RuntimeError):
        task.run(
            {"recording_key": rk}, [only_missing],
            {"output_root": str(tmp_path / "fig2"), "max_bins_per_rec": 30, "n_neighbors": 5},
        )


def test_output_root_required(tmp_path):
    rk = "CX9/260101/PLATE/Network/000001"
    m = _member(tmp_path, rk, "rec0000/well000", seed=1)
    with pytest.raises(ValueError):
        ClusterOverlayByWellTask().run({"recording_key": rk}, [m], {})   # no output_root
