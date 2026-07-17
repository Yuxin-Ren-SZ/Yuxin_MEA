"""End-to-end tests for SpatialMapTask + ConnectivityTask on staged wells."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from yuxin_mea.tasks.connectivity import ConnectivityTask
from yuxin_mea.tasks.spatial_map import SpatialMapTask

_RK = "SampleA/240415/PlateX/Network/001"
_WELL_ID = "rec0000/well000"
_REC = "rec0000"
_WELL = "well000"


def _stage(tmp: Path, n_units=10, with_bursts=True, seed=0):
    rng = np.random.default_rng(seed)
    cur = tmp / "curation" / _RK / _REC / _WELL / "auto_curation"
    cur.mkdir(parents=True)
    # Spike trains: shared burst times drive real STTC coupling.
    burst_centers = np.sort(rng.uniform(2, 58, 20))
    spikes, xs, ys, frs = {}, [], [], []
    for u in range(n_units):
        jit = rng.normal(0, 0.003, burst_centers.size)
        base = np.sort(np.concatenate([burst_centers + jit, rng.uniform(0, 60, 30)]))
        spikes[u] = base
        xs.append(rng.uniform(0, 400)); ys.append(rng.uniform(0, 400))
        frs.append(base.size / 60.0)
    np.save(cur / "curated_spike_times.npy", np.array(spikes, dtype=object),
            allow_pickle=True)
    qm = pd.DataFrame({"loc_x": xs, "loc_y": ys, "firing_rate": frs,
                       "curated": [True] * n_units}, index=list(range(n_units)))
    qm.to_pickle(cur / "quality_metrics.pkl")
    # Network bursts for propagation.
    if with_bursts:
        bd = tmp / "burst" / _RK / _REC / _WELL / "burst_detection"
        bd.mkdir(parents=True)
        pd.DataFrame({"start": burst_centers - 0.1,
                      "end": burst_centers + 0.3}).to_pickle(bd / "network_bursts.pkl")
    return {
        "curation_output_root": str(tmp / "curation"),
        "burst_output_root": str(tmp / "burst"),
        "output_root": str(tmp / "out"),
    }


class TestSpatialMapTask:
    def test_writes_outputs(self):
        with TemporaryDirectory() as t:
            tmp = Path(t)
            base = _stage(tmp)
            out = SpatialMapTask().run(_RK, _WELL_ID, tmp / "x.h5",
                                       {**base, "bins": 32, "sigma_um": 40.0})
            for f in ("activity_field.npy", "burst_propagation.parquet",
                      "spatial_metrics.json", "diagnostics.json"):
                assert (out / f).exists(), f
            field = np.load(out / "activity_field.npy")
            assert field.shape == (32, 32)
            m = json.loads((out / "spatial_metrics.json").read_text())
            assert "activity_gini" in m

    def test_sparse_well_is_complete_empty(self):
        with TemporaryDirectory() as t:
            tmp = Path(t)
            base = _stage(tmp, n_units=0, with_bursts=False)
            out = SpatialMapTask().run(_RK, _WELL_ID, tmp / "x.h5",
                                       {**base, "bins": 16})
            diag = json.loads((out / "diagnostics.json").read_text())
            assert diag["n_units"] == 0
            assert "empty_reason" in diag


class TestConnectivityTask:
    def test_writes_outputs(self):
        with TemporaryDirectory() as t:
            tmp = Path(t)
            base = _stage(tmp, n_units=12)
            out = ConnectivityTask().run(
                _RK, _WELL_ID, tmp / "x.h5",
                {**base, "n_shuffle": 20, "dt": 0.02,
                 "dt_sweep": [0.01, 0.02, 0.05]},
            )
            for f in ("sttc_matrix.npy", "sttc_sweep.npz",
                      "graph_metrics.json", "edges.parquet", "diagnostics.json"):
                assert (out / f).exists(), f
            W = np.load(out / "sttc_matrix.npy")
            assert W.shape == (12, 12)
            gm = json.loads((out / "graph_metrics.json").read_text())
            assert set(["mean_sttc", "edge_density", "modularity",
                        "small_worldness"]).issubset(gm)
            edges = pd.read_parquet(out / "edges.parquet")
            assert list(edges.columns) == ["u", "v", "sttc", "dist_um"]

    def test_sparse_well_is_complete_empty(self):
        with TemporaryDirectory() as t:
            tmp = Path(t)
            base = _stage(tmp, n_units=1)  # < min_units
            out = ConnectivityTask().run(_RK, _WELL_ID, tmp / "x.h5",
                                         {**base, "n_shuffle": 5})
            gm = json.loads((out / "graph_metrics.json").read_text())
            assert gm["n_edges"] == 0
            diag = json.loads((out / "diagnostics.json").read_text())
            assert diag["n_units"] == 0 and "empty_reason" in diag
            # Empty artifacts must still be readable.
            assert pd.read_parquet(out / "edges.parquet").empty
