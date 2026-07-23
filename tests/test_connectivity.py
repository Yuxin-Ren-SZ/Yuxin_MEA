"""STTC connectivity: hand-checked pair values, matrix invariants, graph metrics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from yuxin_mea.analysis.connectivity import (
    ConnectivityConfig,
    ConnectivityError,
    compute_connectivity,
    empty_connectivity_results,
    graph_metrics,
    sttc_matrix,
    sttc_pair,
    write_connectivity,
)
from yuxin_mea.tasks.connectivity import ConnectivityTask


class TestSttcPair:
    def test_identical_single_spike_trains_give_one(self):
        a = np.array([1.0])
        assert sttc_pair(a, a.copy(), dt=0.1, T=10.0) == pytest.approx(1.0)

    def test_identical_multi_spike_trains_give_one(self):
        a = np.array([1.0, 2.0, 3.5, 7.2])
        assert sttc_pair(a, a.copy(), dt=0.05, T=10.0) == pytest.approx(1.0)

    def test_disjoint_far_apart_is_non_positive(self):
        a = np.array([1.0])
        b = np.array([9.0])
        val = sttc_pair(a, b, dt=0.1, T=10.0)
        assert val <= 0.0
        assert val == pytest.approx(-0.02, abs=1e-9)

    def test_empty_train_returns_zero(self):
        assert sttc_pair(np.array([]), np.array([1.0]), dt=0.1, T=10.0) == 0.0

    def test_pencil_computed_partial_overlap(self):
        # a=[1,2], b=[1.05,5], dt=0.1, T=10:
        #   T_A = 0.4/10 = 0.04, T_B = 0.04
        #   P_A = 1/2, P_B = 1/2
        #   term = (0.5 - 0.04)/(1 - 0.5*0.04) = 0.46/0.98 = 0.4693877...
        a = np.array([1.0, 2.0])
        b = np.array([1.05, 5.0])
        assert sttc_pair(a, b, dt=0.1, T=10.0) == pytest.approx(0.4693877551, abs=1e-9)

    def test_symmetric_in_arguments(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.02, 2.5, 8.0])
        assert sttc_pair(a, b, dt=0.1, T=10.0) == pytest.approx(
            sttc_pair(b, a, dt=0.1, T=10.0)
        )


class TestSttcMatrix:
    def test_diagonal_is_one_and_symmetric(self):
        rng = np.random.default_rng(0)
        spikes = {u: np.sort(rng.uniform(0, 10, size=40)) for u in range(4)}
        W = sttc_matrix(spikes, dt=0.02, T=10.0, t_start=0.0)
        assert W.shape == (4, 4)
        assert np.allclose(np.diag(W), 1.0)
        assert np.allclose(W, W.T)

    def test_copied_unit_is_perfectly_coupled(self):
        rng = np.random.default_rng(1)
        base = np.sort(rng.uniform(0, 10, size=50))
        spikes = {0: base, 1: base.copy(), 2: np.sort(rng.uniform(0, 10, size=50))}
        W = sttc_matrix(spikes, dt=0.02, T=10.0, t_start=0.0, unit_ids=[0, 1, 2])
        assert W[0, 1] == pytest.approx(1.0)


class TestGraphMetrics:
    def test_triangle_graph(self):
        # 3 nodes, all-connected (STTC=1 everywhere above the mask).
        W = np.ones((3, 3))
        adj = ~np.eye(3, dtype=bool)
        gm = graph_metrics(W, adjacency=adj)
        assert gm["n_edges"] == 3
        assert gm["edge_density"] == pytest.approx(1.0)
        assert gm["mean_degree"] == pytest.approx(2.0)
        assert gm["clustering_coeff"] == pytest.approx(1.0)
        assert gm["mean_sttc"] == pytest.approx(1.0)

    def test_empty_adjacency_is_safe(self):
        W = np.zeros((3, 3))
        adj = np.zeros((3, 3), dtype=bool)
        gm = graph_metrics(W, adjacency=adj)
        assert gm["n_edges"] == 0
        assert gm["edge_density"] == 0.0


class TestComputeConnectivity:
    def test_end_to_end_two_coupled_clusters(self):
        rng = np.random.default_rng(7)
        # Two tightly-coupled trios sharing burst times → strong within-cluster STTC.
        burst_a = np.sort(rng.uniform(0, 60, size=30))
        burst_b = np.sort(rng.uniform(0, 60, size=30))
        spikes = {}
        for u in range(3):
            spikes[u] = np.sort(burst_a + rng.normal(0, 0.002, burst_a.size))
        for u in range(3, 6):
            spikes[u] = np.sort(burst_b + rng.normal(0, 0.002, burst_b.size))
        cfg = ConnectivityConfig(dt=0.02, n_shuffle=20, min_spikes_per_unit=5)
        res = compute_connectivity(spikes, positions=None, config=cfg)
        assert res.sttc_matrix.shape == (6, 6)
        # within-cluster coupling >> cross-cluster
        assert res.sttc_matrix[0, 1] > res.sttc_matrix[0, 3]
        assert set(res.sttc_sweep.keys()) == {0.005, 0.01, 0.02, 0.05}
        assert res.diagnostics["n_units"] == 6

    def test_too_few_units_raises(self):
        spikes = {0: np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        with pytest.raises(ConnectivityError):
            compute_connectivity(spikes, config=ConnectivityConfig())

    def test_from_task_params_casts_and_ignores_extras(self):
        cfg = ConnectivityConfig.from_task_params(
            {"dt": "0.01", "n_shuffle": "50", "output_root": "/tmp/x"}
        )
        assert cfg.dt == 0.01
        assert cfg.n_shuffle == 50


def _two_cluster_spikes(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    burst_a = np.sort(rng.uniform(0, 60, size=30))
    burst_b = np.sort(rng.uniform(0, 60, size=30))
    spikes = {}
    for u in range(3):
        spikes[u] = np.sort(burst_a + rng.normal(0, 0.002, burst_a.size))
    for u in range(3, 6):
        spikes[u] = np.sort(burst_b + rng.normal(0, 0.002, burst_b.size))
    return spikes


class TestNodeMetricsPass:
    """Node cartography is produced by the connectivity task itself.

    Before this, node_metrics.parquet / node_scalars.json had no producer in the
    repo at all — the files on disk came from uncommitted code, so a recomputed
    well silently lost them.
    """

    _CFG = ConnectivityConfig(dt=0.02, n_shuffle=20, min_spikes_per_unit=5)

    def test_compute_attaches_node_metrics(self):
        res = compute_connectivity(_two_cluster_spikes(), config=self._CFG)
        assert res.node_metrics is not None and res.node_scalars is not None
        assert len(res.node_metrics) == 6
        assert {"role", "participation", "z_within", "degree"} <= set(
            res.node_metrics.columns)
        assert res.node_scalars["n_nodes"] == 6
        assert "node_metrics_error" not in res.diagnostics

    def test_writer_emits_both_files(self, tmp_path):
        res = compute_connectivity(_two_cluster_spikes(), config=self._CFG)
        write_connectivity(res, tmp_path)
        nm_path = tmp_path / "node_metrics.parquet"
        sc_path = tmp_path / "node_scalars.json"
        assert nm_path.exists() and sc_path.exists()
        pd.testing.assert_frame_equal(pd.read_parquet(nm_path), res.node_metrics)
        # NaN != NaN, so compare key-wise (assortativity/rich_club are NaN on a
        # graph this small).
        round_tripped = json.loads(sc_path.read_text())
        assert set(round_tripped) == set(res.node_scalars)
        for k, v in res.node_scalars.items():
            if isinstance(v, float) and np.isnan(v):
                assert np.isnan(round_tripped[k]), k
            else:
                assert round_tripped[k] == pytest.approx(v), k

    def test_node_failure_does_not_sink_connectivity(self, tmp_path, monkeypatch):
        """graph_nodes needs python-louvain, which graph_metrics does not.

        A missing optional dep (or any other failure) must leave an otherwise-good
        connectivity result intact, and must be visible in diagnostics rather than
        silently yielding a well with no node metrics.
        """
        import yuxin_mea.analysis.graph_nodes as gn

        def _boom(*_a, **_k):
            raise ModuleNotFoundError("No module named 'community'")

        monkeypatch.setattr(gn, "compute_node_metrics", _boom)
        res = compute_connectivity(_two_cluster_spikes(), config=self._CFG)

        assert res.node_metrics is None and res.node_scalars is None
        assert "community" in res.diagnostics["node_metrics_error"]
        # the rest of the bundle is unaffected
        assert res.sttc_matrix.shape == (6, 6)
        assert res.graph_metrics["n_nodes"] == 6

        write_connectivity(res, tmp_path)
        # absent, not a stub a reader would mistake for a real empty graph
        assert not (tmp_path / "node_metrics.parquet").exists()
        assert not (tmp_path / "node_scalars.json").exists()
        assert (tmp_path / "graph_metrics.json").exists()

    def test_empty_results_carry_empty_node_results(self, tmp_path):
        res = empty_connectivity_results(reason="too few units")
        assert res.node_metrics is not None and res.node_scalars is not None
        assert len(res.node_metrics) == 0
        assert res.node_scalars["n_nodes"] == 0
        write_connectivity(res, tmp_path)
        assert (tmp_path / "node_metrics.parquet").exists()


def test_connectivity_params_are_frozen():
    """Guard the 1898-well recompute trap.

    PipelineManager.is_task_complete re-runs a COMPLETE task whose stored config
    differs from the current one, and the dashboard's config builder writes every
    schema key on save. So adding a param here silently invalidates every
    completed well and forces a full STTC recompute (100 shuffles/well, hours).
    Node metrics were therefore folded in with no new params. If this fails,
    that is the cost you are about to pay -- make it a deliberate decision.
    """
    expected = {
        "curation_output_root", "output_root", "dt", "dt_sweep", "min_units",
        "min_spikes_per_unit", "n_shuffle", "thresh_percentile", "random_state",
        "n_jobs",
    }
    actual = set(ConnectivityTask.default_params())
    assert actual == expected, (
        f"ConnectivityTask params changed (added {actual - expected}, "
        f"removed {expected - actual}). This invalidates every COMPLETE "
        f"connectivity well and forces a full STTC recompute."
    )
    assert set(ConnectivityTask.params_schema()) == expected
