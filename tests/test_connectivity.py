"""STTC connectivity: hand-checked pair values, matrix invariants, graph metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yuxin_mea.analysis.connectivity import (
    ConnectivityConfig,
    ConnectivityError,
    compute_connectivity,
    graph_metrics,
    sttc_matrix,
    sttc_pair,
)


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
