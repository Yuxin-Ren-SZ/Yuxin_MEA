"""STTC connectivity: hand-checked pair values, matrix invariants, graph metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yuxin_mea.analysis.connectivity import (
    CCG_COLUMNS,
    ConnectivityConfig,
    ConnectivityError,
    attach_ccg,
    ccg_lags,
    ccg_pair,
    ccg_stats,
    ccg_summary,
    compute_ccg,
    compute_connectivity,
    empty_connectivity_results,
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
            {"dt": "0.01", "n_shuffle": "50", "output_root": "/tmp/x",
             "ccg_bin": "0.002", "ccg_max_pairs": "7", "ccg_enable": 0}
        )
        assert cfg.dt == 0.01
        assert cfg.n_shuffle == 50
        assert cfg.ccg_bin == 0.002
        assert cfg.ccg_max_pairs == 7
        assert cfg.ccg_enable is False


def _lagged_pair(lag_s: float, n: int = 1500, jitter: float = 0.0, seed: int = 0):
    """Reference train + a target that fires ``lag_s`` after every reference spike."""
    rng = np.random.default_rng(seed)
    ref = np.sort(rng.uniform(0, 120, n))
    tgt = np.sort(ref + lag_s + (rng.normal(0, jitter, n) if jitter else 0.0))
    return ref, tgt


class TestCcgPair:
    """Sign convention first: reference = u, histogram of (t_v - t_u)."""

    def test_positive_lag_means_reference_leads(self):
        cfg = ConnectivityConfig()
        lags = ccg_lags(cfg.ccg_bin, cfg.ccg_window)
        ref, tgt = _lagged_pair(0.005)
        c = ccg_pair(ref, tgt, cfg.ccg_bin, cfg.ccg_window)
        assert lags[int(np.argmax(c))] == pytest.approx(0.005)

    def test_swapping_arguments_mirrors_the_histogram(self):
        ref, tgt = _lagged_pair(0.005, n=400, seed=3)
        fwd = ccg_pair(ref, tgt, 0.001, 0.05)
        rev = ccg_pair(tgt, ref, 0.001, 0.05)
        assert np.array_equal(fwd, rev[::-1])

    def test_bin_count_is_symmetric_and_odd(self):
        c = ccg_pair(np.array([1.0]), np.array([1.0]), 0.001, 0.1)
        assert c.size == 201 == ccg_lags(0.001, 0.1).size
        assert c[100] == 1  # zero-lag bin is the centre

    def test_empty_train_is_all_zero(self):
        c = ccg_pair(np.array([]), np.array([1.0, 2.0]), 0.001, 0.1)
        assert c.size == 201 and not c.any()

    def test_merged_unit_train_must_be_sorted_first(self):
        """A merged unit is two sorted trains concatenated — i.e. not sorted.

        ``searchsorted`` would silently under-count on it, so ``compute_ccg``
        sorts once per unit. This pins the failure mode rather than the fix.
        """
        ref, tgt = _lagged_pair(0.005, n=600, seed=5)
        good = ccg_pair(ref, tgt, 0.001, 0.05)
        unsorted = np.concatenate([tgt[300:], tgt[:300]])
        assert not np.array_equal(ccg_pair(ref, unsorted, 0.001, 0.05), good)
        assert np.array_equal(ccg_pair(ref, np.sort(unsorted), 0.001, 0.05), good)


class TestCcgAgainstSpikeInterface:
    """Oracle check.

    ``analysis.connectivity`` deliberately hand-rolls the histogram (it works on
    the ``{unit_id -> times}`` contract and must not import SpikeInterface — the
    dashboard depends on that), so SI is used only to validate the numbers.
    SI's ``correlograms[i, j]`` is the mirror of this module's ``(i=ref, j=tgt)``
    orientation, and its bin *edges* land on integers where ours are bin
    *centres* — hence the one-bin tolerance.
    """

    def test_matches_compute_correlograms(self):
        sc = pytest.importorskip("spikeinterface.core")
        pp = pytest.importorskip("spikeinterface.postprocessing")

        fs = 20000.0
        ref, tgt = _lagged_pair(0.005, n=800, jitter=0.0008, seed=0)
        ref = ref[ref > 0.1]
        tgt = tgt[-ref.size:]
        samples = np.round(np.concatenate([ref, tgt]) * fs).astype("int64")
        labels = np.array([0] * ref.size + [1] * tgt.size)
        order = np.argsort(samples)
        try:
            sorting = sc.NumpySorting.from_samples_and_labels(
                [samples[order]], [labels[order]], sampling_frequency=fs)
        except AttributeError:  # older SpikeInterface
            pytest.skip("NumpySorting.from_samples_and_labels unavailable")
        ccgs, bins = pp.compute_correlograms(sorting, window_ms=100.0, bin_ms=1.0)
        centres = (bins[:-1] + bins[1:]) / 2.0

        mine = ccg_pair(ref, tgt, 0.001, 0.05)
        lags_ms = ccg_lags(0.001, 0.05) * 1000.0
        assert abs(lags_ms[int(np.argmax(mine))]
                   - centres[int(np.argmax(ccgs[1, 0]))]) <= 1.0
        inner_mine = np.abs(lags_ms) <= 40
        inner_si = np.abs(centres) <= 40
        assert mine[inner_mine].sum() == pytest.approx(
            ccgs[1, 0][inner_si].sum(), rel=0.02)


class TestCcgStats:
    def test_coupled_pair_is_significant_and_directed(self):
        cfg = ConnectivityConfig()
        lags = ccg_lags(cfg.ccg_bin, cfg.ccg_window)
        ref, tgt = _lagged_pair(0.003, jitter=0.0005, seed=11)
        st = ccg_stats(ccg_pair(ref, tgt, cfg.ccg_bin, cfg.ccg_window), lags, cfg)
        assert st["sig"] is True
        assert st["syn_dir"] == "u->v"
        assert st["peak_lag_ms"] == pytest.approx(3.0, abs=1.0)
        assert st["asymmetry"] > 0.5

    def test_independent_poisson_pair_is_not_significant(self):
        cfg = ConnectivityConfig()
        lags = ccg_lags(cfg.ccg_bin, cfg.ccg_window)
        rng = np.random.default_rng(2)
        a = np.sort(rng.uniform(0, 300, 3000))
        b = np.sort(rng.uniform(0, 300, 3000))
        st = ccg_stats(ccg_pair(a, b, cfg.ccg_bin, cfg.ccg_window), lags, cfg)
        assert st["sig"] is False
        assert st["syn_dir"] == ""

    def test_sparse_pair_keeps_z_finite(self):
        """Near-empty bins must not divide by a zero baseline."""
        cfg = ConnectivityConfig()
        lags = ccg_lags(cfg.ccg_bin, cfg.ccg_window)
        st = ccg_stats(
            ccg_pair(np.array([1.0, 2.0]), np.array([1.004]),
                     cfg.ccg_bin, cfg.ccg_window), lags, cfg)
        assert np.isfinite(st["peak_z"])

    def test_all_zero_correlogram_is_safe(self):
        cfg = ConnectivityConfig()
        lags = ccg_lags(cfg.ccg_bin, cfg.ccg_window)
        st = ccg_stats(np.zeros(lags.size), lags, cfg)
        assert st["sig"] is False and np.isnan(st["peak_z"])


class TestComputeCcg:
    @staticmethod
    def _spikes_and_edges(seed: int = 4):
        rng = np.random.default_rng(seed)
        base = np.sort(rng.uniform(0, 120, 800))
        spikes = {u: np.sort(base + u * 0.003 + rng.normal(0, 0.0005, base.size))
                  for u in range(4)}
        rows = [{"u": u, "v": v, "sttc": 1.0 - 0.1 * (v - u), "dist_um": 100.0}
                for u in range(4) for v in range(u + 1, 4)]
        return spikes, pd.DataFrame(rows)

    def test_row_count_matches_edges_and_columns_attach(self):
        spikes, edges = self._spikes_and_edges()
        out = compute_ccg(spikes, edges, ConnectivityConfig())
        assert out["counts"].shape == (len(edges), 201)
        assert out["pairs"].shape == (len(edges), 2)
        joined = attach_ccg(edges, out["per_pair"])
        assert list(joined.columns) == ["u", "v", "sttc", "dist_um", *CCG_COLUMNS]
        assert joined["ccg_syn_dir"].eq("u->v").all()

    def test_cap_drops_pairs_and_leaves_them_na(self):
        spikes, edges = self._spikes_and_edges()
        cfg = ConnectivityConfig(ccg_max_pairs=2)
        out = compute_ccg(spikes, edges, cfg)
        assert out["counts"].shape[0] == 2
        assert out["meta"]["ccg_pairs_capped"] == len(edges) - 2
        joined = attach_ccg(edges, out["per_pair"])
        # Rows survive; only the ccg_* values are missing for the capped pairs.
        assert len(joined) == len(edges)
        assert int(joined["ccg_peak_z"].isna().sum()) == len(edges) - 2
        # The kept pairs are the highest-STTC ones, and the join is by (u, v).
        kept = {tuple(p) for p in out["pairs"].tolist()}
        assert kept == set(
            map(tuple, edges.nlargest(2, "sttc")[["u", "v"]].to_numpy().tolist()))

    def test_unsorted_merged_unit_gives_the_sorted_answer(self):
        spikes, edges = self._spikes_and_edges()
        good = compute_ccg(spikes, edges, ConnectivityConfig())["counts"]
        shuffled = {u: np.concatenate([s[400:], s[:400]]) for u, s in spikes.items()}
        got = compute_ccg(shuffled, edges, ConnectivityConfig())["counts"]
        assert np.array_equal(got, good)

    def test_disabled_returns_empty_block(self):
        spikes, edges = self._spikes_and_edges()
        out = compute_ccg(spikes, edges, ConnectivityConfig(ccg_enable=False))
        assert out["counts"].shape == (0, 201)
        assert out["meta"]["ccg_enabled"] is False

    def test_summary_scalars(self):
        spikes, edges = self._spikes_and_edges()
        out = compute_ccg(spikes, edges, ConnectivityConfig())
        s = ccg_summary(out["per_pair"])
        assert s["n_ccg_sig_pairs"] >= 1
        assert 0.0 <= s["frac_ccg_sig"] <= 1.0
        assert 0.0 <= s["ccg_flow_asymmetry"] <= 1.0

    def test_summary_of_nothing_is_nan_not_a_crash(self):
        s = ccg_summary(pd.DataFrame(columns=["u", "v", *CCG_COLUMNS]))
        assert s["n_ccg_sig_pairs"] == 0 and np.isnan(s["frac_ccg_sig"])


class TestCcgInResults:
    def test_end_to_end_carries_ccg(self):
        rng = np.random.default_rng(9)
        base = np.sort(rng.uniform(0, 120, 700))
        spikes = {u: np.sort(base + u * 0.004 + rng.normal(0, 0.0005, base.size))
                  for u in range(4)}
        res = compute_connectivity(
            spikes, config=ConnectivityConfig(n_shuffle=10))
        assert res.ccg_counts.shape[0] == len(res.edges) == res.ccg_pairs.shape[0]
        assert set(CCG_COLUMNS) <= set(res.edges.columns)
        assert res.diagnostics["ccg_n_pairs"] == len(res.edges)
        assert "frac_ccg_sig" in res.graph_metrics
        # unit u leads unit v for every edge in this construction
        assert res.edges.loc[res.edges["ccg_sig"].fillna(False),
                             "ccg_syn_dir"].eq("u->v").all()

    def test_empty_results_share_the_full_schema(self):
        empty = empty_connectivity_results(ConnectivityConfig(), reason="sparse")
        full_keys = set(compute_connectivity(
            {u: np.sort(np.random.default_rng(1).uniform(0, 60, 40)) + u * 0.01
             for u in range(3)},
            config=ConnectivityConfig(n_shuffle=5)).graph_metrics)
        assert full_keys <= set(empty.graph_metrics)
        assert list(empty.edges.columns)[-len(CCG_COLUMNS):] == CCG_COLUMNS
        assert empty.ccg_counts.shape == (0, 201)
