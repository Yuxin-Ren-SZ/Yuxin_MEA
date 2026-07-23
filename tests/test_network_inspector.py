"""Tests for the network-inspector library and its dashboard page.

The fixture deliberately reproduces the shape of the live data rather than an
idealised one:

* ``pipeline_cache.json`` records ``auto_curation`` and ``connectivity`` but
  **not** ``criticality`` / ``directed_connectivity`` — exactly as production
  does, because those two were run out of band. Discovery for them therefore has
  to go through the convention-path reconstruction, not the manifest.
* one well has connectivity but no criticality/directed output,
* one well has empty (sparse-well) results,
* one well has a ``quality_metrics.pkl`` with no ``loc_x``/``loc_y``, so the
  three position-dependent figures must hit their own empty state,
* ``node_metrics.parquet`` is present for only one well (it has no committed
  producer, so it is optional everywhere).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from yuxin_mea.analysis import network_inspector as ni


RK = "CXTEST/260401/T000001/Network/000001"
REC = "rec0000"
FULL, NO_CRIT, EMPTY, NO_POS = "well001", "well002", "well003", "well004"
ALL_WELLS = (FULL, NO_CRIT, EMPTY, NO_POS)

# Fill used for a well that has no value for the selected metric.
_CELL_EMPTY_BG = "var(--bg-deep)"


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #
def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _conn_dir(root: Path, well: str) -> Path:
    return root / "connectivity_data" / RK / REC / well / "connectivity"


def _write_ccg(d: Path, edges: pd.DataFrame, n_pairs: int | None = None) -> pd.DataFrame:
    """Stage a ccg.npz for the first ``n_pairs`` edges; returns edges + ccg cols.

    ``n_pairs`` below ``len(edges)`` reproduces the ``ccg_max_pairs`` cap, where
    ccg.npz is SHORTER than edges.parquet and the two can only be joined on (u, v).
    """
    lags_ms = np.arange(-100, 101, dtype=float)
    take = len(edges) if n_pairs is None else min(n_pairs, len(edges))
    rng = np.random.default_rng(3)
    counts = rng.poisson(4.0, size=(take, lags_ms.size)).astype(np.int32)
    peak_bins = [100 + 4 + (k % 3) for k in range(take)]
    for k, pb in enumerate(peak_bins):
        counts[k, pb] += 200
    baseline = np.full((take, lags_ms.size), 4.0, dtype=np.float32)
    pairs = edges[["u", "v"]].to_numpy()[:take]
    np.savez(d / "ccg.npz", counts=counts, baseline=baseline, lags_ms=lags_ms,
             pairs=pairs, n_ref_spikes=np.full(take, 500),
             n_tgt_spikes=np.full(take, 500), bin_ms=np.asarray(1.0),
             window_ms=np.asarray(100.0), syn_lo_ms=np.asarray(0.8),
             syn_hi_ms=np.asarray(8.0))
    out = edges.copy()
    out["ccg_peak_lag_ms"] = [float(peak_bins[i] - 100) if i < take else np.nan
                              for i in range(len(out))]
    out["ccg_peak_z"] = [50.0 - i if i < take else np.nan for i in range(len(out))]
    out["ccg_sig"] = pd.array([i < take for i in range(len(out))], dtype="boolean")
    out["ccg_syn_dir"] = ["u->v" if i < take else None for i in range(len(out))]
    out["ccg_asymmetry"] = [0.7 if i < take else np.nan for i in range(len(out))]
    return out


def _write_connectivity(root: Path, well: str, n: int, *, with_nodes: bool,
                        with_positions: bool = True, with_ccg: bool = False,
                        ccg_pairs: int | None = None) -> None:
    d = _conn_dir(root, well)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(well)) % 2**32)
    W = np.zeros((n, n))
    if n:
        W = rng.random((n, n)) * 0.4
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
    unit_ids = np.arange(n)
    np.save(d / "sttc_matrix.npy", W)
    np.savez(d / "sttc_sweep.npz", **{
        "dt_5ms": W * 0.5, "dt_10ms": W * 0.8, "dt_20ms": W,
        "dt_50ms": W * 1.1, "unit_ids": unit_ids,
    })
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0.25:
                rows.append({"u": i, "v": j, "sttc": float(W[i, j]),
                             "dist_um": float(50 * abs(i - j))
                             if with_positions else float("nan")})
    edges = pd.DataFrame(rows, columns=["u", "v", "sttc", "dist_um"])
    gm = {
        "mean_sttc": float(W.mean()) if n else 0.0, "edge_density": 0.3,
        "mean_degree": 4.0, "clustering_coeff": 0.5, "modularity": 0.2,
        "global_efficiency": 0.6, "small_worldness": 1.2,
        "n_nodes": n, "n_edges": len(rows),
    }
    diag = {
        "dt_primary": 0.02, "dt_sweep": [0.005, 0.01, 0.02, 0.05],
        "n_units": n, "n_edges": len(rows), "T": 300.0,
    }
    if with_ccg and len(edges):
        edges = _write_ccg(d, edges, ccg_pairs)
        n_ccg = len(edges) if ccg_pairs is None else min(ccg_pairs, len(edges))
        gm.update({"n_ccg_sig_pairs": n_ccg,
                   "frac_ccg_sig": n_ccg / len(edges),
                   "mean_abs_ccg_lag_ms": 5.0, "ccg_flow_asymmetry": 0.7})
        diag.update({"ccg_enabled": True, "ccg_bin_ms": 1.0,
                     "ccg_window_ms": 100.0, "ccg_n_pairs": n_ccg,
                     "ccg_pairs_capped": len(edges) - n_ccg})
    edges.to_parquet(d / "edges.parquet")
    _write_json(d / "graph_metrics.json", gm)
    _write_json(d / "diagnostics.json", diag)
    if with_nodes and n:
        roles = ["connector_hub", "provincial_hub", "peripheral", "leaf"]
        nm = pd.DataFrame({
            "degree": rng.integers(1, 10, n),
            "strength": rng.random(n),
            "betweenness": rng.random(n) * 0.1,
            "clustering": rng.random(n),
            "local_efficiency": rng.random(n),
            "module": rng.integers(0, 3, n),
            "z_within": rng.normal(0, 1.5, n),
            "participation": rng.random(n),
            "role": [roles[i % 4] for i in range(n)],
        }, index=pd.Index(unit_ids, name="unit_id"))
        nm.to_parquet(d / "node_metrics.parquet")
        _write_json(d / "node_scalars.json", {
            "n_nodes": n, "n_edges": len(rows), "hub_fraction": 0.25,
            "leaf_fraction": 0.25, "participation_mean": 0.5,
            "mean_betweenness": 0.05, "assortativity": -0.1, "rich_club": 0.8,
            "degree_cv": 0.4, "degree_skew": 0.1,
        })


def _write_criticality(root: Path, well: str) -> None:
    d = root / "criticality_data" / RK / REC / well / "criticality"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    size = rng.pareto(1.5, 800) * 4 + 1
    dur = np.clip((size / 2).astype(int) + 1, 1, None)
    pd.DataFrame({"size": size.astype(int), "duration": dur}).to_parquet(
        d / "avalanches.parquet")
    _write_json(d / "criticality_metrics.json", {
        "branching_ratio_mr": 0.97, "branching_ratio_naive": 0.7,
        "aval_tau": 2.1, "aval_alpha": 2.6, "gamma_fit": 1.3,
        "gamma_pred": 1.45, "dcc": 0.15, "n_avalanches": 800,
        "mean_aval_size": 4.7, "mean_aval_duration_bins": 2.4,
    })
    _write_json(d / "powerlaw_fits.json",
                {"size": {"exponent": 2.1, "xmin": 4.0, "n": 800}})
    _write_json(d / "diagnostics.json",
                {"dt_s": 0.012, "n_units": 8, "bin_mode": "mean_iei"})


def _write_directed(root: Path, well: str, n: int) -> None:
    d = root / "directed_connectivity_data" / RK / REC / well / "directed"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    TE = rng.random((n, n)) * 0.001
    np.fill_diagonal(TE, 0.0)
    np.save(d / "te_matrix.npy", TE)
    np.save(d / "unit_ids.npy", np.arange(n))
    ii, jj = np.where(TE > 0.0005)
    pd.DataFrame({"source": ii, "target": jj, "te": TE[ii, jj]},
                 columns=["source", "target", "te"]).to_parquet(
        d / "directed_edges.parquet")
    _write_json(d / "directed_metrics.json", {
        "mean_te": 0.0007, "te_edge_density": 0.18, "degree_asymmetry": 2.9,
        "reciprocity": 0.67, "flow_hierarchy": 0.04, "causal_density": 0.18,
    })
    _write_json(d / "diagnostics.json", {"n_units": n, "bin_s": 0.005})


def _write_spatial(root: Path, well: str) -> None:
    d = root / "spatial_map_data" / RK / REC / well / "spatial_map"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    np.save(d / "activity_field.npy", rng.random((16, 16)))
    pd.DataFrame({
        "burst_index": np.arange(5), "origin_x": rng.random(5) * 1000,
        "origin_y": rng.random(5) * 1000, "speed_um_per_ms": rng.random(5) * 200,
        "direction_rad": rng.random(5), "r2_planar_fit": rng.random(5),
        "n_participating": rng.integers(3, 20, 5), "delay_ms": rng.random(5) * 50,
    }).to_parquet(d / "burst_propagation.parquet")
    _write_json(d / "spatial_metrics.json", {
        "active_area_frac": 0.11,
        # bin coordinates over the 16x16 field, as spatial_summary writes them
        "activity_centroid": [7.5, 6.25],
        "activity_gini": 0.87, "mean_prop_speed_um_ms": 139.0,
        "prop_planarity": 0.13, "mean_prop_delay_ms": 60.0,
    })
    _write_json(d / "diagnostics.json", {
        "n_units": 8, "bins": 16, "sigma_um": 50.0, "n_bursts_used": 5,
        "field_extent": [0.0, 1000.0, 0.0, 800.0],
    })


def _write_curation(root: Path, well: str, n: int, *, with_positions: bool) -> None:
    d = root / "curation_data" / RK / REC / well / "auto_curation"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    cols: dict = {"firing_rate": rng.random(n)}
    if with_positions:
        cols["loc_x"] = rng.random(n) * 1000
        cols["loc_y"] = rng.random(n) * 800
    pd.DataFrame(cols, index=pd.Index(np.arange(n), name="unit_id")).to_pickle(
        d / "quality_metrics.pkl")


@pytest.fixture
def analysis_root(tmp_path: Path) -> Path:
    """A miniature analysis root reproducing the live data's gaps."""
    root = tmp_path / "analysis"
    root.mkdir()

    _write_connectivity(root, FULL, 8, with_nodes=True, with_ccg=True)
    _write_criticality(root, FULL)
    _write_directed(root, FULL, 8)
    _write_spatial(root, FULL)
    _write_curation(root, FULL, 8, with_positions=True)

    # connectivity + spatial only — no criticality, no directed, no node metrics
    _write_connectivity(root, NO_CRIT, 6, with_nodes=False)
    _write_spatial(root, NO_CRIT)
    _write_curation(root, NO_CRIT, 6, with_positions=True)

    # sparse well: empty results written by the COMPLETE-but-empty path
    _write_connectivity(root, EMPTY, 0, with_nodes=False)
    _write_curation(root, EMPTY, 0, with_positions=True)

    # positions absent: quality_metrics.pkl carries no loc_x / loc_y. Its CCG
    # block is capped, so ccg.npz is shorter than edges.parquet.
    _write_connectivity(root, NO_POS, 6, with_nodes=True, with_positions=False,
                        with_ccg=True, ccg_pairs=2)
    _write_directed(root, NO_POS, 6)
    _write_curation(root, NO_POS, 6, with_positions=False)

    # The manifest knows auto_curation + connectivity only — criticality and
    # directed_connectivity are absent, exactly as in production.
    cache = {}
    for well in ALL_WELLS:
        cache[f"{RK}/{REC}/{well}"] = {
            "recording_key": RK,
            "well_id": f"{REC}/{well}",
            "tasks": {
                "auto_curation": {
                    "status": "complete",
                    "output_path": str(root / "curation_data" / RK / REC / well
                                       / "auto_curation"),
                },
                "connectivity": {
                    "status": "complete",
                    "output_path": str(_conn_dir(root, well)),
                },
                "spatial_map": {
                    "status": "complete",
                    "output_path": str(root / "spatial_map_data" / RK / REC / well
                                       / "spatial_map"),
                },
            },
        }
    (root / "pipeline_cache.json").write_text(json.dumps(cache))
    ni.clear_cache()
    return root


def _dirs_for(root: Path, well: str) -> dict:
    stage_dirs = ni.all_stage_dirs(root, RK)
    return {stage: d.get(well) for stage, d in stage_dirs.items()}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_lists_recordings_with_connectivity(analysis_root: Path):
    assert ni.list_network_recordings(analysis_root) == [RK]


def test_wells_in_recording_keeps_rec_name(analysis_root: Path):
    """The compound well id must survive — build_output_path needs rec_name."""
    assert ni.wells_in_recording(analysis_root, RK) == [
        (REC, w) for w in ALL_WELLS
    ]


def test_manifest_discovery_for_connectivity_and_spatial(analysis_root: Path):
    conn = ni.stage_well_dirs(analysis_root, RK, "connectivity")
    assert set(conn) == set(ALL_WELLS)
    spatial = ni.stage_well_dirs(analysis_root, RK, "spatial_map")
    assert set(spatial) == {FULL, NO_CRIT}


def test_convention_discovery_for_stages_absent_from_the_manifest(analysis_root: Path):
    """criticality / directed are in no manifest — only reconstruction finds them."""
    from yuxin_mea.analysis.burst_inspector import well_output_dirs_from_cache

    assert well_output_dirs_from_cache(analysis_root, RK, "criticality") == {}
    assert well_output_dirs_from_cache(analysis_root, RK, "directed_connectivity") == {}

    assert set(ni.stage_well_dirs(analysis_root, RK, "criticality")) == {FULL}
    assert set(ni.stage_well_dirs(analysis_root, RK, "directed")) == {FULL, NO_POS}


def test_stage_output_root_config_override_wins(analysis_root: Path):
    default = ni.stage_output_root(analysis_root, "criticality", {})
    assert default == analysis_root / "criticality_data"
    override = ni.stage_output_root(
        analysis_root, "criticality", {"criticality": "/somewhere/else"})
    assert override == Path("/somewhere/else")
    relative = ni.stage_output_root(
        analysis_root, "criticality", {"criticality": "crit"})
    assert relative == analysis_root / "crit"


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
def test_full_well_loads_every_stage(analysis_root: Path):
    b = ni.load_network_bundle(_dirs_for(analysis_root, FULL),
                               recording_key=RK, rec_name=REC, well_id=FULL)
    assert b.stages_present == ["connectivity", "spatial_map", "criticality",
                                "directed"]
    assert b.sttc_matrix.shape == (8, 8)
    assert b.unit_ids is not None and len(b.unit_ids) == 8
    assert b.node_metrics is not None and len(b.node_metrics) == 8
    assert b.avalanches is not None and len(b.avalanches) == 800
    assert b.te_matrix.shape == (8, 8)
    assert b.activity_field.shape == (16, 16)
    assert b.positions is not None and len(b.positions) == 8


def test_absent_stages_stay_none(analysis_root: Path):
    b = ni.load_network_bundle(_dirs_for(analysis_root, NO_CRIT))
    assert b.stages_present == ["connectivity", "spatial_map"]
    assert b.criticality_metrics is None
    assert b.avalanches is None
    assert b.te_matrix is None
    # node metrics have no committed producer -> optional even with connectivity
    assert b.node_metrics is None


def test_positions_absent_is_independent_of_stage_presence(analysis_root: Path):
    b = ni.load_network_bundle(_dirs_for(analysis_root, NO_POS))
    assert b.has("connectivity") and b.has("directed")
    assert b.positions is None


def test_bundle_cache_busts_on_mtime(analysis_root: Path):
    dirs = _dirs_for(analysis_root, FULL)
    first = ni.load_network_bundle_cached(dirs, well_id=FULL)
    assert ni.load_network_bundle_cached(dirs, well_id=FULL) is first
    gm = _conn_dir(analysis_root, FULL) / "graph_metrics.json"
    gm.write_text(json.dumps({"mean_sttc": 0.9, "n_nodes": 8, "n_edges": 3}))
    assert ni.load_network_bundle_cached(dirs, well_id=FULL) is not first


# --------------------------------------------------------------------------- #
# Figures — every builder, on every kind of well
# --------------------------------------------------------------------------- #
_BUILDERS = [
    "fig_sttc_matrix", "fig_sttc_graph", "fig_sttc_distance", "fig_dt_sweep",
    "fig_ccg_heatmap", "fig_ccg_small_multiples",
    "fig_cartography", "fig_node_degree", "fig_avalanche_dist", "fig_crackling",
    "fig_te_matrix", "fig_te_graph", "fig_activity_field", "fig_burst_propagation",
]


@pytest.mark.parametrize("well", ALL_WELLS)
@pytest.mark.parametrize("builder", _BUILDERS)
def test_every_builder_returns_a_figure(analysis_root: Path, well: str, builder: str):
    """No well shape may raise — a missing stage renders an empty state."""
    b = ni.load_network_bundle(_dirs_for(analysis_root, well), well_id=well)
    fig = getattr(ni, builder)(b)
    assert isinstance(fig, go.Figure)


def test_position_dependent_figures_report_missing_positions(analysis_root: Path):
    b = ni.load_network_bundle(_dirs_for(analysis_root, NO_POS), well_id=NO_POS)
    for builder in ("fig_sttc_graph", "fig_te_graph", "fig_sttc_distance"):
        fig = getattr(ni, builder)(b)
        text = " ".join(a.text or "" for a in fig.layout.annotations)
        assert "no unit positions" in text, builder


def test_ccg_absent_renders_an_empty_state(analysis_root: Path):
    """Wells computed before CCGs landed have no ccg.npz — that must not raise."""
    b = ni.load_network_bundle(_dirs_for(analysis_root, NO_CRIT), well_id=NO_CRIT)
    assert b.ccg_counts is None
    for builder in ("fig_ccg_heatmap", "fig_ccg_small_multiples"):
        fig = getattr(ni, builder)(b)
        text = " ".join(a.text or "" for a in fig.layout.annotations)
        assert "no correlograms" in text, builder


def test_ccg_figures_use_the_pair_key_not_row_order(analysis_root: Path):
    """NO_POS is staged capped: ccg.npz has 2 rows, edges.parquet has more."""
    b = ni.load_network_bundle(_dirs_for(analysis_root, NO_POS), well_id=NO_POS)
    assert b.ccg_counts.shape[0] == 2 < len(b.edges)
    heat = ni.fig_ccg_heatmap(b)
    assert len(heat.data[0].y) == 2                     # one row per computed pair
    assert len(heat.data[0].z) == 2
    small = ni.fig_ccg_small_multiples(b, n=12)
    # bar + baseline line per computed pair — never one per edge
    assert len(small.data) == 4


def test_ccg_heatmap_sorts_rows_by_peak_lag(analysis_root: Path):
    b = ni.load_network_bundle(_dirs_for(analysis_root, FULL), well_id=FULL)
    fig = ni.fig_ccg_heatmap(b)
    lags = b.edges.set_index(
        b.edges["u"].astype(str) + "→" + b.edges["v"].astype(str)
    )["ccg_peak_lag_ms"]
    shown = [lags[label] for label in fig.data[0].y]
    assert shown == sorted(shown)


def test_full_well_figures_carry_data(analysis_root: Path):
    b = ni.load_network_bundle(_dirs_for(analysis_root, FULL), well_id=FULL)
    assert len(ni.fig_sttc_matrix(b).data) == 1
    assert len(ni.fig_cartography(b).data) == 4          # four roles present
    assert len(ni.fig_te_graph(b).layout.annotations) > 0  # arrows drawn
    assert len(ni.fig_avalanche_dist(b).data) >= 2


def test_activity_field_centroid_is_mapped_from_bin_to_micron_coords(
    analysis_root: Path,
):
    """`activity_centroid` is stored in BIN coordinates (spatial_map writes
    ``(xs * field).sum() / field.sum()`` over indices), while the heatmap is drawn
    on a micron axis. The marker must go through the same transform, and row 0
    must sit at ymin (no vertical flip)."""
    b = ni.load_network_bundle(_dirs_for(analysis_root, FULL), well_id=FULL)
    field = b.activity_field
    x0, x1, y0, y1 = b.spatial_diagnostics["field_extent"]

    # a bin-space centroid that is unambiguously off-centre in both axes
    b.spatial_metrics = dict(b.spatial_metrics, activity_centroid=[3.0, 1.0])
    fig = ni.fig_activity_field(b)
    heat, marker = fig.data[0], fig.data[1]

    rows, cols = field.shape
    dx, dy = (x1 - x0) / cols, (y1 - y0) / rows
    assert heat.x0 == pytest.approx(x0) and heat.dx == pytest.approx(dx)
    assert heat.y0 == pytest.approx(y0) and heat.dy == pytest.approx(dy)
    assert heat.dy > 0, "row 0 must map to ymin — a negative dy flips the field"
    assert marker.x[0] == pytest.approx(x0 + 3.5 * dx)
    assert marker.y[0] == pytest.approx(y0 + 1.5 * dy)
    # ...and it lands inside the plotted field, not in a corner
    assert x0 < marker.x[0] < x1 and y0 < marker.y[0] < y1


def test_activity_field_without_extent_stays_in_bin_units(analysis_root: Path):
    b = ni.load_network_bundle(_dirs_for(analysis_root, FULL), well_id=FULL)
    b.spatial_diagnostics = {k: v for k, v in b.spatial_diagnostics.items()
                             if k != "field_extent"}
    fig = ni.fig_activity_field(b)
    assert "bin" in fig.layout.xaxis.title.text
    # centroid is already in bins here -> plotted as-is (bin centre)
    cx, cy = b.spatial_metrics["activity_centroid"]
    assert fig.data[1].x[0] == pytest.approx(cx + 0.5)
    assert fig.data[1].y[0] == pytest.approx(cy + 0.5)


def test_node_table_and_kv(analysis_root: Path):
    b = ni.load_network_bundle(_dirs_for(analysis_root, FULL), well_id=FULL)
    tbl = ni.node_table(b)
    assert len(tbl) == 8 and "role" in tbl.columns
    kv = ni.metrics_kv(b, "connectivity")
    assert kv["mean STTC"] and kv["n units"] == "8"
    assert ni.node_table(ni.load_network_bundle(
        _dirs_for(analysis_root, NO_CRIT))).empty


# --------------------------------------------------------------------------- #
# Plate roll-up + colouring
# --------------------------------------------------------------------------- #
def test_plate_scalars_merges_every_stage(analysis_root: Path):
    scalars = ni.plate_scalars(ni.all_stage_dirs(analysis_root, RK))
    assert set(scalars) == set(ALL_WELLS)
    assert scalars[FULL]["mean_sttc"] is not None
    assert scalars[FULL]["dcc"] == pytest.approx(0.15)
    assert scalars[FULL]["flow_hierarchy"] == pytest.approx(0.04)
    assert scalars[FULL]["activity_gini"] == pytest.approx(0.87)
    # NO_CRIT has no criticality output at all
    assert "dcc" not in scalars[NO_CRIT]


def test_plate_scalars_busts_on_mtime(analysis_root: Path):
    dirs = ni.all_stage_dirs(analysis_root, RK)
    first = ni.plate_scalars(dirs)
    assert ni.plate_scalars(dirs) is first
    _write_json(_conn_dir(analysis_root, FULL) / "graph_metrics.json",
                {"mean_sttc": 0.99, "n_nodes": 8, "n_edges": 3})
    second = ni.plate_scalars(dirs)
    assert second is not first
    assert second[FULL]["mean_sttc"] == pytest.approx(0.99)


def test_plate_colors_sequential_and_diverging():
    seq, vmin, vmax = ni.plate_colors({"a": 0.0, "b": 1.0, "c": None}, "mean_sttc")
    assert set(seq) == {"a", "b"} and (vmin, vmax) == (0.0, 1.0)
    assert seq["a"][0] != seq["b"][0]

    # A diverging metric is symmetric about its centre, so equal deviations
    # either side of 1.0 land on opposite ends of the ramp.
    div, dmin, dmax = ni.plate_colors(
        {"lo": 0.5, "mid": 1.0, "hi": 1.5}, "branching_ratio_mr")
    assert dmin == pytest.approx(0.5) and dmax == pytest.approx(1.5)
    assert div["mid"][0].lower() == "#e6e1d2"      # neutral grey midpoint
    assert div["lo"][0] != div["hi"][0]


def test_plate_colors_all_missing():
    colors, vmin, vmax = ni.plate_colors({"a": None, "b": float("nan")}, "dcc")
    assert colors == {}
    assert np.isnan(vmin) and np.isnan(vmax)


def test_every_plate_metric_has_a_label():
    for metric in ni.PLATE_METRICS:
        assert metric in ni.METRIC_LABELS, metric


# --------------------------------------------------------------------------- #
# Page smoke test
# --------------------------------------------------------------------------- #
@pytest.fixture
def dash_app(tmp_path: Path):
    """Build the real app once.

    ``dash.register_page`` refuses to run before an app exists, so the page
    module cannot simply be imported on its own — building the app is what
    imports every page, and is therefore also the import test.
    """
    from yuxin_mea.dashboard import build_app

    cfg = tmp_path / "pipeline_config.json"
    cfg.write_text(json.dumps({
        "global": {"data_root": str(tmp_path / "raw"),
                   "analysis_root": str(tmp_path),
                   "figure_root": str(tmp_path / "figures")},
        "tasks": {},
    }))
    return build_app(cfg)


def test_page_module_imports(dash_app):
    """A typo in a callback signature fails here, not in the user's browser."""
    from yuxin_mea.dashboard.pages import network_inspector

    assert network_inspector.layout is not None


def test_page_registered_at_expected_path(dash_app):
    import dash

    paths = {p["path"] for p in dash.page_registry.values()}
    assert "/network-inspector" in paths
    names = {p["name"] for p in dash.page_registry.values()}
    assert "Network inspector" in names


def _grid_for(dash_app, analysis_root: Path, metric: str):
    """Render the plate grid through the real callback."""
    from yuxin_mea.dashboard.pages import network_inspector as page

    dash_app.server.config["YUXIN_MEA"]["analysis_root"] = analysis_root
    with dash_app.server.test_request_context():
        roots = [str(ni.stage_output_root(analysis_root, s, {}))
                 for s, _ in page._ROOT_INPUTS]
        return page._render_grid(1, RK, metric, *roots)


def test_grid_paints_every_well_for_a_universal_metric(dash_app, analysis_root: Path):
    grid, context, legend = _grid_for(dash_app, analysis_root, "mean_sttc")
    assert [c.id["index"] for c in grid.children] == list(ALL_WELLS)
    assert context["ok_wells"] == list(ALL_WELLS)
    assert "4/4 wells" in legend


def test_wells_stay_clickable_when_the_metric_is_missing(dash_app, analysis_root: Path):
    """A well without `dcc` still has connectivity worth opening — it must not
    drop out of the grid or out of prev/next stepping."""
    grid, context, legend = _grid_for(dash_app, analysis_root, "dcc")

    # every cell is a Button (clickable), not an inert Div
    assert all(type(c).__name__ == "Button" for c in grid.children)
    assert [c.id["index"] for c in grid.children] == list(ALL_WELLS)
    # ...and prev/next can reach all of them
    assert context["ok_wells"] == list(ALL_WELLS)
    # only FULL actually has a dcc value
    assert "1/4 wells" in legend
    painted = [c for c in grid.children
               if c.style["background"] != _CELL_EMPTY_BG]
    assert [c.id["index"] for c in painted] == [FULL]


def test_grid_excludes_curation_only_wells(dash_app, analysis_root: Path):
    """A well with auto_curation but no network stage is not a grid cell."""
    curation_only = "well009"
    _write_curation(analysis_root, curation_only, 4, with_positions=True)
    cache = json.loads((analysis_root / "pipeline_cache.json").read_text())
    cache[f"{RK}/{REC}/{curation_only}"] = {
        "recording_key": RK, "well_id": f"{REC}/{curation_only}",
        "tasks": {"auto_curation": {
            "status": "complete",
            "output_path": str(analysis_root / "curation_data" / RK / REC
                               / curation_only / "auto_curation"),
        }},
    }
    (analysis_root / "pipeline_cache.json").write_text(json.dumps(cache))
    ni.clear_cache()

    assert (REC, curation_only) in ni.wells_in_recording(analysis_root, RK)
    grid, context, _ = _grid_for(dash_app, analysis_root, "mean_sttc")
    assert curation_only not in [c.id["index"] for c in grid.children]
    assert curation_only not in context["ok_wells"]


def test_rail_places_the_page_in_the_network_section():
    from yuxin_mea.dashboard.components.layout import _NAV_META, _SECTION_ORDER

    section, glyph = _NAV_META["Network inspector"]
    assert section == "network" and section in _SECTION_ORDER
    assert glyph
