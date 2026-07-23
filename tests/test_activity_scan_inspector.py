"""Viewer-library tests: loaders, figure builders, empty states, thumbnails.

Builds a tiny on-disk output tree with :func:`write_activity_scan` (not the
whole compute path) so the viewer is tested against the exact files it reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from yuxin_mea.analysis import activity_scan_inspector as vi
from yuxin_mea.analysis.activity_scan import (
    GRID_COLS,
    GRID_ROWS,
    WellActivityResults,
    write_activity_scan,
)

REC = "CXTEST/260401/T000001/ActivityScan/000001"


def _synthetic_results(n: int = 30, seed: int = 0) -> WellActivityResults:
    rng = np.random.default_rng(seed)
    electrodes = np.arange(500, 500 + n)
    rows = electrodes // GRID_COLS
    cols = electrodes % GRID_COLS
    fr = np.abs(rng.normal(0.5, 0.4, n))
    df = pd.DataFrame({
        "x_um": cols * 17.5, "y_um": rows * 17.5, "row": rows, "col": cols,
        "n_configs": 1, "n_spikes": (fr * 30).astype(int), "duration_s": 30.0,
        "firing_rate_hz": fr, "noise_uv": np.abs(rng.normal(5.5, 0.5, n)),
        "rms_uv": np.abs(rng.normal(6.0, 0.5, n)),
        "amp_median_uv": -np.abs(rng.normal(30, 8, n)),
        "amp_p10_uv": -np.abs(rng.normal(40, 8, n)),
        "amp_max_uv": -np.abs(rng.normal(55, 8, n)),
        "online_n_spikes": rng.integers(0, 20, n),
    }, index=pd.Index(electrodes, name="electrode"))

    def _grid(col, fill=np.nan):
        g = np.full((GRID_ROWS, GRID_COLS), fill, dtype=float)
        g[rows, cols] = df[col].to_numpy()
        return g

    maps = {"firing_rate": _grid("firing_rate_hz"),
            "amp_median": _grid("amp_median_uv"),
            "noise": _grid("noise_uv"),
            "coverage": _grid("n_configs", fill=0.0)}
    stats = {"n_scanned": n, "coverage_frac": n / (GRID_ROWS * GRID_COLS),
             "n_active": int((fr >= 0.1).sum()), "active_frac": float((fr >= 0.1).mean()),
             "active_area_mm2": 1.23, "active_min_rate_hz": 0.1,
             "fr_mean_active_hz": float(fr[fr >= 0.1].mean()), "fr_gini": 0.4,
             "amp_median_uv": -30.0, "noise_median_uv": 5.5,
             "activity_centroid_um": [float(cols.mean() * 17.5), float(rows.mean() * 17.5)],
             "n_configs": 1, "duration_total_s": 30.0, "online_total_spikes": 42,
             "largest_active_component": 5, "fr_median_active_hz": 0.5,
             "fr_max_hz": float(fr.max()), "total_rate_hz": float(fr.sum())}
    diag = {"per_config": [{"store": "data0000", "n_channels": n, "n_samples": 300000,
                            "fs": 10000.0, "duration_s": 30.0,
                            "noise_median_uv": 5.5, "n_spikes": int(df["n_spikes"].sum())}]}
    return WellActivityResults(df, maps, stats, diag)


@pytest.fixture
def output_tree(tmp_path: Path) -> Path:
    root = tmp_path / "analysis"
    for well in ("well000", "well001", "well005"):
        write_activity_scan(_synthetic_results(seed=hash(well) % 100),
                            root / vi.DEFAULT_OUTPUT_SUBDIR / REC / well)
    (root / vi.DEFAULT_OUTPUT_SUBDIR / REC / "manifest.json").write_text(
        json.dumps({"recording_key": REC, "wells": {}}))
    return root


def test_discovery_lists_recordings_and_wells(output_tree: Path):
    assert vi.recordings_with_activity_scan(output_tree) == [REC]
    assert vi.wells_in_recording(output_tree, REC) == ["well000", "well001", "well005"]


def test_discovery_empty_when_absent(tmp_path: Path):
    assert vi.recordings_with_activity_scan(tmp_path) == []
    assert vi.wells_in_recording(tmp_path, REC) == []


def test_load_well_and_missing(output_tree: Path):
    res = vi.load_well(output_tree, REC, "well000")
    assert res is not None and res.stats["n_scanned"] == 30
    assert vi.load_well(output_tree, REC, "well099") is None


def test_well_name_labels():
    assert vi.well_name("well000") == "A1"
    assert vi.well_name("well005") == "A6"
    assert vi.well_name("well006") == "B1"
    assert vi.well_name("well023") == "D6"


@pytest.mark.parametrize("metric", vi.MAP_ORDER)
def test_fig_map_returns_figure(output_tree: Path, metric: str):
    res = vi.load_well(output_tree, REC, "well000")
    fig = vi.fig_map(res, metric)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_fig_map_centroid_is_in_microns(output_tree: Path):
    """The centroid marker must sit on the micron axis, not in bin space."""
    res = vi.load_well(output_tree, REC, "well000")
    fig = vi.fig_map(res, "firing_rate")
    marker = [t for t in fig.data if getattr(t, "mode", "") == "markers"]
    assert marker, "centroid marker missing"
    mx = marker[0].x[0]
    cx = res.stats["activity_centroid_um"][0]
    assert mx == pytest.approx(cx)
    # and it lies inside the micron extent, nowhere near a 0..220 bin index
    assert 0 <= mx <= GRID_COLS * 17.5


def test_active_mask_respects_threshold(output_tree: Path):
    res = vi.load_well(output_tree, REC, "well000")
    lo = vi.displayed_map(res, "active", active_min_rate_hz=0.01)
    hi = vi.displayed_map(res, "active", active_min_rate_hz=5.0)
    assert np.nansum(lo) >= np.nansum(hi)


def test_amplitude_map_is_magnitude(output_tree: Path):
    res = vi.load_well(output_tree, REC, "well000")
    grid = vi.displayed_map(res, "amplitude")
    finite = grid[np.isfinite(grid)]
    assert (finite >= 0).all()          # displayed as |µV|


@pytest.mark.parametrize("metric", ["firing_rate", "amplitude", "noise"])
def test_fig_histogram(output_tree: Path, metric: str):
    res = vi.load_well(output_tree, REC, "well000")
    assert isinstance(vi.fig_histogram(res, metric), go.Figure)


def test_fig_active_area_curve_and_config_coverage(output_tree: Path):
    res = vi.load_well(output_tree, REC, "well000")
    assert isinstance(vi.fig_active_area_curve(res), go.Figure)
    assert isinstance(vi.fig_config_coverage(res), go.Figure)


def test_empty_states_do_not_raise():
    fig = vi.fig_map(None, "firing_rate")
    assert isinstance(fig, go.Figure)
    assert vi.well_stats_table(None) == []
    assert isinstance(vi.fig_histogram(None, "noise"), go.Figure)


def test_well_stats_table_rows(output_tree: Path):
    res = vi.load_well(output_tree, REC, "well000")
    rows = vi.well_stats_table(res)
    labels = {r["metric"] for r in rows}
    assert "electrodes scanned" in labels
    assert "active area (mm²)" in labels


def test_plate_scalars_and_colors(output_tree: Path):
    vals = vi.plate_scalars(output_tree, REC, "active_frac")
    assert set(vals) == {"well000", "well001", "well005"}
    colors = vi.plate_colors(vals, "active_frac")
    assert set(colors) == set(vals)
    assert all(c.startswith("#") for c in colors.values())


def test_thumbnail_data_uri_and_cache(output_tree: Path, tmp_path: Path):
    res = vi.load_well(output_tree, REC, "well000")
    rdir = vi.well_dir(output_tree, REC, "well000")
    cache = tmp_path / "cache"
    uri = vi.well_png_data_uri(res, "firing_rate", cache, rdir)
    assert uri.startswith("data:image/png;base64,")
    # a second call hits the cached PNG, not a re-render
    pngs = list((cache / vi._RASTER_SUBDIR).glob("*.png"))
    assert len(pngs) == 1
    uri2 = vi.well_png_data_uri(res, "firing_rate", cache, rdir)
    assert uri2 == uri
    assert len(list((cache / vi._RASTER_SUBDIR).glob("*.png"))) == 1


def test_continuous_thumbnail_ignores_threshold(output_tree: Path, tmp_path: Path):
    """Moving the active-cut slider must not regenerate the FR/amp/noise PNGs."""
    res = vi.load_well(output_tree, REC, "well000")
    rdir = vi.well_dir(output_tree, REC, "well000")
    cache = tmp_path / "cache"
    vi.well_png_data_uri(res, "firing_rate", cache, rdir, 0.1)
    vi.well_png_data_uri(res, "firing_rate", cache, rdir, 1.5)   # different threshold
    assert len(list((cache / vi._RASTER_SUBDIR).glob("*.png"))) == 1
    # the active mask, by contrast, is threshold-dependent -> two PNGs
    vi.well_png_data_uri(res, "active", cache, rdir, 0.1)
    vi.well_png_data_uri(res, "active", cache, rdir, 1.5)
    assert len(list((cache / vi._RASTER_SUBDIR).glob("*.png"))) == 3


def test_shared_plate_range_pools_wells(output_tree: Path):
    wells = [vi.load_well(output_tree, REC, w)
             for w in vi.wells_in_recording(output_tree, REC)]
    rng = vi.plate_value_range(wells, "firing_rate")
    assert rng is not None and rng[1] > rng[0]
    # the shared range must cover every well's max (within the 98th pctile clip)
    per_well_max = max(w.electrodes["firing_rate_hz"].max() for w in wells)
    assert rng[1] <= per_well_max + 1e-6
    assert vi.plate_value_range(wells, "active") is None


def test_well_subtitle_tracks_threshold(output_tree: Path):
    res = vi.load_well(output_tree, REC, "well000")
    lo = vi.well_subtitle(res, "active", 0.01)
    hi = vi.well_subtitle(res, "active", 5.0)
    assert lo.endswith("active") and hi.endswith("active")
    assert int(lo.split("%")[0]) >= int(hi.split("%")[0])
    assert "µV" in vi.well_subtitle(res, "noise", 0.1)
