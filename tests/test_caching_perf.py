"""Tests for the dashboard caching + raster-PNG performance work (Caching guide).

Covers Tier 0 (path manifest from pipeline_cache.json replacing globbing),
Tier 1 (signature-keyed in-process memoize), and the overview raster-PNG path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yuxin_mea.analysis.burst_inspector import well_output_dirs_from_cache
from yuxin_mea.analysis.plate_raster_synchrony import (
    PlateViewerConfig,
    WellRecord,
    build_plate_figure,
    build_single_well_figure,
    load_plate_data,
)
from yuxin_mea.analysis.raster_image import (
    png_to_data_uri,
    render_overview_pngs,
    render_well_png,
)
from yuxin_mea.dashboard import data_cache
from yuxin_mea.dashboard.data_cache import (
    clear_cache,
    data_sig,
    load_plate_data_cached,
)


def _make_recording(root: Path, n_ok: int = 3):
    """Lay out a synthetic recording (burst + curation outputs + pipeline cache).

    Returns ``(analysis_root, recording_key, burst_root, curation_root)``.
    """
    ar = root / "analysis"
    ar.mkdir()
    rk = "CX/1/2/Network/9"
    rec = "rec0000"
    burst_root = ar / "burst_detection_data"
    curation_root = ar / "curation_data"
    t = np.linspace(0, 300, 800)
    cache: dict[str, dict] = {}
    for i in range(n_ok):
        wid = f"well{i:03d}"
        bdir = burst_root / rk / rec / wid / "burst_detection"
        bdir.mkdir(parents=True)
        cdir = curation_root / rk / rec / wid / "auto_curation"
        cdir.mkdir(parents=True)
        np.save(bdir / "plot_signals.npy",
                {"t": t, "participation_signal": np.abs(np.sin(t / 7 + i))})
        pd.DataFrame({"start": [10.0 + i], "end": [12.0 + i]}).to_pickle(
            bdir / "network_bursts.pkl")
        np.save(cdir / "curated_spike_times.npy",
                {f"u{j}": np.array([1.0 * j, 2.0 * j, 3.0]) for j in range(6)})
        cache[f"{rk}/{rec}/{wid}"] = {
            "recording_key": rk, "well_id": f"{rec}/{wid}", "created_at": 0.0,
            "tasks": {
                "burst_detection": {"status": "complete", "dependencies": [],
                                    "output_path": str(bdir), "last_updated": 0.0,
                                    "error": None, "config": {}},
                "auto_curation": {"status": "complete", "dependencies": [],
                                  "output_path": str(cdir), "last_updated": 0.0,
                                  "error": None, "config": {}},
            },
        }
    (ar / "pipeline_cache.json").write_text(json.dumps(cache))
    return ar, rk, burst_root, curation_root


def test_data_sig_identity_and_bust(tmp_path):
    p = tmp_path / "a.npy"
    np.save(p, {"x": np.arange(3)})
    s1 = data_sig([p])
    assert data_sig([p]) == s1  # unchanged → identical
    np.save(p, {"x": np.arange(5)})  # rewrite changes size/mtime
    assert data_sig([p]) != s1
    # missing file → stable, distinct component (never raises)
    assert data_sig([tmp_path / "nope.npy"]) == ((str(tmp_path / "nope.npy"), -1, -1),)


def test_well_output_dirs_from_cache(tmp_path):
    ar, rk, burst_root, curation_root = _make_recording(tmp_path, n_ok=3)
    burst_dirs = well_output_dirs_from_cache(ar, rk, "burst_detection")
    cur_dirs = well_output_dirs_from_cache(ar, rk, "auto_curation")
    assert set(burst_dirs) == {"well000", "well001", "well002"}
    assert burst_dirs["well000"] == burst_root / rk / "rec0000" / "well000" / "burst_detection"
    assert cur_dirs["well001"] == curation_root / rk / "rec0000" / "well001" / "auto_curation"
    # missing cache → empty (caller falls back to discovery)
    assert well_output_dirs_from_cache(tmp_path / "nope", rk, "burst_detection") == {}


def test_manifest_matches_glob(tmp_path):
    """Manifest fast-path must resolve the same records as legacy globbing."""
    ar, rk, burst_root, curation_root = _make_recording(tmp_path, n_ok=3)
    burst_dirs = well_output_dirs_from_cache(ar, rk, "burst_detection")
    cur_dirs = well_output_dirs_from_cache(ar, rk, "auto_curation")

    legacy = load_plate_data(burst_root, curation_root, rk)
    manifest = load_plate_data(burst_root, curation_root, rk,
                               burst_well_dirs=burst_dirs, curation_well_dirs=cur_dirs)
    assert [w.status for w in legacy] == [w.status for w in manifest]
    assert legacy[0].status == "ok" and manifest[0].status == "ok"
    assert set(legacy[0].spike_times) == set(manifest[0].spike_times)
    assert legacy[3].status == "missing" and manifest[3].status == "missing"


def test_load_plate_data_cached_hit_and_bust(tmp_path):
    ar, rk, burst_root, curation_root = _make_recording(tmp_path, n_ok=2)
    burst_dirs = well_output_dirs_from_cache(ar, rk, "burst_detection")
    cur_dirs = well_output_dirs_from_cache(ar, rk, "auto_curation")
    clear_cache()
    kw = dict(recording_key=rk, source="traditional", burst_root=burst_root,
              curation_root=curation_root, burst_terminal="burst_detection",
              experiment_cache_path=None, burst_well_dirs=burst_dirs,
              curation_well_dirs=cur_dirs)
    a = load_plate_data_cached(**kw)
    assert load_plate_data_cached(**kw) is a  # identity on cache hit
    # rewrite one well → signature busts → fresh object
    np.save(cur_dirs["well000"] / "curated_spike_times.npy", {"only": np.array([9.0])})
    c = load_plate_data_cached(**kw)
    assert c is not a and set(c[0].spike_times) == {"only"}


def test_render_overview_pngs_cache_and_bust(tmp_path):
    t = np.linspace(0, 300, 500)
    wr = WellRecord(
        well_id="well000", well_name="A1", groupname="NPH",
        plot_signals={"t": t, "participation_signal": np.abs(np.sin(t))},
        spike_times={f"u{i}": np.sort(np.random.RandomState(i).uniform(0, 300, 200))
                     for i in range(10)},
        status="ok",
    )
    missing = WellRecord(well_id="well001", well_name="A2", groupname="NPH", status="missing")
    cache_root = tmp_path / "cache"
    m = render_overview_pngs([wr, missing], cache_root, well_sigs={"well000": "s1"},
                             recording_key="CX/1", source="traditional")
    assert set(m) == {"well000"}  # missing well skipped
    png = Path(m["well000"])
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    mtime = png.stat().st_mtime_ns
    # same signature → cache hit, no re-render
    render_overview_pngs([wr], cache_root, well_sigs={"well000": "s1"},
                         recording_key="CX/1", source="traditional")
    assert Path(m["well000"]).stat().st_mtime_ns == mtime
    # new signature → new PNG file
    m2 = render_overview_pngs([wr], cache_root, well_sigs={"well000": "s2"},
                              recording_key="CX/1", source="traditional")
    assert Path(m2["well000"]) != png
    assert png_to_data_uri(png).startswith("data:image/png;base64,")


def test_render_well_png_is_read_only(tmp_path):
    spikes = {"u0": np.array([1.0, 2.0, 3.0])}
    wr = WellRecord(well_id="well000", well_name="A1", groupname="G",
                    plot_signals={"t": np.arange(3.0), "participation_signal": np.arange(3.0)},
                    spike_times=spikes, status="ok")
    render_well_png(wr, tmp_path / "w.png")
    assert list(spikes["u0"]) == [1.0, 2.0, 3.0]  # inputs untouched


def _ok_record(wid="well000"):
    t = np.linspace(0, 10, 200)
    return WellRecord(
        well_id=wid, well_name="A1", groupname="NPH",
        plot_signals={"t": t, "participation_signal": np.abs(np.sin(t)),
                      "participation_threshold": 0.5},
        spike_times={"u0": np.array([1.0, 2.0]), "u1": np.array([3.0])},
        event_intervals={"network_bursts": [{"start": 1.0, "end": 2.0}],
                         "burstlets": [], "superbursts": []},
        status="ok",
    )


def test_build_single_well_figure():
    fig = build_single_well_figure(_ok_record(), PlateViewerConfig())
    assert len(fig.data) >= 1
    assert {tr.type for tr in fig.data} == {"scattergl"}
    # missing well → annotation, no crash
    miss = WellRecord(well_id="well005", well_name="B1", groupname="G", status="missing")
    assert len(build_single_well_figure(miss, PlateViewerConfig()).layout.annotations) >= 1


def test_build_plate_figure_still_builds():
    """The refactor (extracting _add_well_traces) must not change 24-well output."""
    recs = [_ok_record(f"well{i:03d}") if i < 3
            else WellRecord(well_id=f"well{i:03d}", well_name="x", groupname="G", status="missing")
            for i in range(24)]
    fig = build_plate_figure(recs, PlateViewerConfig())
    assert len(fig.data) >= 3


# ---------------------------------------------------------------------------
# Stage C: Tier 2 persistence + Tier 3 viewer bundles
# ---------------------------------------------------------------------------


def test_viewer_bundle_roundtrip_and_staleness(tmp_path):
    from yuxin_mea.analysis.viewer_bundle import (
        bundle_path, read_viewer_bundle, write_viewer_bundle,
    )
    recs = [_ok_record("well000"),
            WellRecord(well_id="well001", well_name="x", groupname="G", status="missing")]
    bp = bundle_path(tmp_path / "bundles", "CX/1/2/Network/9", "traditional")
    write_viewer_bundle(recs, bp, signature="sigA")
    got = read_viewer_bundle(bp, expected_signature="sigA")
    assert got is not None and [w.status for w in got] == ["ok", "missing"]
    assert read_viewer_bundle(bp, expected_signature="STALE") is None  # sig mismatch
    assert read_viewer_bundle(tmp_path / "nope.pkl") is None  # absent


def test_load_plate_data_cached_prefers_fresh_bundle(tmp_path, monkeypatch):
    from yuxin_mea.analysis.viewer_bundle import bundle_path, write_viewer_bundle
    ar, rk, burst_root, curation_root = _make_recording(tmp_path, n_ok=2)
    bwd = well_output_dirs_from_cache(ar, rk, "burst_detection")
    cwd = well_output_dirs_from_cache(ar, rk, "auto_curation")
    clear_cache()
    kw = dict(recording_key=rk, source="traditional", burst_root=burst_root,
              curation_root=curation_root, burst_terminal="burst_detection",
              experiment_cache_path=None, burst_well_dirs=bwd, curation_well_dirs=cwd,
              bundle_dir=tmp_path / "bundles")
    # assemble once (no bundle yet), write a bundle stamped with the live signature
    records = load_plate_data_cached(**kw)
    sig = data_cache.manifest_signature(bwd, cwd, None)
    write_viewer_bundle(records, bundle_path(tmp_path / "bundles", rk, "traditional"), signature=sig)

    calls = {"n": 0}
    real = data_cache.load_plate_data
    monkeypatch.setattr(data_cache, "load_plate_data",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1])
    clear_cache()
    load_plate_data_cached(**kw)
    assert calls["n"] == 0  # bundle hit → no per-well assembly
    # rewrite a source file → signature changes → stale bundle → fall back
    import time
    time.sleep(0.01)
    np.save(cwd["well000"] / "curated_spike_times.npy", {"z": np.array([1.0])})
    clear_cache()
    load_plate_data_cached(**kw)
    assert calls["n"] == 1


def test_tier2_persistence_survives_restart(tmp_path):
    import json

    import yuxin_mea.dashboard.cache as cmod
    from yuxin_mea.dashboard.app import build_app

    ar, rk, burst_root, curation_root = _make_recording(tmp_path, n_ok=2)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"global": {"analysis_root": str(ar),
                                          "cache_root": str(tmp_path / "cache")}}))
    app = build_app(cfg)
    assert cmod.is_active(), "flask cache should activate with cache_root set"

    bwd = well_output_dirs_from_cache(ar, rk, "burst_detection")
    cwd = well_output_dirs_from_cache(ar, rk, "auto_curation")
    calls = {"n": 0}
    real = data_cache.load_plate_data
    data_cache.load_plate_data = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1]
    try:
        with app.server.app_context():
            kw = dict(recording_key=rk, source="traditional", burst_root=burst_root,
                      curation_root=curation_root, burst_terminal="burst_detection",
                      experiment_cache_path=None, burst_well_dirs=bwd, curation_well_dirs=cwd)
            load_plate_data_cached(**kw)
            assert calls["n"] == 1
            clear_cache()  # simulate restart: drop in-process L1
            load_plate_data_cached(**kw)
            assert calls["n"] == 1  # served from persistent L2, no reload
    finally:
        data_cache.load_plate_data = real
        clear_cache()
