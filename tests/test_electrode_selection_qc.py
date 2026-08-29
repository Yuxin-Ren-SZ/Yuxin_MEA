"""Electrode-selection QC on a synthetic h5 built to the real Network layout.

The fixture writes ``/data_store/dataNNNN`` nodes the way a MaxTwo Network file
does — ``well_id``, ``settings/mapping``, ``groups/routed/channels`` — with
electrode sets *planted* per (well, scan), so extraction and every metric is
checked against ground truth rather than against another run of the same code.

The exact toroidal-shift null is the one piece of real machinery here, so it is
checked against a brute-force ``np.roll`` sweep rather than trusted.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from yuxin_mea.analysis import electrode_selection_inspector as VI
from yuxin_mea.analysis import electrode_selection_qc as Q
from yuxin_mea.analysis.activity_scan import GRID_COLS, GRID_ROWS, N_ELECTRODES


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #
def _write_node(store: h5py.Group, name: str, *, well_id: int,
                electrodes: list[int], routed: list[int] | None = None) -> None:
    """One (well, configuration) node. ``routed`` defaults to every channel."""
    node = store.create_group(name)
    node.create_dataset("well_id", data=np.array([well_id], dtype=np.int32))
    node.create_dataset("recording_id", data=np.array([0], dtype=np.int32))

    n_ch = len(electrodes)
    mapping = np.zeros(n_ch, dtype=[("channel", "<i4"), ("electrode", "<i4"),
                                    ("x", "<f8"), ("y", "<f8")])
    for i, el in enumerate(electrodes):
        row, col = divmod(el, GRID_COLS)
        mapping[i] = (i, el, col * 17.5, row * 17.5)
    node.create_group("settings").create_dataset("mapping", data=mapping)

    chans = np.arange(n_ch, dtype=np.uint16) if routed is None else \
        np.asarray(routed, dtype=np.uint16)
    node.create_group("groups").create_group("routed").create_dataset(
        "channels", data=chans)


def _write_file(path: Path, per_well: dict[int, list[int]],
                routed: dict[int, list[int]] | None = None) -> Path:
    with h5py.File(path, "w") as h5:
        store = h5.create_group("data_store")
        for i, (well, electrodes) in enumerate(sorted(per_well.items())):
            _write_node(store, f"data{i:04d}", well_id=well, electrodes=electrodes,
                        routed=(routed or {}).get(well))
    return path


@pytest.fixture
def dataset(tmp_path):
    """Three dates x two wells, with a known selection per (date, well).

    ``well000`` keeps a fixed core of 10 electrodes and swaps the rest; the two
    later scans are made deliberately far apart so overlap decays with lag.
    ``well001`` reproduces its selection exactly on the last date, which is what
    an operator reusing a base configuration does.
    """
    core = list(range(0, 10))
    plans = {
        "260301": {0: core + list(range(100, 140)), 1: list(range(1000, 1050))},
        "260305": {0: core + list(range(200, 240)), 1: list(range(2000, 2050))},
        "260309": {0: core + list(range(300, 340)), 1: list(range(2000, 2050))},
    }
    entries = {}
    raw = tmp_path / "raw"
    raw.mkdir()
    for i, (date, per_well) in enumerate(sorted(plans.items())):
        rel = Path("CXTEST") / date / "PLATE1" / "Network" / f"{i:06d}" / "data.raw.h5"
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        _write_file(raw / rel, per_well)
        entries[f"CXTEST/{date}/PLATE1/Network/{i:06d}"] = {
            "sample_id": "CXTEST", "date": date, "plate_id": "PLATE1",
            "scan_type": "Network", "run_id": f"{i:06d}",
            "data_path": str(rel),
            "metadata": {"tag": f"Run #{i:06d}"},
            "h5_recordings": {"rec0000": ["well000", "well001"]},
            "wells": {
                "well000": {"metadata": {"groupname": "Control",
                                         "Plating Date": "20.2.2026"}},
                "well001": {"metadata": {"groupname": "Treated",
                                         "Plating Date": "20.2.2026"}},
            },
        }
    return {"entries": entries, "data_root": raw, "tmp": tmp_path, "core": core}


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_extraction_recovers_planted_selection(dataset):
    index, scans = Q.extract_selection_index(dataset["entries"], dataset["data_root"])
    assert len(scans) == 6                       # 3 dates x 2 wells
    assert set(scans["well_uid"]) == {"CXTEST|PLATE1|well000", "CXTEST|PLATE1|well001"}

    first = index[(index.well_uid == "CXTEST|PLATE1|well000")
                  & (index.recording_key == "CXTEST/260301/PLATE1/Network/000000")]
    assert sorted(first["electrode"]) == dataset["core"] + list(range(100, 140))
    assert set(scans[scans.well_id == "well000"]["n_routed"]) == {50}


def test_row_col_match_electrode_identity(dataset):
    index, _ = Q.extract_selection_index(dataset["entries"], dataset["data_root"])
    e = index["electrode"].to_numpy()
    assert np.array_equal(index["row"].to_numpy(), e // GRID_COLS)
    assert np.array_equal(index["col"].to_numpy(), e % GRID_COLS)


def test_channels_mapping_mismatch_keeps_the_intersection(tmp_path):
    """A mapping row whose channel is not routed must not be counted."""
    path = _write_file(tmp_path / "d.h5", {0: [10, 20, 30, 40]},
                       routed={0: [0, 2]})       # channels 0 and 2 only
    got = Q.read_selection(path)
    assert got[0].tolist() == [10, 30]


def test_missing_data_store_raises(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("something_else")
    with pytest.raises(Q.ElectrodeSelectionError):
        Q.read_selection(path)


def test_zero_byte_file_is_skipped_not_fatal(dataset):
    key = "CXTEST/260305/PLATE1/Network/000001"
    stub = dataset["data_root"] / dataset["entries"][key]["data_path"]
    stub.write_bytes(b"")
    _index, scans = Q.extract_selection_index(dataset["entries"], dataset["data_root"])
    assert key not in set(scans["recording_key"])
    assert len(scans) == 4


# --------------------------------------------------------------------------- #
# Reuse detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tag,expected", [
    ("Run #000019 Using 015 as base 15H", "000015"),
    ("Run #000019 using 0015 as base", "000015"),
    ("Run #000004", None),
    ("", None),
    (None, None),
])
def test_parse_base_run(tag, expected):
    assert Q.parse_base_run(tag) == expected


def test_identical_selection_pairs_are_flagged_and_excluded(dataset):
    """well001 repeats its selection, so the raw mean must exceed the dedup mean."""
    res = _qc_for(dataset, "well001", Q.WindowSpec("all", None, None))
    s = res.stability
    assert s["n_identical_pairs"] == 1
    assert s["mean_jaccard_raw"] > s["mean_jaccard_dedup"]
    assert any(p["reuse"] for p in res.pairs)


def test_declared_base_run_flags_a_pair_without_identical_sets(dataset):
    """A near-copy scores J < 1 but is still caught by the run tag."""
    entries = json.loads(json.dumps(dataset["entries"]))
    last = "CXTEST/260309/PLATE1/Network/000002"
    entries[last]["metadata"]["tag"] = "Run #000002 Using 000 as base"
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None),
                  entries=entries)
    assert res.stability["n_identical_pairs"] == 0
    assert res.stability["n_declared_base_pairs"] == 1


# --------------------------------------------------------------------------- #
# The shift null
# --------------------------------------------------------------------------- #
def test_fft_null_matches_brute_force_roll():
    rng = np.random.default_rng(0)
    a = Q.electrode_mask(rng.choice(N_ELECTRODES, 300, replace=False))
    b = Q.electrode_mask(rng.choice(N_ELECTRODES, 400, replace=False))
    corr = np.rint(np.fft.irfft2(
        np.fft.rfft2(a.astype(float)) * np.conj(np.fft.rfft2(b.astype(float))),
        s=a.shape)).astype(int)
    for dr, dc in [(0, 0), (1, 0), (0, 1), (37, 113), (GRID_ROWS - 1, GRID_COLS - 1)]:
        brute = int((a & np.roll(np.roll(b, dr, 0), dc, 1)).sum())
        assert corr[dr, dc] == brute, (dr, dc)


def test_shift_null_on_identical_masks_is_maximal():
    rng = np.random.default_rng(1)
    a = Q.electrode_mask(rng.choice(N_ELECTRODES, 500, replace=False))
    out = Q.shift_null(a, a)
    assert out["observed"] == pytest.approx(1.0)
    # Nothing can beat the zero shift, so p sits at its floor.
    assert out["p"] == pytest.approx(1.0 / (GRID_ROWS * GRID_COLS))
    assert out["enrichment"] > 10


def test_shift_null_on_independent_masks_is_unenriched():
    rng = np.random.default_rng(2)
    a = Q.electrode_mask(rng.choice(N_ELECTRODES, 800, replace=False))
    b = Q.electrode_mask(rng.choice(N_ELECTRODES, 800, replace=False))
    out = Q.shift_null(a, b)
    assert 0.5 < out["enrichment"] < 2.0
    assert out["p"] > 0.01


def test_shift_null_handles_an_empty_mask():
    a = Q.electrode_mask(np.array([1, 2, 3]))
    empty = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    out = Q.shift_null(a, empty)
    assert np.isnan(out["observed"])


# --------------------------------------------------------------------------- #
# Lag decay
# --------------------------------------------------------------------------- #
def test_lag_curve_recovers_a_known_half_life():
    half_life = 5.0
    pairs = [{"lag_days": lag, "jaccard": 0.4 * 0.5 ** (lag / half_life),
              "reuse": False} for lag in range(0, 21)]
    curve = Q._lag_curve(pairs, dedup=True)
    assert curve["half_life_days"] == pytest.approx(half_life, rel=1e-6)


def test_lag_curve_dedup_drops_reused_pairs():
    pairs = [{"lag_days": 0, "jaccard": 1.0, "reuse": True},
             {"lag_days": 0, "jaccard": 0.2, "reuse": False},
             {"lag_days": 4, "jaccard": 0.1, "reuse": False}]
    raw = Q._lag_curve(pairs, dedup=False)
    dedup = Q._lag_curve(pairs, dedup=True)
    assert raw["mean_jaccard"][0] == pytest.approx(0.6)
    assert dedup["mean_jaccard"][0] == pytest.approx(0.2)
    assert dedup["n_pairs"][0] == 1


def test_half_life_needs_enough_bins():
    assert np.isnan(Q._lag_curve(
        [{"lag_days": 0, "jaccard": 0.5, "reuse": False}], dedup=True
    )["half_life_days"])


# --------------------------------------------------------------------------- #
# Window resolution
# --------------------------------------------------------------------------- #
@pytest.fixture
def tmap():
    return {
        ("CXTEST", "PLATE1", "Control"): {"treatment_date": "260305",
                                          "plating_date": "260220",
                                          "canonical_group": "Control",
                                          "role": "control"},
        ("CXTEST", "PLATE1", "Other"): {"treatment_date": "260307",
                                        "plating_date": "260220",
                                        "canonical_group": "Other",
                                        "role": "treatment"},
    }


def test_window_exact_anchor(tmap):
    w = Q.resolve_window(Q.WindowSpec("tau", 0, 21), tmap, "CXTEST", "PLATE1",
                         "Control")
    assert (w.anchor, w.treatment_date, w.kind) == ("exact", "260305", "tau")


def test_window_plate_fallback_uses_earliest_anchor(tmap):
    w = Q.resolve_window(Q.WindowSpec("tau", 0, 21), tmap, "CXTEST", "PLATE1",
                         "GroupNotInMap")
    assert w.anchor == "plate"
    assert w.treatment_date == "260305"          # earliest of 260305 / 260307


def test_window_unanchored_falls_back_to_all_scans():
    w = Q.resolve_window(Q.WindowSpec("tau", 0, 21), {}, "CXTEST", "PLATE1", "X")
    assert w.anchor == "unanchored"
    assert w.kind == "all"
    assert "using all scans" in w.note


def test_div_window_can_use_the_wells_own_plating_metadata():
    w = Q.resolve_window(Q.WindowSpec("div", 0, 30), {}, "CXTEST", "PLATE1", "X",
                         plating_fallback="260220")
    assert w.kind == "div" and w.plating_date == "260220"


def test_select_window_filters_on_tau(dataset, tmap):
    _index, scans = Q.extract_selection_index(dataset["entries"], dataset["data_root"])
    rows = scans[scans.well_id == "well000"].to_dict("records")
    w = Q.resolve_window(Q.WindowSpec("tau", 0, 21), tmap, "CXTEST", "PLATE1",
                         "Control")
    kept = Q.select_window(rows, w)
    # treatment on 260305 -> the 260301 scan is at tau -4 and drops out
    assert [r["date"] for r in kept] == ["260305", "260309"]
    assert [r["tau"] for r in kept] == [0, 4]


def test_well_groupname_prefers_the_most_recent_label():
    scans = [{"raw_groupname": "Control"}, {"raw_groupname": "Control"},
             {"raw_groupname": "NPH"}]
    latest, seen = Q.well_groupname(scans)
    assert latest == "NPH"
    assert seen == ["Control", "NPH"]


# --------------------------------------------------------------------------- #
# Stability + verdict
# --------------------------------------------------------------------------- #
def _qc_for(dataset, well_id, spec, entries=None, min_scans=3):
    entries = entries if entries is not None else dataset["entries"]
    index, scans = Q.extract_selection_index(entries, dataset["data_root"])
    uid = f"CXTEST|PLATE1|{well_id}"
    rows = scans[scans.well_uid == uid].sort_values(["date", "run_id"]).to_dict("records")
    sets = {k: g["electrode"].to_numpy(dtype=np.int64)
            for k, g in index[index.well_uid == uid].groupby("recording_key")}
    window = Q.resolve_window(spec, {}, "CXTEST", "PLATE1", "Control")
    selected = Q.select_window(rows, window)
    return Q.compute_stability(uid, selected, sets, window=window,
                               window_spec=spec, min_scans=min_scans)


def test_core_set_matches_the_planted_core(dataset):
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None))
    s = res.stability
    assert s["n_scans"] == 3
    assert s["core_counts"]["1"] == len(dataset["core"])     # in every scan
    assert s["union_electrodes"] == 10 + 3 * 40
    assert s["n_never_selected"] == N_ELECTRODES - s["union_electrodes"]


def test_verdict_is_withheld_below_min_scans(dataset):
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None), min_scans=99)
    assert res.stability["verdict"] == "insufficient"
    assert "no stability verdict" in res.stability["verdict_text"]


def test_count_grid_leaves_never_selected_as_nan(dataset, tmp_path):
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None))
    out = Q.write_well_qc(res, tmp_path / "out")
    payload = Q.load_well_qc(out)
    grid = VI.count_grid(payload)
    assert np.isnan(grid).sum() == N_ELECTRODES - res.stability["union_electrodes"]
    assert np.nanmin(grid) >= 1                  # never a zero on the ramp
    assert np.nanmax(grid) == 3


def test_fraction_grid_is_bounded_by_one(dataset, tmp_path):
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None))
    payload = Q.load_well_qc(Q.write_well_qc(res, tmp_path / "out"))
    frac = VI.fraction_grid(payload)
    assert np.nanmax(frac) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Persistence + read side
# --------------------------------------------------------------------------- #
def test_round_trip_write_and_load(dataset, tmp_path):
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None))
    out = Q.write_well_qc(res, tmp_path / "out")
    assert (out / "stability.json").exists()
    assert (out / "counts.npz").exists()
    assert (out / "pairs.csv").exists()

    payload = Q.load_well_qc(out)
    assert payload["well_uid"] == res.well_uid
    assert payload["stability"]["n_scans"] == 3
    assert payload["counts"].shape == (GRID_ROWS, GRID_COLS)


def test_load_missing_well_returns_none(tmp_path):
    assert Q.load_well_qc(tmp_path / "nope") is None
    assert VI.load_well(tmp_path, "CXTEST", "PLATE1", "well000") is None


def test_discovery_finds_written_wells(dataset, tmp_path):
    root = tmp_path / "root"
    for well in ("well000", "well001"):
        res = _qc_for(dataset, well, Q.WindowSpec("all", None, None))
        Q.write_well_qc(res, Q.well_dir(root, "CXTEST", "PLATE1", well))
    assert VI.samples(root) == ["CXTEST"]
    assert VI.plates(root, "CXTEST") == ["PLATE1"]
    assert VI.wells(root, "CXTEST", "PLATE1") == ["well000", "well001"]
    assert VI.discover_wells(tmp_path / "absent") == []


def test_thumbnail_cache_key_tracks_the_output(dataset, tmp_path):
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None))
    out = Q.write_well_qc(res, tmp_path / "out")
    cache = tmp_path / "cache"
    first = VI._thumbnail_path(cache, out, VI.DEFAULT_CMAP)
    assert first == VI._thumbnail_path(cache, out, VI.DEFAULT_CMAP)
    assert first != VI._thumbnail_path(cache, out, "viridis")
    # A recomputed well must not serve the old PNG.
    npz = out / "counts.npz"
    npz.write_bytes(npz.read_bytes() + b"\0")
    assert VI._thumbnail_path(cache, out, VI.DEFAULT_CMAP) != first


def test_renderers_write_files(dataset, tmp_path):
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None))
    out = Q.write_well_qc(res, tmp_path / "out")
    payload = Q.load_well_qc(out)

    written = VI.render_count_map(payload, out / "map", formats=("png",))
    assert written[0].exists() and written[0].suffix == ".png"
    thumb = VI.render_count_thumbnail(payload, out / "thumb.png")
    assert thumb.exists()
    uri = VI.count_png_data_uri(payload, tmp_path / "cache", out)
    assert uri.startswith("data:image/png;base64,")


def test_history_frames_fade_with_temporal_distance(dataset):
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None))
    frames = Q.history_frames(res, decay_days=4.0)
    assert frames.shape == (3, GRID_ROWS, GRID_COLS, 3)
    assert frames[0].min() < 1.0                 # something was drawn

    # On the last frame, the scan 4 days back must be darker than the one 8 days
    # back: the fade tracks days, not frame index.
    last = frames[-1]
    recent = Q.electrode_mask(res.sets[res.scans[1]["recording_key"]])
    old = Q.electrode_mask(res.sets[res.scans[0]["recording_key"]])
    current = Q.electrode_mask(res.sets[res.scans[2]["recording_key"]])
    recent_only = recent & ~current
    old_only = old & ~current & ~recent
    assert last[recent_only].mean() < last[old_only].mean()


def test_report_markdown_covers_the_essentials(dataset, tmp_path):
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None))
    payload = Q.load_well_qc(Q.write_well_qc(res, tmp_path / "out"))
    md = VI.report_markdown(payload, map_rel="selection_count_map.png",
                            anim_hint="run me")
    assert "# Electrode-selection QC — CXTEST|PLATE1|well000" in md
    assert "## Verdict" in md and "## Scans" in md and "## Stability" in md
    assert "![cumulative selection map](selection_count_map.png)" in md
    assert "run me" in md                        # hint replaces a missing animation
    assert "260301" in md                        # the scan table is populated


def test_report_markdown_on_missing_payload():
    assert "No output" in VI.report_markdown(None)


def test_window_label_names_a_soft_anchor(dataset, tmp_path):
    res = _qc_for(dataset, "well000", Q.WindowSpec("tau", 0, 21))
    payload = Q.load_well_qc(Q.write_well_qc(res, tmp_path / "out"))
    # No treatment map was supplied, so the window degraded and must say so.
    assert payload["window"]["anchor"] == "unanchored"
    assert "unanchored" in VI.window_label(payload)


def test_well_name_maps_the_plate():
    assert VI.well_name("well000") == "A1"
    assert VI.well_name("well023") == "D6"


# --------------------------------------------------------------------------- #
# Colour bands
# --------------------------------------------------------------------------- #
def _geometric_counts():
    """The reference well's real count distribution: 43% at 1, 98% at <=8."""
    return np.repeat(np.arange(1, 16),
                     [2105, 1106, 634, 402, 272, 134, 106, 56, 37, 30, 14, 7, 3, 5, 2])


def test_band_edges_clip_the_empty_tail():
    edges, top_open = VI.band_edges(_geometric_counts())
    assert edges == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert top_open is True                       # counts 9-15 fold into "8+"


def test_band_ticks_label_the_open_top():
    edges, top_open = VI.band_edges(_geometric_counts())
    ticks, labels = VI.band_ticks(edges, top_open)
    assert labels[0] == "1" and labels[-1] == "8+"
    assert len(ticks) == len(edges)
    assert all(edges[i] < ticks[i] < edges[i] + 1.001 for i in range(len(edges)))


def test_band_count_is_capped_and_the_step_widens():
    edges, _ = VI.band_edges(np.repeat(np.arange(1, 41), 50))
    assert len(edges) <= VI.MAX_BANDS
    steps = {round(b - a, 6) for a, b in zip(edges, edges[1:])}
    assert len(steps) == 1 and steps.pop() > 1    # widened, not truncated


def test_top_open_is_false_when_nothing_exceeds_the_clip():
    edges, top_open = VI.band_edges(np.array([1, 1, 2, 2, 3, 3]))
    assert top_open is False
    assert edges[-1] == 3.0


def test_vmax_floors_at_two_so_one_versus_more_survives():
    """98% at count 1 must still separate '1' from '2+', not collapse to one band."""
    counts = np.array([1] * 990 + [2] * 8 + [5] * 2)
    edges, top_open = VI.band_edges(counts)
    assert len(edges) >= 2
    assert top_open is True


def test_degenerate_inputs_do_not_raise():
    assert VI.band_edges(np.array([])) == ([1.0], False)
    assert VI.band_edges(np.array([1, 1, 1])) == ([1.0], False)
    # zeros are absent data, not a value on the scale
    assert VI.band_edges(np.array([0, 0, 0])) == ([1.0], False)


def _rgb_distance(a, b) -> float:
    return float(np.linalg.norm(np.array(a[:3]) - np.array(b[:3])) / np.sqrt(3))


def test_counts_one_two_three_get_distinct_colours():
    """The actual complaint: adjacent low counts used to render identically."""
    edges, top_open = VI.band_edges(_geometric_counts())
    cmap, norm = VI._band_norm(edges, top_open, VI.DEFAULT_CMAP)
    new = [cmap(norm(v)) for v in (1, 2, 3)]
    assert len({tuple(c) for c in new}) == 3

    # Under the old linear 1..18 ramp these three were nearly the same red.
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    lin_cmap, lin = colormaps[VI.DEFAULT_CMAP], Normalize(vmin=1, vmax=18)
    old = [lin_cmap(lin(v)) for v in (1, 2, 3)]

    # Measured: adjacent separation 0.165/0.172 now versus 0.092/0.091 before,
    # a 1.81x gain in plain sRGB distance. The perceptual gain is larger than
    # that number suggests — the old three were all dark reds (#a50026,
    # #c21c27, #dc3b2c) while the new three cross red -> orange-red -> orange.
    worst_new = min(_rgb_distance(new[i], new[j]) for i, j in ((0, 1), (1, 2)))
    worst_old = min(_rgb_distance(old[i], old[j]) for i, j in ((0, 1), (1, 2)))
    assert worst_new > 1.5 * worst_old
    assert worst_new > 0.15


def test_every_band_is_separable_from_the_white_ground():
    """RdYlGn's pale centre would otherwise read as 'never selected'."""
    white = (1.0, 1.0, 1.0)
    for n in (2, 3, 5, 8, VI.MAX_BANDS):
        for colour in VI.band_colors(n, VI.DEFAULT_CMAP):
            assert _rgb_distance(colour, white) > 0.2


def test_never_selected_still_renders_white():
    cmap, _ = VI._band_norm([1.0, 2.0], False, VI.DEFAULT_CMAP)
    rgba = cmap(np.ma.masked_invalid(np.array([np.nan])))[0]
    assert tuple(np.round(rgba[:3], 6)) == (1.0, 1.0, 1.0)


def test_plotly_band_scale_has_hard_stops():
    scale = VI._plotly_band_scale([1.0, 2.0, 3.0], VI.DEFAULT_CMAP)
    assert len(scale) == 6                        # two entries per band
    assert scale[0][1] == scale[1][1]             # colour repeated => hard step
    assert scale[0][0] == 0.0 and scale[-1][0] == 1.0


def test_plate_band_edges_bracket_every_well(dataset, tmp_path):
    payloads = []
    for well in ("well000", "well001"):
        res = _qc_for(dataset, well, Q.WindowSpec("all", None, None))
        payloads.append(Q.load_well_qc(Q.write_well_qc(
            res, Q.well_dir(tmp_path / "root", "CXTEST", "PLATE1", well))))
    edges, _top = VI.plate_band_edges(payloads)
    assert len(edges) >= 1
    for payload in payloads:
        frac = VI.fraction_grid(payload)
        finite = frac[np.isfinite(frac)]
        assert edges[0] <= finite.min() + 1e-9


def test_thumbnail_cache_key_tracks_the_scale(dataset, tmp_path):
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None))
    out = Q.write_well_qc(res, tmp_path / "out")
    cache = tmp_path / "cache"
    per_well = VI._thumbnail_path(cache, out, VI.DEFAULT_CMAP, None)
    unified = VI._thumbnail_path(cache, out, VI.DEFAULT_CMAP, [0.0, 0.2, 0.4])
    other = VI._thumbnail_path(cache, out, VI.DEFAULT_CMAP, [0.0, 0.3, 0.6])
    assert len({per_well, unified, other}) == 3


def test_map_still_paints_a_white_ground(dataset, tmp_path):
    from PIL import Image

    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None))
    out = Q.write_well_qc(res, tmp_path / "out")
    payload = Q.load_well_qc(out)
    written = VI.render_count_map(payload, out / "map", formats=("png",), dpi=80)
    im = Image.open(written[0]).convert("RGB")
    assert im.getpixel((2, 2)) == (255, 255, 255)


def test_linear_scale_remains_available(dataset, tmp_path):
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None))
    out = Q.write_well_qc(res, tmp_path / "out")
    payload = Q.load_well_qc(out)
    written = VI.render_count_map(payload, out / "lin", formats=("png",),
                                  dpi=80, scale="linear")
    assert written[0].exists()


def test_thumbnail_accepts_shared_and_per_well_edges(dataset, tmp_path):
    res = _qc_for(dataset, "well000", Q.WindowSpec("all", None, None))
    out = Q.write_well_qc(res, tmp_path / "out")
    payload = Q.load_well_qc(out)
    a = VI.render_count_thumbnail(payload, out / "a.png")
    b = VI.render_count_thumbnail(payload, out / "b.png",
                                  edges=[0.0, 0.25, 0.5], top_open=True)
    assert a.exists() and b.exists()
    assert a.read_bytes() != b.read_bytes()
