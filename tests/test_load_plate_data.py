"""Tests for ``yuxin_mea.analysis.plate_raster_synchrony.load_plate_data``.

These tests migrated from ``tests/test_api_compatibility.py`` (the six
``test_plate_viewer_*`` cases) when Phase 5 promoted
``BasePlateViewer._load_well_records`` to a public free function. They
exercise the same code paths (compound upstream paths, cache metadata,
auto rec_name discovery, event intervals, missing-file tolerance) but
now hit the public surface that the new dashboard page consumes.

The original "fallback cache-path resolution" test is intentionally
dropped: that fallback logic lived on the task base class. The promoted
loader takes a single ``experiment_cache_path`` argument; the dashboard
caller resolves the path.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from yuxin_mea.analysis.plate_raster_synchrony import load_plate_data


_RECORDING_KEY = "SampleA/240415/PlateX/Network/001"


def _setup_well_outputs(
    root: Path,
    rec_name: str,
    well_id: str,
    *,
    with_plot_signals: bool = True,
    with_spike_times: bool = True,
    burst_event_tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[Path, Path]:
    """Create burst + curation output directories for one (rec, well) pair."""
    burst_root = root / "burst"
    curation_root = root / "curation"
    burst_dir = burst_root / _RECORDING_KEY / rec_name / well_id / "burst_detection"
    curation_dir = curation_root / _RECORDING_KEY / rec_name / well_id / "auto_curation"
    burst_dir.mkdir(parents=True)
    curation_dir.mkdir(parents=True)

    if with_plot_signals:
        np.save(
            burst_dir / "plot_signals.npy",
            {"t": np.array([0.0]), "rate_signal": np.array([1.0])},
        )
    if with_spike_times:
        np.save(
            curation_dir / "curated_spike_times.npy",
            {"unit_0": np.array([0.1])},
        )
    if burst_event_tables:
        for event_key, table in burst_event_tables.items():
            table.to_pickle(burst_dir / f"{event_key}.pkl")
    return burst_root, curation_root


def _find_record(records, well_id: str):
    return next(r for r in records if r.well_id == well_id)


# ---------------------------------------------------------------------------
# Compound-path loading + cache metadata surfacing
# ---------------------------------------------------------------------------


def test_load_plate_data_reads_compound_upstream_paths():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        burst_root, curation_root = _setup_well_outputs(root, "rec0000", "well000")
        cache_path = root / "experiment_cache.json"
        cache_path.write_text(json.dumps({
            _RECORDING_KEY: {
                "wells": {
                    "well000": {"metadata": {"well_name": "A1", "groupname": "control"}},
                },
                "h5_recordings": {"rec0000": ["well000"]},
            },
        }))

        records = load_plate_data(
            burst_detection_root=burst_root,
            curation_output_root=curation_root,
            recording_key=_RECORDING_KEY,
            rec_name="rec0000",
            experiment_cache_path=cache_path,
        )

    assert len(records) == 24
    rec = _find_record(records, "well000")
    assert rec.status == "ok"
    assert rec.well_name == "A1"
    assert rec.groupname == "control"
    assert rec.plot_signals is not None
    assert rec.spike_times is not None


def test_load_plate_data_marks_unpopulated_wells_missing():
    """Wells without burst/curation outputs return placeholder records."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        burst_root, curation_root = _setup_well_outputs(root, "rec0000", "well000")

        records = load_plate_data(
            burst_detection_root=burst_root,
            curation_output_root=curation_root,
            recording_key=_RECORDING_KEY,
            rec_name="rec0000",
            experiment_cache_path=None,
        )

    # well000 populated, every other well missing
    assert _find_record(records, "well000").status == "ok"
    missing = [r for r in records if r.status == "missing"]
    assert len(missing) == 23


# ---------------------------------------------------------------------------
# Auto rec_name discovery
# ---------------------------------------------------------------------------


def test_load_plate_data_auto_discovers_rec_name_across_outputs():
    """`rec_name='auto'` should find wells living under any rec* directory."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        burst_root, curation_root = _setup_well_outputs(root, "rec0001", "well006")

        records = load_plate_data(
            burst_detection_root=burst_root,
            curation_output_root=curation_root,
            recording_key=_RECORDING_KEY,
            rec_name="auto",
            experiment_cache_path=None,
        )

    rec = _find_record(records, "well006")
    assert rec.status == "ok"
    assert rec.plot_signals is not None
    assert rec.spike_times is not None


def test_load_plate_data_cache_provides_rec_name_when_disk_lookup_fails():
    """When the user-supplied rec_name doesn't exist on disk, the cache's mapping is used."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        burst_root, curation_root = _setup_well_outputs(root, "rec0003", "well023")
        cache_path = root / "experiment_cache.json"
        cache_path.write_text(json.dumps({
            _RECORDING_KEY: {
                "wells": {
                    "well023": {"metadata": {"well_name": "D6", "groupname": "treated"}},
                },
                "h5_recordings": {"rec0003": ["well023"]},
            },
        }))

        records = load_plate_data(
            burst_detection_root=burst_root,
            curation_output_root=curation_root,
            recording_key=_RECORDING_KEY,
            rec_name="bogus_user_hint",   # ← invalid hint
            experiment_cache_path=cache_path,
        )

    rec = _find_record(records, "well023")
    assert rec.status == "ok"
    assert rec.well_name == "D6"


# ---------------------------------------------------------------------------
# Event intervals
# ---------------------------------------------------------------------------


def test_load_plate_data_loads_burst_event_intervals():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        burst_root, curation_root = _setup_well_outputs(
            root, "rec0000", "well000",
            burst_event_tables={
                "burstlets": pd.DataFrame([
                    {"start": 0.1, "end": 0.2, "peak_time": 0.15, "total_spikes": 10},
                ]),
                "network_bursts": pd.DataFrame([{"start": 0.3, "end": 0.6}]),
                "superbursts": pd.DataFrame(),  # empty → keeps an empty list
            },
        )

        records = load_plate_data(
            burst_detection_root=burst_root,
            curation_output_root=curation_root,
            recording_key=_RECORDING_KEY,
            rec_name="rec0000",
            experiment_cache_path=None,
        )

    rec = _find_record(records, "well000")
    assert rec.event_intervals["burstlets"] == [
        {"start": 0.1, "end": 0.2, "peak_time": 0.15, "total_spikes": 10}
    ]
    assert rec.event_intervals["network_bursts"] == [{"start": 0.3, "end": 0.6}]
    assert rec.event_intervals["superbursts"] == []


def test_load_plate_data_tolerates_no_event_tables():
    """Missing event-table .pkl files yield empty lists, not errors."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        burst_root, curation_root = _setup_well_outputs(root, "rec0000", "well000")

        records = load_plate_data(
            burst_detection_root=burst_root,
            curation_output_root=curation_root,
            recording_key=_RECORDING_KEY,
            rec_name="rec0000",
            experiment_cache_path=None,
        )

    rec = _find_record(records, "well000")
    assert rec.event_intervals["burstlets"] == []
    assert rec.event_intervals["network_bursts"] == []
    assert rec.event_intervals["superbursts"] == []


# ---------------------------------------------------------------------------
# Cache absence: graceful defaults
# ---------------------------------------------------------------------------


def test_load_plate_data_no_cache_defaults_well_name_to_question_mark():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        burst_root, curation_root = _setup_well_outputs(root, "rec0000", "well000")

        records = load_plate_data(
            burst_detection_root=burst_root,
            curation_output_root=curation_root,
            recording_key=_RECORDING_KEY,
            rec_name="rec0000",
            experiment_cache_path=None,
        )

    rec = _find_record(records, "well000")
    assert rec.well_name == "?"
    assert rec.groupname == "?"
