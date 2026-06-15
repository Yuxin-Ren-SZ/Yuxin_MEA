"""Tests for the library module yuxin_mea.analysis.burst_diagnostic.

Focus areas:
- Cache helpers (cache_key stability, load_or_run_batch behavior).
- Kilosort source discovery on edge cases.

Does NOT cover figure correctness — those have visual contracts best tested
manually against real data. The Dash page is covered separately in
tests/test_dashboard.py (smoke import + page registration).
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from yuxin_mea.analysis.burst_diagnostic import (
    BatchResults,
    cache_key,
    cache_path,
    discover_real_spike_sources,
    is_kilosort_dir,
    load_or_run_batch,
)


def _make_fake_kilosort_dir(parent: Path, name: str = "rec0") -> Path:
    """Create the three sentinel files `is_kilosort_dir` checks for."""
    ks = parent / name
    ks.mkdir()
    (ks / "spike_times.npy").write_bytes(b"")
    (ks / "spike_clusters.npy").write_bytes(b"")
    (ks / "params.py").write_text("sample_rate = 20000\n")
    return ks


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def test_cache_key_stable_for_same_path():
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "ks"
        root.mkdir()
        assert cache_key(root) == cache_key(root)


def test_cache_key_differs_for_different_paths():
    with TemporaryDirectory() as tmp:
        a = Path(tmp) / "a"
        b = Path(tmp) / "b"
        a.mkdir()
        b.mkdir()
        assert cache_key(a) != cache_key(b)


def test_cache_path_under_burst_diagnostic_cache_dir():
    p = cache_path(Path("/some/analysis"), "v1_abcdef")
    assert p == Path("/some/analysis/burst_diagnostic_cache/v1_abcdef.pkl")


# ---------------------------------------------------------------------------
# load_or_run_batch
# ---------------------------------------------------------------------------


def test_load_or_run_batch_raises_on_missing_sources():
    with TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="No Kilosort sources"):
            load_or_run_batch(Path(tmp) / "empty", Path(tmp))


def test_load_or_run_batch_writes_and_reads_cache():
    """First call → fresh run + write. Second call → reads from cache."""
    with TemporaryDirectory() as tmp:
        analysis_root = Path(tmp) / "analysis"
        analysis_root.mkdir()
        ks_root = Path(tmp) / "ks_root"
        ks_root.mkdir()
        _make_fake_kilosort_dir(ks_root, "rec0")

        fake_batch = BatchResults()
        fake_batch.spike_times["rec0"] = {"cluster_000": [0.1, 0.2]}

        # First call: cache miss → run_batch invoked, result pickled.
        with patch(
            "yuxin_mea.analysis.burst_diagnostic.run_batch",
            return_value=fake_batch,
        ) as mock_run:
            batch1, from_cache1 = load_or_run_batch(ks_root, analysis_root)
        assert from_cache1 is False
        assert mock_run.call_count == 1
        assert cache_path(analysis_root, f"{cache_key(ks_root)}_iterative").exists()

        # Second call: cache hit → run_batch NOT called.
        with patch(
            "yuxin_mea.analysis.burst_diagnostic.run_batch",
            return_value=fake_batch,
        ) as mock_run:
            batch2, from_cache2 = load_or_run_batch(ks_root, analysis_root)
        assert from_cache2 is True
        assert mock_run.call_count == 0
        assert batch2.spike_times["rec0"] == fake_batch.spike_times["rec0"]


def test_load_or_run_batch_force_recompute_bypasses_cache_and_overwrites():
    with TemporaryDirectory() as tmp:
        analysis_root = Path(tmp) / "analysis"
        analysis_root.mkdir()
        ks_root = Path(tmp) / "ks_root"
        ks_root.mkdir()
        _make_fake_kilosort_dir(ks_root, "rec0")

        # Seed the cache with a batch identifiable as "first".
        first = BatchResults()
        first.spike_times["first"] = {}
        with patch(
            "yuxin_mea.analysis.burst_diagnostic.run_batch",
            return_value=first,
        ):
            load_or_run_batch(ks_root, analysis_root)

        cache_file = cache_path(analysis_root, f"{cache_key(ks_root)}_iterative")
        assert cache_file.exists()

        # force_recompute=True → call run_batch even though cache exists,
        # and *overwrite* the cache with the new batch.
        second = BatchResults()
        second.spike_times["second"] = {}
        with patch(
            "yuxin_mea.analysis.burst_diagnostic.run_batch",
            return_value=second,
        ) as mock_run:
            returned, from_cache = load_or_run_batch(
                ks_root, analysis_root, force_recompute=True,
            )
        assert from_cache is False
        assert mock_run.call_count == 1
        assert "second" in returned.spike_times

        # Subsequent cache-hit load must return the *new* contents, proving
        # the cache file was overwritten (not just bypassed).
        with patch(
            "yuxin_mea.analysis.burst_diagnostic.run_batch"
        ) as mock_run_again:
            cached, from_cache2 = load_or_run_batch(ks_root, analysis_root)
        assert from_cache2 is True
        assert mock_run_again.call_count == 0
        assert "second" in cached.spike_times
        assert "first" not in cached.spike_times


def test_load_or_run_batch_none_analysis_root_skips_cache_io():
    """analysis_root=None must not touch disk and must not raise."""
    with TemporaryDirectory() as tmp:
        ks_root = Path(tmp) / "ks_root"
        ks_root.mkdir()
        _make_fake_kilosort_dir(ks_root, "rec0")

        with patch(
            "yuxin_mea.analysis.burst_diagnostic.run_batch",
            return_value=BatchResults(),
        ) as mock_run:
            batch, from_cache = load_or_run_batch(ks_root, None)
        assert from_cache is False
        assert mock_run.call_count == 1
        # Nothing written under the (nonexistent) cache directory.
        assert not (Path(tmp) / "burst_diagnostic_cache").exists()


# ---------------------------------------------------------------------------
# discover_real_spike_sources edge cases
# ---------------------------------------------------------------------------


def test_discover_real_spike_sources_missing_root():
    assert discover_real_spike_sources(Path("/definitely/not/here")) == []


def test_discover_real_spike_sources_none_root():
    assert discover_real_spike_sources(None) == []


def test_discover_real_spike_sources_finds_kilosort_dir():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        ks = _make_fake_kilosort_dir(root, "rec0")
        assert is_kilosort_dir(ks)
        assert ks in discover_real_spike_sources(root)
