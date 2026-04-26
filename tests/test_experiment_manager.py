"""Unit tests for the experiment_manager module."""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from experiment_manager.cache_store import JsonCacheStore
from experiment_manager.manager import ExperimentManager
from experiment_manager.metadata_extractor import RecordingMetadata, WellMetadata
from experiment_manager.recording_entry import RecordingEntry, WellEntry


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def temp_data_root():
    """Create a temporary data root directory structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_analysis_dir():
    """Create a temporary analysis directory for cache storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_recording_entry(temp_data_root):
    """Create a sample RecordingEntry for testing."""
    # Create the directory structure
    data_dir = (
        temp_data_root / "SampleA" / "240415" / "PlateX" / "ScanType1" / "001"
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    data_file = data_dir / "data.raw.h5"
    data_file.write_bytes(b"mock_data" * 100)  # ~900 bytes

    entry = RecordingEntry.from_path(
        data_path=data_file,
        data_root=temp_data_root,
        discovered_at=time.time(),
    )
    return entry


@pytest.fixture
def mock_cache_store(temp_analysis_dir):
    """Create a mock cache store for testing."""
    return JsonCacheStore(temp_analysis_dir)


# ==============================================================================
# RecordingEntry Tests
# ==============================================================================


class TestRecordingEntry:
    """Test RecordingEntry creation and properties."""

    def test_recording_entry_creation(self, sample_recording_entry):
        """Test that a RecordingEntry is created with correct attributes."""
        entry = sample_recording_entry
        assert entry.sample_id == "SampleA"
        assert entry.date == "240415"
        assert entry.plate_id == "PlateX"
        assert entry.scan_type == "ScanType1"
        assert entry.run_id == "001"
        assert entry.data_path.name == "data.raw.h5"
        assert entry.file_size > 0
        assert entry.mtime > 0
        assert entry.discovered_at > 0

    def test_cache_key_format(self, sample_recording_entry):
        """Test that cache_key is formatted correctly."""
        entry = sample_recording_entry
        expected_key = "SampleA/240415/PlateX/ScanType1/001"
        assert entry.cache_key == expected_key

    def test_from_path_root_level(self, temp_data_root):
        """Test parsing a path at root level."""
        # Create directory structure for root-level layout
        data_dir = (
            temp_data_root / "SampleB" / "240420" / "PlateY" / "ScanType2" / "002"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / "data.raw.h5"
        data_file.write_bytes(b"test_data")

        entry = RecordingEntry.from_path(data_file, temp_data_root)

        assert entry.sample_id == "SampleB"
        assert entry.date == "240420"
        assert entry.plate_id == "PlateY"
        assert entry.scan_type == "ScanType2"
        assert entry.run_id == "002"

    def test_from_path_sample_level(self, temp_data_root):
        """Test parsing a path at sample level."""
        # Create directory structure for sample-level layout
        data_dir = temp_data_root / "240415" / "PlateX" / "ScanType1" / "001"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / "data.raw.h5"
        data_file.write_bytes(b"test_data")

        entry = RecordingEntry.from_path(
            data_file, temp_data_root, sample_id_override="SampleA"
        )

        assert entry.sample_id == "SampleA"
        assert entry.date == "240415"

    def test_from_path_invalid_pattern_raises_error(self, temp_data_root):
        """Test that from_path raises ValueError for invalid paths."""
        invalid_dir = temp_data_root / "invalid" / "path"
        invalid_dir.mkdir(parents=True, exist_ok=True)
        data_file = invalid_dir / "data.raw.h5"
        data_file.write_bytes(b"test_data")

        with pytest.raises(ValueError, match="does not match"):
            RecordingEntry.from_path(data_file, temp_data_root)

    def test_recording_entry_frozen(self, sample_recording_entry):
        """Test that RecordingEntry is immutable (frozen dataclass)."""
        entry = sample_recording_entry
        with pytest.raises(AttributeError):
            entry.sample_id = "NewSample"


# ==============================================================================
# JsonCacheStore Tests
# ==============================================================================


class TestJsonCacheStore:
    """Test the JsonCacheStore cache persistence."""

    def test_load_empty_cache(self, temp_analysis_dir):
        """Test loading when cache file doesn't exist."""
        store = JsonCacheStore(temp_analysis_dir)
        cache = store.load()
        assert cache == {}

    def test_save_and_load_cache(self, temp_analysis_dir, sample_recording_entry):
        """Test saving and loading cache entries."""
        store = JsonCacheStore(temp_analysis_dir)
        entry = sample_recording_entry
        entry.metadata["runid"] = "001"
        entry.metadata["chipid"] = "chip-123"
        entry.wells["well000"] = WellEntry(
            well_id="well000",
            metadata={"groupname": "control", "density": 10000.0},
        )

        # Save
        cache_dict = {entry.cache_key: entry}
        store.save(cache_dict)

        # Load
        loaded = store.load()
        assert len(loaded) == 1
        assert entry.cache_key in loaded
        loaded_entry = loaded[entry.cache_key]
        assert loaded_entry.sample_id == entry.sample_id
        assert loaded_entry.date == entry.date
        assert loaded_entry.file_size == entry.file_size
        assert loaded_entry.metadata == entry.metadata
        assert loaded_entry.wells["well000"].metadata == entry.wells["well000"].metadata

    def test_cache_file_location(self, temp_analysis_dir):
        """Test that cache file is created at correct location."""
        store = JsonCacheStore(temp_analysis_dir)
        cache_file = temp_analysis_dir / "experiment_cache.json"
        assert not cache_file.exists()

        store.save({})
        assert cache_file.exists()

    def test_save_atomic_write(self, temp_analysis_dir, sample_recording_entry):
        """Test that save operation is atomic (no partial writes on failure)."""
        store = JsonCacheStore(temp_analysis_dir)
        entry = sample_recording_entry
        cache_dict = {entry.cache_key: entry}

        # First save should succeed
        store.save(cache_dict)
        cache_file = temp_analysis_dir / "experiment_cache.json"
        assert cache_file.exists()

        # Verify no temp files are left behind
        temp_files = list(temp_analysis_dir.glob(".cache_tmp_*"))
        assert len(temp_files) == 0


# ==============================================================================
# ExperimentManager Tests
# ==============================================================================


class TestExperimentManager:
    """Test the ExperimentManager class."""

    class StubMetadataExtractor:
        """Deterministic extractor used to verify metadata population."""

        def get(self, metadata_path: Path) -> RecordingMetadata:
            return RecordingMetadata(
                fields={
                    "metadata_path": str(metadata_path),
                    "chipid": "chip-123",
                    "runid": "001",
                },
                wells=[
                    WellMetadata(
                        well_id="well000",
                        fields={"groupname": "control", "density": 10000.0},
                    ),
                    WellMetadata(
                        well_id="well001",
                        fields={"groupname": "treatment", "density": 15000.0},
                    ),
                ],
            )

    def test_initialization_with_empty_data_root(self, temp_data_root, temp_analysis_dir):
        """Test initializing manager with empty data root."""
        manager = ExperimentManager(temp_data_root, temp_analysis_dir)
        assert len(manager.recordings) == 0

    def test_recordings_property(self, temp_data_root, temp_analysis_dir, sample_recording_entry):
        """Test that recordings property returns cached entries."""
        # Create directory for first recording
        data_dir = (
            temp_data_root / "SampleA" / "240415" / "PlateX" / "ScanType1" / "001"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / "data.raw.h5"
        data_file.write_bytes(b"test_data_a")

        manager = ExperimentManager(temp_data_root, temp_analysis_dir)
        recordings = manager.recordings
        assert len(recordings) == 1
        assert recordings[0].sample_id == "SampleA"

    def test_recording_metadata_and_wells_populated(
        self, temp_data_root, temp_analysis_dir
    ):
        """Test that extractor output populates recording metadata and wells."""
        data_dir = (
            temp_data_root / "SampleA" / "240415" / "PlateX" / "ScanType1" / "001"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "data.raw.h5").write_bytes(b"test_data")

        manager = ExperimentManager(
            temp_data_root,
            temp_analysis_dir,
            metadata_extractor=self.StubMetadataExtractor(),
        )

        entry = manager.recordings[0]
        assert entry.metadata["chipid"] == "chip-123"
        assert entry.metadata["runid"] == "001"
        assert entry.metadata["metadata_path"].endswith(
            "SampleA/240415/PlateX/ScanType1/001/mxassay.metadata"
        )
        assert set(entry.wells) == {"well000", "well001"}
        assert entry.wells["well000"].metadata["groupname"] == "control"
        assert entry.wells["well001"].metadata["density"] == 15000.0

    def test_get_by_equals(self, temp_data_root, temp_analysis_dir):
        """Test filtering recordings with == operator."""
        # Create multiple recordings
        for sample in ["SampleA", "SampleB"]:
            for scan in ["ScanType1", "ScanType2"]:
                data_dir = (
                    temp_data_root / sample / "240415" / "PlateX" / scan / "001"
                )
                data_dir.mkdir(parents=True, exist_ok=True)
                data_file = data_dir / "data.raw.h5"
                data_file.write_bytes(b"test_data")

        manager = ExperimentManager(temp_data_root, temp_analysis_dir)

        # Filter by sample_id
        results = manager.get_by("sample_id", "SampleA", "==")
        assert len(results) == 2
        assert all(r.sample_id == "SampleA" for r in results)

        # Filter by scan_type
        results = manager.get_by("scan_type", "ScanType1", "==")
        assert len(results) == 2
        assert all(r.scan_type == "ScanType1" for r in results)

    def test_get_by_not_equals(self, temp_data_root, temp_analysis_dir):
        """Test filtering recordings with != operator."""
        for sample in ["SampleA", "SampleB"]:
            data_dir = temp_data_root / sample / "240415" / "PlateX" / "ScanType1" / "001"
            data_dir.mkdir(parents=True, exist_ok=True)
            data_file = data_dir / "data.raw.h5"
            data_file.write_bytes(b"test_data")

        manager = ExperimentManager(temp_data_root, temp_analysis_dir)

        results = manager.get_by("sample_id", "SampleA", "!=")
        assert len(results) == 1
        assert results[0].sample_id == "SampleB"

    def test_get_by_comparison_operators(self, temp_data_root, temp_analysis_dir):
        """Test filtering recordings with <, <=, >, >= operators."""
        # Create recordings with different dates
        for date in ["240410", "240415", "240420"]:
            data_dir = (
                temp_data_root / "SampleA" / date / "PlateX" / "ScanType1" / "001"
            )
            data_dir.mkdir(parents=True, exist_ok=True)
            data_file = data_dir / "data.raw.h5"
            data_file.write_bytes(b"test_data")

        manager = ExperimentManager(temp_data_root, temp_analysis_dir)

        # Test <
        results = manager.get_by("date", "240415", "<")
        assert len(results) == 1
        assert results[0].date == "240410"

        # Test >
        results = manager.get_by("date", "240415", ">")
        assert len(results) == 1
        assert results[0].date == "240420"

        # Test <=
        results = manager.get_by("date", "240415", "<=")
        assert len(results) == 2

    def test_get_by_contain_operators(self, temp_data_root, temp_analysis_dir):
        """Test filtering recordings with 'contain' and 'not contain' operators."""
        for i, plate in enumerate(["PlateX", "PlateY", "PlateZ"]):
            data_dir = (
                temp_data_root / "SampleA" / "240415" / plate / "ScanType1" / "001"
            )
            data_dir.mkdir(parents=True, exist_ok=True)
            data_file = data_dir / "data.raw.h5"
            data_file.write_bytes(b"test_data" * (i + 1))

        manager = ExperimentManager(temp_data_root, temp_analysis_dir)

        # Test contain
        results = manager.get_by("plate_id", "Plate", "contain")
        assert len(results) == 3

        # Test not contain
        results = manager.get_by("plate_id", "Z", "not contain")
        assert len(results) == 2

    def test_get_by_invalid_key_raises_error(self, temp_data_root, temp_analysis_dir):
        """Test that get_by raises ValueError for invalid key."""
        manager = ExperimentManager(temp_data_root, temp_analysis_dir)

        with pytest.raises(ValueError, match="Unknown key"):
            manager.get_by("invalid_key", "value", "==")

    def test_get_by_invalid_method_raises_error(self, temp_data_root, temp_analysis_dir):
        """Test that get_by raises ValueError for invalid method."""
        manager = ExperimentManager(temp_data_root, temp_analysis_dir)

        with pytest.raises(ValueError, match="Unknown method"):
            manager.get_by("sample_id", "value", "invalid_method")

    def test_refresh_clears_and_rescans(self, temp_data_root, temp_analysis_dir):
        """Test that refresh clears cache and re-scans all directories."""
        # Create initial recording
        data_dir = (
            temp_data_root / "SampleA" / "240415" / "PlateX" / "ScanType1" / "001"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / "data.raw.h5"
        data_file.write_bytes(b"test_data_1")

        manager = ExperimentManager(temp_data_root, temp_analysis_dir)
        assert len(manager.recordings) == 1

        # Add a new recording
        data_dir2 = (
            temp_data_root / "SampleB" / "240420" / "PlateY" / "ScanType2" / "002"
        )
        data_dir2.mkdir(parents=True, exist_ok=True)
        data_file2 = data_dir2 / "data.raw.h5"
        data_file2.write_bytes(b"test_data_2")

        # Refresh should find the new recording
        manager.refresh()
        assert len(manager.recordings) == 2

    def test_sample_level_layout_detection(self):
        """Test that sample-level layout is correctly detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "SampleA"
            data_root.mkdir()

            # Create sample-level structure
            data_dir = data_root / "240415" / "PlateX" / "ScanType1" / "001"
            data_dir.mkdir(parents=True, exist_ok=True)
            data_file = data_dir / "data.raw.h5"
            data_file.write_bytes(b"test_data")

            with tempfile.TemporaryDirectory() as analysis_dir:
                manager = ExperimentManager(
                    data_root, Path(analysis_dir)
                )
                assert len(manager.recordings) == 1
                # SampleID should be inferred from directory name
                assert manager.recordings[0].sample_id == "SampleA"

    def test_root_level_layout_detection(self, temp_data_root, temp_analysis_dir):
        """Test that root-level layout is correctly detected."""
        # Create root-level structure
        data_dir = (
            temp_data_root / "SampleA" / "240415" / "PlateX" / "ScanType1" / "001"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / "data.raw.h5"
        data_file.write_bytes(b"test_data")

        manager = ExperimentManager(temp_data_root, temp_analysis_dir)
        assert len(manager.recordings) == 1
        assert manager.recordings[0].sample_id == "SampleA"

    def test_missing_data_file_ignored(self, temp_data_root, temp_analysis_dir):
        """Test that directories without data.raw.h5 are ignored."""
        # Create directory structure but without data.raw.h5
        data_dir = (
            temp_data_root / "SampleA" / "240415" / "PlateX" / "ScanType1" / "001"
        )
        data_dir.mkdir(parents=True, exist_ok=True)

        manager = ExperimentManager(temp_data_root, temp_analysis_dir)
        assert len(manager.recordings) == 0

    def test_cache_persistence_across_instances(self, temp_data_root, temp_analysis_dir):
        """Test that cache persists across manager instances."""
        # Create recording with first manager
        data_dir = (
            temp_data_root / "SampleA" / "240415" / "PlateX" / "ScanType1" / "001"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / "data.raw.h5"
        data_file.write_bytes(b"test_data")

        manager1 = ExperimentManager(
            temp_data_root,
            temp_analysis_dir,
            metadata_extractor=self.StubMetadataExtractor(),
        )
        assert len(manager1.recordings) == 1

        # Create second manager instance - should load from cache
        manager2 = ExperimentManager(temp_data_root, temp_analysis_dir)
        assert len(manager2.recordings) == 1
        assert manager2.recordings[0].sample_id == manager1.recordings[0].sample_id
        assert manager2.recordings[0].metadata == manager1.recordings[0].metadata
        assert manager2.recordings[0].wells["well000"].metadata == {
            "groupname": "control",
            "density": 10000.0,
        }

    def test_date_format_validation(self, temp_data_root, temp_analysis_dir):
        """Test that only 6-digit dates are recognized."""
        # Create directory with 6-digit date (valid)
        valid_dir = (
            temp_data_root / "SampleA" / "240415" / "PlateX" / "ScanType1" / "001"
        )
        valid_dir.mkdir(parents=True, exist_ok=True)
        (valid_dir / "data.raw.h5").write_bytes(b"test_data")

        # Create directory with non-6-digit date (invalid)
        invalid_dir = temp_data_root / "SampleB" / "2404" / "PlateY" / "ScanType1" / "001"
        invalid_dir.mkdir(parents=True, exist_ok=True)
        (invalid_dir / "data.raw.h5").write_bytes(b"test_data")

        manager = ExperimentManager(temp_data_root, temp_analysis_dir)
        recordings = manager.recordings
        assert len(recordings) == 1
        assert recordings[0].date == "240415"


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestExperimentManagerIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self, temp_data_root, temp_analysis_dir):
        """Test complete workflow: create manager, scan, filter, cache."""
        # Create diverse recording structure
        recordings_data = [
            ("SampleA", "240410", "PlateX", "ScanType1", "001"),
            ("SampleA", "240410", "PlateX", "ScanType2", "002"),
            ("SampleA", "240415", "PlateY", "ScanType1", "001"),
            ("SampleB", "240415", "PlateX", "ScanType1", "001"),
        ]

        for sample, date, plate, scan, run in recordings_data:
            data_dir = temp_data_root / sample / date / plate / scan / run
            data_dir.mkdir(parents=True, exist_ok=True)
            data_file = data_dir / "data.raw.h5"
            data_file.write_bytes(b"test_data")

        # Initialize manager
        manager = ExperimentManager(temp_data_root, temp_analysis_dir)
        assert len(manager.recordings) == 4

        # Test various filters
        sample_a = manager.get_by("sample_id", "SampleA", "==")
        assert len(sample_a) == 3

        date_240415 = manager.get_by("date", "240415", "==")
        assert len(date_240415) == 2

        scan_type1 = manager.get_by("scan_type", "ScanType1", "==")
        assert len(scan_type1) == 3

        # Verify cache was saved
        cache_file = temp_analysis_dir / "experiment_cache.json"
        assert cache_file.exists()

        # Create new manager instance and verify it loads cached data
        manager2 = ExperimentManager(temp_data_root, temp_analysis_dir)
        assert len(manager2.recordings) == 4

    def test_incremental_scan_on_new_dates(self, temp_data_root, temp_analysis_dir):
        """Test that only new dates are scanned on subsequent initializations."""
        # Create first batch of recordings
        data_dir1 = (
            temp_data_root / "SampleA" / "240410" / "PlateX" / "ScanType1" / "001"
        )
        data_dir1.mkdir(parents=True, exist_ok=True)
        (data_dir1 / "data.raw.h5").write_bytes(b"test_data_1")

        manager1 = ExperimentManager(temp_data_root, temp_analysis_dir)
        assert len(manager1.recordings) == 1

        # Add recordings with new date
        data_dir2 = (
            temp_data_root / "SampleA" / "240415" / "PlateX" / "ScanType1" / "001"
        )
        data_dir2.mkdir(parents=True, exist_ok=True)
        (data_dir2 / "data.raw.h5").write_bytes(b"test_data_2")

        # Create new manager - should find new date without re-scanning old
        manager2 = ExperimentManager(temp_data_root, temp_analysis_dir)
        assert len(manager2.recordings) == 2

    def test_manual_data_dir_generates_cache(self):
        """Test cache generation against a manually specified data directory."""
        data_root_value = os.environ.get("EXPERIMENT_MANAGER_TEST_DATA_ROOT")
        if not data_root_value:
            pytest.skip(
                "Set EXPERIMENT_MANAGER_TEST_DATA_ROOT to a real data directory to run this test."
            )

        data_root = Path(data_root_value).expanduser()
        if not data_root.is_dir():
            pytest.fail(
                f"EXPERIMENT_MANAGER_TEST_DATA_ROOT is not a directory: {data_root}"
            )

        analysis_dir = Path(__file__).parent / "../temp"
        cache_file = analysis_dir / "experiment_cache.json"
        if cache_file.exists():
            cache_file.unlink()

        manager = ExperimentManager(data_root, analysis_dir)
        loaded_cache = JsonCacheStore(analysis_dir).load()

        assert cache_file.exists()
        assert len(loaded_cache) == len(manager.recordings)
        assert set(loaded_cache) == {entry.cache_key for entry in manager.recordings}


# ==============================================================================
# Run Tests
# ==============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
