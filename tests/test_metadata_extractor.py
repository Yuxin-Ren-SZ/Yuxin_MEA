"""Unit tests for the metadata_extractor module."""

import json
from pathlib import Path

import pytest

from dataset_manager.metadata_extractor import MxassayMetadataExtractor


class TestMetadataExtractorManualReview:
    """Manual-review tests for real metadata files."""

    def test_test_data_root_outputs_dict_for_review(self):
        """Write one extracted metadata dict per test file to ./temp."""
        data_root = Path("data/test_data/metadata_extractor")
        if not data_root.is_dir():
            pytest.skip(
                f"Expected metadata test-data root does not exist: {data_root}"
            )

        metadata_paths = sorted(path for path in data_root.rglob("*") if path.is_file())
        if not metadata_paths:
            pytest.skip(f"No test files found under {data_root}")

        output_dir = Path("temp/metadata_extractor")
        output_dir.mkdir(parents=True, exist_ok=True)
        extractor = MxassayMetadataExtractor()
        output_paths = []

        for metadata_path in metadata_paths:
            recording_metadata = extractor.get(metadata_path)
            extracted = {
                "fields": recording_metadata.fields,
                "wells": {
                    well.well_id: well.fields for well in recording_metadata.wells
                },
            }
            relative_path = metadata_path.relative_to(data_root)
            output_path = (output_dir / relative_path).with_suffix(".json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(extracted, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            output_paths.append(output_path)

            saved = json.loads(output_path.read_text(encoding="utf-8"))
            assert saved == extracted

        assert len(output_paths) == len(metadata_paths)
        assert all(path.exists() for path in output_paths)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
