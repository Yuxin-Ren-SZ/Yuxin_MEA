"""Unit tests for the metadata_extractor module."""

import json
from pathlib import Path

import pytest

from yuxin_mea.dataset.metadata import (
    MxassayMetadataExtractor,
    parse_hours_since_media,
)


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


# ---------------------------------------------------------------------------
# Assay tag -> hours since the last media change
# ---------------------------------------------------------------------------
# Every case below is a tag that actually appears in the live experiment cache
# (or a near-miss of one), because the parser's whole job is to survive what
# people really type at the scanner.
@pytest.mark.parametrize("tag,expected", [
    # the common case, and its lowercase / unit-less variants
    ("Run #000007 24H", 24.0),
    ("Run #000012 24h", 24.0),
    # a bare number carries no unit, so it is not an interval
    ("Run #000022 24", None),
    ("Run #000012 48H", 48.0),
    ("Run #000003 72H", 72.0),
    # "Imm" alone is the media change itself
    ("Run #000009 Imm", 0.0),
    ("Run #000014 Imm 0H", 0.0),
    ("Run #000058 Imm with Drug", 0.0),
    # an explicit number beats a bare "Imm" in the same tag
    ("Run #000009 Imm 0.5H", 0.5),
    ("Run #000043 NU Imm 1.5H", 1.5),
    # the interval can trail a sentence, and the LAST number is the interval
    ("Run #000019 Using 015 as base 15H", 15.0),
    ("Run #000064 24H Post Drug", 24.0),
    ("Run #000056 24H after Regular Media", 24.0),
    # minutes
    ("Run #000054 5 min post drug. using 052 conf", 5.0 / 60.0),
    # no interval recorded at all -> None, never 0.0
    ("Run #000017 Old_config", None),
    ("Run #000021 Baseline conf", None),
    ("Run #000002 Test", None),
    ("Run #000061", None),
    ("", None),
    (None, None),
])
def test_parse_hours_since_media(tag, expected):
    got = parse_hours_since_media(tag)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_untimed_tag_is_none_not_zero():
    """'Nobody wrote it down' must not read as 'recorded at the media change'.

    Collapsing the two would let untimed recordings pass an acute-timepoint
    filter, which is the one mistake this column exists to prevent.
    """
    assert parse_hours_since_media("Run #000017 Old_config") is None
    assert parse_hours_since_media("Run #000009 Imm") == 0.0


def test_only_labelled_hours_count():
    """An interval is only an interval when the tag labels it ``H``."""
    assert parse_hours_since_media("Run #000022 24") is None
    assert parse_hours_since_media("Run #000022 24H") == 24.0
    # ...so a run id can never be mistaken for an interval either
    assert parse_hours_since_media("Run #000048") is None
    assert parse_hours_since_media("#000048") is None


def test_hour_label_may_sit_mid_tag():
    """The unit is not anchored to the end of the string."""
    assert parse_hours_since_media("Run #000064 24H Post Drug") == 24.0
    assert parse_hours_since_media("Run #000056 24H after Regular Media") == 24.0
    assert parse_hours_since_media("Run #000019 Using 015 as base 15H") == 15.0
