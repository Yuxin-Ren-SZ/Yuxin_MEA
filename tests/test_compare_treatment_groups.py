"""Tests for the roll-up script's use of the recording-level assay tag.

The script has no other test file; these cover only the tag path added with
``hours_since_media``, not the metric assembly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from compare_treatment_groups import (  # noqa: E402
    _hours_lookup,
    _treatment_from_flip,
)


def _cache(**tags):
    """A minimal experiment_cache shaped like the real JSON."""
    return {key: {"metadata": {"tag": tag}} for key, tag in tags.items()}


def test_hours_lookup_is_keyed_by_recording_not_well():
    """One assay tag per scan — every well of a recording shares its value."""
    cache = _cache(**{
        "S1/260101/P1/Network/001": "Run #000001 24H",
        "S1/260102/P1/Network/002": "Run #000002 Imm 0.5H",
    })
    assert _hours_lookup(cache) == {
        "S1/260101/P1/Network/001": 24.0,
        "S1/260102/P1/Network/002": 0.5,
    }


def test_hours_lookup_reports_none_for_untimed_tags():
    cache = _cache(**{"S1/260101/P1/Network/001": "Run #000001 Old_config"})
    assert _hours_lookup(cache) == {"S1/260101/P1/Network/001": None}


def test_hours_lookup_tolerates_missing_metadata():
    """A cache entry with no metadata block must not crash the roll-up."""
    cache = {"S1/260101/P1/Network/001": {}}
    assert _hours_lookup(cache) == {"S1/260101/P1/Network/001": None}


# --------------------------------------------------------------------------- #
# _treatment_from_flip now shares the parser
# --------------------------------------------------------------------------- #
def test_treatment_date_back_dates_by_whole_days():
    # 24 h after the change -> the change was the previous day.
    assert _treatment_from_flip("260601", "Run #000001 24H") == "260531"
    # 48 h -> two days back.
    assert _treatment_from_flip("260601", "Run #000001 48H") == "260530"
    # Recorded at the change -> same day.
    assert _treatment_from_flip("260325", "Run #000014 Imm 0H") == "260325"


def test_treatment_date_handles_intervals_the_old_ladder_missed():
    """72 h and 32 h used to fall through to the 1-day default and mis-date a chip."""
    assert _treatment_from_flip("260601", "Run #000001 72H") == "260529"
    assert _treatment_from_flip("260601", "Run #000001 32H") == "260530"


def test_treatment_date_falls_back_to_the_protocol_default():
    """No interval recorded -> assume the routine next-day scan, as before."""
    assert _treatment_from_flip("260203", "Run #000001 Test") == "260202"


def test_treatment_date_is_empty_when_the_date_is_unparseable():
    assert _treatment_from_flip("not-a-date", "Run #000001 24H") == ""
