"""Scope: group_by keying, encoding, and where-filtering."""

import pytest

from yuxin_mea.pipeline.scope import (
    SCOPE_RECORDING,
    SCOPE_SAMPLE_GROUP,
    SCOPE_WELL_UID,
    Scope,
)
from yuxin_mea.pipeline.well_dims import WellDims

RK = "CX169/260602/T003346/Network/000004"


def _dims(rk=RK, cw="rec0000/well002", group="Control"):
    return WellDims.from_entry(rk, cw, group)


def test_welldims_parse():
    d = _dims()
    assert d.sample_id == "CX169"
    assert d.date == "260602"
    assert d.plate_id == "T003346"
    assert d.scan_type == "Network"
    assert d.run_id == "000004"
    assert d.rec_name == "rec0000"
    assert d.well_id == "well002"
    assert d.well_uid == "CX169|T003346|well002"
    assert d.compound_well_id == "rec0000/well002"


def test_welldims_no_slash_well():
    d = WellDims.from_entry(RK, "well007", "?")
    assert d.rec_name == ""
    assert d.well_id == "well007"


def test_scope_keys():
    d = _dims()
    assert SCOPE_RECORDING.well_key(d) == (RK,)
    assert SCOPE_RECORDING.scope_key_dict(d) == {"recording_key": RK}
    assert SCOPE_WELL_UID.scope_key_dict(d) == {"well_uid": "CX169|T003346|well002"}
    assert SCOPE_SAMPLE_GROUP.scope_key_dict(d) == {"sample_id": "CX169", "groupname": "Control"}


def test_encode_path_slugs_unsafe_chars():
    assert Scope.encode_path({"recording_key": RK}) == "recording_key=CX169-260602-T003346-Network-000004"
    assert Scope.encode_path({"well_uid": "CX169|T003346|well002"}) == "well_uid=CX169-T003346-well002"
    assert Scope.encode_path({"sample_id": "CX169", "groupname": "10 uM H2O2"}) == \
        "sample_id=CX169__groupname=10-uM-H2O2"
    assert Scope.encode_path({}) == "__all__"


def test_where_filter():
    net = _dims()
    act = WellDims.from_entry("CX169/260602/T003346/ActivityScan/000003", "rec0000/well002", "Control")
    assert SCOPE_RECORDING.matches(net) is True
    assert SCOPE_RECORDING.matches(act) is False           # scan_type != Network
    assert SCOPE_SAMPLE_GROUP.matches(_dims(group="?")) is False   # groupname == "?"


def test_bad_group_by_rejected():
    with pytest.raises(ValueError):
        Scope("bad", ("not_a_dimension",))
