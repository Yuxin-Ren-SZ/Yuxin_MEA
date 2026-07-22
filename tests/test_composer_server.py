"""Composer HTTP layer, against a synthetic dataset and Flask's test client.

The server is meant to be thin — every real operation lives in ``spec`` or
``compose``. These tests check the translation: that a request reaches the right
function, that a bad spec comes back as a 4xx with a message rather than a 500,
and that the export → resume loop closes through the API the page actually uses.

``/api/preview`` is deliberately not exercised here: it spawns a process pool and
re-reads the real analysis tree, which belongs in the manual pass, not the suite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _tidy() -> pd.DataFrame:
    rows = []
    for w in range(10):
        for tau in (-3, 3):
            rows.append({
                "well_uid": f"CHIP|w{w}", "sample_id": f"CHIP{w % 3}",
                "well_name": f"W{w}", "recording_key": "K", "rec_name": "r",
                "well_id": f"well{w:03d}", "DIV": 14 + tau, "tau": tau,
                "canonical_group": "Control" if w % 2 else "IVH_Late",
                "nb_rate": 0.5 + 0.1 * w, "nb_count": 10 + w,
                "nb_duration_mean": 0.3, "nb_spikes_per_burst_mean": 20.0,
                "nb_ibi_mean": 5.0, "median_firing_rate": 1.0 + 0.05 * w,
                "n_curated": 10 + w,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """An app wired to a synthetic tidy table and a temporary figure root."""
    figure_root = tmp_path / "figures"
    (figure_root / "treatment_comparison").mkdir(parents=True)
    _tidy().to_csv(figure_root / "treatment_comparison" / "tidy_long.csv", index=False)
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(
        {"global": {"analysis_root": str(tmp_path / "analysis"),
                    "figure_root": str(figure_root)}}))
    (tmp_path / "analysis").mkdir()

    from scripts.report.composer.server import create_app

    app = create_app(str(config), cache_dir=tmp_path / "cache")
    app.config.update(TESTING=True)
    with app.test_client() as c:
        c.figure_root = figure_root
        c.tmp_path = tmp_path
        yield c


def _default(client) -> dict:
    return client.get("/api/bootstrap").get_json()["default"]


def _post(client, path, body):
    return client.post(path, json=body)


# --------------------------------------------------------------------------- #
def test_page_renders(client):
    html = client.get("/").get_data(as_text=True)
    assert "Figure composer" in html
    assert "gridstack-all.js" in html          # the vendored copy is wired up


def test_bootstrap_describes_the_registry_and_the_data(client):
    d = client.get("/api/bootstrap").get_json()
    assert len(d["panels"]) >= 20
    assert {"id", "title", "section", "params", "default_w"} <= set(d["panels"][0])
    assert "Connectivity" in d["sections"]
    assert d["groups"] == ["Control", "IVH_Late"]
    assert d["data"]["tidy_rows"] == 20
    assert d["default"]["figures"]


def test_bootstrap_reports_per_group_metric_coverage(client):
    cov = client.get("/api/bootstrap").get_json()["coverage"]
    assert cov["Control"]["burst"] > 0
    # the synthetic table has no connectivity columns at all
    assert cov["Control"]["connectivity"] == 0
    assert cov["IVH_Late"]["_wells"] == 5


def test_validate_reports_warnings_without_failing(client):
    comp = _default(client)
    comp["figures"][0]["panels"][0]["panel"] = "no.such.panel"
    r = _post(client, "/api/validate", comp)
    assert r.status_code == 200
    assert any("unknown panel" in w for w in r.get_json()["warnings"])


def test_validate_rejects_an_out_of_grid_panel(client):
    comp = _default(client)
    comp["figures"][0]["panels"][0]["w"] = 99
    r = _post(client, "/api/validate", comp)
    assert r.status_code == 400
    assert "past the grid" in r.get_json()["error"]


def test_save_then_load_round_trips(client):
    comp = _default(client)
    comp["name"] = "my layout"
    saved = _post(client, "/api/spec", comp).get_json()
    assert saved["name"] == "my_layout"          # sanitised for the filesystem
    assert "my_layout" in saved["specs"]

    loaded = client.get("/api/spec/my_layout").get_json()
    assert [f["id"] for f in loaded["composition"]["figures"]] == \
           [f["id"] for f in comp["figures"]]
    assert loaded["data_delta"] == []


def test_loading_an_unknown_spec_is_a_404(client):
    assert client.get("/api/spec/nope").status_code == 404


def test_split_and_merge_go_through_the_api(client):
    comp = _default(client)
    fid = comp["figures"][0]["id"]
    uids = [p["uid"] for p in comp["figures"][0]["panels"]][-1:]

    after = _post(client, "/api/split",
                  {"composition": comp, "figure": fid, "uids": uids}).get_json()
    comp2 = after["composition"]
    assert len(comp2["figures"]) == len(comp["figures"]) + 1
    new_id = comp2["figures"][1]["id"]

    merged = _post(client, "/api/merge",
                   {"composition": comp2, "figures": [fid, new_id],
                    "title": "back together"}).get_json()["composition"]
    assert len(merged["figures"]) == len(comp["figures"])
    assert merged["figures"][0]["title"] == "back together"


def test_split_that_would_empty_a_figure_is_a_400(client):
    comp = _default(client)
    fid = comp["figures"][0]["id"]
    uids = [p["uid"] for p in comp["figures"][0]["panels"]]
    r = _post(client, "/api/split", {"composition": comp, "figure": fid, "uids": uids})
    assert r.status_code == 400
    assert "empty the source" in r.get_json()["error"]


# --------------------------------------------------------------------------- #
# Export -> resume, the loop the user actually depends on
# --------------------------------------------------------------------------- #
def test_export_writes_outputs_and_registers_the_spec_for_loading(client):
    comp = _default(client)
    comp["name"] = "session1"
    comp["figures"] = comp["figures"][:1]        # one figure keeps this fast
    out = client.tmp_path / "out"

    m = _post(client, "/api/export",
              {"composition": comp, "out_dir": str(out),
               "formats": ["png"]}).get_json()

    assert Path(m["combined_pdf"]).exists()
    assert Path(m["spec"]).exists()
    assert (Path(m["out_dir"]) / "manifest.json").exists()
    # exporting is the end-of-session action; it must leave the layout loadable
    assert "session1" in m["specs"]
    assert client.get("/api/spec/session1").status_code == 200


def test_exported_spec_can_be_loaded_back_by_path(client):
    comp = _default(client)
    comp["name"] = "byPath"
    comp["figures"] = comp["figures"][:1]
    m = _post(client, "/api/export",
              {"composition": comp, "out_dir": str(client.tmp_path / "bp"),
               "formats": ["png"]}).get_json()

    by_file = _post(client, "/api/spec_from_path", {"path": m["spec"]}).get_json()
    by_dir = _post(client, "/api/spec_from_path", {"path": m["out_dir"]}).get_json()
    assert by_file["composition"] == by_dir["composition"]
    assert by_file["composition"]["name"] == "byPath"
    assert by_file["data_delta"] == []


def test_load_by_path_rejects_a_missing_file(client):
    r = _post(client, "/api/spec_from_path", {"path": "/definitely/not/here.json"})
    assert r.status_code == 404
    assert "no such file" in r.get_json()["error"]


def test_load_by_path_needs_a_path(client):
    assert _post(client, "/api/spec_from_path", {}).status_code == 400


def test_export_of_an_invalid_spec_is_a_400_not_a_500(client):
    comp = _default(client)
    comp["figures"][0]["rows"] = 0
    r = _post(client, "/api/export", {"composition": comp})
    assert r.status_code == 400


def test_panel_png_for_an_unrendered_key_is_a_404(client):
    assert client.get("/api/panel/deadbeef.png").status_code == 404
