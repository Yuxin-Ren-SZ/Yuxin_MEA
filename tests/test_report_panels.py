"""Contract tests for the standalone report panels and the statistics behind them.

Three things must not drift:

* every panel is runnable on its own and produces a caption-free image plus a
  caption ``.txt`` (that separation is the whole point of the panel split);
* the significance glyphs follow the stated rule — ``*`` from the FDR q-value,
  ``△`` from the uncorrected p — with nothing else able to promote a mark;
* the per-time-point difference-in-differences reduces to the pooled one when
  there is a single post-treatment bin, so the new estimator is a refinement of
  the old rather than a different quantity.
"""
from __future__ import annotations

import importlib
import pkgutil
import types

import numpy as np
import pandas as pd
import pytest

from scripts.report import panel_base, panels as panels_pkg, stats as S
from scripts.report.report_style import fdr_stars, sig_glyph

PANEL_MODULES = sorted(
    m.name for m in pkgutil.iter_modules(panels_pkg.__path__)
    if not m.name.startswith("_")
)


# --------------------------------------------------------------------------- #
# Panel contract
# --------------------------------------------------------------------------- #
def test_panels_exist():
    assert PANEL_MODULES, "no panel modules discovered"


@pytest.mark.parametrize("mod_name", PANEL_MODULES)
def test_panel_declares_contract(mod_name):
    mod = importlib.import_module(f"{panels_pkg.__name__}.{mod_name}")
    assert getattr(mod, "NAME", ""), f"{mod_name}: missing NAME"
    builders = [b for b in ("build", "build_fig", "make_fig") if hasattr(mod, b)]
    assert builders, f"{mod_name}: needs build / build_fig / make_fig"
    caption = getattr(mod, "CAPTION", "")
    assert isinstance(caption, str) and len(caption) > 80, (
        f"{mod_name}: CAPTION must explain the panel, not label it")


def test_panel_names_unique():
    names = [importlib.import_module(f"{panels_pkg.__name__}.{m}").NAME
             for m in PANEL_MODULES]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate panel NAMEs: {dupes}"


def test_render_panel_writes_image_and_caption_without_drawing_it(tmp_path):
    """The image must stay clean; the caption goes to its own file."""
    mod = types.SimpleNamespace(
        __name__="fake_panel",
        NAME="X1a_fake",
        CAPTION="A caption that must never be drawn on the image itself.",
        FIGSIZE=(2.0, 1.5),
        build=lambda ax, ctx: ax.plot([0, 1], [0, 1]) and None,
    )
    ctx = types.SimpleNamespace(figure_root=tmp_path)
    rec = panel_base.render_panel(mod, ctx, formats=("png",),
                                  out_dir=tmp_path / "panels")
    assert rec["status"] == "ok", rec
    assert (tmp_path / "panels" / "X1a_fake.png").exists()
    cap = tmp_path / "report" / "captions" / "X1a_fake.txt"
    assert cap.exists() and cap.read_text().strip() == mod.CAPTION


def test_render_panel_reports_failure_without_raising(tmp_path):
    def boom(ax, ctx):
        raise ValueError("no data on disk")

    mod = types.SimpleNamespace(__name__="fake", NAME="X1b_fake",
                                CAPTION="x" * 100, FIGSIZE=(1.0, 1.0), build=boom)
    ctx = types.SimpleNamespace(figure_root=tmp_path)
    rec = panel_base.render_panel(mod, ctx, formats=("png",), out_dir=tmp_path)
    assert rec["status"] == "error" and "no data on disk" in rec["warning"]
    assert rec["files"] == []


# --------------------------------------------------------------------------- #
# Significance glyphs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q,expected", [
    (0.0005, "***"), (0.005, "**"), (0.04, "*"), (0.05, ""), (0.9, ""),
    (np.nan, ""), (None, ""),
])
def test_fdr_stars(q, expected):
    assert fdr_stars(q) == expected


def test_glyph_star_requires_q_and_triangle_requires_p_only():
    # FDR-significant -> a star, regardless of how small p is.
    assert sig_glyph(0.001, 0.005)[0] == "**"
    # p clears 0.05 but q does not -> the suggestive triangle.
    assert sig_glyph(0.03, 0.4)[0] == "△"
    # neither -> no mark at all.
    assert sig_glyph(0.2, 0.9)[0] == ""
    # A missing p must not silently become significant.
    assert sig_glyph(np.nan, np.nan)[0] == ""


def test_glyph_ignores_direction_consistency():
    """The rule is p/q only — no extra clause may suppress or promote a mark."""
    assert sig_glyph(0.04, 0.6)[0] == "△"
    assert sig_glyph(0.04, 0.005)[0] == "**"


# --------------------------------------------------------------------------- #
# Per-time-point difference-in-differences
# --------------------------------------------------------------------------- #
def _tidy(seed: int = 0) -> pd.DataFrame:
    """Three chips x (2 control + 2 treated) wells, one pre and one post recording."""
    rng = np.random.default_rng(seed)
    rows = []
    # Per-chip jitter on the growth factors: without it every chip's DiD is
    # identical, the one-sample t-test has zero variance and returns NaN by
    # design — a degenerate fixture, not a real effect.
    for chip in ("A", "B", "C"):
        ctrl_gain = 2.0 * (1 + 0.05 * rng.standard_normal())
        treat_gain = 3.0 * (1 + 0.05 * rng.standard_normal())
        for arm in ("Control", "IVH_Early"):
            gain = ctrl_gain if arm == "Control" else treat_gain
            for w in range(2):
                wuid = f"{chip}|w{arm[:2]}{w}"
                base = 1.0 + 0.1 * rng.standard_normal()
                for tau, val, grp in ((-2, base, "Control"),
                                      (10, base * gain, arm)):
                    rows.append(dict(well_uid=wuid, sample_id=chip, tau=tau,
                                     DIV=10 + tau, canonical_group=grp,
                                     nb_rate=val))
    return pd.DataFrame(rows)


def test_response_by_tau_matches_pooled_response_for_a_single_bin():
    tidy = _tidy()
    pooled = S.well_response(tidy, metrics=["nb_rate"])
    by_tau = S.well_response_by_tau(tidy, metrics=["nb_rate"])
    assert by_tau.tauc.nunique() == 1, "fixture should land in one post bin"
    merged = pooled.merge(by_tau, on=["well_uid", "metric"],
                          suffixes=("_pooled", "_tau"))
    assert len(merged) == len(pooled)
    np.testing.assert_allclose(merged.response_pooled, merged.response_tau)


def test_did_at_endpoint_recovers_the_simulated_effect():
    tidy = _tidy()
    resp = S.well_response_by_tau(tidy, metrics=["nb_rate"])
    did = S.did_at_endpoint(resp, focus_arms=["IVH_Early"])
    assert len(did) == 1
    row = did.iloc[0]
    assert row.n_chips_paired == 3
    # Treated wells triple while Control doubles: log2(3) - log2(2) = 0.585.
    assert row.did == pytest.approx(np.log2(3) - np.log2(2), abs=0.15)
    assert row.p_vs_control < 0.05          # a clean, consistent effect
    assert row.chip_consistent


def test_did_q_corrects_across_the_whole_family_not_one_metric():
    """Passing metrics one at a time would make q == p — the family must be whole."""
    tidy = _tidy()
    tidy = tidy.assign(other=tidy.nb_rate * 1.7)
    resp = S.well_response_by_tau(tidy, metrics=["nb_rate", "other"])
    single = S.did_at_endpoint(
        S.well_response_by_tau(tidy, metrics=["nb_rate"]), focus_arms=["IVH_Early"])
    family = S.did_at_endpoint(resp, focus_arms=["IVH_Early"])
    q_single = float(single.q_bh.iloc[0])
    q_family = float(family.loc[family.metric == "nb_rate", "q_bh"].iloc[0])
    assert q_family >= q_single


def test_primary_endpoint_is_the_latest_bin_all_chips_reach():
    tidy = _tidy()
    # One chip stops early: its extra late recording is absent, so the endpoint
    # must stay at the bin every chip still contributes to.
    late = tidy[(tidy.sample_id != "C") & (tidy.tau == 10)].copy()
    late["tau"] = 22
    tidy2 = pd.concat([tidy, late], ignore_index=True)
    resp = S.well_response_by_tau(tidy2, metrics=["nb_rate"])
    assert sorted(resp.tauc.unique()) == [10.0, 22.0]
    assert S.primary_endpoint(resp, ["IVH_Early"]) == 10.0


# ---------------------------------------------------------------------------
# Time-since-media-change window
# ---------------------------------------------------------------------------
_WINDOW = {"pre": (24.0, 48.0), "post": (18.0, 30.0), "assume_unknown_hours": 24.0}


def _rows(*specs):
    """``(well, tau, hours)`` triples -> a tidy-shaped frame."""
    return pd.DataFrame(
        [{"well_uid": w, "tau": t, "hours_since_media": h, "sample_id": "S1"}
         for w, t, h in specs]
    )


def test_media_window_keeps_the_routine_scans_and_drops_the_acute_ones():
    from scripts.report.panel_data import _restrict_media_window

    tidy = _rows(("w1", 3, 24.0), ("w1", 5, 0.0), ("w1", 7, 0.5), ("w1", 9, 48.0))
    out = _restrict_media_window(tidy, _WINDOW)
    assert list(out.tau) == [3]


def test_media_window_is_wider_before_treatment_than_after():
    """A baseline may be a day or two old; a treatment-period scan may not."""
    from scripts.report.panel_data import _restrict_media_window

    tidy = _rows(("w1", -2, 48.0), ("w1", 4, 48.0))
    out = _restrict_media_window(tidy, _WINDOW)
    assert list(out.tau) == [-2]


def test_media_window_bounds_are_inclusive():
    """The unknown-tag default sits exactly on the pre window's lower edge."""
    from scripts.report.panel_data import _restrict_media_window

    tidy = _rows(("w1", -1, 24.0), ("w1", -2, 48.0), ("w1", 2, 18.0), ("w1", 3, 30.0))
    out = _restrict_media_window(tidy, _WINDOW)
    assert len(out) == 4


def test_media_window_admits_untimed_rows_on_the_protocol_default():
    from scripts.report.panel_data import _restrict_media_window

    tidy = _rows(("w1", -3, np.nan), ("w1", 5, np.nan))
    out = _restrict_media_window(tidy, _WINDOW)
    assert len(out) == 2, "untimed rows are kept, not silently dropped"


def test_media_window_rejects_tau_zero():
    """Every tau-0 recording is a 0 h scan by construction — the treatment date
    is back-dated from that very scan's tag — so the post window rejects it."""
    from scripts.report.panel_data import _restrict_media_window

    tidy = _rows(("w1", 0, 0.0))
    assert _restrict_media_window(tidy, _WINDOW).empty


def test_media_window_disabled_returns_everything():
    from scripts.report.panel_data import _restrict_media_window

    tidy = _rows(("w1", 5, 0.0), ("w1", 7, 72.0))
    assert len(_restrict_media_window(tidy, None)) == 2


def test_media_window_is_a_noop_without_the_column():
    """An older tidy_long.csv must degrade to unfiltered, not to empty."""
    from scripts.report.panel_data import _restrict_media_window

    tidy = pd.DataFrame({"well_uid": ["w1"], "tau": [5], "sample_id": ["S1"]})
    assert len(_restrict_media_window(tidy, _WINDOW)) == 1


def test_media_window_reports_a_well_that_loses_its_baseline(caplog):
    """The one failure mode that would not look like a filter."""
    from scripts.report.panel_data import _restrict_media_window

    tidy = _rows(("w1", -1, 0.5), ("w1", 4, 24.0))
    with caplog.at_level("WARNING"):
        out = _restrict_media_window(tidy, _WINDOW)
    assert list(out.tau) == [4]
    assert any("pre or post coverage" in r.message for r in caplog.records)


def test_media_window_is_part_of_the_cache_key(tmp_path, monkeypatch):
    """Toggling the window must invalidate the pickles, not silently reuse them.

    The fingerprint used to hash only the on-disk table, so switching
    ``media_window`` off returned a cache built under the old cohort with nothing
    on screen to say so.
    """
    import json

    from scripts.report import panel_data as D

    tidy = pd.DataFrame({"well_uid": ["w1"], "tau": [4], "sample_id": ["S1"],
                         "hours_since_media": [24.0]})
    monkeypatch.setattr(D.L, "load_tidy", lambda _root: tidy)
    monkeypatch.setattr(D, "_backfill", lambda t, _root: t)
    monkeypatch.setattr(D, "_backfill_hours", lambda t, _root: t)
    monkeypatch.setattr(D, "resolve_roots", lambda _p: (tmp_path, tmp_path))

    on = tmp_path / "on.json"
    on.write_text(json.dumps({"global": {}}))
    off = tmp_path / "off.json"
    off.write_text(json.dumps({"global": {}, "media_window": None}))

    ctx_on = D.PanelContext(str(on))
    ctx_off = D.PanelContext(str(off))
    assert ctx_on.cache.fingerprint != ctx_off.cache.fingerprint
    assert ctx_off.media_window is None
