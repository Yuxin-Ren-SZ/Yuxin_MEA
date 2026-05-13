"""Tests for ``yuxin_mea.analysis.curation_summary``."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from yuxin_mea.analysis.curation_summary import (
    aggregate_curation_summaries,
    format_curation_summary,
    summarize_curation,
)


def _write_qm_pkl(dir_: Path, n_curated: int, n_rejected: int) -> None:
    rows = []
    for i in range(n_curated):
        rows.append({
            "unit_id": i,
            "curated": True,
            "presence_ratio": 0.9,
            "rp_contamination": 0.05,
            "firing_rate": 1.5,
            "amplitude_median": -55.0,
        })
    for i in range(n_curated, n_curated + n_rejected):
        rows.append({
            "unit_id": i,
            "curated": False,
            "presence_ratio": 0.4,
            "rp_contamination": 0.30,
            "firing_rate": 0.02,
            "amplitude_median": -15.0,
        })
    pd.DataFrame(rows).to_pickle(dir_ / "quality_metrics.pkl")


def _write_rl_pkl(dir_: Path, reason_lists: list[str]) -> None:
    pd.DataFrame({"reasons": reason_lists}).to_pickle(dir_ / "rejection_log.pkl")


# ---------------------------------------------------------------------------
# summarize_curation
# ---------------------------------------------------------------------------


def test_summarize_curation_missing_qm_raises():
    with TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="quality_metrics.pkl"):
            summarize_curation(Path(tmp))


def test_summarize_curation_counts_and_pct():
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        _write_qm_pkl(out, n_curated=7, n_rejected=3)
        _write_rl_pkl(
            out,
            ["presence_ratio < 0.75",
             "presence_ratio < 0.75; firing_rate < 0.05",
             "rp_contamination > 0.15"],
        )
        s = summarize_curation(out)

        assert s["n_total"] == 10
        assert s["n_curated"] == 7
        assert s["n_rejected"] == 3
        assert s["pct_kept"] == 70.0
        assert s["rejection_reasons"]["presence_ratio < 0.75"] == 2
        assert s["rejection_reasons"]["firing_rate < 0.05"] == 1
        assert s["rejection_reasons"]["rp_contamination > 0.15"] == 1
        assert s["metric_stats"] is not None
        # describe() returns 8 rows (count/mean/std/min/25%/50%/75%/max)
        assert s["metric_stats"].shape == (8, 4)


def test_summarize_curation_no_rejection_log():
    """Missing rejection_log.pkl must produce an empty reasons dict, not error."""
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        _write_qm_pkl(out, n_curated=5, n_rejected=0)
        s = summarize_curation(out)
        assert s["rejection_reasons"] == {}
        assert s["n_rejected"] == 0


def test_summarize_curation_legacy_format_raises_clear_error():
    """A quality_metrics.pkl without a `curated` column gets a clear error."""
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        pd.DataFrame({"unit_id": [0, 1]}).to_pickle(out / "quality_metrics.pkl")
        with pytest.raises(KeyError, match="no `curated` column"):
            summarize_curation(out)


# ---------------------------------------------------------------------------
# format_curation_summary
# ---------------------------------------------------------------------------


def test_format_curation_summary_contains_key_lines():
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        _write_qm_pkl(out, n_curated=4, n_rejected=1)
        _write_rl_pkl(out, ["firing_rate < 0.05"])
        text = format_curation_summary(summarize_curation(out))

        assert "Total units:    5" in text
        assert "Passed:         4" in text
        assert "Rejected:       1" in text
        assert "% kept:         80.0%" in text
        assert "Rejection reasons:" in text
        assert "firing_rate < 0.05: 1" in text


# ---------------------------------------------------------------------------
# aggregate_curation_summaries
# ---------------------------------------------------------------------------


def test_aggregate_curation_summaries_skips_missing():
    with TemporaryDirectory() as tmp:
        a = Path(tmp) / "a"; a.mkdir()
        b = Path(tmp) / "b"; b.mkdir()
        c = Path(tmp) / "c"; c.mkdir()  # no qm file → skipped
        _write_qm_pkl(a, n_curated=2, n_rejected=8)
        _write_qm_pkl(b, n_curated=10, n_rejected=0)

        df = aggregate_curation_summaries([a, b, c])
        assert len(df) == 2
        assert set(df["n_total"]) == {10}
        assert df.loc[df["output_dir"] == str(a), "n_curated"].item() == 2
        assert df.loc[df["output_dir"] == str(b), "pct_kept"].item() == 100.0
        assert df.loc[df["output_dir"] == str(b), "median_firing_rate"].item() == 1.5


def test_aggregate_curation_summaries_empty_input():
    df = aggregate_curation_summaries([])
    assert df.empty
    # Empty result must still have the expected columns so downstream
    # consumers don't break on a fresh analysis_root.
    assert "n_curated" in df.columns
    assert "pct_kept" in df.columns
