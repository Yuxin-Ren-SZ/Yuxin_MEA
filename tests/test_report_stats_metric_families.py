"""Guard the RATIO/DIFF metric classification in ``scripts.report.stats``.

A log2 ratio needs strictly positive baseline *and* post. When either is zero
the ``_EPS`` guard fabricates a response of ~±30 log2 units, which then drives
the forest-plot medians and CIs in F5/F7/F7b/F8. These tests pin the invariant:
any metric that is exactly zero in a meaningful share of wells must be a DIFF
metric, so the guard is never load-bearing for a bounded graph descriptor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report import stats as S  # noqa: E402

# Bounded fractions / normalised graph descriptors: exactly zero whenever a
# well's thresholded graph comes out empty (or has no hubs / no leaves).
ZERO_PRONE_METRICS = [
    "edge_density", "clustering_coeff", "global_efficiency",
    "hub_fraction", "leaf_fraction", "mean_betweenness", "participation_mean",
    "degree_cv", "rich_club",
    "mean_te", "te_edge_density", "degree_asymmetry",
]


def _synthetic_tidy(metric: str, n_wells: int = 12) -> pd.DataFrame:
    """Paired pre/post wells where half have a zero baseline for ``metric``."""
    rows = []
    for w in range(n_wells):
        zero_baseline = w % 2 == 0
        for tau, val in ((-3, 0.0 if zero_baseline else 0.4), (3, 0.25)):
            rows.append({
                "well_uid": f"CHIP|w{w}", "sample_id": f"CHIP{w % 3}",
                "canonical_group": "Control" if w % 3 == 0 else "IVH_Late",
                "tau": tau, "DIV": 14 + tau, metric: val,
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("metric", ZERO_PRONE_METRICS)
def test_zero_prone_metrics_are_diff_metrics(metric):
    assert metric in S.DIFF_METRICS, (
        f"{metric!r} is exactly zero in a non-trivial share of wells; as a "
        f"RATIO metric the _EPS guard fabricates a ~30 log2-unit response"
    )
    assert metric not in S.RATIO_METRICS


@pytest.mark.parametrize("metric", ZERO_PRONE_METRICS)
def test_zero_baseline_does_not_explode_the_response(metric):
    resp = S.well_response(_synthetic_tidy(metric), metrics=[metric])
    assert not resp.empty
    assert resp.response.abs().max() <= 1.0, (
        f"{metric!r} response reached {resp.response.abs().max():.1f} on wells "
        f"with a zero baseline — the metric is being treated as a ratio"
    )
    assert not resp.eps_floored.any()


def test_eps_floored_flags_a_genuine_ratio_floor():
    """A rate going to zero is a legitimate floor, but it must be flagged."""
    resp = S.well_response(_synthetic_tidy("nb_rate"), metrics=["nb_rate"])
    assert "nb_rate" in S.RATIO_METRICS
    floored = resp[resp.eps_floored]
    assert len(floored) > 0
    assert (floored.response.abs() > 10).all()      # the guard is load-bearing
    assert not resp[~resp.eps_floored].eps_floored.any()


def test_metric_families_are_disjoint_and_complete():
    assert not set(S.RATIO_METRICS) & set(S.DIFF_METRICS)
    assert len(S.ALL_METRICS) == len(set(S.ALL_METRICS))


@pytest.mark.parametrize("metric", ZERO_PRONE_METRICS)
def test_diff_response_is_a_plain_difference(metric):
    resp = S.well_response(_synthetic_tidy(metric), metrics=[metric])
    expected = {0.25 - 0.0, 0.25 - 0.4}
    assert set(np.round(resp.response.unique(), 10)) == {
        round(v, 10) for v in expected}
