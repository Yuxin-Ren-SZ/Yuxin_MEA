"""F7m — Connectivity difference-in-differences over treatment time."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import TIMECOURSE_METHOD, did_timecourse_build

NAME = "F7m_did_timecourse"
FIGSIZE = (7.6, 4.4)
CAPTION = ("Connectivity treatment effect over time. " + TIMECOURSE_METHOD)

build_fig = did_timecourse_build(D.F7_METRICS, ncols=3,
                                 ylabel="DiD vs Control  (log₂ or Δ)")


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
