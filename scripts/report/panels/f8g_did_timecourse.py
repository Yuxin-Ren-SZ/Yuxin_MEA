"""F8g — Criticality difference-in-differences over treatment time."""
from __future__ import annotations

from .. import panel_data as D
from ..panel_base import panel_main
from ._did import TIMECOURSE_METHOD, did_timecourse_build

NAME = "F8g_did_timecourse"
FIGSIZE = (6.4, 4.4)
CAPTION = ("Criticality treatment effect over time. " + TIMECOURSE_METHOD)

build_fig = did_timecourse_build(D.F8_METRICS, ncols=2,
                                 ylabel="DiD vs Control  (log₂ or Δ)")


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
