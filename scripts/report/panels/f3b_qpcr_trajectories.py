"""F3b — Per-gene expression trajectories across the sampling days."""
from __future__ import annotations

NAME = "F3b_qpcr_trajectories"
CAPTION = (
    "Per-gene expression trajectories across the four sampling days (day 0 = "
    "treatment day, then 7, 14 and 21). One small panel per gene, on a log2 "
    "fold-change axis where a symmetric linear scale is the readable choice for "
    "curves. The upper row shows each arm against the shared pre-treatment "
    "calibrator, so the Control curve is the developmental trajectory; the lower "
    "row shows each treated arm against the time-matched Control, which removes "
    "that development and leaves the treatment effect. Points that failed quality "
    "control are marked rather than dropped. Descriptive only: n = 1 chip with "
    "technical duplicates, so no error bars and no statistics — the data layer "
    "computes across-chip means and SEMs and will start drawing them once a "
    "replicate chip exists."
)

from ..panel_base import panel_main  # noqa: E402


def make_fig(ctx):
    from .. import report_style as RS
    from ..fig3_qpcr import load_qpcr_dir
    from ..fig_s6_qpcr_trajectories import build_s6

    if ctx.qpcr_dir is None:
        raise RuntimeError("no qPCR plate directory (pass --qpcr-dir)")
    tidy = load_qpcr_dir(ctx.qpcr_dir)
    was = RS._CAPTION_DRAW
    RS._CAPTION_DRAW = False
    try:
        fig = build_s6(tidy)
    finally:
        RS._CAPTION_DRAW = was
    fig.suptitle("")
    return fig, None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
