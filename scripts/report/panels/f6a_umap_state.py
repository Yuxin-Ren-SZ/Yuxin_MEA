"""F6a — Feature-space embedding of one recording's time bins."""
from __future__ import annotations

from ..panel_base import panel_main
from ..report_style import MUTED, STATE_BURST, UMAP_REST_GREY

NAME = "F6a_umap_state"
FIGSIZE = (3.0, 3.0)
ARM = "IVH_Late"
CAPTION = (
    "The burst manifold the detector recovers. Each point is one time bin of a "
    "single recording, described by the 26 features the detector computes and "
    "projected to two dimensions with UMAP; bins the clustering assigned to a "
    "burst state are coloured, the rest are grey. Bursts occupy a distinct, "
    "low-density region of feature space rather than sitting on a threshold along "
    "one axis, which is why the detector clusters in this space instead of "
    "thresholding a population firing rate. Axes are UMAP coordinates and carry "
    "no units — only relative position is meaningful. Illustrative single "
    "recording, not a group statistic."
)


def build(ax, ctx):
    from ..fig6_ml import _panel_umap

    row = ctx.ml_trace_row(ARM)
    if row is None:
        raise RuntimeError(f"no {ARM} recording with an ML debug trace on disk")
    _panel_umap(ax, ctx.ml_trace(row))
    ax.set_box_aspect(1)
    ax.scatter([], [], s=10, c=UMAP_REST_GREY, label="rest bins")
    ax.scatter([], [], s=10, c=STATE_BURST, label="burst bins")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.set_title(f"{row.well_uid.split('|')[0]} · {ARM.replace('_', ' ')} · "
                 f"τ{int(row.tau):+d}", loc="left", fontsize=6.5, color=MUTED)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
