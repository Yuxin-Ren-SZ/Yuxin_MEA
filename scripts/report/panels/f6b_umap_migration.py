"""F6b — Developmental migration of the resting and bursting manifolds."""
from __future__ import annotations

NAME = "F6b_umap_migration"
CAPTION = (
    "How a well's network-state structure moves after treatment. One row per arm "
    "(a chip-matched trio, so the comparison is not confounded by chip); each "
    "column is a developmental stage, spaced from the treatment day (τ0) to that "
    "well's latest recording, with its actual day annotated. Every point is a "
    "time bin embedded in that well's pooled UMAP — crucially the embedding is "
    "fitted once on all days together, so the axes are identical across columns "
    "and a bin landing in the same place means the same thing at every time "
    "point; separate per-day embeddings would have incomparable axes. Colour "
    "marks the network state (resting versus bursting), the X marks the centroid "
    "of the resting bins and the arrow tracks how that centroid migrates between "
    "consecutive stages. Illustrative single wells, not a group statistic."
)

from ..panel_base import panel_main  # noqa: E402


def make_fig(ctx):
    from .. import report_style as RS
    from ..fig_umap_migration import build_umap_migration

    # The composed-figure builder bakes in a title and an on-image caption; a
    # standalone panel carries neither (the caption goes to its own .txt).
    was = RS._CAPTION_DRAW
    RS._CAPTION_DRAW = False
    try:
        fig, _meta = build_umap_migration(ctx.tidy, ctx.analysis_root)
    finally:
        RS._CAPTION_DRAW = was
    fig.suptitle("")
    return fig, None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
