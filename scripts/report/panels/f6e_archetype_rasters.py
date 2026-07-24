"""F6e — Example rasters, several per burst archetype."""
from __future__ import annotations

from ..panel_base import panel_main
from ..report_style import MUTED
from . import _f6_common as A

NAME = "F6e_archetype_rasters"
N_EXAMPLES = 3
CAPTION = (
    "What each burst archetype actually looks like. Every column is one "
    "archetype and every row one example: the {n} bursts whose feature vectors "
    "sit closest to that archetype's centre, drawn as curated-unit spike rasters "
    "from 0.5 s before burst onset to 0.5 s after burst end, with the detected "
    "burst window shaded in the archetype's colour. Several examples per "
    "archetype rather than one, so the reader sees the spread within an archetype "
    "instead of taking a single burst as its definition. Illustrative examples "
    "drawn from across the cohort; the feature signature that defines each "
    "archetype is F6d."
).format(n=N_EXAMPLES)


def make_fig(ctx):
    import matplotlib.pyplot as plt

    from ..fig6c_bursttypes import _burst_raster

    arch = A.archetypes(ctx)
    fig = plt.figure(figsize=(1.7 * arch.k + 0.6, 1.5 * N_EXAMPLES + 0.5))
    gs = fig.add_gridspec(N_EXAMPLES, arch.k, hspace=0.55, wspace=0.35)
    for a in range(arch.k):
        reps = A.representative_bursts(arch, a, N_EXAMPLES)
        colour = A.arch_color(a)
        for j in range(N_EXAMPLES):
            ax = fig.add_subplot(gs[j, a])
            if j >= len(reps):
                ax.set_visible(False)
                continue
            _burst_raster(ax, ctx.analysis_root, reps.iloc[j], colour, "")
            ax.set_title("")
            if j == 0:
                ax.set_title(A.arch_label(arch, a), loc="left", fontsize=7.5,
                             color=colour, fontweight="bold", pad=3)
            if a:
                ax.set_ylabel("")
            if j < N_EXAMPLES - 1:
                ax.set_xlabel("")
    fig.text(0.99, 0.005, "shaded = detected burst window", ha="right",
             va="bottom", fontsize=5.5, color=MUTED)
    return fig, None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
