"""F6d — What defines each burst archetype (feature profile)."""
from __future__ import annotations

import numpy as np

from ..panel_base import panel_main
from ..report_style import fc_div_cmap
from . import _f6_common as A

NAME = "F6d_archetype_profiles"
FIGSIZE = (4.4, 2.2)
CAPTION = (
    "The feature signature of each cohort-wide burst archetype. Rows are the "
    "burst-shape features the clustering used, columns the archetypes; colour is "
    "the archetype's mean value of that feature in standard deviations from the "
    "cohort mean, so red means the archetype scores high on that feature and blue "
    "low. Reading down a column gives the definition of that archetype and the "
    "name it is given in F6c. Features are the biologically interpretable burst "
    "properties only — duration, within-burst firing rate, participation, spikes "
    "per burst, peak population rate, peak synchrony and intensity — with the "
    "detector's internal likelihood and posterior features deliberately excluded "
    "so the archetypes describe network behaviour rather than the algorithm."
)


def build(ax, ctx):
    from ..fig6c_bursttypes import FEAT_LABELS

    arch = A.archetypes(ctx)
    # Transposed relative to the original layout: features on the y-axis and
    # archetypes on the x. With few archetypes and many features, the old
    # orientation produced a tall column of enormous cells that dominated the
    # figure and still did not say what each archetype looked like.
    prof = arch.profile.T
    im = ax.imshow(prof, aspect="auto", cmap=fc_div_cmap(), vmin=-1.5, vmax=1.5)
    ax.set_yticks(range(len(FEAT_LABELS)))
    ax.set_yticklabels(FEAT_LABELS, fontsize=6)
    ax.set_xticks(range(arch.k))
    ax.set_xticklabels([A.arch_label(arch, a) for a in range(arch.k)],
                       fontsize=7, fontweight="bold")
    for a, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(A.arch_color(a))
    for a in range(arch.k):                     # value in the cell, not just hue
        for f in range(prof.shape[0]):
            v = prof[f, a]
            ax.text(a, f, f"{v:+.1f}", ha="center", va="center", fontsize=5,
                    color=("white" if abs(v) > 1.0 else "0.25"))
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("z-score", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    return None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
