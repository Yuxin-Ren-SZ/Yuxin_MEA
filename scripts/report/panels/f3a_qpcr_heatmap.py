"""F3a — qPCR marker expression: treatment effect with maturation removed."""
from __future__ import annotations

NAME = "F3a_qpcr_heatmap"
CAPTION = (
    "Marker-gene expression by RT-qPCR, as fold change relative to the "
    "pre-treatment sample. Rows are genes grouped by the cell type they mark; "
    "columns are the sampling days 7, 14 and 21. Day 0 — the treatment day — is "
    "not a column because it is the reference every cell is expressed against: "
    "all arms were split from a single culture on that day, so there is one "
    "shared pre-treatment sample rather than a per-arm day 0, and its fold change "
    "is 1x by construction. The left block is the change the untreated Control "
    "culture undergoes on its own, which is development, not treatment. The right "
    "block is each treated arm against the **time-matched** Control, so the "
    "developmental change is subtracted and what remains is the treatment effect "
    "— the same difference-in-differences logic the electrophysiology figures "
    "use, and the reason the calibrator cancels exactly. Colour is spaced "
    "logarithmically about 1x because fold change is asymmetric (a doubling is "
    "2.0, a halving 0.5) and a linear ramp would crush every downregulated cell "
    "against zero. Grey cells failed quality control and are excluded from the "
    "colour scale rather than silently dropped; † marks a day whose Control "
    "reference had to be interpolated. Descriptive only: n = 1 chip with "
    "technical duplicates, so there are no error bars and no statistics."
)

from ..panel_base import panel_main  # noqa: E402


def make_fig(ctx):
    from .. import report_style as RS
    from ..fig3_qpcr import build_f3, load_qpcr_dir, treatment_effects
    from ..report_style import report_dir

    if ctx.qpcr_dir is None:
        raise RuntimeError("no qPCR plate directory (pass --qpcr-dir)")
    tidy = load_qpcr_dir(ctx.qpcr_dir)
    was = RS._CAPTION_DRAW
    RS._CAPTION_DRAW = False          # standalone panel: caption goes to its .txt
    try:
        fig = build_f3(tidy)
    finally:
        RS._CAPTION_DRAW = was
    fig.suptitle("")
    # The underlying tables travel with the panel — the figure is a summary, the
    # CSVs are the record (they cover the whole gene panel, not just what is drawn).
    out = report_dir(ctx.figure_root, "panels")
    tidy.to_csv(out / "F3a_qpcr_tidy.csv", index=False)
    treatment_effects(tidy).to_csv(out / "F3a_qpcr_effects.csv", index=False)
    return fig, None


if __name__ == "__main__":
    raise SystemExit(panel_main(__name__))
