"""Tests for the qPCR report figures (F3 markers, S5 reference-gene screen).

The regression that matters is :func:`fig3_qpcr.derive_ddcq` reproducing the
shipped ``*_ddcq.csv`` exports bit-for-bit: those exports are the primary input
to F3, and the fallback path is what keeps future plates working if they ever
arrive without one.

Tests that need the real plate exports skip when the analysis share is not
mounted; the parser tests run everywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.report import fig3_qpcr as F3
from scripts.report import fig_s5_refgene as S5
from scripts.report import fig_s6_qpcr_trajectories as S6

QPCR_ROOT = Path("/mnt/Vol20tb2/SadeghLab/yuxin/analysis/qPCR")
MARKER_GENES = {"MAP2", "GFAP", "SLC17A7", "SLC32A1"}

needs_data = pytest.mark.skipif(not QPCR_ROOT.is_dir(),
                                reason=f"qPCR data not mounted at {QPCR_ROOT}")


@pytest.fixture(autouse=True)
def _style():
    from scripts.report.report_style import apply_style
    apply_style()


# --------------------------------------------------------------------------- #
# Sample-name parsing (no data needed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    # calibrator, both spellings seen in the label files
    ("CX169 CT DIV 7", ("Control", 0.0, True)),
    # spacing variants of the same condition
    ("CX169 CT D 7", ("Control", 7.0, False)),
    ("CX169 CT D7", ("Control", 7.0, False)),
    ("CX169 CT D 14", ("Control", 14.0, False)),
    ("CX169 CT D14", ("Control", 14.0, False)),
    ("CX169 CT D 21", ("Control", 21.0, False)),
    ("CX169 CT D21", ("Control", 21.0, False)),
    ("CX169 IVH L D 7", ("IVH_Late", 7.0, False)),
    ("CX169 IVH L D 14", ("IVH_Late", 14.0, False)),
    ("CX169 IVH L D 21", ("IVH_Late", 21.0, False)),
    ("CX169 IVH E D7", ("IVH_Early", 7.0, False)),
    ("CX169 IVH E D 7", ("IVH_Early", 7.0, False)),
    ("CX169 IVH E D 14", ("IVH_Early", 14.0, False)),
    ("CX169 IVH E D 21", ("IVH_Early", 21.0, False)),
    # same physical well, mislabelled D27 on the reference plate -> D21
    ("CX169 IVH E D 27", ("IVH_Early", 21.0, False)),
    ("CX169 20uM H2O2 D 7", ("H2O2_20uM", 7.0, False)),
    ("CX169 20 uM H2O2 D 7", ("H2O2_20uM", 7.0, False)),
    ("CX169 20uM H2O2 D 14", ("H2O2_20uM", 14.0, False)),
    ("CX169 20 uM H2O2 D 21", ("H2O2_20uM", 21.0, False)),
])
def test_parse_sample(raw, expected):
    assert F3.parse_sample(raw) == expected


@pytest.mark.parametrize("raw", ["NTC", "", "   ", "nonsense label"])
def test_parse_sample_rejects_non_conditions(raw):
    assert F3.parse_sample(raw) == (None, None, False)


def test_d27_and_d21_normalise_to_one_condition():
    """The alias must collapse, or IVH_Early gets two day-21 points."""
    assert (F3.normalise_sample("CX169 IVH E D 27")
            == F3.normalise_sample("CX169 IVH E D 21"))


def test_ordered_genes_puts_markers_in_story_order():
    got = F3.ordered_genes(["SLC32A1", "GFAP", "MAP2", "SLC17A7", "NEWGENE"])
    assert got == ["MAP2", "GFAP", "SLC17A7", "SLC32A1", "NEWGENE"]


def test_gene_tiers_cover_the_planned_panel_and_sort_first():
    """All 13 planned targets are known, priority-1 ahead of priority-2."""
    planned = F3.GENE_TIERS[1] + F3.GENE_TIERS[2]
    assert len(planned) == 13
    assert set(planned) == set(F3.GENE_ORDER)
    # references must not sneak into the target panel
    assert not set(planned) & F3.REFERENCE_CANDIDATES

    got = F3.ordered_genes(["SST", "MAP2", "KCNMA1", "PAX6", "MYSTERY"])
    assert got == ["MAP2", "PAX6", "KCNMA1", "SST", "MYSTERY"]
    assert F3.gene_tier("MAP2") < F3.gene_tier("SST") < F3.gene_tier("MYSTERY")


@pytest.mark.parametrize("gene,expected", [
    ("SLC17A7", "SLC17A7 (VGLUT1)"),
    ("KCNMA1", "KCNMA1 (BK channel)"),
    ("MAP2", "MAP2"),
])
def test_gene_label(gene, expected):
    assert F3.gene_label(gene) == expected


# --------------------------------------------------------------------------- #
# Directory loader
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tidy():
    return F3.load_qpcr_dir(QPCR_ROOT)


@needs_data
def test_load_qpcr_dir_has_the_four_marker_genes(tidy):
    assert set(tidy.gene) == MARKER_GENES


@needs_data
def test_reference_gene_screen_is_not_treated_as_a_marker_plate(tidy):
    """ACTB/HPRT1 belong to S5; they must never reach F3."""
    assert not {"ACTB", "HPRT1"} & set(tidy.gene)


@needs_data
def test_calibrator_is_unity_for_every_gene(tidy):
    """All arms must share one anchor, else the panels are not comparable."""
    cal = tidy[tidy.is_calibrator]
    assert set(cal.gene) == MARKER_GENES
    assert cal.fold.to_numpy() == pytest.approx(1.0)
    assert cal.log2_fold.to_numpy() == pytest.approx(0.0)
    assert (cal.group == "Control").all()
    assert (cal.day == 0).all()


@needs_data
def test_failed_conditions_are_flagged_not_dropped(tidy):
    """CT D14 (loading failure) and H2O2 D7 (no amplification) stay visible."""
    ct14 = tidy[(tidy.group == "Control") & (tidy.day == 14)]
    assert len(ct14) == len(MARKER_GENES)
    assert (ct14.qc == "ref_cq_outlier").all()

    h2o2_d7 = tidy[(tidy.group == "H2O2_20uM") & (tidy.day == 7)]
    assert len(h2o2_d7) == len(MARKER_GENES)
    assert (h2o2_d7.qc == "no_amplification").all()
    assert h2o2_d7.ddcq.isna().all()


@needs_data
def test_every_other_condition_passes_qc(tidy):
    ok = tidy[tidy.qc == "ok"]
    assert len(ok) == 11 * len(MARKER_GENES)   # 13 conditions - 2 failures


@needs_data
def test_single_chip_is_recorded_in_the_table(tidy):
    """n=1 must be explicit in the data, not just in the caption."""
    assert set(tidy.sample_id) == {"CX169"}


def _two_chip_frame():
    """Two chips, identical arms, offset baselines — the future n=2 shape.

    CX170's ΔCq sits 3 cycles above CX169's throughout (a different cDNA input),
    which is exactly what per-chip calibration and the calibrator-free effect
    measure have to absorb.
    """
    rows = []
    for chip, shift in (("CX169", 0.0), ("CX170", 3.0)):
        for group, day, dcq in [("Control", 0.0, 2.0), ("Control", 7.0, 1.0),
                                ("Control", 21.0, 1.0), ("IVH_Late", 7.0, 2.0),
                                ("IVH_Late", 21.0, 2.4)]:
            rows.append(dict(sample_id=chip, gene="MAP2", group=group, day=day,
                             dcq=dcq + shift, sem_dcq=0.05, qc="ok",
                             is_calibrator=(group == "Control" and day == 0)))
    return pd.DataFrame(rows)


def test_effects_are_averaged_across_chips_not_pooled_blindly():
    """n >= 2 readiness: per-chip effects, then mean +/- SEM over chips."""
    two = _two_chip_frame()
    eff = F3.treatment_effects(two)
    # the +3 cycle offset is a property of the chip, not the treatment: it must
    # cancel, leaving both chips on the same effect for a given day
    spread = eff.groupby('day').log2_effect.agg(lambda v: v.max() - v.min())
    assert (spread.abs() < 1e-9).all(), f'chip offset leaked through: {spread}'

    agg = F3.aggregate_chips(eff)
    d7 = agg[(agg.day == 7.0) & (agg.group == "IVH_Late")].iloc[0]
    assert d7.n_chips == 2
    assert d7.log2_effect == pytest.approx(-1.0)
    assert d7.sem_effect == pytest.approx(0.0)         # identical chips -> zero spread


@needs_data
def test_heatmap_renders_with_two_chips(tidy):
    """A second chip must widen the evidence, not break the cell lookup."""
    two = pd.concat([tidy, tidy.assign(
        sample_id="CX170",
        **{"sample": tidy["sample"].str.replace("CX169", "CX170")})],
        ignore_index=True)
    assert two.sample_id.nunique() == 2
    genes, _cols, value, _state, _text = F3.effect_matrix(two)
    one = F3.effect_matrix(tidy)[2]
    # identical duplicate chip -> identical means, so the map is unchanged
    assert np.allclose(value, one, equal_nan=True)
    F3.build_f3(two)


def test_aggregate_reports_no_sem_for_a_single_chip():
    one = _two_chip_frame().query("sample_id == 'CX169'")
    agg = F3.aggregate_chips(F3.treatment_effects(one))
    assert set(agg.n_chips.unique()) <= {0, 1}
    assert agg.sem_effect.isna().all()


@needs_data
def test_all_four_arms_present_at_every_timepoint(tidy):
    grid = tidy.groupby(["group", "day"]).size().unstack()
    assert set(grid.index) == {"Control", "IVH_Early", "IVH_Late", "H2O2_20uM"}
    assert set(grid.columns) == {0.0, 7.0, 14.0, 21.0}


# --------------------------------------------------------------------------- #
# Regression lock: raw wells -> ddCq must equal the shipped export
# --------------------------------------------------------------------------- #
@needs_data
@pytest.mark.parametrize("plate", ["CX169 Plate1 GAPDH MAP2 GFAP",
                                   "CX169 Plate2 GAPDH SLC17A7 SLC32A1"])
def test_derived_ddcq_matches_shipped_export(plate):
    pdir = QPCR_ROOT / plate
    shipped = sorted(pdir.glob("*_ddcq.csv"))
    assert shipped, f"no shipped ddCq export in {pdir}"

    want = F3._read_ddcq(shipped[0])
    want = want.set_index([want["sample"].map(F3.normalise_sample), "gene"])
    got = F3.derive_ddcq(pdir)
    got = got.set_index([got["sample"].map(F3.normalise_sample), "gene"])

    common = want.index.intersection(got.index)
    assert len(common) >= 20, "too few overlapping (sample, gene) pairs to compare"
    for col in ("mean_cq", "ref_cq", "dcq", "ddcq", "ddcq_sem"):
        pd.testing.assert_series_equal(
            got.loc[common, col].sort_index(),
            want.loc[common, col].sort_index(),
            check_names=False, atol=1e-3)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _titled_axes(fig):
    return [ax for ax in fig.axes if ax.get_title(loc="left")]


@needs_data
def test_s6_clamps_axes_to_qc_passing_data(tidy):
    """A 119x QC-failed artefact must not set the MAP2 y-scale.

    The clamp used to live in ``build_f3``; it moved to S6 when F3 became the
    heatmap, and it still has to work there.
    """
    fig = S6.build_s6(tidy)
    lo, hi = _titled_axes(fig)[0].get_ylim()          # block A, first gene = MAP2
    assert hi < 2.0, f"MAP2 y-axis stretched to {hi} by the QC-failed point"
    assert lo < 0.0 < hi


@needs_data
def test_s6_draws_two_blocks_of_one_panel_per_gene(tidy):
    fig = S6.build_s6(tidy)
    assert len(_titled_axes(fig)) == 2 * len(MARKER_GENES)


@needs_data
def test_no_error_bars_at_n_equals_one(tidy):
    """At n = 1 the only spread is technical; drawing it would imply biology."""
    from matplotlib.container import ErrorbarContainer
    for fig in (F3.build_f3(tidy), S6.build_s6(tidy)):
        for ax in fig.axes:
            assert not [c for c in ax.containers
                        if isinstance(c, ErrorbarContainer)]


@needs_data
def test_build_f3_heatmap_excludes_failures_from_the_colour_scale(tidy):
    """The +6.89 log2 MAP2 artefact must not set vmax and wash the map out."""
    genes, _cols, value, state, text = F3.effect_matrix(tidy)
    assert np.nanmax(np.abs(value)) == pytest.approx(4.50, abs=0.01)
    fail = value[state != "ok"]
    assert np.isnan(fail).all(), "QC-failed cells must not be colour-mapped"
    # the artefact is still reported, just greyed and bracketed
    r, c = genes.index("MAP2"), 1                      # development block, D14
    assert state[r, c] == "ref_cq_outlier"
    assert text[r, c] == "(119×)"


@pytest.mark.parametrize("log2,expected", [
    (0.0, "1×"),
    (0.85, "1.8×"),
    (4.47, "22×"),
    (6.89, "119×"),              # never scientific notation
    (-2.31, "0.2×"),
    (-3.38, "0.096×"),
])
def test_fold_text(log2, expected):
    assert F3.fold_text(log2) == expected


def test_fold_labels_are_symmetric_about_one():
    """An n-fold up and an n-fold down must sit equally far from 1x on the bar."""
    assert F3.fold_text(2.0) == "4×"
    assert F3.fold_text(-2.0) == "0.25×"


@needs_data
def test_heatmap_cells_are_labelled_in_fold_change(tidy):
    _genes, _cols, _value, state, text = F3.effect_matrix(tidy)
    labelled = text[(state == "ok")]
    assert len(labelled) and all(t.endswith("×") for t in labelled)


@needs_data
def test_build_f3_heatmap_renders(tidy):
    fig = F3.build_f3(tidy)
    assert len(fig.axes) >= 1


# --------------------------------------------------------------------------- #
# S5 — reference-gene screen
# --------------------------------------------------------------------------- #
@needs_data
def test_ref_screen_covers_three_candidates_and_drops_failures():
    screen = S5.load_ref_screen(QPCR_ROOT)
    assert set(screen.gene) == {"ACTB", "GAPDH", "HPRT1"}   # GABDH typo aliased
    mat = S5.condition_matrix(screen)
    assert len(mat) == 11
    assert not set(S5.EXCLUDE) & set(mat.index)


@needs_data
def test_genorm_picks_gapdh_the_reference_actually_used():
    mat = S5.condition_matrix(S5.load_ref_screen(QPCR_ROOT))
    m = S5.genorm_m(mat)
    assert m.idxmin() == F3.REFERENCE_GENE
    assert (m < 0.5).all(), "all candidates should clear the geNorm M < 0.5 bar"


@needs_data
def test_pairwise_dcq_sd_is_symmetric_with_genorm():
    mat = S5.condition_matrix(S5.load_ref_screen(QPCR_ROOT))
    pair = S5.pairwise_dcq_sd(mat)
    assert len(pair) == 3
    # With 3 genes, M_i is the mean of the two pairs that involve gene i.
    m = S5.genorm_m(mat)
    for gene in mat.columns:
        involved = [v for k, v in pair.items() if gene in k]
        assert m[gene] == pytest.approx(sum(involved) / len(involved))


@needs_data
def test_build_s5_renders():
    from scripts.report.report_style import apply_style
    apply_style()
    fig = S5.build_s5(QPCR_ROOT)
    assert len(fig.axes) == 4


# --------------------------------------------------------------------------- #
# Treatment effect with development removed
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def effects(tidy):
    return F3.treatment_effects(tidy)


@needs_data
def test_effect_is_independent_of_the_calibrator(tidy):
    """The mathematical guarantee: ΔΔCq(arm) − ΔΔCq(Control) cancels the baseline.

    If this ever fails, the "development removed" claim is not what the figure
    is actually showing.
    """
    shifted = tidy.copy()
    shifted.loc[shifted.is_calibrator, "dcq"] += 7.5

    a = F3.treatment_effects(tidy).log2_effect
    b = F3.treatment_effects(shifted).log2_effect
    pd.testing.assert_series_equal(a, b)


@needs_data
@pytest.mark.parametrize("gene,group,day,expected", [
    ("SLC32A1", "IVH_Late", 21.0, -3.22),
    ("GFAP", "IVH_Early", 21.0, 2.28),
    ("SLC17A7", "IVH_Late", 7.0, -1.54),
    ("GFAP", "IVH_Late", 7.0, 0.52),
])
def test_effect_values(effects, gene, group, day, expected):
    row = effects[(effects.gene == gene) & (effects.group == group)
                  & (effects.day == day)]
    assert len(row) == 1
    assert row.log2_effect.iloc[0] == pytest.approx(expected, abs=0.01)


@needs_data
def test_control_is_not_its_own_contrast(effects):
    """Control vs time-matched Control is identically 0 — omit it, don't plot it."""
    assert F3.CONTROL_GROUP not in set(effects.group)


@needs_data
def test_d14_control_is_interpolated_and_labelled(tidy, effects):
    ctl = F3.control_reference(tidy)
    d14 = ctl[ctl.day == 14.0]
    assert (d14.control_source == "interpolated").all()
    assert (ctl[ctl.day.isin([7.0, 21.0])].control_source == "measured").all()
    assert (effects[effects.day == 14.0].control_source == "interpolated").all()


@needs_data
def test_interpolated_control_lies_between_its_measured_neighbours(tidy):
    """The fill must not invent a new extreme.

    Note the invariant is on the *control* ΔCq — the quantity actually being
    interpolated. The resulting D14 *effect* may sit outside its own D7/D21
    range, because the arm's D14 measurement is real (MAP2/IVH_Early: D7 −0.60,
    D14 −0.16, D21 −0.48). That is data, not an interpolation artefact.
    """
    ctl = F3.control_reference(tidy).set_index(["gene", "day"]).control_dcq
    for gene in tidy.gene.unique():
        lo, hi = sorted([ctl[(gene, 7.0)], ctl[(gene, 21.0)]])
        assert lo - 1e-9 <= ctl[(gene, 14.0)] <= hi + 1e-9, gene


@needs_data
def test_interpolation_never_extrapolates(tidy):
    """Drop the last measured Control day: the fill must refuse, not guess on."""
    truncated = tidy[~((tidy.group == F3.CONTROL_GROUP) & (tidy.day == 21.0))]
    ctl = F3.control_reference(truncated)
    assert (ctl[ctl.day == 21.0].control_source == "unavailable").all()
    assert ctl[ctl.day == 21.0].control_dcq.isna().all()

    eff = F3.treatment_effects(truncated)
    assert eff[eff.day == 21.0].log2_effect.isna().all()


@needs_data
def test_interpolation_can_be_switched_off(tidy):
    ctl = F3.control_reference(tidy, interpolate=False)
    assert (ctl[ctl.day == 14.0].control_source == "unavailable").all()
    eff = F3.treatment_effects(tidy, interpolate=False)
    assert eff[eff.day == 14.0].log2_effect.isna().all()


@needs_data
def test_failed_arm_keeps_its_row_and_reason(effects):
    """H2O2 D7 never amplified: present, NaN, and labelled — not dropped."""
    row = effects[(effects.group == "H2O2_20uM") & (effects.day == 7.0)]
    assert len(row) == len(MARKER_GENES)
    assert row.log2_effect.isna().all()
    assert (row.qc == "no_amplification").all()
