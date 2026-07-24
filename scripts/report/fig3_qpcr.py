"""F3 — qPCR marker expression (CX169, preliminary).

Two input modes:

**Directory mode** (``--qpcr-dir``, preferred). Points at the qPCR analysis root
that holds one sub-directory per plate, each with the AriaMx exports plus a
pre-computed ``*_ddcq.csv``::

    <root>/CX169 Plate1 GAPDH MAP2 GFAP/CX169_Plate1_..._ddcq.csv
    <root>/CX169 Plate2 GAPDH SLC17A7 SLC32A1/..._ddcq.csv
    <root>/CX169 all sample ref gene test/...            <- feeds S5, not F3

``*_ddcq.csv`` columns: ``Sample, Target, Mean Cq, Ref Cq, dCq, ddCq, ddCq SD,
ddCq SEM, Fold (2^-ddCq), Fold low, Fold high``. Reference gene is GAPDH,
calibrator is ``CX169 CT DIV 7`` (= tau 0, the treatment day, DIV 7). The
reference-gene-test plate is excluded automatically because its targets are the
candidate reference genes, not markers.

If a plate directory ships no ``_ddcq.csv``, the loader falls back to deriving
ΔΔCq from ``* - Tabular Results.txt`` (FAM rows, ``Cq (∆R)``) joined to the
hand-made ``*_labels.csv``. That path reproduces the shipped ddCq files exactly
and is locked by ``tests/test_qpcr_report.py``.

**Table mode** (``--qpcr-table``, legacy). A generic long/wide table; see
:func:`load_qpcr`.

Caveats baked into the figure: n = 1 chip, technical duplicates only, no
statistics, ΔΔCq assumes ~100 % amplification efficiency (no standard curves).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .report_style import caption, group_color, ordered_groups, report_dir, save_fig

_GENE_COLS = ["gene", "target", "marker"]
_GROUP_COLS = ["group", "condition", "arm", "treatment", "canonical_group"]
_VALUE_COLS = ["value", "fold_change", "fold", "rq", "ddct", "expression",
               "relative_expression", "rel_expr"]

#: Reference gene used on the marker plates (its Cq lands in the ``Ref Cq``
#: column of the ddCq exports, never as a ``Target`` row).
REFERENCE_GENE = "GAPDH"

#: Sample whose ΔCq anchors every ΔΔCq. tau 0 == treatment day == DIV 7.
CALIBRATOR = "CT DIV 7"

#: Source-file gene-name typos -> canonical symbol.
GENE_ALIASES = {"GABDH": "GAPDH"}

#: Candidate reference genes; any plate whose targets are a subset of this set
#: is a reference-gene screen and is skipped by F3 (it feeds S5 instead).
REFERENCE_CANDIDATES = {"ACTB", "GAPDH", "HPRT1", "B2M", "TBP", "RPLP0", "18S"}

#: Arms parsed from the plate but kept out of the figures. The oxidative-stress
#: arms are a separate line of enquiry from the CSF comparison the report makes,
#: and on a single chip with technical duplicates they add columns without adding
#: evidence. They stay in ``F3a_qpcr_tidy.csv`` / ``F3a_qpcr_effects.csv`` — the
#: record is complete; only the drawn figure is narrowed. Empty this set to bring
#: them back.
EXCLUDED_GROUPS = {"H2O2_10uM", "H2O2_20uM"}

#: Arm prefix (post-normalisation) -> canonical_group in ``GROUP_PALETTE``.
GROUP_ALIASES = {
    "CT": "Control",
    "IVH E": "IVH_Early",
    "IVH L": "IVH_Late",
    "20 UM H2O2": "H2O2_20uM",
    "10 UM H2O2": "H2O2_10uM",
    "NPH": "NPH",
    "ARAC": "AraC",
}

#: Hand-resolved label conflicts. Well G1 is the same physical sample on all
#: three plates but is written ``IVH E D 27`` on the reference-gene plate and
#: ``IVH E D 21`` on plates 1-2; the user confirmed D21 is correct.
SAMPLE_ALIASES = {"IVH E D 27": "IVH E D 21"}

#: Reference-Cq deviation (cycles, vs the plate median) above which a sample is
#: called a loading/cDNA failure rather than biology.
REF_CQ_OUTLIER_CYCLES = 5.0

#: The planned 13-target panel, split into the two priority tiers. Plates 1-2 of
#: 7 are run so far (MAP2/GFAP/SLC17A7/SLC32A1); the rest arrive as their export
#: directories are dropped under ``--qpcr-dir``. Figures order genes by tier then
#: by position here; anything unlisted sorts last, in first-seen order.
GENE_TIERS: dict[int, list[str]] = {
    1: ["MAP2", "GFAP", "SLC17A7", "SLC32A1", "PAX6", "STX1A"],
    2: ["OLIG2", "KCNMA1", "KCNN2", "HCN1", "SOX2", "PVALB", "SST"],
}
GENE_ORDER = [g for tier in sorted(GENE_TIERS) for g in GENE_TIERS[tier]]

#: Widely-used protein/common names, shown after the gene symbol. Several targets
#: are referred to informally (BK/SK/HCN channel), so map the symbol to the name
#: people actually say.
GENE_ALT_NAMES = {
    "SLC17A7": "VGLUT1",
    "SLC32A1": "VGAT",
    "STX1A": "syntaxin-1A",
    "KCNMA1": "BK channel",
    "KCNN2": "SK channel",
    "HCN1": "HCN channel",
    "PVALB": "parvalbumin",
    "SST": "somatostatin",
}

#: Cell-type family per gene, used to group heatmap rows so the story reads as a
#: pattern: astrocyte marker (GFAP, ↑) first, then the neuronal/synaptic markers
#: (↓). Genes not listed fall into "Other" and sort last.
GENE_CELLTYPE = {
    "GFAP": "Astrocyte", "OLIG2": "Oligodendrocyte",
    "MAP2": "Neuronal", "PAX6": "Neural progenitor", "SOX2": "Neural progenitor",
    "SLC17A7": "Excitatory synapse", "STX1A": "Synaptic",
    "SLC32A1": "Inhibitory synapse", "PVALB": "Interneuron", "SST": "Interneuron",
    "KCNMA1": "Ion channel", "KCNN2": "Ion channel", "HCN1": "Ion channel",
}
#: Row order of the cell-type families (astrocyte first → synaptic → channels).
CELLTYPE_ORDER = ["Astrocyte", "Oligodendrocyte", "Neuronal",
                  "Neural progenitor", "Excitatory synapse", "Inhibitory synapse",
                  "Synaptic", "Interneuron", "Ion channel", "Other"]


def celltype_ordered_genes(genes) -> list[str]:
    """Order genes by cell-type family (GENE_CELLTYPE / CELLTYPE_ORDER), keeping
    the within-family order from GENE_ORDER. Puts GFAP (astrocyte) on top."""
    present = list(dict.fromkeys(genes))
    def key(g):
        fam = GENE_CELLTYPE.get(g, "Other")
        return (CELLTYPE_ORDER.index(fam) if fam in CELLTYPE_ORDER else 99,
                GENE_ORDER.index(g) if g in GENE_ORDER else 99, g)
    return sorted(present, key=key)


# --------------------------------------------------------------------------- #
# Sample-name parsing
# --------------------------------------------------------------------------- #
def normalise_sample(name: str) -> str:
    """Strip the chip prefix and collapse spelling variants to one form.

    ``'CX169 20uM H2O2 D 14'`` and ``'CX169 20 uM H2O2 D 14'`` both become
    ``'20 UM H2O2 D 14'``; ``'CX169 IVH E D7'`` becomes ``'IVH E D 7'``.
    """
    s = str(name).strip().upper()
    s = re.sub(r"^CX\d+\s*", "", s)
    s = re.sub(r"(\d)\s*UM", r"\1 UM", s)          # 20UM -> 20 UM
    s = re.sub(r"\bD\s*(\d+)", r"D \1", s)         # D7 / D  7 -> D 7
    s = re.sub(r"\bDIV\s*(\d+)", r"DIV \1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return SAMPLE_ALIASES.get(s, s)


def parse_sample(name: str) -> tuple[str | None, float | None, bool]:
    """``sample name -> (canonical_group, day_post_treatment, is_calibrator)``.

    Returns ``(None, None, False)`` for NTC / unparseable labels so callers can
    drop them without special-casing.
    """
    s = normalise_sample(name)
    if not s or s == "NTC":
        return None, None, False

    # Calibrator: "CT DIV 7" -> Control at day 0 (treatment day).
    m = re.match(r"^(?P<arm>.+?)\s+DIV\s+(?P<div>\d+)$", s)
    if m:
        group = GROUP_ALIASES.get(m.group("arm").strip())
        return group, 0.0, True

    m = re.match(r"^(?P<arm>.+?)\s+D\s+(?P<day>\d+)$", s)
    if not m:
        return None, None, False
    group = GROUP_ALIASES.get(m.group("arm").strip())
    return group, float(m.group("day")), False


# --------------------------------------------------------------------------- #
# Directory loader (primary)
# --------------------------------------------------------------------------- #
def _read_ddcq(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "sample": df["Sample"].astype(str),
        "gene": df["Target"].astype(str).str.strip().map(
            lambda g: GENE_ALIASES.get(g.upper(), g.upper())),
        "mean_cq": pd.to_numeric(df["Mean Cq"], errors="coerce"),
        "ref_cq": pd.to_numeric(df["Ref Cq"], errors="coerce"),
        "dcq": pd.to_numeric(df["dCq"], errors="coerce"),
        "ddcq": pd.to_numeric(df["ddCq"], errors="coerce"),
        "ddcq_sem": pd.to_numeric(df.get("ddCq SEM"), errors="coerce"),
        "fold": pd.to_numeric(df["Fold (2^-ddCq)"], errors="coerce"),
        "fold_lo": pd.to_numeric(df.get("Fold low"), errors="coerce"),
        "fold_hi": pd.to_numeric(df.get("Fold high"), errors="coerce"),
    })
    return out


def read_plate_cq(plate_dir: Path) -> pd.DataFrame:
    """Per-well Cq for one plate: ``Well, Sample, Gene, Cq`` (FAM channel).

    Joins the AriaMx tabular export to the hand-made ``*_labels.csv``. Some
    directories ship two ``Tabular Results`` exports; only one carries the
    ``Cq (∆R)`` column (the other is a melt-curve dump), so pick by column.
    """
    plate_dir = Path(plate_dir)
    labels = sorted(plate_dir.glob("*_labels.csv"))
    if not labels:
        raise FileNotFoundError(f"no *_labels.csv in {plate_dir}")
    lab = pd.read_csv(labels[0])
    lab["Well"] = lab["Well"].astype(str).str.strip()

    for tab in sorted(plate_dir.glob("* - Tabular Results*.txt")):
        raw = pd.read_csv(tab, sep="\t", engine="python")
        raw.columns = [c.strip() for c in raw.columns]
        cq_col = next((c for c in raw.columns if c.startswith("Cq (")), None)
        if cq_col is None:
            continue
        raw["Well"] = raw["Well"].astype(str).str.strip()
        raw = raw[raw["Dye"].astype(str).str.strip() == "FAM"]
        cq = raw.assign(Cq=pd.to_numeric(raw[cq_col], errors="coerce"))[["Well", "Cq"]]
        out = lab.merge(cq, on="Well", how="left")
        out["Gene"] = out["Gene"].astype(str).str.strip().str.upper().map(
            lambda g: GENE_ALIASES.get(g, g))
        return out
    raise FileNotFoundError(f"no tabular export with a Cq column in {plate_dir}")


def derive_ddcq(plate_dir: Path, reference: str = REFERENCE_GENE,
                calibrator: str = CALIBRATOR) -> pd.DataFrame:
    """Recompute ΔΔCq from raw wells — fallback when no ``_ddcq.csv`` exists.

    Technical duplicates are averaged per (sample, gene); ΔCq = target − the
    same sample's reference; ΔΔCq = ΔCq − the calibrator's ΔCq. Reproduces the
    shipped ``_ddcq.csv`` values exactly (see ``tests/test_qpcr_report.py``),
    including its error convention: the quoted SEM is the target's own
    technical-duplicate SEM (``SD/sqrt(n)``), *not* a propagation of target and
    reference SDs. Keeping the two paths on one convention matters — otherwise
    the F3 error bars would silently change meaning for a plate that happens to
    ship without a ddCq export.
    """
    wells = read_plate_cq(plate_dir).dropna(subset=["Sample", "Gene"])
    wells = wells[wells["Sample"].astype(str).str.upper() != "NTC"]
    agg = (wells.groupby(["Sample", "Gene"]).Cq
           .agg(mean_cq="mean", sd="std", n="count").reset_index())

    ref = agg[agg.Gene == reference].set_index("Sample")
    tgt = agg[agg.Gene != reference].copy()
    tgt["ref_cq"] = tgt.Sample.map(ref["mean_cq"])
    tgt["ref_sd"] = tgt.Sample.map(ref["sd"])
    tgt["dcq"] = tgt.mean_cq - tgt.ref_cq

    cal = {g: d.dcq.iloc[0] for g, d in
           tgt[tgt.Sample.map(normalise_sample) == calibrator].groupby("Gene")}
    tgt["ddcq"] = tgt.dcq - tgt.Gene.map(cal)
    tgt["ddcq_sem"] = tgt.sd / np.sqrt(tgt.n.clip(lower=1))
    tgt["fold"] = 2.0 ** (-tgt.ddcq)
    tgt["fold_lo"] = 2.0 ** (-(tgt.ddcq + tgt.ddcq_sem))
    tgt["fold_hi"] = 2.0 ** (-(tgt.ddcq - tgt.ddcq_sem))
    return tgt.rename(columns={"Sample": "sample", "Gene": "gene"})[
        ["sample", "gene", "mean_cq", "ref_cq", "dcq", "ddcq", "ddcq_sem",
         "fold", "fold_lo", "fold_hi"]]


def duplicate_sd(plate_dir: Path,
                 reference: str = REFERENCE_GENE) -> pd.DataFrame:
    """Technical-duplicate SD/n per (sample, gene), target *and* reference.

    The shipped ``_ddcq.csv`` quotes only the target's SD, so a properly
    propagated ΔCq error — ``hypot(target_sd/sqrt(n), ref_sd/sqrt(n))`` — cannot
    be recovered from it. That propagated term is what the effects CSV reports
    and what the future multi-chip path needs, so pull it from the raw wells.

    Returns ``sample, gene, target_sd, n_target, ref_sd, n_ref``.
    """
    wells = read_plate_cq(plate_dir).dropna(subset=["Sample", "Gene"])
    wells = wells[wells["Sample"].astype(str).str.upper() != "NTC"]
    agg = (wells.groupby(["Sample", "Gene"]).Cq
           .agg(sd="std", n="count").reset_index())
    ref = agg[agg.Gene == reference].set_index("Sample")
    tgt = agg[agg.Gene != reference].copy()
    tgt["ref_sd"] = tgt.Sample.map(ref["sd"])
    tgt["n_ref"] = tgt.Sample.map(ref["n"])
    return tgt.rename(columns={"Sample": "sample", "Gene": "gene",
                               "sd": "target_sd", "n": "n_target"})[
        ["sample", "gene", "target_sd", "n_target", "ref_sd", "n_ref"]]


def qc_flags(df: pd.DataFrame,
             tol: float = REF_CQ_OUTLIER_CYCLES) -> pd.Series:
    """Per-row QC verdict: ``ok`` | ``no_amplification`` | ``ref_cq_outlier``.

    A sample whose reference Cq sits > ``tol`` cycles from its plate median has
    failed on input (cDNA/RNA loading), not biology — every gene on that sample
    shifts together, so the ΔΔCq is not trustworthy even though it normalises.
    """
    flag = pd.Series("ok", index=df.index, dtype=object)
    for _, idx in df.groupby("plate").groups.items():
        ref = df.loc[idx, "ref_cq"]
        med = ref.median()
        flag.loc[idx[(ref - med).abs() > tol]] = "ref_cq_outlier"
    flag[df.mean_cq.isna() | df.ref_cq.isna() | df.ddcq.isna()] = "no_amplification"
    return flag


def _labelled_samples(plate_dir: Path) -> list[str]:
    """Sample names on a plate's ``_labels.csv``, NTC and blanks removed."""
    labels = sorted(Path(plate_dir).glob("*_labels.csv"))
    if not labels:
        return []
    s = pd.read_csv(labels[0])["Sample"].dropna().astype(str)
    return [x for x in dict.fromkeys(s) if x.strip().upper() != "NTC"]


def _add_missing_rows(tab: pd.DataFrame, plate_dir: Path) -> pd.DataFrame:
    """Re-insert samples that were plated but never made it into the ddCq table.

    A sample that failed to amplify (CX169 20uM H2O2 D 7: no Cq on any target)
    is simply absent upstream. Adding it back as an all-NaN row keeps the gap
    visible on the figure instead of silently dropping a condition.
    """
    plated = {normalise_sample(s): s for s in _labelled_samples(plate_dir)}
    present = {normalise_sample(s) for s in tab["sample"]}
    missing = [orig for key, orig in plated.items() if key not in present]
    if not missing:
        return tab
    rows = [{"sample": s, "gene": g} for s in missing for g in tab.gene.unique()]
    return pd.concat([tab, pd.DataFrame(rows)], ignore_index=True)


def load_qpcr_dir(root, reference: str = REFERENCE_GENE) -> pd.DataFrame:
    """Tidy marker-gene ΔΔCq table pooled over every plate directory in ``root``.

    Columns: ``plate, sample, sample_id, group, day, gene, mean_cq, ref_cq, dcq,
    ddcq, ddcq_sem, target_sd, n_target, ref_sd, n_ref, sem_dcq, fold, fold_lo,
    fold_hi, log2_fold, is_calibrator, qc``.
    Reference-gene screens (targets ⊆ :data:`REFERENCE_CANDIDATES`) are skipped.

    **ΔΔCq is calibrated per chip** — each ``sample_id`` against its own
    ``CT DIV 7`` — so several chips can be loaded together. Nothing here averages
    them; see :func:`aggregate_chips`. (The treatment-effect measure in
    :func:`treatment_effects` needs no calibrator at all, so it is chip-internal
    by construction.)
    """
    root = Path(root)
    plate_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not plate_dirs:                       # a single plate dir was passed
        plate_dirs = [root]

    frames = []
    for pdir in plate_dirs:
        ddcq = sorted(pdir.glob("*_ddcq.csv"))
        try:
            tab = _read_ddcq(ddcq[0]) if ddcq else derive_ddcq(pdir, reference)
        except (FileNotFoundError, KeyError):
            continue
        if tab.empty:
            continue
        if set(tab.gene.unique()) <= REFERENCE_CANDIDATES:
            continue                          # reference-gene screen -> S5
        tab = _add_missing_rows(tab, pdir)
        try:
            sds = duplicate_sd(pdir, reference)
            tab = tab.merge(sds, on=["sample", "gene"], how="left")
        except (FileNotFoundError, KeyError):
            pass                              # no raw export; SEM stays absent
        tab.insert(0, "plate", pdir.name)
        frames.append(tab)

    if not frames:
        raise ValueError(f"no marker-gene qPCR plates found under {root}")

    df = pd.concat(frames, ignore_index=True)
    parsed = df["sample"].map(parse_sample)
    df["sample_id"] = df["sample"].str.extract(r"(CX\d+)", expand=False)
    df["group"] = [p[0] for p in parsed]
    df["day"] = [p[1] for p in parsed]
    df["is_calibrator"] = [p[2] for p in parsed]
    df = df[df.group.notna()].copy()
    for col in ("target_sd", "n_target", "ref_sd", "n_ref"):
        if col not in df:
            df[col] = np.nan
    df["sem_dcq"] = np.hypot(df.target_sd / np.sqrt(df.n_target.clip(lower=1)),
                             df.ref_sd / np.sqrt(df.n_ref.clip(lower=1)))
    df = _calibrate_per_chip(df)
    df["log2_fold"] = -df["ddcq"] + 0.0       # log2(2^-ddCq) == -ddCq
    df["qc"] = qc_flags(df)
    return df.sort_values(["sample_id", "gene", "group", "day"]).reset_index(drop=True)


def _calibrate_per_chip(df: pd.DataFrame) -> pd.DataFrame:
    """Re-anchor ΔΔCq to each chip's own calibrator sample.

    The shipped ``_ddcq.csv`` is already single-chip calibrated, so for one chip
    this is a no-op that reproduces the export. With several chips it stops
    CX170's ΔΔCq being expressed against CX169's baseline, which is what makes
    the chips comparable at all.
    """
    if df.sample_id.nunique(dropna=True) < 2:
        return df
    out = []
    for _, chunk in df.groupby("sample_id", dropna=False):
        cal = (chunk[chunk.is_calibrator].dropna(subset=["dcq"])
               .set_index("gene").dcq)
        chunk = chunk.copy()
        chunk["ddcq"] = chunk.dcq - chunk.gene.map(cal)
        chunk["fold"] = 2.0 ** (-chunk.ddcq)
        out.append(chunk)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Treatment effect with development removed (time-matched-control contrast)
# --------------------------------------------------------------------------- #
CONTROL_GROUP = "Control"


def drawable(tidy: pd.DataFrame) -> pd.DataFrame:
    """``tidy`` without the arms held out of the figures (:data:`EXCLUDED_GROUPS`).

    Applied by the figure builders rather than by the loader, so the exported
    tables and any interactive use still see every arm that was on the plate.
    """
    if not EXCLUDED_GROUPS or "group" not in tidy.columns:
        return tidy
    return tidy[~tidy.group.isin(EXCLUDED_GROUPS)].copy()


def control_reference(tidy: pd.DataFrame,
                      interpolate: bool = True) -> pd.DataFrame:
    """Per ``(sample_id, gene, day)`` Control ΔCq, with failed days filled in.

    Returns ``sample_id, gene, day, control_dcq, control_sem, control_source``
    where ``control_source`` is ``measured`` | ``interpolated`` | ``unavailable``.

    ``CT D14`` on this dataset is a cDNA-loading failure, so D14 has no measured
    control. Control ΔCq is nearly flat across D7→D21 (SLC17A7 6.600 → 6.570),
    which is what makes filling it defensible — but the fill is **never
    extrapolated** past the last good day, since a trend that has not been
    observed cannot be assumed. Days outside the measured range stay
    ``unavailable`` and their effects come out NaN.
    """
    days = sorted(tidy.day.dropna().unique())
    rows = []
    for (chip, gene), sub in tidy.groupby(["sample_id", "gene"], dropna=False):
        ctl = sub[sub.group == CONTROL_GROUP]
        good = ctl[(ctl.qc == "ok") & ctl.dcq.notna()].sort_values("day")
        measured = dict(zip(good.day, good.dcq))
        sem = dict(zip(good.day, good.sem_dcq))
        for day in days:
            if day in measured:
                rows.append((chip, gene, day, measured[day],
                             sem.get(day, np.nan), "measured"))
            elif (interpolate and len(good) >= 2
                  and good.day.min() < day < good.day.max()):
                rows.append((chip, gene, day,
                             float(np.interp(day, good.day, good.dcq)),
                             float(np.nanmax(good.sem_dcq)) if good.sem_dcq.notna().any()
                             else np.nan,
                             "interpolated"))
            else:
                rows.append((chip, gene, day, np.nan, np.nan, "unavailable"))
    return pd.DataFrame(rows, columns=["sample_id", "gene", "day", "control_dcq",
                                       "control_sem", "control_source"])


def treatment_effects(tidy: pd.DataFrame,
                      interpolate: bool = True) -> pd.DataFrame:
    """Treatment effect with developmental change removed.

    Every arm was split from one culture at the treatment day, so they share a
    single baseline sample and there is no per-arm day 0. The
    difference-in-differences therefore collapses to a **time-matched-control
    contrast**, and the ``CT DIV 7`` calibrator cancels exactly::

        effect(arm, d) = ΔΔCq(arm, d) − ΔΔCq(Control, d)
                       = ΔCq(arm, d)  − ΔCq(Control, d)
        log2 fold      = −effect

    Same estimand as the MEA difference-in-differences in ``stats.py`` (each
    treated arm against Control's own change over the same interval), so the
    qPCR and MEA figures rest on one logic.

    Returns ``sample_id, gene, group, day, log2_effect, sem_effect, fold_effect,
    control_source, qc``. Control itself is omitted — it is identically 0 by
    construction. Rows whose *arm* failed QC keep the row and the reason with a
    NaN effect, so a failed condition stays visible rather than vanishing.
    """
    ctl = control_reference(tidy, interpolate=interpolate)
    arms = tidy[tidy.group != CONTROL_GROUP].copy()
    m = arms.merge(ctl, on=["sample_id", "gene", "day"], how="left")

    m["log2_effect"] = -(m.dcq - m.control_dcq)
    m["sem_effect"] = np.hypot(m.sem_dcq, m.control_sem)
    m.loc[m.qc != "ok", "log2_effect"] = np.nan
    m.loc[m.control_source == "unavailable", "log2_effect"] = np.nan
    m["fold_effect"] = 2.0 ** m.log2_effect
    return m[["sample_id", "gene", "group", "day", "log2_effect", "sem_effect",
              "fold_effect", "control_source", "qc"]].sort_values(
        ["sample_id", "gene", "group", "day"]).reset_index(drop=True)


def aggregate_chips(effects: pd.DataFrame) -> pd.DataFrame:
    """Collapse :func:`treatment_effects` over chips: ``mean, sem, n_chips``.

    **Provisional.** With one chip this returns the single value and a NaN SEM,
    which is the only behaviour exercised by real data so far; the multi-chip
    branch is covered by a synthetic test only. When real replicate chips exist,
    revisit whether a plain across-chip SEM is the right summary and whether the
    ``stats.py`` Wilcoxon/BH-FDR machinery should be wired in — do not read an
    n = 2 SEM as an inferential result.
    """
    g = effects.groupby(["gene", "group", "day"], dropna=False)
    out = g.agg(log2_effect=("log2_effect", "mean"),
                sd=("log2_effect", "std"),
                n_chips=("log2_effect", "count"),
                control_source=("control_source", "first"),
                qc=("qc", "first")).reset_index()
    # named sem_effect, not sem — `df.sem` is a DataFrame method
    out["sem_effect"] = out.sd / np.sqrt(out.n_chips.where(out.n_chips > 1))
    return out.drop(columns="sd")


# --------------------------------------------------------------------------- #
# Generic table loader (legacy ``--qpcr-table``)
# --------------------------------------------------------------------------- #
def _first_match(cols, candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def load_qpcr(path) -> pd.DataFrame:
    """Return a tidy ``gene, group, value`` frame from a CSV/xlsx table."""
    path = str(path)
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) \
        else pd.read_csv(path)
    gene = _first_match(df.columns, _GENE_COLS)
    group = _first_match(df.columns, _GROUP_COLS)
    value = _first_match(df.columns, _VALUE_COLS)
    if group is None:
        raise ValueError(f"qPCR table needs a group column ({_GROUP_COLS}); "
                         f"got {list(df.columns)}")
    if gene is not None and value is not None:
        out = df[[gene, group, value]].copy()
        out.columns = ["gene", "group", "value"]
    else:
        # wide: genes are the numeric columns other than the group column
        num = [c for c in df.columns if c != group
               and pd.api.types.is_numeric_dtype(df[c])]
        out = df.melt(id_vars=[group], value_vars=num,
                      var_name="gene", value_name="value")
        out = out.rename(columns={group: "group"})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
_FOOTNOTE = (
    "PRELIMINARY — n = 1 chip (CX169), technical duplicates only, no statistics. "
    "ΔΔCq vs GAPDH, calibrated to CT DIV 7 (treatment day); assumes ~100 % "
    "amplification efficiency (no standard curves run). 4 of 13 planned target genes."
)


def ordered_genes(present) -> list[str]:
    """Sort gene symbols by :data:`GENE_TIERS`; unknowns last, first-seen order."""
    seen = list(dict.fromkeys(present))
    known = [g for g in GENE_ORDER if g in seen]
    return known + [g for g in seen if g not in known]


def gene_label(gene: str) -> str:
    """``'SLC17A7'`` -> ``'SLC17A7 (VGLUT1)'``; unknown symbols pass through."""
    alt = GENE_ALT_NAMES.get(gene)
    return f"{gene} ({alt})" if alt else gene


def gene_tier(gene: str) -> int:
    """Priority tier of a gene; unlisted genes sort after every known tier."""
    for tier, genes in GENE_TIERS.items():
        if gene in genes:
            return tier
    return max(GENE_TIERS) + 1


# --------------------------------------------------------------------------- #
# F3 — effect heatmap
# --------------------------------------------------------------------------- #
#: Cell fill for a condition that failed QC. Deliberately outside the diverging
#: colormap so it cannot be read as a large positive or negative value.
_FAIL_COLOR = "0.85"


def fold_text(log2_value: float) -> str:
    """Format a log2 fold-change as a fold-change label: ``-3.38 -> '0.096x'``.

    Two significant figures, but never scientific notation — ``2^6.89`` has to
    read as ``119×``, not ``1.2e+02×``.
    """
    if pd.isna(log2_value):
        return "n.a."
    fold = 2.0 ** log2_value
    return f"{fold:,.0f}×" if fold >= 10 else f"{fold:.2g}×"


def effect_matrix(tidy: pd.DataFrame, genes: list[str] | None = None):
    """Build the F3 heatmap arrays.

    Returns ``(genes, columns, value, state, text)`` where ``columns`` is a list
    of ``(block, label, group_or_None, day)`` and the three arrays are
    ``len(genes) x len(columns)``. ``state`` is ``ok`` | ``ref_cq_outlier`` |
    ``no_amplification`` | ``missing``; only ``ok`` cells are colour-mapped.

    ``value`` stays in **log2** because that is what the colour scale needs:
    fold change is asymmetric (2x up = 2.0, 2x down = 0.5), so mapping colour
    linearly in fold would compress every downregulated cell against zero while
    a single large increase claimed the whole warm half. ``text`` carries the
    fold-change labels the figure actually shows.
    """
    genes = genes or ordered_genes(tidy.gene)
    days = [d for d in sorted(tidy.day.dropna().unique()) if d > 0]
    arms = [g for g in ordered_groups(tidy.group.dropna().unique())
            if g != CONTROL_GROUP]

    eff = aggregate_chips(treatment_effects(tidy))
    # Averaged over chips exactly like the effect block, so a second chip widens
    # the evidence rather than duplicating (gene, day) keys and breaking lookup.
    dev = (tidy[(tidy.group == CONTROL_GROUP) & (tidy.day > 0)]
           .groupby(["gene", "day"], as_index=False)
           .agg(log2_fold=("log2_fold", "mean"), qc=("qc", "first")))

    columns = [("dev", f"D{int(d)}", None, d) for d in days]
    columns += [("eff", f"D{int(d)}", arm, d) for arm in arms for d in days]

    value = np.full((len(genes), len(columns)), np.nan)
    state = np.full((len(genes), len(columns)), "missing", dtype=object)
    text = np.full((len(genes), len(columns)), "", dtype=object)

    dev_i = dev.set_index(["gene", "day"])
    eff_i = eff.set_index(["gene", "group", "day"])
    for r, gene in enumerate(genes):
        for c, (block, _lbl, arm, day) in enumerate(columns):
            key = (gene, day) if block == "dev" else (gene, arm, day)
            src = dev_i if block == "dev" else eff_i
            if key not in src.index:
                text[r, c] = "–"
                continue
            row = src.loc[key]
            val = row.log2_fold if block == "dev" else row.log2_effect
            state[r, c] = row.qc
            if row.qc == "ok" and pd.notna(val):
                value[r, c] = val
                text[r, c] = fold_text(val)
            elif row.qc == "no_amplification":
                text[r, c] = "n.a."
            else:                       # ref_cq_outlier: show what it was, greyed
                text[r, c] = "n.a." if pd.isna(val) else f"({fold_text(val)})"
    return genes, columns, value, state, text


def build_f3(tidy: pd.DataFrame, genes: list[str] | None = None):
    """F3 — treatment effect with developmental change removed.

    A gene x condition heatmap in two blocks sharing one diverging colourbar:
    the developmental change the Control arm undergoes on its own (left), and
    each treated arm against the **time-matched** Control (right), which is the
    difference-in-differences that removes exactly that development.

    Scales to the full 13-gene panel: rows grow, layout does not change.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch, Rectangle

    from .report_style import MUTED, QC_NA, fc_div_cmap

    tidy = drawable(tidy)
    # Story order: group rows by cell type (GFAP astrocyte first, then the
    # neuronal / synaptic markers) so the pattern reads as a pattern.
    if genes is None:
        genes = celltype_ordered_genes(tidy.gene.dropna().unique())
    genes, columns, value, state, text = effect_matrix(tidy, genes)
    n_row, n_col = value.shape

    # Log-symmetric diverging fill, clipped at ±16× (log2 = ±4) so the QC-failed
    # loading artefacts (MAP2 +6.89 = 119×) can never set the scale.
    vmax = 4.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = fc_div_cmap()

    fig, ax = plt.subplots(figsize=(1.35 + 0.62 * n_col, 2.1 + 0.34 * n_row))
    ax.imshow(np.ma.masked_invalid(np.clip(value, -vmax, vmax)), cmap=cmap,
              norm=norm, aspect="auto", zorder=1)

    for r in range(n_row):
        for c in range(n_col):
            if state[r, c] != "ok":
                # QC-fail / n.a. → diagonal hatch (never a grey that reads as low)
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=QC_NA,
                                       edgecolor="#9A917C", hatch="////", lw=0.0,
                                       zorder=2))
            if not text[r, c]:
                continue
            ok = state[r, c] == "ok"
            shade = min(abs(value[r, c]), vmax) / vmax if ok else 0.0
            ax.text(c, r, text[r, c], ha="center", va="center", fontsize=5.8,
                    color="white" if shade > 0.62 else ("0.12" if ok else "0.40"),
                    zorder=4)

    # --- column labels + block headers -------------------------------------
    interp = _interpolated_days(tidy)
    labels = [f"{lbl}†" if (block == "eff" and day in interp) else lbl
              for block, lbl, _arm, day in columns]
    ax.set_xticks(range(n_col))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_yticks(range(n_row))
    ax.set_yticklabels([gene_label(g) for g in genes], fontsize=7)
    ax.set_xlim(-0.5, n_col - 0.5)
    ax.set_ylim(n_row - 0.5, -0.5)

    n_dev = sum(1 for b, *_ in columns if b == "dev")
    # The maturation block is the baseline being SUBTRACTED OUT, not a result:
    # set it apart with a wide gap line + a translucent overlay that mutes it, and
    # a grey "removed" header (value labels stay legible on top).
    ax.axvline(n_dev - 0.5, color="black", lw=2.2)
    ax.add_patch(Rectangle((-0.5, -0.5), n_dev, n_row, facecolor="white",
                           alpha=0.34, edgecolor="none", zorder=2.5))
    _block_header(ax, 0, n_dev, "development — removed (context)", MUTED)
    c0 = n_dev
    for arm in [a for a in dict.fromkeys(col[2] for col in columns) if a]:
        width = sum(1 for col in columns if col[2] == arm)
        _block_header(ax, c0, c0 + width, arm, group_color(arm))
        if c0 > n_dev:
            ax.axvline(c0 - 0.5, color="white", lw=1.4)
        c0 += width

    # rule between cell-type families (rows are grouped GFAP↑ then neuronal↓)
    fams = [GENE_CELLTYPE.get(g, "Other") for g in genes]
    for r in range(1, n_row):
        if fams[r] != fams[r - 1]:
            ax.axhline(r - 0.5, color="black", lw=1.0)

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.022, pad=0.015, extend="both")
    # Ticks are placed on the (log-spaced) colour axis but labelled in fold
    # change, so a 4x increase and a 4x decrease sit equally far from 1x.
    ticks = [t for t in range(-4, 5)]
    cb.set_ticks(ticks)
    cb.set_ticklabels([fold_text(t) for t in ticks])
    cb.set_label("fold-change (clip ±16×)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)

    # QC-fail / n.a. hatch key (below the heatmap, clear of the block headers)
    ax.legend(handles=[Patch(facecolor=QC_NA, edgecolor="#9A917C", hatch="////",
                             label="QC-fail / n.a. (excluded)")],
              loc="upper right", bbox_to_anchor=(1.0, -0.06), fontsize=6,
              frameon=False)

    fig.suptitle("Figure 3 — qPCR: treatment effect with development removed",
                 fontsize=9, x=0.02, ha="left", fontweight="bold")
    fig.text(0.985, 0.965, "PRELIMINARY · n=1 chip · technical duplicates · "
             "no statistics", ha="right", va="top", fontsize=6.5,
             color="#A83B00", fontweight="bold")
    # Story callout printed on the figure. Names the arms actually drawn — it
    # claimed consistency with H2O2 while those columns were on the figure, and a
    # callout that outlives the columns it describes is a wrong claim, not a
    # stale one.
    drawn = list(dict.fromkeys(c[2] for c in columns if c[0] == "eff" and c[2]))
    shown = " and ".join(a.replace("_", " ") for a in drawn) or "the treated arms"
    fig.text(0.5, 0.028, "GFAP ↑ (astrogliosis) while MAP2 / VGLUT1 / VGAT ↓ "
             f"(neuronal & synaptic loss) — consistent across {shown}.",
             ha="center", va="bottom", fontsize=7.5, color="#5A5140",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FBF3E0",
                       edgecolor="#C79A2E", lw=0.8))
    fig.tight_layout(rect=(0, 0.075, 1, 0.92))
    caption(fig, _heatmap_footnote(tidy, interp).replace("\n", " "))
    return fig


def _block_header(ax, c0, c1, label, colour):
    """Colour bar + label spanning columns ``[c0, c1)`` above the heatmap."""
    mid = (c0 + c1 - 1) / 2
    ax.plot([c0 - 0.42, c1 - 0.58], [-0.72, -0.72], lw=3.0, color=colour,
            solid_capstyle="butt", clip_on=False, zorder=4)
    ax.annotate(label, xy=(mid, -0.85), xycoords=("data", "data"), ha="center",
                va="bottom", fontsize=6.5, color=colour, fontweight="bold",
                annotation_clip=False)


def _interpolated_days(tidy: pd.DataFrame) -> set[float]:
    """Days whose time-matched control had to be interpolated."""
    ctl = control_reference(tidy)
    return set(ctl.loc[ctl.control_source == "interpolated", "day"])


def _heatmap_footnote(tidy: pd.DataFrame, interp: set[float]) -> str:
    n_chips = tidy.sample_id.nunique(dropna=True)
    chips = ", ".join(sorted(tidy.sample_id.dropna().unique()))
    n_genes = tidy.gene.nunique()
    parts = [
        f"Gene × condition heatmap; each cell = treatment effect as fold-change "
        f"(2^-ΔΔCq) vs the time-matched Control (>1 up, <1 down). "
        f"PRELIMINARY — n = {n_chips} chip ({chips}), technical duplicates only, "
        "no statistics. Left block: Control's own change since the treatment day "
        "(the maturation being removed). Right blocks: each arm minus the "
        "time-matched Control, so the CT DIV 7 calibrator cancels exactly.",
        f"Values are fold-change (2^-ΔΔCq) vs GAPDH; assumes ~100 % amplification "
        f"efficiency (no standard curves run). {n_genes} of {len(GENE_ORDER)} "
        "planned target genes. Colour is spaced logarithmically about 1× so that "
        "an n-fold increase and an n-fold decrease are equally far from no change.",
        "Hatched = failed QC (reference-Cq shift, value in brackets) or no "
        "amplification (n.a.); excluded from the colour scale.",
    ]
    if interp:
        days = ", ".join(f"D{int(d)}" for d in sorted(interp))
        parts.append(f"† {days}: no measured control that day (CT D14 failed) — "
                     "Control ΔCq linearly interpolated from the flanking days.")
    return "\n".join(parts)


def build_f3_bars(qpcr: pd.DataFrame):
    """Legacy grouped-bar form, used by the generic ``--qpcr-table`` path."""
    import matplotlib.pyplot as plt

    genes = list(dict.fromkeys(qpcr.gene))
    groups = ordered_groups(qpcr.group.unique())
    n_g = len(groups)
    x = np.arange(len(genes))
    width = 0.8 / max(n_g, 1)

    fig, ax = plt.subplots(figsize=(max(5.0, 1.1 * len(genes) + 2), 3.4))
    for gi, grp in enumerate(groups):
        means, sems = [], []
        for gene in genes:
            v = qpcr[(qpcr.gene == gene) & (qpcr.group == grp)].value.to_numpy(float)
            means.append(np.nanmean(v) if len(v) else np.nan)
            sems.append(np.nanstd(v, ddof=1) / np.sqrt(len(v))
                        if len(v) > 1 else 0.0)
        ax.bar(x + gi * width - 0.4 + width / 2, means, width, yerr=sems,
               capsize=2, color=group_color(grp), alpha=0.85, label=grp,
               error_kw=dict(lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=30, ha="right")
    ax.set_ylabel("relative expression (mean ± SEM)")
    ax.axhline(1.0, ls="--", color="0.6", lw=0.7)
    ax.legend(fontsize=6, ncol=2, title="group", title_fontsize=6)
    ax.set_title("Figure 3 — qPCR marker expression", loc="left",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def synthetic_table(seed: int = 0) -> pd.DataFrame:
    """Small synthetic qPCR table for smoke-testing the builder."""
    rng = np.random.default_rng(seed)
    genes = ["GFAP", "IBA1", "MAP2", "SOX2", "TNF"]
    groups = ["Control", "IVH_Early", "IVH_Late", "H2O2_20uM"]
    rows = []
    for gene in genes:
        for grp in groups:
            base = 1.0 if grp == "Control" else rng.uniform(0.5, 3.0)
            for _ in range(3):
                rows.append(dict(gene=gene, group=grp,
                                 value=max(0.05, base * rng.normal(1, 0.15))))
    return pd.DataFrame(rows)


def render(figure_root, table_path=None, qpcr_dir=None, genes=None):
    """Directory input wins; then a generic table; else a synthetic placeholder.

    Alongside the figure, writes both tables: the per-condition ΔΔCq
    (``F3_qpcr_tidy.csv``) and the maturation-controlled effects
    (``F3_qpcr_effects.csv``). The CSVs stay complete for the whole gene panel
    regardless of what the figure is asked to draw.
    """
    if qpcr_dir:
        tidy = load_qpcr_dir(qpcr_dir)
        fig = build_f3(tidy, genes)
        paths = save_fig(fig, "F3_qpcr", figure_root, subdir="main")
        out = report_dir(figure_root, "main")
        tidy.to_csv(out / "F3_qpcr_tidy.csv", index=False)
        treatment_effects(tidy).to_csv(out / "F3_qpcr_effects.csv", index=False)
        return paths + [out / "F3_qpcr_tidy.csv", out / "F3_qpcr_effects.csv"]

    qpcr = load_qpcr(table_path) if table_path else synthetic_table()
    fig = build_f3_bars(qpcr)
    name = "F3_qpcr" if table_path else "F3_qpcr_SYNTHETIC_placeholder"
    return save_fig(fig, name, figure_root, subdir="main")
