"""S5 — reference-gene selection screen (CX169).

Supplement to F3. Before the marker plates were run, all 13 CX169 conditions
were assayed for three candidate reference genes (ACTB, GAPDH, HPRT1) on one
plate — ``<qpcr_root>/CX169 all sample ref gene test/``. This figure documents
that screen and the resulting choice of **GAPDH**.

Panels
------
A  Per-condition Cq spread of each candidate, centred on that gene's own mean
   (the candidates sit ~8 cycles apart, which would otherwise hide the spread);
   the absolute mean Cq is printed under each x label.
B  Stability: raw Cq SD and **geNorm M**. The two metrics disagree here, so both
   are shown; GAPDH has the lowest M and is the reference used by F3.
C  Pairwise ΔCq SD — the quantity geNorm M averages.
D  QC transparency: reference Cq per condition on every plate, plus NTC calls.
   Two conditions are excluded from the stability maths in A-C:
   ``CT D 14`` (all three candidates shift ~+10 Cq on both marker plates too →
   cDNA/RNA loading failure, not biology) and ``20 uM H2O2 D 7`` (no
   amplification anywhere).

geNorm M (Vandesompele et al. 2002) for gene *i* is the mean over all *j ≠ i* of
``SD_samples(Cq_i − Cq_j)``; lower is more stable, M < 0.5 is the conventional
threshold for homogeneous sample panels.
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .fig3_qpcr import (REFERENCE_GENE, normalise_sample, parse_sample,
                        read_plate_cq)
from .report_style import OKABE_ITO, caption, ordered_groups, save_fig

#: Directory name fragment identifying the reference-gene screen plate.
REF_PLATE_HINT = "ref gene"

#: Conditions dropped from the stability maths (technical failures, not biology).
EXCLUDE = ["CT D 14", "20 UM H2O2 D 7"]

#: Candidate colours — distinct from the group palette (these are genes, not arms).
CANDIDATE_COLORS = {
    "ACTB": OKABE_ITO["purple"],
    "GAPDH": OKABE_ITO["green"],
    "HPRT1": OKABE_ITO["sky"],
}


def find_ref_plate(root) -> Path:
    """The reference-gene-screen sub-directory of a qPCR analysis root."""
    root = Path(root)
    for p in sorted(root.iterdir()):
        if p.is_dir() and REF_PLATE_HINT in p.name.lower():
            return p
    raise FileNotFoundError(f"no '*{REF_PLATE_HINT}*' plate directory under {root}")


def load_ref_screen(root) -> pd.DataFrame:
    """Per-well Cq of the reference screen: ``sample, key, group, day, gene, cq``.

    ``key`` is the normalised condition name; NTC wells are returned separately
    by :func:`ntc_calls`, not here.
    """
    wells = read_plate_cq(find_ref_plate(root)).dropna(subset=["Sample", "Gene"])
    wells = wells[wells.Sample.astype(str).str.upper() != "NTC"].copy()
    wells["key"] = wells.Sample.map(normalise_sample)
    parsed = wells.Sample.map(parse_sample)
    wells["group"] = [p[0] for p in parsed]
    wells["day"] = [p[1] for p in parsed]
    return wells.rename(columns={"Sample": "sample", "Gene": "gene", "Cq": "cq"})[
        ["sample", "key", "group", "day", "gene", "cq"]]


def ntc_calls(root) -> pd.DataFrame:
    """NTC wells across every plate: ``plate, gene, cq`` (NaN = no amplification)."""
    rows = []
    for pdir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        try:
            wells = read_plate_cq(pdir)
        except FileNotFoundError:
            continue
        ntc = wells[wells.Sample.astype(str).str.upper() == "NTC"]
        for _, r in ntc.dropna(subset=["Gene"]).iterrows():
            rows.append({"plate": pdir.name, "gene": r.Gene, "cq": r.Cq})
    return pd.DataFrame(rows)


def marker_plate_reference(root, reference: str = REFERENCE_GENE
                           ) -> dict[str, pd.Series]:
    """``plate tag -> mean reference Cq per condition`` for the marker plates.

    Lets S5 panel D show that a failed condition fails on *every* plate, i.e.
    it is a cDNA/RNA input problem rather than one bad well.
    """
    out: dict[str, pd.Series] = {}
    ref_dir = find_ref_plate(root)
    for pdir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if pdir == ref_dir:
            continue
        try:
            wells = read_plate_cq(pdir)
        except FileNotFoundError:
            continue
        wells = wells[(wells.Gene == reference)
                      & (wells.Sample.astype(str).str.upper() != "NTC")]
        wells = wells.dropna(subset=["Sample"])
        if wells.empty:
            continue
        out[_plate_tag(pdir.name)] = (wells.assign(key=wells.Sample.map(normalise_sample))
                                      .groupby("key").Cq.mean())
    return out


def condition_matrix(screen: pd.DataFrame,
                     exclude: list[str] | None = None) -> pd.DataFrame:
    """``condition x candidate-gene`` mean Cq, technical duplicates averaged."""
    exclude = EXCLUDE if exclude is None else exclude
    keep = screen[~screen.key.isin(exclude)].dropna(subset=["cq"])
    return keep.groupby(["key", "gene"]).cq.mean().unstack()


def genorm_m(mat: pd.DataFrame) -> pd.Series:
    """geNorm M per gene: mean over j!=i of ``SD_samples(Cq_i - Cq_j)``."""
    genes = list(mat.columns)
    return pd.Series(
        {a: float(np.mean([mat[a].sub(mat[b]).std(ddof=1)
                           for b in genes if b != a])) for a in genes},
        name="genorm_m")


def pairwise_dcq_sd(mat: pd.DataFrame) -> pd.Series:
    """SD across conditions of every candidate pair's ΔCq."""
    return pd.Series(
        {f"{a}−{b}": float(mat[a].sub(mat[b]).std(ddof=1))
         for a, b in itertools.combinations(mat.columns, 2)},
        name="pairwise_sd")


def _candidate_color(gene: str) -> str:
    return CANDIDATE_COLORS.get(gene, OKABE_ITO["grey"])


def _pretty(key: str) -> str:
    """``'20 UM H2O2 D 14'`` -> ``'H2O2 D14'``, ``'IVH L D 7'`` -> ``'IVH-L D7'``."""
    s = key.replace("20 UM H2O2", "H2O2").replace("10 UM H2O2", "H2O2-10")
    s = s.replace("IVH E", "IVH-E").replace("IVH L", "IVH-L")
    s = re.sub(r"\bDIV\s*(\d+)", r"DIV\1", s)
    s = re.sub(r"\bD\s*(\d+)", r"D\1", s)
    return re.sub(r"\s+", " ", s).replace("DIV", "DIV ").strip()


def _plate_tag(name: str) -> str:
    """Short label for a plate directory name (``'…Plate1 GAPDH…'`` -> ``'Plate1'``)."""
    for tok in name.split():
        if tok.lower().startswith("plate"):
            return tok
    return "ref plate"


def _sorted_conditions(screen: pd.DataFrame) -> list[str]:
    """Conditions ordered by arm (canonical group order) then day."""
    meta = (screen.dropna(subset=["group"])
            .drop_duplicates("key").set_index("key")[["group", "day"]])
    order = {g: i for i, g in enumerate(ordered_groups(meta.group.unique()))}
    return list(meta.sort_values(by=["group", "day"],
                                key=lambda s: s.map(order) if s.name == "group" else s
                                ).index)


_SELECTED_COLOR = "#117733"


def _flag_selected(ax, genes, best, y=None):
    """Bold the winner's tick label and stamp a ✓ selected flag (every panel)."""
    if best not in genes:
        return
    bi = genes.index(best)
    for lbl in ax.get_xticklabels():
        if lbl.get_text().split("\n")[0] == best:
            lbl.set_fontweight("bold"); lbl.set_color(_SELECTED_COLOR)
    if y is None:                       # over the column, top of the axes
        ax.annotate("✓ selected", xy=(bi, 0.98), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=6, color=_SELECTED_COLOR,
                    fontweight="bold")
    else:                               # above the geNorm M bar (at x + 0.19)
        ax.annotate("✓ selected reference", xy=(bi + 0.19, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.5, color=_SELECTED_COLOR, fontweight="bold")


def build_s5(root):
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    screen = load_ref_screen(root)
    mat = condition_matrix(screen)
    genes = [g for g in ["ACTB", "GAPDH", "HPRT1"] if g in mat.columns]
    genes += [g for g in mat.columns if g not in genes]
    mat = mat[genes]
    m = genorm_m(mat)
    sd = mat.std(ddof=1)
    pair = pairwise_dcq_sd(mat)
    best = m.idxmin()

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0))

    # --- A. spread per candidate -------------------------------------------
    # Absolute Cq differs by ~8 cycles between candidates (ACTB 18, GAPDH 20,
    # HPRT1 26), which would hide the within-gene spread that stability is
    # about — so plot deviation from each gene's own mean and print the mean.
    ax = axes[0, 0]
    centred = mat - mat.mean()
    for i, gene in enumerate(genes):
        v = centred[gene].to_numpy(float)
        jit = (np.arange(len(v)) % 5 - 2) * 0.035
        ax.plot(np.full(len(v), i) + jit, v, "o", ms=3.5, mfc=_candidate_color(gene),
                mec="none", alpha=0.85, zorder=3)
    # one faint line per condition — parallel lines mean the conditions move
    # together (shared cDNA input), not that one gene is drifting.
    for _, row in centred.iterrows():
        ax.plot(range(len(genes)), row[genes].to_numpy(float), "-", color="0.8",
                lw=0.5, zorder=1)
    ax.axhline(0.0, ls="--", color="0.6", lw=0.7, zorder=2)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels([f"{g}\nmean Cq {mat[g].mean():.1f}" for g in genes])
    ax.set_ylabel("Cq − gene mean (cycles)")
    ax.set_title(f"A  Candidate spread (n = {len(mat)} conditions)",
                 loc="left", fontweight="bold")
    _flag_selected(ax, genes, best)   # ✓ selected reference on the winner

    # --- B. stability metrics ---------------------------------------------
    ax = axes[0, 1]
    x = np.arange(len(genes))
    ax.bar(x - 0.19, [sd[g] for g in genes], 0.36, label="raw Cq SD",
           color="0.72", edgecolor="none")
    ax.bar(x + 0.19, [m[g] for g in genes], 0.36, label="geNorm M",
           color=OKABE_ITO["blue"], edgecolor="none")
    ax.axhline(0.5, ls="--", color="0.5", lw=0.7)
    ax.annotate("geNorm M < 0.5 = stable", xy=(len(genes) - 0.5, 0.5),
                xytext=(0, 2), textcoords="offset points", ha="right",
                fontsize=6, color="0.4")
    ax.set_xticks(x)
    ax.set_xticklabels(genes)
    ax.set_ylabel("cycles")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title(f"B  Stability — {best}: lowest geNorm M ({m[best]:.3f})",
                 loc="left", fontweight="bold", fontsize=7.5)
    _flag_selected(ax, genes, best, y=m[best])

    # --- C. pairwise ΔCq SD -------------------------------------------------
    ax = axes[1, 0]
    ax.barh(range(len(pair)), pair.to_numpy(float), 0.55, color="0.55",
            edgecolor="none")
    ax.set_yticks(range(len(pair)))
    ax.set_yticklabels(pair.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("SD of ΔCq across conditions (cycles)")
    ax.set_title("C  Pairwise ΔCq stability", loc="left", fontweight="bold")

    # --- D. QC: reference Cq per condition, every plate ---------------------
    ax = axes[1, 1]
    order = _sorted_conditions(screen)
    pos = {k: i for i, k in enumerate(order)}
    xs = list(range(len(order)))
    series = [(f"{g} (ref plate)", _candidate_color(g), "-",
               screen[screen.gene == g].groupby("key").cq.mean().reindex(order))
              for g in genes]
    series += [(f"{REFERENCE_GENE} ({tag})", grey, ls, s.reindex(order))
               for (tag, s), grey, ls in zip(marker_plate_reference(root).items(),
                                             ("0.30", "0.60"), ("--", ":"))]
    for label, colour, ls, s in series:
        ax.plot(xs, s.to_numpy(float), "o", ls=ls, ms=3, lw=0.9, color=colour,
                label=label)

    # Scale to the passing conditions; the failures are called out in text so a
    # single no-amplification Cq (~39) cannot squash the informative range.
    valid = pd.concat([s.drop(index=EXCLUDE, errors="ignore") for _, _, _, s in series])
    lo, hi = valid.min(), valid.max()
    ax.set_ylim(lo - 1.5, hi + 3.5)
    for key in EXCLUDE:
        if key in pos:
            ax.axvspan(pos[key] - 0.45, pos[key] + 0.45, color=OKABE_ITO["vermillion"],
                       alpha=0.12, lw=0, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([_pretty(k) for k in order], rotation=60, ha="right",
                       fontsize=5.5)
    ax.set_ylabel("Cq")
    ax.legend(fontsize=5.5, loc="lower left", ncol=2, frameon=True,
          facecolor="white", edgecolor="none", framealpha=0.9)
    for key, note in [("CT D 14", "all refs ≈ +10 Cq\non every plate"),
                      ("20 UM H2O2 D 7", "no\namplification")]:
        if key in pos:
            ax.annotate(note, xy=(pos[key], 0.98), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=5.5,
                        color=OKABE_ITO["vermillion"],
                        bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))
    ax.set_title("D  QC — excluded conditions shaded", loc="left", fontweight="bold")

    ntc = ntc_calls(root)
    amp = ntc.dropna(subset=["cq"])
    ntc_txt = ("NTC: all no-Cq" if amp.empty else
               "NTC: " + "; ".join(
                   f"{r.gene} Cq {r.cq:.1f} on {_plate_tag(r.plate)}"
                   for _, r in amp.iterrows())
               + f" (late, of 45 cycles); {len(ntc) - len(amp)}/{len(ntc)} no-Cq")
    fig.suptitle("Figure S5 — reference-gene selection (CX169)", fontsize=9,
                 x=0.02, ha="left", fontweight="bold")
    # Shared candidate colour legend (A & D use these colours).
    fig.legend(handles=[mlines.Line2D([], [], marker="o", ls="", ms=5,
                                      color=_candidate_color(g), label=g)
                        for g in genes],
               loc="upper right", ncol=len(genes), bbox_to_anchor=(0.99, 0.995),
               fontsize=6.5, frameon=False, title="candidate", title_fontsize=6)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    caption(fig,
        "Selection of the qPCR reference (housekeeping) gene from the candidate "
        "screen. (A) Spread of raw Cq across conditions for each candidate. "
        "(B) Expression stability by the geNorm M statistic (lower = more "
        "stable); the winner anchors the marker plates. (C) Pairwise ΔCq "
        "stability between candidates. (D) QC — excluded conditions shaded. "
        f"{ntc_txt}. Excluded from A-C: "
        + ", ".join(_pretty(k) for k in EXCLUDE)
        + f". Marker plates (F3) use {REFERENCE_GENE} as the reference gene.")
    return fig


def render(figure_root, qpcr_dir):
    fig = build_s5(qpcr_dir)
    return save_fig(fig, "S5_reference_genes", figure_root, subdir="supp")
