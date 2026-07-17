"""F3 — qPCR marker expression.

The only wet-lab figure that is data-generated. Reads a user-supplied table
(CSV/xlsx) and draws grouped bars (mean ± SEM) per gene per condition, coloured
with the shared group palette so it matches the MEA figures.

Expected table (long format, one row per replicate)::

    gene,group,value
    GFAP,Control,1.00
    GFAP,IVH_Early,2.31
    ...

Column auto-detection (case-insensitive):
  * gene   : one of {gene, target, marker}
  * group  : one of {group, condition, arm, treatment}
  * value  : one of {value, fold_change, fold, rq, ddct, expression,
             relative_expression}
A wide table (genes as columns, a group column, replicate rows) is auto-melted.
Provide ``--qpcr-table PATH`` to ``make_report``. Group labels are matched to
the canonical palette; unknown labels fall back to grey.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .report_style import group_color, ordered_groups, save_fig

_GENE_COLS = ["gene", "target", "marker"]
_GROUP_COLS = ["group", "condition", "arm", "treatment", "canonical_group"]
_VALUE_COLS = ["value", "fold_change", "fold", "rq", "ddct", "expression",
               "relative_expression", "rel_expr"]


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


def build_f3(qpcr: pd.DataFrame):
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


def render(figure_root, table_path=None):
    qpcr = load_qpcr(table_path) if table_path else synthetic_table()
    fig = build_f3(qpcr)
    name = "F3_qpcr" if table_path else "F3_qpcr_SYNTHETIC_placeholder"
    return save_fig(fig, name, figure_root, subdir="main")
