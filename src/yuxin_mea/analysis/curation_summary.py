"""Summarize ``AutoCurationTask`` outputs for one or many wells.

Reads `quality_metrics.pkl` and `rejection_log.pkl` written by
``yuxin_mea.tasks.AutoCurationTask`` and returns structured summaries
suitable for notebook display or dashboard surfacing.

Extracted from inline cells in
``notebooks/05_auto_curation.ipynb`` (cells 17–19) so the logic is
testable and reusable. The functions are pure readers — they do not
mutate any cache file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


_METRIC_COLUMNS = ("presence_ratio", "rp_contamination", "firing_rate", "amplitude_median")


def summarize_curation(curation_output_dir: Path) -> dict[str, Any]:
    """Summarize one well's curation output.

    Returns a dict with the following keys (None if absent on disk):

    - ``n_total``       : total unit count
    - ``n_curated``     : units that passed all thresholds
    - ``n_rejected``    : total − curated
    - ``pct_kept``      : 100 * n_curated / n_total (or None if n_total == 0)
    - ``rejection_reasons`` : ``{reason: count}`` from ``rejection_log.pkl``
    - ``metric_stats``  : ``pandas.DataFrame.describe()`` of the four
      standard metrics restricted to curated units
    - ``output_dir``    : the input path (for downstream rendering)

    Raises ``FileNotFoundError`` if ``quality_metrics.pkl`` is missing —
    the well almost certainly hasn't been curated yet.
    """
    curation_output_dir = Path(curation_output_dir)
    qm_path = curation_output_dir / "quality_metrics.pkl"
    if not qm_path.exists():
        raise FileNotFoundError(f"quality_metrics.pkl not found under {curation_output_dir}")

    qm = pd.read_pickle(qm_path)
    n_total = len(qm)
    if "curated" not in qm.columns:
        # Defensive: an older format would lack the boolean flag column.
        raise KeyError(
            f"quality_metrics.pkl at {qm_path} has no `curated` column; "
            "regenerate the curation output with the current AutoCurationTask."
        )
    n_curated = int(qm["curated"].sum())
    n_rejected = n_total - n_curated

    rejection_reasons = _load_rejection_reasons(curation_output_dir / "rejection_log.pkl")
    metric_stats = _curated_metric_stats(qm)

    return {
        "output_dir": curation_output_dir,
        "n_total": n_total,
        "n_curated": n_curated,
        "n_rejected": n_rejected,
        "pct_kept": round(100 * n_curated / n_total, 1) if n_total else None,
        "rejection_reasons": rejection_reasons,
        "metric_stats": metric_stats,
    }


def format_curation_summary(summary: dict[str, Any]) -> str:
    """Render a ``summarize_curation`` dict as a multi-line plain-text block."""
    lines = [
        f"Output dir:  {summary['output_dir']}",
        f"  Total units:    {summary['n_total']}",
        f"  Passed:         {summary['n_curated']}",
        f"  Rejected:       {summary['n_rejected']}",
    ]
    if summary["pct_kept"] is not None:
        lines.append(f"  % kept:         {summary['pct_kept']}%")
    if summary["rejection_reasons"]:
        lines.append("\nRejection reasons:")
        for reason, count in summary["rejection_reasons"].items():
            lines.append(f"  {reason}: {count}")
    if summary["metric_stats"] is not None and not summary["metric_stats"].empty:
        lines.append("\nMetric summary (curated units):")
        lines.append(summary["metric_stats"].round(3).to_string())
    return "\n".join(lines)


def aggregate_curation_summaries(curation_output_dirs: list[Path]) -> pd.DataFrame:
    """Build a one-row-per-well summary DataFrame.

    Skips directories that don't have a ``quality_metrics.pkl``. Returns
    an empty DataFrame (with the expected columns) when every directory is
    skipped — useful for notebook display before any curation has run.
    """
    rows: list[dict[str, Any]] = []
    for output_dir in curation_output_dirs:
        output_dir = Path(output_dir)
        qm_path = output_dir / "quality_metrics.pkl"
        if not qm_path.exists():
            continue
        qm = pd.read_pickle(qm_path)
        if "curated" not in qm.columns:
            continue
        n_total = len(qm)
        n_curated = int(qm["curated"].sum())
        curated_rows = qm.loc[qm["curated"]] if n_curated else qm.iloc[:0]
        rows.append({
            "output_dir": str(output_dir),
            "n_total": n_total,
            "n_curated": n_curated,
            "pct_kept": round(100 * n_curated / n_total, 1) if n_total else None,
            "median_firing_rate": _median(curated_rows, "firing_rate"),
            "median_amplitude":   _median(curated_rows, "amplitude_median"),
        })
    columns = [
        "output_dir", "n_total", "n_curated", "pct_kept",
        "median_firing_rate", "median_amplitude",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_rejection_reasons(rl_path: Path) -> dict[str, int]:
    """Return ``{reason: count}`` parsed from ``rejection_log.pkl`` semicolons.

    Returns an empty dict if the file is absent or the log is empty.
    """
    if not rl_path.exists():
        return {}
    rl = pd.read_pickle(rl_path)
    if rl.empty or "reasons" not in rl.columns:
        return {}
    counts = (
        rl["reasons"].astype(str).str.split("; ").explode().value_counts()
    )
    return {reason: int(count) for reason, count in counts.items()}


def _curated_metric_stats(qm: pd.DataFrame) -> pd.DataFrame | None:
    cols = [c for c in _METRIC_COLUMNS if c in qm.columns]
    curated = qm.loc[qm["curated"], cols] if "curated" in qm.columns else qm[cols]
    if curated.empty or not cols:
        return None
    return curated.describe()


def _median(curated_rows: pd.DataFrame, column: str) -> float | None:
    if column not in curated_rows.columns or curated_rows.empty:
        return None
    val = curated_rows[column].median()
    return None if pd.isna(val) else round(float(val), 3)
