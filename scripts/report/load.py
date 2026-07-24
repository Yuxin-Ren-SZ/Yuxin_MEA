"""Thin data loaders for the journal report.

Two families of source:

* **Rolled-up tables** already produced by ``compare_treatment_groups.py`` and
  ``compare_burst_methods.py`` under ``<figure_root>/`` — one ``read_csv`` each.
* **Per-well artifacts** on the analysis tree. Every row of ``tidy_long.csv``
  carries enough identifiers to reconstruct the artifact path deterministically::

      <analysis_root>/<stage>/<recording_key>/<rec_name>/<well_id>/<subdir>/<file>

  e.g. ``recording_key = CX118/260203/T003346/Network/000002`` +
  ``rec_name = rec0000`` + ``well_id = well000``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from .report_style import figures_base, resolve_roots

# --------------------------------------------------------------------------- #
# Rolled-up tables
# --------------------------------------------------------------------------- #
_TC = "treatment_comparison"
_BMC = "burst_method_comparison"


def load_tidy(figure_root: Path) -> pd.DataFrame:
    """Master per-well-recording metric table (tidy_long.csv)."""
    return pd.read_csv(figures_base(figure_root) / _TC / "tidy_long.csv")


def load_per_well(figure_root: Path) -> pd.DataFrame:
    """Per-well baseline-vs-post response table (per_well_summary.csv)."""
    return pd.read_csv(figures_base(figure_root) / _TC / "per_well_summary.csv")


def load_stats(figure_root: Path) -> pd.DataFrame:
    """Per-metric arm-vs-Control stats (stats.csv: Cliff's delta + CI, q_bh)."""
    return pd.read_csv(figures_base(figure_root) / _TC / "stats.csv")


def load_method_compare(figure_root: Path) -> pd.DataFrame:
    """ML-vs-traditional per-well burst comparison (per_well.csv)."""
    return pd.read_csv(figures_base(figure_root) / _BMC / "per_well.csv")


# --------------------------------------------------------------------------- #
# Per-well artifact paths
# --------------------------------------------------------------------------- #
def artifact_path(
    analysis_root: Path,
    stage: str,
    row: pd.Series | dict[str, Any],
    subdir: str,
    filename: str,
) -> Path:
    """Reconstruct the on-disk path of a per-well artifact from a tidy row."""
    rk = row["recording_key"]
    return (
        Path(analysis_root) / stage / rk
        / row["rec_name"] / row["well_id"] / subdir / filename
    )


def ml_trace_path(analysis_root: Path, row: pd.Series | dict[str, Any]) -> Path:
    return artifact_path(
        analysis_root, "ml_burst_data_umap", row,
        "ml_burst_detection", "debug_trace.pkl",
    )


def quality_metrics_path(analysis_root: Path, row: pd.Series | dict[str, Any]) -> Path:
    return artifact_path(
        analysis_root, "curation_data", row,
        "auto_curation", "quality_metrics.pkl",
    )


def load_ml_trace(analysis_root: Path, row: pd.Series | dict[str, Any]):
    """Unpickle an :class:`MLBurstTrace` for one well-recording.

    Imports the class first so pickle can resolve
    ``yuxin_mea.analysis.ml_burst_detector.MLBurstTrace``.
    """
    from yuxin_mea.analysis.ml_burst_detector import MLBurstTrace  # noqa: F401

    p = ml_trace_path(analysis_root, row)
    with p.open("rb") as fh:
        return pickle.load(fh)


def load_curated_spikes(
    analysis_root: Path, row: pd.Series | dict[str, Any],
) -> dict[int, "np.ndarray"]:
    """Return ``{unit_id: spike_times_seconds}`` for one well-recording.

    ``curated_spike_times.npy`` is a 0-d object array wrapping a dict.
    """
    import numpy as np

    p = artifact_path(
        analysis_root, "curation_data", row,
        "auto_curation", "curated_spike_times.npy",
    )
    arr = np.load(p, allow_pickle=True)
    return arr.item() if arr.ndim == 0 else dict(enumerate(arr))


def load_ml_bursts(analysis_root: Path, row: pd.Series | dict[str, Any]) -> pd.DataFrame:
    """Per-burst table (``network_bursts.pkl``) for one well-recording.

    Columns include the raw burst-shape features F6c clusters on: duration_s,
    within_burst_fr, participation, total_spikes, burst_peak, peak_synchrony,
    synchrony_energy; the ML-internal detector features: llr_aggregate, llr_peak,
    posterior_peak, posterior_mean, ff_peak, n_distinct_clusters; plus the
    per-well burst_type label. (Written by ``ml_burst_detector`` — see the event
    dict there for the authoritative schema.)
    """
    p = artifact_path(
        analysis_root, "ml_burst_data_umap", row,
        "ml_burst_detection", "network_bursts.pkl",
    )
    return pd.read_pickle(p)


def load_ccg(analysis_root: Path, row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Per-edge cross-correlograms for one well-recording.

    Returns the arrays written by ``connectivity.compute_ccg``: ``counts`` and
    ``baseline`` are ``(n_pairs, n_bins)``, ``lags_ms`` the bin centres, ``pairs``
    the ``(reference, target)`` unit ids — **rows are keyed by ``pairs``, never by
    position in ``edges.parquet``**, which can be longer when the pair cap trips.
    ``syn_lo_ms``/``syn_hi_ms`` bound the causal window the significance test uses.
    """
    import numpy as np

    p = artifact_path(analysis_root, "connectivity_data", row,
                      "connectivity", "ccg.npz")
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def load_edges(analysis_root: Path, row: pd.Series | dict[str, Any]) -> pd.DataFrame:
    """Significant STTC edges for one well-recording (``edges.parquet``).

    Columns: ``u, v, sttc, dist_um`` plus the ``ccg_*`` summaries
    (``ccg_peak_lag_ms, ccg_peak_z, ccg_sig, ccg_syn_dir, ccg_asymmetry``) when
    cross-correlograms were computed.
    """
    return pd.read_parquet(artifact_path(analysis_root, "connectivity_data", row,
                                         "connectivity", "edges.parquet"))


def load_templates(analysis_root: Path, row: pd.Series | dict[str, Any]):
    """Average waveform templates ``(n_units, n_samples, n_channels)``."""
    import numpy as np

    p = artifact_path(
        analysis_root, "analyzer_data", row,
        "analyzer/extensions/templates", "average.npy",
    )
    return np.load(p)


def load_units(analysis_root: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """Concatenate per-unit ``quality_metrics.pkl`` for the given tidy rows.

    Each unit row is annotated with its well identifiers + ``canonical_group``
    so downstream figures can group/colour without a second join. Missing
    pickles are skipped (logged count returned via attrs).

    Reads every row it is given. There used to be a ``limit`` that quietly took
    the first ``n`` — no caller ever passed it, and a truncation this easy to
    reach is the same footgun that made the S2 QC histograms describe 2% of the
    cohort. Narrow ``rows`` at the call site, where the choice is visible.
    """
    frames: list[pd.DataFrame] = []
    n_missing = 0
    for _, r in rows.iterrows():
        p = quality_metrics_path(analysis_root, r)
        if not p.exists():
            n_missing += 1
            continue
        qm = pd.read_pickle(p)
        qm = qm.copy()
        qm["well_uid"] = r["well_uid"]
        qm["canonical_group"] = r["canonical_group"]
        qm["DIV"] = r["DIV"]
        qm["tau"] = r["tau"]
        qm["sample_id"] = r["sample_id"]
        frames.append(qm)
    if not frames:
        out = pd.DataFrame()
    else:
        out = pd.concat(frames, ignore_index=True)
    out.attrs["n_missing"] = n_missing
    out.attrs["n_wells"] = len(rows) - n_missing
    return out


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def load_all(config_path: Path | str | None = None):
    """Resolve roots + load the three core tables in one call.

    Returns ``(analysis_root, figure_root, tidy, per_well, stats)``.
    """
    analysis_root, figure_root = resolve_roots(config_path)
    return (
        analysis_root,
        figure_root,
        load_tidy(figure_root),
        load_per_well(figure_root),
        load_stats(figure_root),
    )
