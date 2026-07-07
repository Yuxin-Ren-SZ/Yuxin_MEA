"""Read helpers for ml_burst_detection debug traces.

These load the per-well ``debug_trace.pkl`` (the 26-D bin feature matrix + HDBSCAN
labels + scaler stats persisted when ``debug=True``) and its sibling
``diagnostics.json``. Extracted so both the standalone overlay script and the
in-pipeline cluster-overlay tasks share one implementation and nothing in
``src/`` imports from ``scripts/``.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    try:
        with Path(path).open() as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def load_trace(pkl_path: Path):
    """Unpickle a ``debug_trace.pkl`` (an ``MLBurstTrace``). None if absent/broken."""
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        return None
    try:
        with pkl_path.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load debug_trace %s: %s", pkl_path, exc)
        return None


def burst_label_set(diagnostics: dict[str, Any] | None) -> set[int]:
    """Cluster ids selected as burst.

    Prefers multi-cluster ``cluster_burst_labels``; falls back to the legacy
    single ``cluster_burst_label``. Empty set when neither is present.
    """
    if not diagnostics:
        return set()
    labels = diagnostics.get("cluster_burst_labels")
    if labels:
        return {int(c) for c in labels if c is not None and int(c) != -1}
    bl = diagnostics.get("cluster_burst_label")
    if bl is not None and int(bl) != -1:
        return {int(bl)}
    return set()


def znorm_from_trace(trace) -> np.ndarray | None:
    """Reconstruct the z-normed feature matrix HDBSCAN saw, via saved scaler stats.

    Falls back to whole-matrix z-norm when scaler stats are absent (only for very
    early debug builds). None when there is no usable feature matrix.
    """
    X = np.asarray(trace.feature_matrix, dtype=float) if getattr(trace, "feature_matrix", None) is not None else None
    if X is None or X.ndim != 2 or X.shape[0] == 0:
        return None
    mu = getattr(trace, "scaler_mean", None)
    sd = getattr(trace, "scaler_std", None)
    if mu is None or sd is None:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
    mu = np.asarray(mu, float)
    sd = np.asarray(sd, float)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (X - mu) / sd
