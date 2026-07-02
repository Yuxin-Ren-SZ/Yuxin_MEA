"""Per-recording "viewer bundle" — one file holding all 24 wells (Caching guide Tier 3).

The dashboard normally assembles a recording from per-well outputs
(``plot_signals.npy`` + event tables + ``curated_spike_times.npy``). A viewer
bundle collapses those "hundreds of tiny files" into a **single sequential
read** — the access pattern a network filesystem is actually good at — so the
first-ever load is fast even before any in-process/Tier-2 cache is warm.

The bundle is a pickled ``list[WellRecord]`` (the exact objects
:func:`build_plate_figure` / :func:`build_single_well_figure` consume) plus a
staleness ``signature``. On read we re-check the signature against the live
source files and ignore a stale bundle — same "bust by signature, never by
timer" rule as every other cache here.

Trust boundary: bundles are written by this project's own tooling and live
alongside the analysis outputs (not user-supplied), so unpickling them is no
more privileged than the ``pd.read_pickle`` / ``np.load(allow_pickle=True)``
the loaders already do on task outputs.

This module is standalone — it is NOT wired into the pipeline DAG (that would
need plate-level task enumeration the pipeline doesn't have yet). Generate
bundles opt-in with ``scripts/build_viewer_bundles.py``; the dashboard reads
them automatically when present and fresh.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yuxin_mea.analysis.plate_raster_synchrony import WellRecord

# Bump on ANY change to WellRecord's fields: the signature guards the envelope +
# source freshness, not the pickled dataclass shape, and unpickling bypasses the
# _make_well_record tolerance — a stale-shaped bundle would otherwise load dirty.
_BUNDLE_VERSION = 1
_BUNDLE_SUFFIX = ".viewer.pkl"


def bundle_path(bundle_root: str | Path, recording_key: str, source: str) -> Path:
    """Location of a recording+source bundle under ``bundle_root``."""
    return Path(bundle_root) / recording_key / f"{source}{_BUNDLE_SUFFIX}"


def write_viewer_bundle(
    well_records: list["WellRecord"],
    out_path: str | Path,
    signature: str = "",
) -> Path:
    """Write the 24 well records + signature to ``out_path`` atomically."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _BUNDLE_VERSION,
        "signature": str(signature),
        "records": list(well_records),
    }
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out_path)  # atomic on the same filesystem
    return out_path


def read_viewer_bundle(
    path: str | Path,
    expected_signature: str | None = None,
) -> list["WellRecord"] | None:
    """Read a bundle, or return ``None`` when absent/corrupt/stale.

    When ``expected_signature`` is given, a bundle whose stored signature differs
    (source files changed since it was built) is treated as a miss.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception:  # noqa: BLE001 — corrupt/partial bundle → miss, rebuild
        return None
    if not isinstance(payload, dict) or payload.get("version") != _BUNDLE_VERSION:
        return None
    if expected_signature is not None and str(payload.get("signature")) != str(expected_signature):
        return None
    records = payload.get("records")
    return records if isinstance(records, list) else None
