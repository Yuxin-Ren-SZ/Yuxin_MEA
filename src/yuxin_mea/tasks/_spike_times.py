from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def load_spike_times(
    curation_dir: Path,
    analyzer_path: Path,
    logger: logging.Logger,
    well_label: str,
) -> dict[Any, Any]:
    """Load spike times, preferring curated data with analyzer fallback.

    Tries ``curation_dir/curated_spike_times.npy`` first.  When that file is
    absent (auto_curation was skipped), falls back to extracting *all* unit
    spike trains from the SortingAnalyzer at *analyzer_path*.
    """
    import numpy as np

    curated_path = curation_dir / "curated_spike_times.npy"
    if curated_path.exists():
        return np.load(curated_path, allow_pickle=True).item()  # type: ignore[union-attr]

    if not analyzer_path.exists():
        raise FileNotFoundError(
            f"No spike-time source for {well_label}: "
            f"curated file not found at {curated_path} and "
            f"SortingAnalyzer not found at {analyzer_path}."
        )

    logger.warning(
        "%s: curated_spike_times.npy not found; "
        "falling back to uncurated spike times from %s",
        well_label,
        analyzer_path,
    )

    import spikeinterface.full as si

    analyzer = si.load_sorting_analyzer(analyzer_path)
    sorting = analyzer.sorting
    fs = float(analyzer.recording.get_sampling_frequency())
    return {
        uid: sorting.get_unit_spike_train(uid, segment_index=0).astype(float) / fs
        for uid in sorting.unit_ids
    }
