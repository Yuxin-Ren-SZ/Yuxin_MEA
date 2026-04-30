from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline_manager import BaseAnalysisTask
from pipeline_tasks.analyzer import AnalyzerTask
from pipeline_tasks.preprocessing import PreprocessingTask


class AutoCurationTask(BaseAnalysisTask):
    """Quality-metric curation of sorted units.

    Loads the SortingAnalyzer from AnalyzerTask, merges quality_metrics,
    template_metrics, and unit_locations into a single pickled DataFrame, and
    applies configurable threshold filters.

    When enabled=True (default), units that fail any threshold are marked
    curated=False in quality_metrics.pkl and recorded in
    rejection_log.pkl.

    When enabled=False, all units are marked curated=True (pass-through).

    Always writes curated_spike_times.npy — a dict of
    {unit_id: np.ndarray[float]} of spike times in seconds — which is the
    hard contract expected by BurstDetectionTask.
    """

    task_name = "auto_curation"
    dependencies = ["analyzer"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "curation_output_root": "./curation_data",
            "analyzer_output_root": "./analyzer_data",
            "enabled": True,
            "presence_ratio_min": 0.75,
            "rp_contamination_max": 0.15,
            "firing_rate_min": 0.05,
            "amplitude_median_max": -20.0,
        }

    @staticmethod
    def split_compound_well_id(well_id: str) -> tuple[str, str]:
        return PreprocessingTask.split_compound_well_id(well_id)

    @staticmethod
    def build_output_path(
        curation_output_root: str | Path,
        recording_key: str,
        rec_name: str,
        well_id: str,
    ) -> Path:
        return (
            Path(curation_output_root)
            / Path(recording_key)
            / rec_name
            / well_id
            / "auto_curation"
        )

    @staticmethod
    def _apply_thresholds(
        metrics: Any,
        p: dict[str, Any],
    ) -> tuple[list[bool], list[dict[str, Any]]]:
        """Returns (keep_flags, rejection_rows) without importing pandas."""
        keep: list[bool] = []
        rejections: list[dict[str, Any]] = []

        for unit_id, row in metrics.iterrows():
            reasons: list[str] = []
            if row.get("presence_ratio", 1.0) < float(p["presence_ratio_min"]):
                reasons.append("low_presence_ratio")
            if row.get("rp_contamination", 0.0) > float(p["rp_contamination_max"]):
                reasons.append("high_rp_contamination")
            if row.get("firing_rate", 0.0) < float(p["firing_rate_min"]):
                reasons.append("low_firing_rate")
            # amplitude_median is negative for real spikes; reject if too close to zero
            if row.get("amplitude_median", -100.0) > float(p["amplitude_median_max"]):
                reasons.append("low_amplitude")

            keep.append(len(reasons) == 0)
            if reasons:
                rejections.append({"unit_id": unit_id, "reasons": "; ".join(reasons)})

        return keep, rejections

    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        import numpy as np
        import pandas as pd
        import spikeinterface.full as si

        p = self.resolve_params(params)
        rec_name, actual_well_id = self.split_compound_well_id(well_id)

        analyzer_path = AnalyzerTask.build_output_path(
            p["analyzer_output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        output_dir = self.build_output_path(
            p["curation_output_root"],
            recording_key,
            rec_name,
            actual_well_id,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        analyzer = si.load_sorting_analyzer(analyzer_path)
        sorting = analyzer.sorting
        fs = float(analyzer.recording.get_sampling_frequency())

        qm = analyzer.get_extension("quality_metrics").get_data()
        tm = analyzer.get_extension("template_metrics").get_data()
        locs = analyzer.get_extension("unit_locations").get_data()

        # Merge into one table; suffix _tm disambiguates any shared column names
        metrics = qm.join(tm, how="left", rsuffix="_tm")
        metrics["loc_x"] = locs[:, 0]
        metrics["loc_y"] = locs[:, 1]

        if p["enabled"]:
            keep, rejection_rows = self._apply_thresholds(metrics, p)
            metrics = metrics.copy()
            metrics["curated"] = keep
            rejection_log = pd.DataFrame(rejection_rows, columns=["unit_id", "reasons"])
        else:
            metrics = metrics.copy()
            metrics["curated"] = True
            rejection_log = pd.DataFrame(columns=["unit_id", "reasons"])

        curated_ids = metrics.index[metrics["curated"]].tolist()

        metrics.to_pickle(output_dir / "quality_metrics.pkl")
        rejection_log.to_pickle(output_dir / "rejection_log.pkl")

        # Hard contract: BurstDetectionTask expects this file at this exact path
        spike_times: dict[Any, Any] = {
            uid: sorting.get_unit_spike_train(uid, segment_index=0).astype(float) / fs
            for uid in curated_ids
        }
        np.save(output_dir / "curated_spike_times.npy", spike_times)  # type: ignore[arg-type]

        return output_dir
