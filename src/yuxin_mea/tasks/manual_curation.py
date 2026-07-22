from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask
from yuxin_mea.tasks.analyzer import AnalyzerTask
from yuxin_mea.tasks.preprocessing import PreprocessingTask


class ManualCurationTask(BaseAnalysisTask):
    """Apply **manual** single-unit curation authored in the dashboard.

    Reads the durable manifest written by the dashboard
    (``<analysis_root>/manual_curation_data/<rk>/<rec>/<well>/manual_curation.json``,
    see :mod:`yuxin_mea.analysis.manual_curation_store`), converts it to a
    SpikeInterface curation-format-v2 dict, and runs ``si.apply_curation`` on the
    analyzer's sorting to produce a curated spike-time set.

    Outputs (under ``applied_output_root``, a tree **separate** from
    ``analyzer_data`` / ``curation_data`` so nothing regenerates the manual work)::

        <applied_output_root>/<rk>/<rec>/<well>/manual_curation/
          curated_spike_times.npy   # {unit_id: spike_times_sec} — downstream contract
          applied_curation.json     # the exact SI curation dict applied
          result.json               # summary (counts, fingerprint, skipped reason)

    Wells with no manifest (the common case — curation is opt-in per well) are a
    graceful COMPLETE-empty: a ``result.json`` marker is written with
    ``skipped=True`` and no analyzer is loaded. A manifest whose fingerprint no
    longer matches the sorting is likewise skipped (``skipped_reason="stale"``)
    rather than misapplied.
    """

    task_name = "manual_curation"
    dependencies = ["analyzer"]

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "manual_curation_root": "./manual_curation_data",
            "analyzer_output_root": "./analyzer_data",
            "applied_output_root": "./manual_curation_applied",
            "split_feature": "amplitude",
        }

    @classmethod
    def params_schema(cls) -> dict[str, ParamSpec]:
        d = cls.default_params()
        return {
            "manual_curation_root": ParamSpec(
                "path", d["manual_curation_root"],
                "Directory holding the dashboard-authored manual_curation.json "
                "manifests. Relative paths resolve under analysis_root.",
            ),
            "analyzer_output_root": ParamSpec(
                "path", d["analyzer_output_root"],
                "Directory with upstream SortingAnalyzer outputs (AnalyzerTask).",
            ),
            "applied_output_root": ParamSpec(
                "path", d["applied_output_root"],
                "Directory where applied-curation outputs are written. Relative "
                "paths resolve under analysis_root.",
            ),
            "split_feature": ParamSpec(
                "str", d["split_feature"],
                "Per-spike feature used to resolve manual split thresholds "
                "(currently 'amplitude').",
            ),
        }

    @staticmethod
    def split_compound_well_id(well_id: str) -> tuple[str, str]:
        return PreprocessingTask.split_compound_well_id(well_id)

    @staticmethod
    def build_output_path(
        applied_output_root: str | Path,
        recording_key: str,
        rec_name: str,
        well_id: str,
    ) -> Path:
        return (Path(applied_output_root) / Path(recording_key) / rec_name
                / well_id / "manual_curation")

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _resolve_root(root: str | Path, analysis_root: Path) -> Path:
        p = Path(root)
        return p if p.is_absolute() else analysis_root / p

    @staticmethod
    def _write_result(output_dir: Path, payload: dict[str, Any]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "result.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return output_dir

    # -- run ---------------------------------------------------------------
    def run(
        self,
        recording_key: str,
        well_id: str,
        data_path: Path,
        params: dict[str, Any],
    ) -> Path:
        import numpy as np

        from yuxin_mea.analysis import manual_curation_store as mc

        p = self.resolve_params(params)
        rec_name, actual_well = self.split_compound_well_id(well_id)

        analyzer_root = Path(p["analyzer_output_root"])
        # analysis_root = parent of the analyzer stage dir (…/analysis/analyzer_data).
        analysis_root = analyzer_root.parent
        manual_root = self._resolve_root(p["manual_curation_root"], analysis_root)
        applied_root = self._resolve_root(p["applied_output_root"], analysis_root)

        output_dir = self.build_output_path(
            applied_root, recording_key, rec_name, actual_well)

        manifest_path = mc.manifest_path(
            manual_root, recording_key, rec_name, actual_well)
        manifest = mc.load_manifest(manifest_path)
        if manifest is None:
            return self._write_result(output_dir, {
                "skipped": True, "skipped_reason": "no_manifest",
                "manifest_path": str(manifest_path)})

        # Load analyzer (heavy — only reached for curated wells).
        import spikeinterface.full as si
        from spikeinterface.curation import apply_curation, validate_curation_dict

        analyzer_path = AnalyzerTask.build_output_path(
            analyzer_root, recording_key, rec_name, actual_well)
        analyzer = si.load_sorting_analyzer(analyzer_path)
        sorting = analyzer.sorting
        unit_ids = [int(u) for u in sorting.unit_ids]
        fs = float(sorting.get_sampling_frequency())

        fingerprint = mc.sorting_fingerprint(analyzer_path, unit_ids)
        if mc.is_stale(manifest, fingerprint):
            return self._write_result(output_dir, {
                "skipped": True, "skipped_reason": "stale",
                "manifest_fingerprint": manifest.get("sorting_fingerprint"),
                "sorting_fingerprint": fingerprint})

        # Resolve feature-threshold splits into per-spike index groups.
        split_indices = self._resolve_splits(
            manifest, analyzer_path, unit_ids, str(p["split_feature"]))

        curation = mc.to_si_curation_dict(manifest, unit_ids, split_indices)
        validate_curation_dict(curation)
        curated = apply_curation(sorting, curation)

        spike_times = {
            int(u): curated.get_unit_spike_train(u, segment_index=0).astype(float) / fs
            for u in curated.unit_ids
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "curated_spike_times.npy", spike_times)  # type: ignore[arg-type]
        with (output_dir / "applied_curation.json").open("w", encoding="utf-8") as fh:
            json.dump(curation, fh, indent=2)

        self._write_result(output_dir, {
            "skipped": False,
            "n_units_in": len(unit_ids),
            "n_units_out": int(len(curated.unit_ids)),
            "n_removed": len(curation["removed"]),
            "n_merges": len(curation["merges"]),
            "n_splits": len(curation["splits"]),
            "n_labeled": len(curation["manual_labels"]),
            "sorting_fingerprint": fingerprint,
        })
        return output_dir

    @staticmethod
    def _resolve_splits(
        manifest: dict[str, Any],
        analyzer_path: Path,
        unit_ids: list[int],
        feature: str,
    ) -> dict[int, list[list[int]]]:
        """Turn ``{unit_id, feature, threshold}`` specs into per-spike index groups.

        Amplitudes are read straight off the analyzer extension (spike order ==
        the unit's spike-train order), so a threshold partitions each unit's own
        spike indices — exactly what SI split ``mode='indices'`` expects.
        """
        specs = manifest.get("splits", [])
        if not specs or feature != "amplitude":
            return {}
        import numpy as np

        amp_path = analyzer_path / "extensions" / "spike_amplitudes" / "amplitudes.npy"
        spikes_path = analyzer_path / "sorting" / "spikes.npy"
        if not (amp_path.exists() and spikes_path.exists()):
            return {}
        amplitudes = np.load(amp_path)
        spikes = np.load(spikes_path)
        out: dict[int, list[list[int]]] = {}
        for spec in specs:
            uid = int(spec["unit_id"])
            if uid not in unit_ids:
                continue
            pos = unit_ids.index(uid)
            amps_u = amplitudes[spikes["unit_index"] == pos]
            thr = float(spec["threshold"])
            g0 = np.nonzero(amps_u <= thr)[0]
            g1 = np.nonzero(amps_u > thr)[0]
            if g0.size and g1.size:
                out[uid] = [g0.tolist(), g1.tolist()]
        return out
