# Yuxin MEA

MEA recording, pipeline, and analysis library for SadeghLab.

**English** | [中文](README.zh.md)

---

## What this is

An end-to-end analysis stack for HD-MEA recordings from MaxWell MaxTwo plates. The pipeline runs:

```
preprocessing (bandpass + common reference)
   → Kilosort4 spike sorting
   → (optional) SpikeInterface auto-merge
   → analyzer (waveforms, unit locations, quality metrics)
   → metric-threshold auto-curation
   → network burst detection
```

Two burst detectors ship with the project: a classic threshold-based detector (`burst_detection`) and an **iterative Fisher-LDA detector** (`iterative_burst_detection`) that refines a composite firing-rate signal across iterations, with optional GMM event clustering for super-burst structure. Outputs feed an interactive Plotly **plate viewer** (one HTML per recording) and a multipage **Dash dashboard** for non-technical browsing of dataset, pipeline status, and burst diagnostics.

## Repo layout

```
src/yuxin_mea/          installable namespace (`pip install -e .`)
  config/               JSON config loader + ParamSpec schemas
  dataset/              raw MEA discovery, mxassay metadata parsing, recording/well cache
  pipeline/             per-well task DAG + JSON status cache
  tasks/                preprocessing, sorting, auto_merge, analyzer,
                        auto_curation, burst_detection,
                        iterative_burst_detection, plate_viewer
  analysis/             algorithm code (burst detectors, burst_diagnostic lib,
                        curation_summary, synthetic_validation)
  dashboard/            multipage Dash app (Home / Recordings / Pipeline /
                        Burst Diagnostic / Settings)
config/                 example pipeline config JSON
notebooks/v2/           canonical post-refactor pipeline notebooks
notebooks/              original notebooks (kept for reference)
tests/                  pytest suite (~200 tests)
scripts/                helper scripts (e.g. strip_notebook_outputs.py)
doc/architecture.md     older architecture note (partly stale; AGENTS.md is authoritative)
AGENTS.md               exhaustive module/symbol map — the canonical reference
```

## Installation

Scientific dependencies (torch+CUDA, Kilosort, SpikeInterface, Dash, Plotly, scikit-learn, h5py, zarr, …) are managed by conda — `pyproject.toml` keeps `dependencies = []` on purpose so `pip install -e .` does not re-resolve them from PyPI.

```bash
conda env create -f environment.yml
conda activate yuxin_mea
pip install -e .
```

The pinned torch wheels target **CUDA 12.8**. For CPU-only or different CUDA, edit the `pip:` block in `environment.yml` before creating the env.

## Quickstart

Three entry points, depending on who you are.

### 1. Dashboard (non-technical users)

```bash
yuxin-mea-dashboard --config config/pipeline_config.example.json
```

Then open `http://127.0.0.1:8050`. Pages:

- **Home** — config path, data roots, cache entry counts
- **Recordings** — sortable/filterable table from `experiment_cache.json`
- **Pipeline** — `(recording × well) × task` status matrix
- **Burst Diagnostic** — batch-run the iterative burst detector and browse diagnostic figures
- **Settings** — schema-driven config editor (validated against each task's `ParamSpec` declarations)

If the config file doesn't exist yet, the dashboard still launches and shows a banner — use the Settings page to bootstrap one.

### 2. Notebooks (v2)

```
notebooks/v2/00_full_pipeline.ipynb   — end-to-end run
notebooks/v2/01_si_preprocessing.ipynb
notebooks/v2/01_plate_viewer.ipynb
notebooks/v2/03_auto_merge.ipynb
notebooks/v2/04_analyzer.ipynb
notebooks/v2/05_auto_curation.ipynb
notebooks/v2/06_iterative_burst_detector_synthetic_validation.ipynb
```

Open in JupyterLab from the `yuxin_mea` conda env. The original `notebooks/` are kept for reference but not maintained post-refactor.

### 3. Library (developers)

```python
from yuxin_mea.dataset import DatasetManager
from yuxin_mea.pipeline import PipelineManager
from yuxin_mea.config import ConfigManager
from yuxin_mea.analysis.iterative_burst_detector import compute_iterative_bursts
```

See `AGENTS.md` for the full public surface (module-by-module symbol map).

## Configuration

A single JSON file holds everything:

```json
{
  "global": {
    "data_root":     "/path/to/raw/recordings",
    "analysis_root": "./data/analysis",
    "figure_root":   "./output/figures"
  },
  "tasks": {
    "preprocessing":  { "bandpass_freq_min": 300, "bandpass_freq_max": 3000, ... },
    "sorting":        { "sorter": "kilosort4", ... },
    "auto_merge":     { "enabled": false, ... },
    "analyzer":       { ... },
    "auto_curation":  { "presence_ratio_min": 0.75, ... },
    "burst_detection":           { ... },
    "iterative_burst_detection": { ... },
    "plate_viewer":              { ... }
  }
}
```

Edit it via the **Settings** tab in the dashboard (each field is validated against the task's `ParamSpec`), or hand-edit against `config/pipeline_config.example.json`. The parity between every task's `params_schema()` and `default_params()` is enforced by `tests/test_params_schema.py`.

## Pipeline flow

```
raw → preprocessing → sorting → (auto_merge) → analyzer → auto_curation → burst_detection
                                                                       ↘ iterative_burst_detection → plate_viewer
```

Design notes:

- `DatasetManager` and `PipelineManager` are **independent** — neither imports the other. The caller links them via `recording_key + "/" + well_id` (the compound `pipeline_key`).
- Each task snapshots its config when it starts running. `is_task_complete()` returns true only if status is `complete` **and** the snapshot equals the current config — so changing a task's config invalidates that task and everything downstream of it automatically.
- All cache writes (`experiment_cache.json`, `pipeline_cache.json`, output JSON) are atomic (tempfile + `os.replace`) to survive interrupted writes on NAS.

## Tests

```bash
conda run -n yuxin_mea pytest
```

~200 tests covering: dataset cache + scanning, every task's params schema and output paths, the burst detector accuracy/reference equivalence, the config-builder form rendering + nested-dict reconstruction, curation summary, synthetic spike-train validation, and `notebooks/v2` execution.

## Further reading

- **`AGENTS.md`** — exhaustive module/symbol map; the canonical reference. Start here when reading the code.
- **`doc/architecture.md`** — older architecture write-up. Naming is partly stale (pre-refactor `stages`/`stage_record` vs current `tasks`/`task_record`), but the dependency-transitivity and config-staleness sections still apply.
- **`TODO.md`** — current work-in-progress.

## Contact

SadeghLab internal — contact the maintainer for access to raw data and infrastructure.
