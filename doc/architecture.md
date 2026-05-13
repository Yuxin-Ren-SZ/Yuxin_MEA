# Yuxin_MEA Architecture

> Reflects the codebase after the 4-phase refactor (commit `8631fde`).
> Chinese version: [`architecture.zh.md`](architecture.zh.md).

## What this project is

Lab software for analyzing **multi-electrode array (MEA)** neural recordings. Full pipeline: raw `.h5` files from MaxWell hardware → preprocessing → spike sorting → unit curation → burst detection → visualization.

---

## Top-level layout

```
src/yuxin_mea/          ← installable Python package (pip install -e .)
├── dataset/            ← "what recordings exist?"
├── pipeline/           ← "how do we schedule tasks?"
├── tasks/              ← "what analysis steps run?"
├── config/             ← "what parameters do we use?"
├── analysis/           ← "low-level algorithms + plotting functions"
├── dashboard/          ← "browser UI"
└── cli/                ← "command-line entry points"

notebooks/v2/           ← runnable tutorials against the new API
tests/                  ← 236 tests (pytest)
config/                 ← example JSON config
AGENTS.md               ← code map for future AI assistants
pyproject.toml          ← package definition
```

---

## Core architectural idea: layers that compose but don't import each other

### Layer 1: `dataset/` — discovery

**Responsibility**: scan MaxWell recordings on disk; cache metadata.

```
dataset/
├── manager.py        DatasetManager  ← main entry point
├── entries.py        RecordingEntry, WellEntry  ← data structures
├── metadata.py       parses mxassay.metadata files
├── cache.py          JsonCacheStore  ← reads/writes experiment_cache.json
└── _mxassay_decoder.py  Qt-serialization-format decoder (implementation detail)
```

**Key invariants**:
- `RecordingEntry` is a **frozen dataclass** (`frozen=True`) — once scanned, its recording-level fields cannot change.
- `cache_key` = `"sample_id/date/plate_id/scan_type/run_id"` — this string is the system's primary key.
- Compound `well_id` = `"rec0000/well000"` — reflects the two-level structure inside the `.h5` file.

**Design notes**:
- **Why frozen?** Recording-level metadata (sample rate, channel map) is immutable after first observation. `frozen=True` enforces that at the Python level — any accidental field mutation crashes immediately instead of silently poisoning downstream analyses.
- **`cache_key` is the connective tissue.** Dataset, Pipeline, and Dashboard layers never import each other's classes. They exchange `cache_key` strings instead. This is "decoupling through IDs" in its textbook form.

---

### Layer 2: `pipeline/` — orchestration

**Responsibility**: track "which task has finished for which well"; expose a DAG-aware scheduler.

```
pipeline/
├── manager.py            PipelineManager  ← scheduler
├── base_task.py          BaseAnalysisTask  ← task base class (each Task subclasses)
├── task_record.py        TaskStatus (not_run/running/complete/failed) + TaskRecord
├── pipeline_entry.py     PipelineEntry: status container per (recording, well) pair
├── work_item.py          WorkItem: schedulable "to-do" triple
├── cache.py              JsonPipelineCacheStore  ← reads/writes pipeline_cache.json
├── config_provider.py    BaseConfigProvider interface
└── well_metadata.py      future per-well metadata provider interface
```

**Key invariants** (from `AGENTS.md`):
- `PipelineManager` and `DatasetManager` are **fully independent** — they never import each other.
- `TaskRecord.config` is **snapshotted** when status transitions to `running` ← this is the reproducibility hinge.
- `is_task_complete()` returns True only when "status == complete AND current config matches snapshot."
- At startup, every non-complete cached task is reset to `not_run`.

**Design notes**:
- **Config snapshot pattern**: when a task starts running, the parameters in effect at that moment are frozen into `TaskRecord.config`. Next time the user asks "is this well done?", the system compares current config vs. the snapshot. If they differ, the task is considered stale even if its stored status is `complete`. This makes "rerun or skip" automatic — the user doesn't have to remember what they changed.
- **Task dependencies live on the class** (`dependencies: list[str]`), not in the scheduler. Which means `PipelineManager` knows nothing about "business" — it just reads class attributes. Adding a new task doesn't touch the scheduler.

---

### Layer 3: `tasks/` — business logic

Every task is a `BaseAnalysisTask` subclass with four core methods:

```python
class SortingTask(BaseAnalysisTask):
    task_name = "sorting"                    # ← unique identifier
    dependencies = ["preprocessing"]         # ← upstream DAG

    @classmethod
    def default_params(cls) -> dict:         # ← default values
        return {"sorter": "kilosort4", ...}

    @classmethod
    def params_schema(cls) -> dict[str, ParamSpec]:   # ← Phase 3: form schema
        return {"sorter": ParamSpec("str", "kilosort4",
                                    choices=["kilosort4", ...]), ...}

    def run(self, recording_key, well_id, data_path, params) -> Path:
        ...                                  # ← actual work
```

Current 7 tasks (DAG order):

```
preprocessing → sorting → auto_merge → analyzer → auto_curation → burst_detection
                                                                ↘
                                                                 iterative_burst_detection
```

(Phase 5 removed `plate_viewer` from the pipeline: visualization isn't
processing. It lives in the dashboard now — see Layer 6.)

---

### Layer 4: `config/` — configuration

```
config/
├── manager.py    ConfigManager  ← JSON config load/save
├── schema.py     ParamSpec dataclass + validate_value()  ← Phase 3
└── globals.py    GLOBALS_SCHEMA: data_root / analysis_root / figure_root
```

**Config file shape** (`pipeline_config.json`):

```json
{
  "global": {
    "data_root": "/path/to/raw",
    "analysis_root": "/path/to/analysis",
    "figure_root": "/path/to/figures"
  },
  "tasks": {
    "preprocessing": {"bandpass_freq_min": 300, ...},
    "sorting": {"sorter": "kilosort4", ...}
  }
}
```

**Design notes**:
- **`default_params()` and `params_schema()` must have identical key sets** — enforced by `tests/test_params_schema.py`. This is deliberately *two* sources of truth, kept in sync by a test. Why? Because `default_params()` returns values, while `params_schema()` returns **types + validation rules + UI hints**. Splitting them lets task classes stay free of any `ParamSpec` dependency, while still rendering as Dashboard forms.
- **The dashboard does NOT `import DatasetManager`** — it reads `JsonCacheStore` directly. Why? Because `DatasetManager.__init__` scans the disk and mutates the cache. The dashboard is read-only; constructing a Manager breaks the "read-only" promise.

---

### Layer 5: `analysis/` — pure algorithms

```
analysis/
├── burst_detector.py              compute_network_bursts (classical detection)
├── iterative_burst_detector.py    compute_iterative_bursts (iterative detection)
├── burst_output.py                BurstResults → pickle/parquet
├── plate_raster_synchrony.py      Plotly figures for multi-well visualization
├── burst_diagnostic.py            ← Phase 2b: diagnostic-dashboard figures + run_batch + caching
├── curation_summary.py            ← Phase 4: curation result aggregation
├── synthetic_validation.py        ← Phase 4: synthetic spike trains + GT scoring
└── (load_plate_data is a public function inside plate_raster_synchrony.py — Phase 5)
```

**Hard rule**: nothing in `analysis/` may import Dash, Pipeline, Dataset, or Config — only numpy / scipy / sklearn / plotly. This lets the algorithms be used standalone in Jupyter or tests.

---

### Layer 6: `dashboard/` — browser UI

```
dashboard/
├── cli.py                  ← yuxin-mea-dashboard command-line entry
├── app.py                  ← build_app(config_path) -> Dash
├── data.py                 ← read-only cache file loaders
├── components/
│   ├── layout.py           ← top navbar + page_container
│   └── form_builder.py     ← ParamSpec → Dash widget renderer
└── pages/                  ← Dash 4.x multipage (pages_folder auto-discovery)
    ├── home.py             /                  (order=0)
    ├── recordings.py       /recordings        (order=1)
    ├── pipeline.py         /pipeline          (order=2)
    ├── plate_viewer.py     /plate-viewer      (order=3) ← Phase 5: was a task; now a page
    ├── burst_diagnostic.py /burst-diagnostic  (order=4)
    └── settings.py         /settings          (order=10)  ← config editor
```

**How to launch**:

```bash
yuxin-mea-dashboard --config pipeline_config.json
# open http://127.0.0.1:8050
```

If the config file doesn't exist, the dashboard launches in **"config-only mode"** — data pages show a "no config yet" banner, but the Settings page is fully usable; the first Save creates the file.

**Design notes**:
- **Pattern-matched IDs `{"form": "...", "field": "..."}`** are what let one set of callbacks in `settings.py` handle 8 different task forms. Dash's `ALL` wildcard matches every component with the same pattern shape — much cleaner than writing 8 near-identical callbacks.
- **`app.server.config["YUXIN_MEA"]`** is the cross-page state stash. Flask's `config` dict is shared across all requests; Dash's `dcc.Store` is per-session. When you need "one config visible to every callback in the app", stashing on the server config is the idiom.

---

## Data flow: one analysis end-to-end

```
raw .h5 files
   │
   ↓  (DatasetManager scans)
experiment_cache.json
   │
   ↓  (user fills config via Dashboard Settings)
pipeline_config.json
   │
   ↓  (PipelineManager.add_well + register_task)
pipeline_cache.json   ← per-(recording, well) task status
   │
   ↓  (task.run(...) loop)
<analysis_root>/preprocessed_data/<rec>/<well>/preprocessed.zarr
<analysis_root>/spikesorted_data/<rec>/<well>/...
<analysis_root>/auto_merge_data/<rec>/<well>/...
<analysis_root>/analyzer_data/<rec>/<well>/...
<analysis_root>/curation_data/<rec>/<well>/quality_metrics.pkl + curated_spike_times.npy
<analysis_root>/burst_detection_data/<rec>/<well>/...
   │
   ↓  (Dashboard reads these read-only)
Recordings / Pipeline / Burst diagnostic pages in the browser
```

---

## How to use the system

### Path A: browser-only (recommended for non-technical users)

```bash
# First time: create the config file
yuxin-mea-dashboard --config pipeline_config.json
# → open browser → Settings → fill in data_root/analysis_root/figure_root + per-task params → Save

# Day-to-day: monitor the pipeline
yuxin-mea-dashboard --config pipeline_config.json
# → Recordings page: which recordings exist
# → Pipeline page: per-task status with color coding (green/blue/red/gray)
# → Burst diagnostic page: interactive exploration of burst detection results
```

### Path B: notebook-based (recommended for research)

The 8 notebooks in `notebooks/v2/` are the new "tutorial + workflow" set:

```
00_full_pipeline.ipynb       ← run the whole pipeline (all tasks)
01_si_preprocessing.ipynb    ← single step: preprocessing
03_auto_merge.ipynb          ← single step: auto-merge
04_analyzer.ipynb            ← single step: analyzer
05_auto_curation.ipynb       ← single step: auto-curation + curation_summary aggregation
06_..._synthetic_validation  ← burst detector validation against synthetic ground truth
01_plate_viewer / 02_generate_default_params  ← redirects to Dashboard
```

Every notebook follows the same template (`03_auto_merge.ipynb` is the canonical example):

1. Imports (always from `yuxin_mea.*`)
2. Load `ConfigManager` (write a template and stop if no config exists)
3. Scan recordings
4. Register tasks + add wells
5. Status overview table
6. Main loop: `get_next_task` → `task.run()` → `update_status`
7. Final status report

---

## How to extend: four common scenarios

### Scenario 1: add a new Task

The most common extension. Full steps:

```python
# 1. In src/yuxin_mea/tasks/my_new_task.py:
from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask

class MyNewTask(BaseAnalysisTask):
    task_name = "my_new_task"
    dependencies = ["analyzer"]               # ← upstream task names

    @classmethod
    def default_params(cls):
        return {"my_param": 42, "output_root": "./my_data"}

    @classmethod
    def params_schema(cls):                   # ← Dashboard auto-renders this
        return {
            "my_param": ParamSpec("int", 42, "what it does", min=0),
            "output_root": ParamSpec("path", "./my_data", "where output goes"),
        }

    def run(self, recording_key, well_id, data_path, params):
        p = self.resolve_params(params)       # ← merge with defaults
        # ... actual logic ...
        return output_path                    # ← must return the output path
```

```python
# 2. In src/yuxin_mea/tasks/__init__.py, add one line:
from .my_new_task import MyNewTask

# 3. In dashboard/pages/settings.py, append it to _TASK_CLASSES
```

**Auto-test coverage**: `test_params_schema.py` is parametrized — your new task gets the "schema vs default_params key parity" check automatically.

**Design notes**:
- **You almost never touch `pipeline/manager.py`.** That's the payoff of base-class + class-attribute-declared dependencies — the scheduler reads `task_name` and `dependencies` and that's enough. Adding a task is a local change (one new file + two import lines).
- **Calling `resolve_params()` at the top of `run()`** is the convention. It merges JSON user values over `default_params()`; JSON wins. This lets users put only the values they want to override in the config — everything else comes from defaults.

### Scenario 2: add a new analysis function

Put it under `src/yuxin_mea/analysis/`:

```python
# src/yuxin_mea/analysis/my_algorithm.py
import numpy as np

def detect_my_thing(spike_times: dict[str, np.ndarray]) -> dict:
    """Pure algorithm — no Dash / Pipeline dependencies."""
    ...
```

Then:
- Import it in a `notebooks/v2/` notebook to demonstrate usage.
- Write `tests/test_my_algorithm.py` if there's a meaningful input/output contract.
- If it should integrate into the pipeline, wrap it in a Task (see Scenario 1).

**Critical discipline**: `analysis/` modules **never import Dash / Pipeline / Dataset / Config**. This lets the algorithm be used outside the entire dashboard stack — pure Jupyter, scripts, etc.

### Scenario 3: add a new Dashboard page

```python
# src/yuxin_mea/dashboard/pages/my_page.py
import dash
from dash import callback, Input, Output, html

dash.register_page(__name__, path="/my-page", name="My Page", order=5)

layout = html.Div([html.H2("My Page"), html.Div(id="my-content")])

@callback(Output("my-content", "children"), Input("my-content", "id"))
def _render(_id):
    return "Hello!"
```

Dash 4.x's `pages_folder="pages"` auto-discovery picks it up — the top navbar grows a new link without any other code change.

### Scenario 4: extend the config schema (add a global)

```python
# Add one line in src/yuxin_mea/config/globals.py:
GLOBALS_SCHEMA = {
    "data_root": ParamSpec(...),
    "analysis_root": ParamSpec(...),
    "figure_root": ParamSpec(...),
    "my_new_global": ParamSpec("path", "", "what it's for"),   # ← here
}
```

The Dashboard Settings page's Globals tab automatically grows a new field. Other code can read it via `cm.get_global("my_new_global")` immediately.

---

## Testing discipline

236 tests. Four categories:

1. **Business tests** (carried over from pre-refactor) — verify each manager/task behavior.
2. **Architectural invariant tests** (added during the refactor) — guard cross-phase contracts:
   - `test_params_schema.py`: schema/default key parity.
   - `test_notebooks_v2.py`: no notebook may import old package names.
   - `test_config_builder.py`: nested-dict forms don't lose fields on save.
3. **Algorithm tests** — `synthetic_validation.py` provides synthetic data + scoring so burst detectors have ground-truth tests.
4. **Dashboard smoke tests** — `build_app(missing_path)` doesn't crash; all pages register.

**Hard rule**: **no test calls `app.run()`**. `app.run()` blocks the entire test process. Every test only constructs the app, asserts on attributes, and exits.

---

## Where to find documentation

- **`AGENTS.md`** (473 lines): code map for future AI assistants. One bullet per public symbol per module. The single source of truth for "what's where."
- **Each task class's docstring** explains the algorithm briefly.
- **`config/pipeline_config.example.json`**: drop-in config example, matches the current schema exactly.
- **`~/.claude/plans/please-review-the-codebase-goofy-shannon.md`**: full multi-phase refactor design doc (per-commit decision history).

---

## Summary: how the architecture is framed

The system's **core abstraction** is three independent subsystems linked by string IDs:

```
DatasetManager  ←(cache_key)→  PipelineManager  ←(task_name)→  TaskRegistry
       ↑                              ↑                              ↑
       │                              │                              │
       └──── ConfigManager ───────────┴───── Dashboard (read-only) ──┘
```

- **No God object** — there's no "main controller" that manages everything.
- **No cyclic dependencies** — the dependency graph is a single-direction DAG (`analysis` depends on nothing; `tasks` depend on `pipeline + config + analysis`; `dashboard` depends on everything; `cli` depends on `dashboard`).
- **Invariants are guarded by tests** — every architectural promise ("schema matches default", "notebooks don't import old packages", "dashboard is read-only") has a corresponding test.
- **Extension points live at the edges** — new task = one new file; new page = one new file; new global = one line. Core schedulers and managers almost never change.

This shape suits lab software well — the business (new algorithms, new analysis steps) changes frequently, while the skeleton ("how do we know what recordings exist", "how do we know what's been processed") needs to stay stable. Put the volatile parts in the edge files (task classes, page modules); put the stable parts in the managers and schemas.
