"""Shared per-callback constructors for DatasetManager / PipelineManager.

Mutating dashboard actions (queue wells, reset cascade) need a live manager;
read-only views keep using `data.py`'s pure cache loaders. We deliberately
*don't* cache the managers on the Flask app — each callback builds a fresh
instance so its in-memory state matches the on-disk cache at click time.
That sidesteps the `PipelineManager.__init__` stale-reset side-effect from
clobbering a worker that's writing the same cache concurrently.

Both constructors return `None` when the relevant global path isn't set in
the config (e.g. a freshly-initialized dashboard before Settings is filled
in) so callbacks can render an "unconfigured" message instead of crashing.
"""

from __future__ import annotations

from pathlib import Path

from flask import current_app

from yuxin_mea.config import ConfigManager
from yuxin_mea.dataset import DatasetManager
from yuxin_mea.pipeline import PipelineManager
from yuxin_mea.tasks import TASK_CLASSES


def _ctx() -> dict:
    return current_app.config.get("YUXIN_MEA", {})


def load_dataset_mgr() -> DatasetManager | None:
    ctx = _ctx()
    data_root = ctx.get("data_root")
    analysis_root = ctx.get("analysis_root")
    if not data_root or not analysis_root:
        return None
    return DatasetManager(Path(data_root), Path(analysis_root))


def load_pipeline_mgr() -> PipelineManager | None:
    """Build a fresh PipelineManager with TASK_CLASSES registered.

    Use this only for write actions (`add_well`, `refresh`). Read-only pages
    should use `dashboard.data.load_pipeline_df` instead — it skips the
    stale-reset that `PipelineManager.__init__` performs.
    """
    ctx = _ctx()
    analysis_root = ctx.get("analysis_root")
    config_path = ctx.get("config_path")
    if not analysis_root:
        return None

    cm = ConfigManager()
    for cls in TASK_CLASSES:
        cm.register_task(cls)
    if config_path is not None and Path(config_path).exists():
        cm.load(config_path)

    pm = PipelineManager(Path(analysis_root), config_provider=cm)
    for cls in TASK_CLASSES:
        try:
            pm.register_task(cls)
        except ValueError:
            # Re-registration on an existing cache is fine — the cache holds the
            # task already and `register_task` raises rather than no-op.
            pass
    return pm
