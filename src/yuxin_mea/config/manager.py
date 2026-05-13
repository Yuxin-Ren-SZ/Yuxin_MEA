from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from pipeline_manager.config_provider import BaseConfigProvider

logger = logging.getLogger(__name__)


class ConfigManager(BaseConfigProvider):
    """Central configuration manager for the MEA analysis pipeline.

    Manages two categories of settings:

    * **Global settings** — environment-specific paths and values such as
      ``data_root``, ``analysis_root``, and ``preprocessing_output_root``.
    * **Per-task parameters** — analysis parameters for each named pipeline task
      (e.g. bandpass frequencies, sorter name, thread counts).

    The manager is a *pure loader/recorder*: it does not merge class-level
    defaults from task classes.  Those defaults live in
    :meth:`BaseAnalysisTask.default_params` and are applied by
    :meth:`BaseAnalysisTask.resolve_params` inside the task's own ``run()``
    method.

    Core workflow (no registration required)::

        cm = ConfigManager()
        cm.load("pipeline_config.json")
        pipeline_mgr = PipelineManager(analysis_dir, config_provider=cm)

    Optional template-generation workflow::

        cm = ConfigManager()
        cm.register_task(PreprocessingTask)   # seeds template with default_params()
        cm.set_global("data_root", "/path/to/raw")
        cm.generate_template("pipeline_config.json")
        # ← user edits the JSON file ←
        cm.load("pipeline_config.json")

    Config file schema::

        {
          "global": {
            "data_root":                 "/path/to/raw/data",
            "analysis_root":             "/path/to/analysis",
            "preprocessing_output_root": "/path/to/preprocessed"
          },
          "tasks": {
            "SI-Preprocessing": {
              "bandpass_freq_min": 300,
              "bandpass_freq_max": 3000
            },
            "sorting": {
              "sorter": "kilosort4"
            }
          }
        }
    """

    def __init__(self) -> None:
        # Programmatic seeds (used as template defaults; not returned by get_config)
        self._global_set: dict[str, Any]     = {}
        self._task_defaults: dict[str, dict] = {}

        # Loaded from JSON (these are the authoritative values)
        self._global_loaded: dict[str, Any]  = {}
        self._task_loaded:   dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Optional registration — seeds generate_template() only
    # ------------------------------------------------------------------

    def register_task(self, task_class: type) -> None:
        """Register a BaseAnalysisTask subclass to seed the config template.

        Reads ``task_class.default_params()`` and stores the result under
        ``task_class.task_name``.  This does **not** affect ``get_config()``
        or ``get_task_params()``; it only populates the template written by
        ``generate_template()``.

        Not required for ``load()`` or ``get_config()`` to work.
        """
        name = task_class.task_name
        self._task_defaults[name] = dict(task_class.default_params())
        logger.debug("Registered task %r for template generation.", name)

    def set_global(self, key: str, value: Any) -> None:
        """Set a global setting to seed the config template.

        Not required for ``load()`` or ``get_global()`` to work.
        """
        self._global_set[key] = value

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_global(self, key: str, default: Any = None) -> Any:
        """Return a global setting.

        Values from a loaded JSON file take precedence over those set via
        ``set_global()``.  Returns *default* if the key is absent in both.
        """
        return self._global_loaded.get(key, self._global_set.get(key, default))

    def get_task_params(self, task_name: str) -> dict:
        """Return the params for *task_name* from the loaded config file.

        Returns an empty dict if the task section is absent or no file has
        been loaded.  Class-level defaults (``BaseAnalysisTask.default_params``)
        are **not** included here — they are applied inside the task's own
        ``resolve_params()`` call.
        """
        return dict(self._task_loaded.get(task_name, {}))

    # ------------------------------------------------------------------
    # BaseConfigProvider interface
    # ------------------------------------------------------------------

    def get_config(self, task_name: str, recording_key: str, well_id: str) -> dict:
        """Return the params for *task_name* to be snapshotted by PipelineManager.

        Called automatically by ``PipelineManager.update_status()`` when a task
        transitions to RUNNING.  The returned dict is frozen into
        ``TaskRecord.config`` for reproducibility tracking.
        """
        return self.get_task_params(task_name)

    # ------------------------------------------------------------------
    # Template generation
    # ------------------------------------------------------------------

    def generate_template(self, path: Path | str) -> None:
        """Write a starter config JSON to *path*.

        The file contains:

        * A ``"global"`` section populated from ``set_global()`` calls
          (empty dict if none were made).
        * A ``"tasks"`` section populated from ``register_task()`` calls
          (empty dict if no tasks were registered).

        Raises :class:`FileExistsError` if *path* already exists — the
        template writer never overwrites an existing config to prevent
        accidental data loss.
        """
        path = Path(path)
        template: dict[str, Any] = {
            "global": dict(self._global_set),
            "tasks": {
                name: dict(defaults)
                for name, defaults in self._task_defaults.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        # "x" mode = exclusive creation; raises FileExistsError if file exists
        try:
            with path.open("x", encoding="utf-8") as fh:
                json.dump(template, fh, indent=2)
                fh.write("\n")
        except FileExistsError:
            raise FileExistsError(
                f"Config template already exists at {path}. "
                "Delete it first if you want a fresh template."
            )
        logger.info("Config template written to %s.", path)

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load(self, path: Path | str) -> None:
        """Load config from *path* (JSON).

        Populates ``_global_loaded`` and ``_task_loaded`` from the file's
        ``"global"`` and ``"tasks"`` sections respectively.  Any previously
        loaded values are replaced.

        Works regardless of whether tasks have been registered.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data: dict = json.load(fh)

        self._global_loaded = dict(data.get("global", {}))
        self._task_loaded   = {
            name: dict(params)
            for name, params in data.get("tasks", {}).items()
        }
        logger.info(
            "Loaded config from %s: %d global key(s), %d task section(s).",
            path,
            len(self._global_loaded),
            len(self._task_loaded),
        )

    def save(self, path: Path | str) -> None:
        """Write the currently loaded config to *path* (atomic write).

        Saves only the values from the last ``load()`` call — not the
        programmatic seeds from ``set_global()`` / ``register_task()``.
        Use ``generate_template()`` to write a template that includes those.

        Uses tempfile + os.replace for an atomic write so the file is never
        left in a partial state.
        """
        path = Path(path)
        payload: dict[str, Any] = {
            "global": dict(self._global_loaded),
            "tasks": {
                name: dict(params)
                for name, params in self._task_loaded.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config_tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                fh.write("\n")
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.info("Config saved to %s.", path)
