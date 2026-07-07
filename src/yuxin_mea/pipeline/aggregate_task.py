"""BaseAggregateTask: a task whose unit of work is a *set* of wells.

Unlike :class:`~yuxin_mea.pipeline.base_task.BaseAnalysisTask` (one well →
``run(recording_key, well_id, data_path, params)``), an aggregate task runs once
per :class:`~yuxin_mea.pipeline.scope.Scope` instance and receives the list of
:class:`~yuxin_mea.pipeline.well_dims.Member` wells whose dependency completed:
``run(scope_key, members, params)``.

Deliberately NOT a subclass of ``BaseAnalysisTask`` — that keeps aggregate tasks
out of everything that iterates ``TASK_CLASSES`` (the per-well registry, its
tests, the manager's ``_forward_deps``) so the per-well pipeline is untouched.
Aggregate tasks live in their own ``AGGREGATE_TASK_CLASSES`` tuple and are driven
by the separate :class:`~yuxin_mea.pipeline.aggregate_scheduler.AggregateScheduler`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .scope import Scope
from .well_dims import Member


class BaseAggregateTask(ABC):
    """ABC for scope-level aggregate tasks.

    Subclasses declare class attributes ``task_name`` (str), ``dependencies``
    (list of upstream task names — per-well or aggregate), and ``scope`` (a
    :class:`Scope`), then implement ``run``.
    """

    task_name: str
    dependencies: list[str]
    scope: Scope

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Mirror BaseAnalysisTask: only validate concrete tasks (those that
        # define run), so an intermediate ABC without run escapes the check.
        if "run" not in cls.__dict__:
            return
        if not isinstance(getattr(cls, "task_name", None), str) or not cls.task_name:
            raise TypeError(f"{cls.__name__} must define a non-empty str class attr 'task_name'.")
        if not isinstance(getattr(cls, "dependencies", None), list):
            raise TypeError(f"{cls.__name__} must define a list class attr 'dependencies'.")
        if not isinstance(getattr(cls, "scope", None), Scope):
            raise TypeError(f"{cls.__name__} must define a Scope class attr 'scope'.")

    # ------------------------------------------------------------------
    # Params (same contract as BaseAnalysisTask)
    # ------------------------------------------------------------------
    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def params_schema(cls) -> dict[str, Any]:
        return {}

    def resolve_params(self, config_params: dict[str, Any]) -> dict[str, Any]:
        return {**self.default_params(), **(config_params or {})}

    # ------------------------------------------------------------------
    # Output path — scoped, never collides with the per-well tree
    # ------------------------------------------------------------------
    @staticmethod
    def build_output_path(
        output_root: str | Path, task_name: str, scope_key: dict[str, str]
    ) -> Path:
        """``<output_root>/__agg__/<task_name>/<encoded_scope_key>/``."""
        return Path(output_root) / "__agg__" / task_name / Scope.encode_path(scope_key)

    # ------------------------------------------------------------------
    # The work
    # ------------------------------------------------------------------
    @abstractmethod
    def run(
        self,
        scope_key: dict[str, str],
        members: list[Member],
        params: dict[str, Any],
    ) -> Path:
        """Aggregate ``members`` (the wells whose dependency COMPLETED) into one
        artifact and return its on-disk path.
        """
