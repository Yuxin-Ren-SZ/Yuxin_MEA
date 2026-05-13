"""Parity tests for every task's `params_schema()` against `default_params()`.

These tests catch drift: if a developer adds a key to `default_params()`
without updating `params_schema()` (or vice versa), the dashboard form
will silently miss the field or render an unmapped default. The parity
check is mechanical but cheap — every task is covered uniformly.
"""

from __future__ import annotations

import pytest

from yuxin_mea.config import ParamSpec, validate_value
from yuxin_mea.tasks import (
    AnalyzerTask,
    AutoCurationTask,
    AutoMergeTask,
    BurstDetectionTask,
    IterativeBurstDetectionTask,
    PlateViewerTask,
    PreprocessingTask,
    SortingTask,
)


ALL_TASKS = [
    PreprocessingTask,
    SortingTask,
    AutoMergeTask,
    AnalyzerTask,
    AutoCurationTask,
    BurstDetectionTask,
    IterativeBurstDetectionTask,
    PlateViewerTask,
]


@pytest.mark.parametrize("task_cls", ALL_TASKS, ids=lambda c: c.task_name)
def test_schema_keys_match_default_params(task_cls):
    """Schema and defaults must have identical key sets."""
    defaults = task_cls.default_params()
    schema = task_cls.params_schema()
    assert set(schema.keys()) == set(defaults.keys()), (
        f"{task_cls.__name__}: schema/defaults key drift. "
        f"In schema but not defaults: {set(schema) - set(defaults)}; "
        f"in defaults but not schema: {set(defaults) - set(schema)}"
    )


@pytest.mark.parametrize("task_cls", ALL_TASKS, ids=lambda c: c.task_name)
def test_schema_entries_are_param_specs(task_cls):
    schema = task_cls.params_schema()
    for name, spec in schema.items():
        assert isinstance(spec, ParamSpec), (
            f"{task_cls.__name__}.params_schema()[{name!r}] is not a ParamSpec"
        )


@pytest.mark.parametrize("task_cls", ALL_TASKS, ids=lambda c: c.task_name)
def test_each_default_satisfies_its_schema(task_cls):
    """Every default value must pass its own schema's validate_value()."""
    defaults = task_cls.default_params()
    schema = task_cls.params_schema()
    for name, spec in schema.items():
        try:
            validate_value(spec, defaults[name])
        except Exception as exc:
            pytest.fail(
                f"{task_cls.__name__}.default_params()[{name!r}] "
                f"({defaults[name]!r}) failed its schema: {exc}"
            )
