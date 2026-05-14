"""Sanity checks on `yuxin_mea.tasks.TASK_CLASSES` — the single source of
truth used by the dashboard's Settings page, the Run page, and the
`yuxin-mea-run` CLI."""

from __future__ import annotations

from yuxin_mea.pipeline import BaseAnalysisTask
from yuxin_mea.tasks import TASK_CLASSES


def test_task_classes_is_non_empty_tuple():
    assert isinstance(TASK_CLASSES, tuple)
    assert len(TASK_CLASSES) >= 1


def test_every_member_is_a_task_subclass():
    for cls in TASK_CLASSES:
        assert isinstance(cls, type)
        assert issubclass(cls, BaseAnalysisTask)
        assert isinstance(cls.task_name, str) and cls.task_name


def test_task_names_are_unique():
    names = [cls.task_name for cls in TASK_CLASSES]
    assert len(names) == len(set(names))


def test_dependency_order_is_topological():
    """Each task's dependencies must appear earlier in TASK_CLASSES so that
    PipelineManager.register_task() accepts the tuple in order.
    """
    seen: set[str] = set()
    for cls in TASK_CLASSES:
        for dep in cls.dependencies:
            assert dep in seen, (
                f"{cls.task_name} depends on {dep!r}, but {dep!r} appears "
                f"later in TASK_CLASSES (or is missing entirely)."
            )
        seen.add(cls.task_name)
