"""Invariants on AGGREGATE_TASK_CLASSES (parallel to test_task_classes_registry)."""

from yuxin_mea.pipeline.aggregate_task import BaseAggregateTask
from yuxin_mea.pipeline.scope import Scope
from yuxin_mea.tasks import (
    AGGREGATE_TASK_CLASSES,
    TASK_CLASSES,
    UNIFIED_AGGREGATE_TASK_CLASSES,
)


def test_nonempty_tuple():
    assert isinstance(AGGREGATE_TASK_CLASSES, tuple)
    assert len(AGGREGATE_TASK_CLASSES) >= 1


def test_all_are_aggregate_tasks():
    for cls in AGGREGATE_TASK_CLASSES:
        assert issubclass(cls, BaseAggregateTask)
        assert isinstance(cls.task_name, str) and cls.task_name
        assert isinstance(cls.dependencies, list)
        assert isinstance(cls.scope, Scope)


def test_unique_names():
    names = [c.task_name for c in AGGREGATE_TASK_CLASSES]
    assert len(names) == len(set(names))


def test_disjoint_from_per_well_registry():
    """Aggregate tasks must NOT be per-well tasks (keeps the per-well pipeline
    and its registry test unaffected)."""
    per_well = {c.task_name for c in TASK_CLASSES}
    agg = {c.task_name for c in AGGREGATE_TASK_CLASSES}
    assert per_well.isdisjoint(agg)
    for cls in AGGREGATE_TASK_CLASSES:
        assert cls not in TASK_CLASSES


def test_dependency_order_topological_among_aggregates():
    """If an aggregate task depends on another aggregate task, the dependency
    must appear earlier in the tuple."""
    seen: set[str] = set()
    agg_names = {c.task_name for c in AGGREGATE_TASK_CLASSES}
    for cls in AGGREGATE_TASK_CLASSES:
        for dep in cls.dependencies:
            if dep in agg_names:
                assert dep in seen, f"{cls.task_name} depends on {dep} declared later"
        seen.add(cls.task_name)


def test_s1_registry_has_no_aggregate_to_aggregate_dep():
    """S1's AggregateScheduler raises on aggregate→aggregate deps; keep the S1
    registry free of them until S1 is retired."""
    agg_names = {c.task_name for c in AGGREGATE_TASK_CLASSES}
    for cls in AGGREGATE_TASK_CLASSES:
        assert agg_names.isdisjoint(cls.dependencies), (
            f"{cls.task_name} has an aggregate dependency — belongs only in "
            "UNIFIED_AGGREGATE_TASK_CLASSES"
        )


def test_unified_registry_superset_and_topological():
    """UNIFIED_AGGREGATE_TASK_CLASSES extends the S1 registry and stays topological,
    and it carries at least one genuine aggregate→aggregate dependency (the
    containment primitive's consumer)."""
    assert set(AGGREGATE_TASK_CLASSES) <= set(UNIFIED_AGGREGATE_TASK_CLASSES)
    names = [c.task_name for c in UNIFIED_AGGREGATE_TASK_CLASSES]
    assert len(names) == len(set(names))
    agg_names = set(names)
    seen: set[str] = set()
    has_agg_dep = False
    for cls in UNIFIED_AGGREGATE_TASK_CLASSES:
        for dep in cls.dependencies:
            if dep in agg_names:
                has_agg_dep = True
                assert dep in seen, f"{cls.task_name} depends on {dep} declared later"
        seen.add(cls.task_name)
    assert has_agg_dep, "unified registry must exercise an aggregate→aggregate dep"
