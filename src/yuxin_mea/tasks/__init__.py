from .analyzer import AnalyzerTask
from .auto_curation import AutoCurationTask
from .auto_merge import AutoMergeTask
from .burst_detection import BurstDetectionTask
from .cluster_overlay_tasks import (
    ClusterOverlayByRecordingTask,
    ClusterOverlayByWellTask,
    SampleOverlaySummaryTask,
)
from .ml_burst_detection import MLBurstDetectionTask
from .preprocessing import PreprocessingTask
from .sorting import SortingTask

# Canonical pipeline order — dashboards, the run CLI, and full-pipeline notebooks
# all read from this tuple so adding a new task only requires editing this file.
TASK_CLASSES = (
    PreprocessingTask,
    SortingTask,
    AutoMergeTask,
    AnalyzerTask,
    AutoCurationTask,
    BurstDetectionTask,
    MLBurstDetectionTask,
)

# Aggregate (scope-level) tasks run once per Scope instance, NOT per well — a
# SEPARATE registry driven by AggregateScheduler, deliberately kept out of
# TASK_CLASSES (and its registry test) so the per-well pipeline is untouched.
#
# NOTE: this S1 registry contains ONLY tasks that depend on per-well tasks. The
# S1 AggregateScheduler raises on aggregate→aggregate deps, so the containment
# task lives in UNIFIED_AGGREGATE_TASK_CLASSES below until S1 is retired (Phase 4).
AGGREGATE_TASK_CLASSES = (
    ClusterOverlayByWellTask,
    ClusterOverlayByRecordingTask,
)

# The S2 unified scheduler's registry: the S1 tasks PLUS the aggregate→aggregate
# tasks it alone can schedule. Order is topological (a task's aggregate
# dependencies precede it) so the containment DAG resolves. In Phase 4 this
# becomes the single registry and AGGREGATE_TASK_CLASSES is retired.
UNIFIED_AGGREGATE_TASK_CLASSES = AGGREGATE_TASK_CLASSES + (
    SampleOverlaySummaryTask,
)

__all__ = [
    "AnalyzerTask",
    "AutoCurationTask",
    "AutoMergeTask",
    "BurstDetectionTask",
    "ClusterOverlayByRecordingTask",
    "ClusterOverlayByWellTask",
    "SampleOverlaySummaryTask",
    "MLBurstDetectionTask",
    "PreprocessingTask",
    "SortingTask",
    "TASK_CLASSES",
    "AGGREGATE_TASK_CLASSES",
    "UNIFIED_AGGREGATE_TASK_CLASSES",
]
