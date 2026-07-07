from .analyzer import AnalyzerTask
from .auto_curation import AutoCurationTask
from .auto_merge import AutoMergeTask
from .burst_detection import BurstDetectionTask
from .cluster_overlay_tasks import (
    ClusterOverlayByRecordingTask,
    ClusterOverlayByWellTask,
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
AGGREGATE_TASK_CLASSES = (
    ClusterOverlayByWellTask,
    ClusterOverlayByRecordingTask,
)

__all__ = [
    "AnalyzerTask",
    "AutoCurationTask",
    "AutoMergeTask",
    "BurstDetectionTask",
    "ClusterOverlayByRecordingTask",
    "ClusterOverlayByWellTask",
    "MLBurstDetectionTask",
    "PreprocessingTask",
    "SortingTask",
    "TASK_CLASSES",
    "AGGREGATE_TASK_CLASSES",
]
