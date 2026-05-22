from .analyzer import AnalyzerTask
from .auto_curation import AutoCurationTask
from .auto_merge import AutoMergeTask
from .burst_detection import BurstDetectionTask
from .iterative_burst_detection import IterativeBurstDetectionTask
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
    IterativeBurstDetectionTask,
    MLBurstDetectionTask,
)

__all__ = [
    "AnalyzerTask",
    "AutoCurationTask",
    "AutoMergeTask",
    "BurstDetectionTask",
    "IterativeBurstDetectionTask",
    "MLBurstDetectionTask",
    "PreprocessingTask",
    "SortingTask",
    "TASK_CLASSES",
]
