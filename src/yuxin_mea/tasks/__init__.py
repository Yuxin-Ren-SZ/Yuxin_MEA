from .analyzer import AnalyzerTask
from .auto_curation import AutoCurationTask
from .auto_merge import AutoMergeTask
from .burst_detection import BurstDetectionTask
from .connectivity import ConnectivityTask
from .criticality import CriticalityTask
from .directed_connectivity import DirectedConnectivityTask
from .manual_curation import ManualCurationTask
from .ml_burst_detection import MLBurstDetectionTask
from .preprocessing import PreprocessingTask
from .sorting import SortingTask
from .spatial_map import SpatialMapTask

# Canonical pipeline order — dashboards, the run CLI, and full-pipeline notebooks
# all read from this tuple so adding a new task only requires editing this file.
# New tasks must follow the tasks they depend on (registration validates deps):
# spatial_map needs auto_curation + burst_detection; connectivity needs auto_curation.
TASK_CLASSES = (
    PreprocessingTask,
    SortingTask,
    AutoMergeTask,
    AnalyzerTask,
    AutoCurationTask,
    ManualCurationTask,
    BurstDetectionTask,
    MLBurstDetectionTask,
    SpatialMapTask,
    ConnectivityTask,
    CriticalityTask,
    DirectedConnectivityTask,
)

__all__ = [
    "AnalyzerTask",
    "AutoCurationTask",
    "AutoMergeTask",
    "BurstDetectionTask",
    "ConnectivityTask",
    "CriticalityTask",
    "DirectedConnectivityTask",
    "ManualCurationTask",
    "MLBurstDetectionTask",
    "PreprocessingTask",
    "SortingTask",
    "SpatialMapTask",
    "TASK_CLASSES",
]
