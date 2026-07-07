from .base_task import BaseAnalysisTask
from .base_plate_level_task import BasePlateLevelTask
from .config_provider import BaseConfigProvider, DummyConfigProvider
from .task_record import TaskStatus, TaskRecord
from .work_item import WorkItem
from .pipeline_entry import PipelineEntry
from .cache import BasePipelineCacheStore, JsonPipelineCacheStore
from .well_metadata import BaseWellMetadataProvider, DummyWellMetadataProvider
from .manager import PipelineManager

# Aggregate (scope-level) task infrastructure — additive; the per-well pipeline
# above is untouched. See aggregate_scheduler.AggregateScheduler.
from .scope import Scope, DIMENSION_NAMES
from .well_dims import WellDims, Member
from .aggregate_task import BaseAggregateTask
from .aggregate_cache import (
    AggregateInstanceRecord,
    BaseAggregateCacheStore,
    JsonAggregateCacheStore,
    instance_key,
)
from .aggregate_scheduler import AggregateScheduler, PlannedInstance

__all__ = [
    "BaseAnalysisTask",
    "BasePlateLevelTask",
    "BaseConfigProvider",
    "DummyConfigProvider",
    "TaskStatus",
    "TaskRecord",
    "WorkItem",
    "PipelineEntry",
    "BasePipelineCacheStore",
    "JsonPipelineCacheStore",
    "BaseWellMetadataProvider",
    "DummyWellMetadataProvider",
    "PipelineManager",
    # aggregate-task infrastructure
    "Scope",
    "DIMENSION_NAMES",
    "WellDims",
    "Member",
    "BaseAggregateTask",
    "AggregateInstanceRecord",
    "BaseAggregateCacheStore",
    "JsonAggregateCacheStore",
    "instance_key",
    "AggregateScheduler",
    "PlannedInstance",
]
