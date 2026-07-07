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
from .scope import Scope, DIMENSION_NAMES, ScopeRelation, FINEST_GROUP_BY
from .well_dims import WellDims, Member
from .aggregate_task import BaseAggregateTask
from .aggregate_cache import (
    AggregateInstanceRecord,
    BaseAggregateCacheStore,
    JsonAggregateCacheStore,
    instance_key,
)
from .aggregate_scheduler import AggregateScheduler, PlannedInstance

# S2 unified instance model + store (additive alongside the two legacy caches).
from .scoped_instance import ScopedInstance, scoped_instance_key
from .instance_store import (
    InstanceStore,
    JsonInstanceStore,
    load_legacy_wells,
    load_legacy_aggregates,
)
from .unified_scheduler import (
    UnifiedScheduler,
    ScopeRegistry,
    WorkUnit,
    InstanceDecision,
)

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
    "ScopeRelation",
    "FINEST_GROUP_BY",
    "WellDims",
    "Member",
    "BaseAggregateTask",
    "AggregateInstanceRecord",
    "BaseAggregateCacheStore",
    "JsonAggregateCacheStore",
    "instance_key",
    "AggregateScheduler",
    "PlannedInstance",
    # S2 unified instance model + store
    "ScopedInstance",
    "scoped_instance_key",
    "InstanceStore",
    "JsonInstanceStore",
    "load_legacy_wells",
    "load_legacy_aggregates",
    "UnifiedScheduler",
    "ScopeRegistry",
    "WorkUnit",
    "InstanceDecision",
]
