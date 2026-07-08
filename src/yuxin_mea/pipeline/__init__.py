from .base_task import BaseAnalysisTask
from .base_plate_level_task import BasePlateLevelTask
from .config_provider import BaseConfigProvider, DummyConfigProvider
from .task_record import TaskStatus, TaskRecord
from .work_item import WorkItem
from .pipeline_entry import PipelineEntry
from .cache import BasePipelineCacheStore, JsonPipelineCacheStore
from .well_metadata import BaseWellMetadataProvider, DummyWellMetadataProvider
from .manager import PipelineManager

# Scope/aggregate-task abstractions. `aggregate_cache` is retained for the finest
# layer's sibling format + one-time migration of any legacy aggregate_cache.json;
# scheduling is the S2 UnifiedScheduler (S1's AggregateScheduler was removed).
from .scope import Scope, DIMENSION_NAMES, ScopeRelation, FINEST_GROUP_BY
from .well_dims import WellDims, Member
from .aggregate_task import BaseAggregateTask
from .aggregate_cache import (
    AggregateInstanceRecord,
    BaseAggregateCacheStore,
    JsonAggregateCacheStore,
    instance_key,
)

# S2 unified instance model + store.
from .scoped_instance import ScopedInstance, scoped_instance_key
from .instance_store import (
    InstanceStore,
    JsonInstanceStore,
    LayeredInstanceStore,
    load_legacy_wells,
    load_legacy_aggregates,
    project_wells_to_entries,
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
    # S2 unified instance model + store
    "ScopedInstance",
    "scoped_instance_key",
    "InstanceStore",
    "JsonInstanceStore",
    "LayeredInstanceStore",
    "load_legacy_wells",
    "load_legacy_aggregates",
    "project_wells_to_entries",
    "UnifiedScheduler",
    "ScopeRegistry",
    "WorkUnit",
    "InstanceDecision",
]
