from __future__ import annotations

from abc import ABC, abstractmethod


class BaseConfigProvider(ABC):
    """Returns the current config dict for a (task, recording, well) triple.

    PipelineManager calls get_config() when a task transitions to RUNNING and
    freezes the result into TaskRecord.config. This snapshot is the reproducibility
    record: is_task_complete() returns True only when status is COMPLETE and the
    frozen config still matches what get_config() returns today.
    """

    @abstractmethod
    def get_config(self, task_name: str, recording_key: str, well_id: str) -> dict:
        """Return the current config dict for this (task, recording, well) triple."""


class DummyConfigProvider(BaseConfigProvider):
    """No-op placeholder. Always returns {}. Use during development or testing."""

    def get_config(self, task_name: str, recording_key: str, well_id: str) -> dict:
        return {}
