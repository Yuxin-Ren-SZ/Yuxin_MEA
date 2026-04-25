from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseConfigProvider(ABC):
    """Returns the *current* configuration for a given (stage, recording, well) triple.

    The dict returned here is compared against the config frozen in the cache to
    determine whether a completed stage is still current (i.e., config unchanged)
    or stale (i.e., must be re-run).
    """

    @abstractmethod
    def get_config(
        self,
        stage_name:    str,
        recording_key: str,
        well_id:       str,
    ) -> dict[str, Any]:
        """Return the active config dict for this stage + recording + well."""


class DummyConfigProvider(BaseConfigProvider):
    """Placeholder until the real config module is built. Always returns {}."""

    def get_config(
        self,
        stage_name:    str,
        recording_key: str,
        well_id:       str,
    ) -> dict[str, Any]:
        return {}
