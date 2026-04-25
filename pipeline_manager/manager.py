from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .cache_store import BasePipelineCacheStore, JsonPipelineCacheStore
from .config_provider import BaseConfigProvider, DummyConfigProvider
from .pipeline_entry import PipelineEntry
from .stage_record import BUILTIN_DEPS, StageRecord, StageStatus
from .well_metadata import BaseWellMetadataProvider, DummyWellMetadataProvider

logger = logging.getLogger(__name__)


class PipelineManager:
    """Tracks per-well analysis progress across an open set of pipeline stages.

    Each (recording_key, well_id) pair has one PipelineEntry.  Stages within
    an entry are keyed by plain strings; built-in names live in stage_record.py,
    custom analysis stages are arbitrary strings supplied by the caller.

    is_stage_complete() is the canonical "can I skip this stage?" test: it returns
    True only when both the status is COMPLETE *and* the config frozen at last run
    matches the current config returned by the injected ConfigProvider.

    is_stage_ready() checks all immediate dependencies pass is_stage_complete(),
    giving transitive readiness for free.
    """

    def __init__(
        self,
        analysis_dir:           Path,
        config_provider:        BaseConfigProvider | None = None,
        well_metadata_provider: BaseWellMetadataProvider | None = None,
        cache_store:            BasePipelineCacheStore | None = None,
    ) -> None:
        self._analysis_dir          = Path(analysis_dir)
        self._config_provider       = config_provider or DummyConfigProvider()
        self._well_meta_provider    = well_metadata_provider or DummyWellMetadataProvider()
        self._store                 = cache_store or JsonPipelineCacheStore(self._analysis_dir)
        self._cache: dict[str, PipelineEntry] = {}

        self._initialise()

    # ------------------------------------------------------------------
    # Public API — entry management
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[PipelineEntry]:
        return list(self._cache.values())

    def get_entry(self, recording_key: str, well_id: str) -> PipelineEntry | None:
        return self._cache.get(f"{recording_key}/{well_id}")

    def get_or_create_entry(self, recording_key: str, well_id: str) -> PipelineEntry:
        key = f"{recording_key}/{well_id}"
        if key not in self._cache:
            self._cache[key] = PipelineEntry(
                recording_key=recording_key,
                well_id=well_id,
                created_at=time.time(),
                stages={},
            )
            self._store.save(self._cache)
        return self._cache[key]

    def get_entries_for_recording(self, recording_key: str) -> list[PipelineEntry]:
        prefix = f"{recording_key}/"
        return [e for k, e in self._cache.items() if k.startswith(prefix)]

    def get_wells(self, recording_key: str) -> list[str]:
        return [e.well_id for e in self.get_entries_for_recording(recording_key)]

    # ------------------------------------------------------------------
    # Public API — stage lifecycle
    # ------------------------------------------------------------------

    def mark_stage_running(
        self,
        recording_key: str,
        well_id:       str,
        stage_name:    str,
        depends_on:    list[str] | None = None,
    ) -> None:
        """Begin a stage: snapshot current config, set status to RUNNING, save.

        depends_on: immediate upstream stage names. None → use BUILTIN_DEPS if
        the stage name is known there, else [].
        """
        entry = self.get_or_create_entry(recording_key, well_id)

        resolved_deps = (
            depends_on
            if depends_on is not None
            else BUILTIN_DEPS.get(stage_name, [])
        )

        # Config is snapshotted NOW and frozen — never refetched.
        config_snapshot = self._config_provider.get_config(stage_name, recording_key, well_id)

        existing = entry.stages.get(stage_name)
        entry.stages[stage_name] = StageRecord(
            status=StageStatus.RUNNING,
            dependencies=resolved_deps,
            output_path=existing.output_path if existing else None,
            last_updated=time.time(),
            config=config_snapshot,
            error=None,
        )
        self._store.save(self._cache)
        logger.info("Stage %s/%s/%s → RUNNING", recording_key, well_id, stage_name)

    def mark_stage_complete(
        self,
        recording_key: str,
        well_id:       str,
        stage_name:    str,
        output_path:   Path,
    ) -> None:
        entry = self._require_entry(recording_key, well_id, stage_name)
        s = entry.stages[stage_name]
        s.status       = StageStatus.COMPLETE
        s.output_path  = Path(output_path)
        s.last_updated = time.time()
        s.error        = None
        self._store.save(self._cache)
        logger.info("Stage %s/%s/%s → COMPLETE", recording_key, well_id, stage_name)

    def mark_stage_failed(
        self,
        recording_key: str,
        well_id:       str,
        stage_name:    str,
        error:         str,
    ) -> None:
        entry = self._require_entry(recording_key, well_id, stage_name)
        s = entry.stages[stage_name]
        s.status       = StageStatus.FAILED
        s.last_updated = time.time()
        s.error        = error
        self._store.save(self._cache)
        logger.warning("Stage %s/%s/%s → FAILED: %s", recording_key, well_id, stage_name, error)

    def reset_stage(self, recording_key: str, well_id: str, stage_name: str) -> None:
        """Set a stage back to NOT_RUN, clearing output_path, config, and error."""
        entry = self._require_entry(recording_key, well_id, stage_name)
        s = entry.stages[stage_name]
        s.status       = StageStatus.NOT_RUN
        s.output_path  = None
        s.last_updated = time.time()
        s.config       = {}
        s.error        = None
        self._store.save(self._cache)
        logger.info("Stage %s/%s/%s → NOT_RUN (reset)", recording_key, well_id, stage_name)

    # ------------------------------------------------------------------
    # Public API — stage queries
    # ------------------------------------------------------------------

    def get_stage(
        self, recording_key: str, well_id: str, stage_name: str
    ) -> StageRecord | None:
        entry = self.get_entry(recording_key, well_id)
        if entry is None:
            return None
        return entry.stages.get(stage_name)

    def is_stage_complete(
        self, recording_key: str, well_id: str, stage_name: str
    ) -> bool:
        """True iff status == COMPLETE and the frozen config matches current config.

        A mismatch means the stage completed with different parameters and must be
        re-run before downstream stages are considered ready.
        """
        stage = self.get_stage(recording_key, well_id, stage_name)
        if stage is None or stage.status != StageStatus.COMPLETE:
            return False
        current = self._config_provider.get_config(stage_name, recording_key, well_id)
        return stage.config == current

    def is_stage_ready(
        self, recording_key: str, well_id: str, stage_name: str
    ) -> bool:
        """True iff all immediate dependencies pass is_stage_complete().

        Transitivity is automatic: each dep's is_stage_complete also checks its
        own config, so a stale ancestor propagates readiness = False upward.
        """
        stage = self.get_stage(recording_key, well_id, stage_name)
        deps = stage.dependencies if stage else BUILTIN_DEPS.get(stage_name, [])
        return all(self.is_stage_complete(recording_key, well_id, dep) for dep in deps)

    # ------------------------------------------------------------------
    # Public API — well metadata
    # ------------------------------------------------------------------

    def get_well_metadata(self, recording_key: str, well_id: str) -> dict[str, Any]:
        """Delegate to the injected WellMetadataProvider (dummy until implemented)."""
        return self._well_meta_provider.get_metadata(recording_key, well_id)

    # ------------------------------------------------------------------
    # Public API — refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the cache from disk, then validate output_paths for complete stages.

        Missing paths are logged as warnings; stage status is NOT changed automatically
        (resetting a stage is an explicit action requiring user intent).
        """
        self._cache = self._store.load()
        logger.info("Reloaded pipeline cache: %d entries", len(self._cache))
        self._validate_output_paths()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _initialise(self) -> None:
        self._cache = self._store.load()
        logger.info(
            "Loaded %d pipeline entries from cache.", len(self._cache)
        )
        self._validate_output_paths()

    def _validate_output_paths(self) -> None:
        for entry in self._cache.values():
            for stage_name, stage in entry.stages.items():
                if (
                    stage.status == StageStatus.COMPLETE
                    and stage.output_path is not None
                    and not stage.output_path.exists()
                ):
                    logger.warning(
                        "Output path for %s/%s stage '%s' no longer exists: %s",
                        entry.recording_key,
                        entry.well_id,
                        stage_name,
                        stage.output_path,
                    )

    def _require_entry(
        self, recording_key: str, well_id: str, stage_name: str
    ) -> PipelineEntry:
        """Return the entry if it exists and already has the stage; raise otherwise."""
        entry = self.get_entry(recording_key, well_id)
        if entry is None:
            raise KeyError(
                f"No pipeline entry for recording={recording_key!r}, well={well_id!r}. "
                "Call mark_stage_running() first."
            )
        if stage_name not in entry.stages:
            raise KeyError(
                f"Stage {stage_name!r} not found in entry for "
                f"{recording_key!r}/{well_id!r}. "
                "Call mark_stage_running() first."
            )
        return entry
