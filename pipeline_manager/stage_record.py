from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Built-in stage name constants
# ---------------------------------------------------------------------------

STAGE_PREPROCESSING = "preprocessing"
STAGE_SORTING       = "sorting"
STAGE_CURATION      = "curation"   # optional — insert between sorting and analyzer if used
STAGE_ANALYZER      = "analyzer"


# Default immediate-dependency graph for built-in stages.
# Analyzer depends on sorting by default.  Callers that run curation before
# the analyzer should override: mark_stage_running(..., depends_on=[STAGE_CURATION]).
BUILTIN_DEPS: dict[str, list[str]] = {
    STAGE_PREPROCESSING: [],
    STAGE_SORTING:       [STAGE_PREPROCESSING],
    STAGE_CURATION:      [STAGE_SORTING],
    STAGE_ANALYZER:      [STAGE_SORTING],
}


# ---------------------------------------------------------------------------
# Stage status
# ---------------------------------------------------------------------------

class StageStatus:
    NOT_RUN  = "not_run"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"


# ---------------------------------------------------------------------------
# StageRecord
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    status:       str            # one of StageStatus constants
    dependencies: list[str]      # immediate upstream stage names (stored in cache)
    output_path:  Path | None    # directory / file produced by this stage
    last_updated: float | None   # POSIX timestamp of most recent status change
    config:       dict[str, Any] # config snapshot frozen at mark_stage_running() time
    error:        str | None     # most recent error message when status == FAILED
