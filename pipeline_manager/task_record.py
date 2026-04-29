from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class TaskStatus:
    NOT_RUN  = "not_run"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"

    _ALL = {NOT_RUN, RUNNING, COMPLETE, FAILED}

    @classmethod
    def validate(cls, value: str) -> None:
        if value not in cls._ALL:
            raise ValueError(
                f"Invalid status {value!r}. Must be one of: {sorted(cls._ALL)}"
            )


@dataclass
class TaskRecord:
    status:       str           # TaskStatus constant
    dependencies: list[str]     # immediate upstream task names
    output_path:  Path | None   # output produced (set on complete)
    last_updated: float | None  # POSIX timestamp of last status change
    error:        str | None    # error message when status == FAILED
    config:       dict          = field(default_factory=dict)  # config snapshot at RUNNING
