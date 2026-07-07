"""Scope: the aggregation level of a task.

A task's Scope declares which wells define ONE task instance. Wells are grouped
by a tuple of dimensions (``group_by``); wells sharing the same group-by values
form one bucket = one instance. An optional ``where`` predicate filters which
wells participate at all.

Well-level tasks (the existing per-well pipeline) are the finest scope:
``group_by=("recording_key", "rec_name", "well_id")``. Every coarser level
(recording / sample / group / longitudinal-well / arbitrary compound) is just a
shorter group_by tuple, optionally with a filter. See ``WellDims`` for the
dimensions a scope may group/filter on.

This module has NO dependency on the per-well scheduler — it is a pure data
abstraction shared by S1 (the additive AggregateScheduler) and reusable verbatim
by the future S2 unified scheduler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .well_dims import WellDims

# Every dimension a scope may group_by / filter on. Must match WellDims fields.
DIMENSION_NAMES: tuple[str, ...] = (
    "recording_key",
    "sample_id",
    "date",
    "plate_id",
    "scan_type",
    "run_id",
    "rec_name",
    "well_id",
    "well_uid",
    "groupname",
)

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str) -> str:
    """Make a value safe for a single filesystem path segment.

    e.g. ``"CX138/260327/T003346/Network/000019"`` -> ``"CX138-260327-...-000019"``,
    ``"CX138|T003346|well000"`` -> ``"CX138-T003346-well000"``.
    """
    return _SLUG_RE.sub("-", str(value)).strip("-") or "_"


@dataclass(frozen=True)
class Scope:
    """The aggregation level of a task.

    Args:
        name: human label (also used in output paths / logs), e.g. ``"recording"``.
        group_by: subset of :data:`DIMENSION_NAMES`; wells sharing these values
            form one instance. Empty tuple = one global instance.
        where: optional predicate over a :class:`WellDims`; wells for which it
            returns False do not participate in any instance of this task.
    """

    name: str
    group_by: tuple[str, ...]
    where: Callable[[WellDims], bool] | None = None

    def __post_init__(self) -> None:
        bad = [d for d in self.group_by if d not in DIMENSION_NAMES]
        if bad:
            raise ValueError(
                f"Scope {self.name!r} has unknown group_by dimension(s) {bad}; "
                f"valid dimensions are {DIMENSION_NAMES}."
            )

    def matches(self, dims: WellDims) -> bool:
        """True if this well participates in the scope (passes ``where``)."""
        return True if self.where is None else bool(self.where(dims))

    def well_key(self, dims: WellDims) -> tuple:
        """The group-by value tuple identifying which instance a well belongs to."""
        return tuple(getattr(dims, d) for d in self.group_by)

    def scope_key_dict(self, dims: WellDims) -> dict[str, str]:
        """The instance identity as an ordered ``{dimension: value}`` mapping."""
        return {d: str(getattr(dims, d)) for d in self.group_by}

    @staticmethod
    def encode_path(scope_key: dict[str, str]) -> str:
        """Encode a scope-key dict into one filesystem-safe directory segment.

        e.g. ``{"sample_id": "CX138", "groupname": "NPH"}`` ->
        ``"sample_id=CX138__groupname=NPH"``. Order follows the dict (which
        :meth:`scope_key_dict` builds in group_by order). Empty key (global
        scope) -> ``"__all__"``.
        """
        if not scope_key:
            return "__all__"
        return "__".join(f"{k}={_slug(v)}" for k, v in scope_key.items())


# --------------------------------------------------------------------------- #
# Presets — freely composable; a "compound" task just builds Scope(...) inline.
# --------------------------------------------------------------------------- #
# NOTE: for a Network recording_key there are 4 rec_names × 6 physical wells = 24
# wells, so group_by=(recording_key,) with the Network filter yields exactly the
# plate's 24 wells. ActivityScan repeats wells across many rec_names, so the
# scan_type filter is what makes the "plate" preset well-defined.

SCOPE_RECORDING = Scope(
    "recording", ("recording_key",), where=lambda d: d.scan_type == "Network"
)
SCOPE_SAMPLE = Scope("sample", ("sample_id",))
SCOPE_SAMPLE_GROUP = Scope(
    "sample_group", ("sample_id", "groupname"),
    where=lambda d: d.groupname != "?",
)
SCOPE_SAMPLE_GROUP_RUN = Scope(
    "sample_group_run", ("sample_id", "groupname", "run_id"),
    where=lambda d: d.groupname != "?",
)
SCOPE_WELL_UID = Scope(
    "longitudinal_well", ("well_uid",), where=lambda d: d.scan_type == "Network"
)
