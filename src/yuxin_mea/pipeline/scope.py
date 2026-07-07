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
from enum import Enum
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

# The finest scope — a fully-identified physical well within one recording.
# Every per-well pipeline task lives here. Coarser scopes are shorter group_bys.
FINEST_GROUP_BY: tuple[str, ...] = ("recording_key", "rec_name", "well_id")

# Structural functional dependencies: a dimension "determines" (⊢) the dimensions
# derivable from it, exactly as WellDims.from_entry splits them (well_dims.py:41,55).
# This is the lattice the scope-containment relation reads — NOT literal name
# intersection, which is wrong (sample_id and recording_key share no name yet
# recording refines sample). Every dimension trivially determines itself.
DIMENSION_DETERMINES: dict[str, frozenset[str]] = {
    "recording_key": frozenset(
        {"recording_key", "sample_id", "date", "plate_id", "scan_type", "run_id"}
    ),
    "well_uid": frozenset({"well_uid", "sample_id", "plate_id", "well_id"}),
    "sample_id": frozenset({"sample_id"}),
    "date": frozenset({"date"}),
    "plate_id": frozenset({"plate_id"}),
    "scan_type": frozenset({"scan_type"}),
    "run_id": frozenset({"run_id"}),
    "rec_name": frozenset({"rec_name"}),
    "well_id": frozenset({"well_id"}),
    "groupname": frozenset({"groupname"}),
}


class ScopeRelation(Enum):
    """How a dependency scope relates to the downstream scope depending on it.

    SAME     — identical group_by (well→well fast path; exactly one dep instance).
    CONTAINS — the dependency is at least as fine as the downstream scope; each
               downstream instance's members map onto a subset of dep instances
               (aggregate→well, aggregate→coarser-aggregate).
    A relation that is neither (coarse-on-coarse, disjoint) is rejected with a
    ``ValueError`` rather than represented here.
    """

    SAME = "same"
    CONTAINS = "contains"


def _closure(group_by: tuple[str, ...]) -> frozenset[str]:
    """All dimensions the given group_by structurally determines.

    Special case: the finest group_by determines EVERY dimension — a fully
    identified well fixes ``well_uid`` (composite) and ``groupname`` (via the
    experiment-cache LUT), neither of which the pairwise splits reproduce.
    """
    if set(group_by) >= set(FINEST_GROUP_BY):
        return frozenset(DIMENSION_NAMES)
    out: set[str] = set()
    for d in group_by:
        out |= DIMENSION_DETERMINES.get(d, frozenset({d}))
    return frozenset(out)


def _partial_dims_from_key(key: dict[str, str]) -> dict[str, str]:
    """Expand a scope_key into every dimension it structurally determines.

    Applies the same splits as ``WellDims.from_entry`` so that, e.g., a key
    carrying ``recording_key`` also yields ``sample_id``/``plate_id``/…  Used only
    by :meth:`Scope.contains` (tests/introspection); the scheduler resolves
    dependency instances by well-set membership, never by parsing keys.
    """
    out = dict(key)
    if "recording_key" in key:
        parts = key["recording_key"].split("/")
        s, dt, pl, sc, rn = (parts + [""] * 5)[:5]
        out.setdefault("sample_id", s)
        out.setdefault("date", dt)
        out.setdefault("plate_id", pl)
        out.setdefault("scan_type", sc)
        out.setdefault("run_id", rn)
    if "well_uid" in key:
        parts = key["well_uid"].split("|")
        s, pl, wid = (parts + [""] * 3)[:3]
        out.setdefault("sample_id", s)
        out.setdefault("plate_id", pl)
        out.setdefault("well_id", wid)
    return out


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

    @property
    def is_finest(self) -> bool:
        """True if this is the finest (per-well) scope."""
        return set(self.group_by) >= set(FINEST_GROUP_BY)

    def relation_to(self, dep: "Scope") -> ScopeRelation:
        """How dependency scope ``dep`` relates to this (downstream) scope.

        Called as ``downstream.relation_to(dependency)``. Returns SAME when the
        group_bys are identical, CONTAINS when ``dep`` is at least as fine as this
        scope (its dimensions determine all of this scope's), and raises
        ``ValueError`` otherwise (coarse-on-coarse or disjoint — the design's
        "explicitly rejected" branch).
        """
        self_set = set(self.group_by)
        dep_set = set(dep.group_by)
        if dep_set == self_set:
            return ScopeRelation.SAME
        if _closure(dep.group_by) >= self_set:
            return ScopeRelation.CONTAINS
        raise ValueError(
            f"Scope {self.name!r} (group_by={self.group_by}) has no containment "
            f"relation to dependency {dep.name!r} (group_by={dep.group_by}): "
            f"the dependency is neither identical nor finer."
        )

    def contains(
        self,
        self_key: dict[str, str],
        other_key: dict[str, str],
        other_scope: "Scope",
    ) -> bool:
        """True if ``other`` instance's members fall under this instance.

        Convenience for tests/introspection only — valid solely when
        ``other_scope`` refines this scope (SAME or CONTAINS). The scheduler's hot
        path resolves dependency instances by well-set membership instead.
        """
        try:
            self.relation_to(other_scope)
        except ValueError:
            return False
        derived = _partial_dims_from_key(other_key)
        return all(derived.get(d) == self_key.get(d) for d in self.group_by)

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

# The finest scope: one physical well within one recording = one per-well pipeline
# entry. Every per-well task lives here; it is the identity level of scheduling.
SCOPE_WELL = Scope("well", FINEST_GROUP_BY)

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
