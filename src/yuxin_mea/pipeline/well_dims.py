"""WellDims + Member: the per-well descriptors an aggregate task groups over.

``WellDims`` carries every dimension a :class:`~yuxin_mea.pipeline.scope.Scope`
may group_by / filter on, all derived from the pipeline entry's ``recording_key``
+ compound ``well_id`` plus the well's ``groupname`` from the experiment cache.

``Member`` pairs a well's dims with the *status and output path of the upstream
dependency task* for that well — read straight from the per-well
``TaskRecord`` (no filesystem globbing). An aggregate task's ``run`` receives the
list of Members whose upstream dependency completed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WellDims:
    """All dimensions of a single well, derived from the pipeline entry."""

    recording_key: str        # "sample/date/plate/scan/run_id"
    sample_id: str
    date: str                 # YYMMDD
    plate_id: str             # physical chip serial (reused across cultures)
    scan_type: str            # Network / ActivityScan / AxonTracking
    run_id: str               # assay / recording run
    rec_name: str             # e.g. "rec0000"
    well_id: str              # PHYSICAL well, e.g. "well000"
    well_uid: str             # f"{sample_id}|{plate_id}|{well_id}"
    groupname: str            # from experiment_cache; "?" when metadata absent
    compound_well_id: str     # "rec_name/well_id" == the pipeline entry's well_id

    @classmethod
    def from_entry(
        cls, recording_key: str, compound_well_id: str, groupname: str = "?"
    ) -> "WellDims":
        parts = recording_key.split("/")
        # Tolerate an unexpected shape rather than crash the whole scheduler.
        sample_id, date, plate_id, scan_type, run_id = (parts + [""] * 5)[:5]
        if "/" in compound_well_id:
            rec_name, well_id = compound_well_id.split("/", 1)
        else:
            rec_name, well_id = "", compound_well_id
        return cls(
            recording_key=recording_key,
            sample_id=sample_id,
            date=date,
            plate_id=plate_id,
            scan_type=scan_type,
            run_id=run_id,
            rec_name=rec_name,
            well_id=well_id,
            well_uid=f"{sample_id}|{plate_id}|{well_id}",
            groupname=groupname if groupname else "?",
            compound_well_id=compound_well_id,
        )


@dataclass(frozen=True)
class Member:
    """One well participating in an aggregate instance, with its upstream state.

    ``upstream_output_path`` is the dependency task's ``TaskRecord.output_path``
    (e.g. the ``.../ml_burst_detection`` dir); the aggregate task reads whatever
    it needs from there (e.g. ``debug_trace.pkl``).

    A member is usually a *well* (the finest scope). S2 also lets a member be an
    upstream *aggregate instance* (an aggregate→aggregate dependency): then
    ``scope_key``/``scope_name`` identify that instance and ``dims`` carries a
    representative member well. For a well member both stay ``None``.
    """

    dims: WellDims
    upstream_status: str
    upstream_output_path: Path | None
    upstream_last_updated: float | None
    scope_key: dict[str, str] | None = None   # set when the member is an aggregate instance
    scope_name: str | None = None

    @property
    def recording_key(self) -> str:
        return self.dims.recording_key

    @property
    def compound_well_id(self) -> str:
        return self.dims.compound_well_id
