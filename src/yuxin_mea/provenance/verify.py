"""Compare pipeline outputs' provenance stamps against the current raw + config.

Reusable by both ``scripts/check_cache.py`` (CLI) and, later, the dashboard
(provenance badges — Stage 2). Read-only: nothing here writes a cache. Per
COMPLETE task it emits one of:

- ``OK``               — stamp matches the current raw fingerprint + config.
- ``RAW-CHANGED``      — the h5 that produced the output differs from disk now.
- ``CONFIG-CHANGED``   — the task's config params changed since it ran.
- ``METADATA-CHANGED`` — ``mxassay.metadata`` changed (labels only; not analysis).
- ``UNKNOWN``          — no stamp (output predates provenance) — unverifiable,
                         not a mismatch.

``RAW-CHANGED`` and ``CONFIG-CHANGED`` are *computational* drift (the output
should be recomputed); ``METADATA-CHANGED`` is *interpretation* drift (re-read
``groupname`` via ``refresh_groupnames``; no re-run). See ``doc/caching.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATUSES = ("OK", "RAW-CHANGED", "CONFIG-CHANGED", "METADATA-CHANGED", "UNKNOWN")


def h5_match(a: dict | None, b: dict | None) -> bool | None:
    """``True``/``False`` when comparable, ``None`` when unknown/incomparable."""
    if not a or not b:
        return None
    if a.get("sha256") and b.get("sha256"):
        # Different fingerprint methods (e.g. sampled vs --full-hash) are not
        # comparable → UNKNOWN, not a false RAW-CHANGED alarm.
        if a.get("method") != b.get("method"):
            return None
        return a["sha256"] == b["sha256"]
    if a.get("file_size") is None or b.get("file_size") is None:
        return None
    return a.get("file_size") == b.get("file_size") and a.get("mtime_ns") == b.get("mtime_ns")


def meta_match(a: dict | None, b: dict | None) -> bool | None:
    if not a or not b:
        return None
    if a.get("sha256") and b.get("sha256"):
        return a["sha256"] == b["sha256"]
    if a.get("size") is None or b.get("size") is None:
        return None
    return a.get("size") == b.get("size") and a.get("mtime_ns") == b.get("mtime_ns")


def classify_task(prov: dict | None, cur_h5: dict | None, cur_meta: dict | None,
                  cur_config_hash: str) -> list[str]:
    """Return the drift labels for one task's stamp (``[]`` means OK).

    ``["UNKNOWN"]`` when there is no stamp. Otherwise any of RAW-/CONFIG-/
    METADATA-CHANGED (a single task can trip more than one).
    """
    if prov is None:
        return ["UNKNOWN"]
    issues: list[str] = []
    if prov.get("config_hash") != cur_config_hash:
        issues.append("CONFIG-CHANGED")
    if h5_match(prov.get("h5"), cur_h5) is False:
        issues.append("RAW-CHANGED")
    if meta_match(prov.get("metadata"), cur_meta) is False:
        issues.append("METADATA-CHANGED")
    return issues


_SEVERITY = {"RAW-CHANGED": 4, "CONFIG-CHANGED": 3, "METADATA-CHANGED": 2,
             "UNKNOWN": 1, "OK": 0}


@dataclass
class VerifyReport:
    counts: dict[str, int] = field(default_factory=lambda: {s: 0 for s in STATUSES})
    details: list[tuple[str, str]] = field(default_factory=list)  # (labels, "rkey/well/task")
    # Computational drift (RAW/CONFIG) as structured items, for --mark-stale:
    #   {"recording_key", "well_id", "task", "labels": [...]}
    drifted: list[dict] = field(default_factory=list)
    # Worst-case status per recording_key (for dashboard badges).
    per_recording: dict[str, str] = field(default_factory=dict)
    # Per recording_key, whether its provenance is an *adopted baseline* only —
    # True when every stamped task is an adopted stamp (no real-run measurement).
    # Reset to False the moment any measured stamp appears (e.g. after a re-run).
    per_recording_adopted: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def computational_drift(self) -> int:
        return self.counts["RAW-CHANGED"] + self.counts["CONFIG-CHANGED"]

    def _bump(self, recording_key: str, status: str) -> None:
        # First status seen registers the recording (even OK — else an all-OK
        # recording would never appear here and its badge would never render);
        # thereafter keep only the worst.
        cur = self.per_recording.get(recording_key)
        if cur is None or _SEVERITY.get(status, 0) > _SEVERITY.get(cur, 0):
            self.per_recording[recording_key] = status


def _well_of(pipeline_key: str) -> str:
    parts = pipeline_key.split("/")
    return "/".join(parts[5:]) if len(parts) > 5 else pipeline_key


def verify_provenance(cm, data_root: Path, analysis_root: Path, *,
                      full_hash: bool = False, hash_mode: str | None = None,
                      network_only: bool = False,
                      use_sidecar: bool = True) -> VerifyReport:
    """Verify all COMPLETE tasks. ``cm`` is a loaded ConfigManager (current config).

    By default the current raw fingerprint comes from ``experiment_cache.json``
    (cheap; reflects the last scan). ``hash_mode`` (``"struct"``/``"content"``/
    ``"full"``) instead re-fingerprints each h5 from disk at that level, in memory
    (never persisted) — costly on the NAS, so it is opt-in. ``full_hash=True`` is
    the legacy alias for ``hash_mode="full"``.

    ``use_sidecar`` falls back to the on-disk ``yuxin_provenance.json`` when a
    task's cache stamp is missing. Set it ``False`` for the dashboard hot path —
    those sidecars live on the NAS and stat-ing one per task is prohibitively slow
    (the cache copy in ``pipeline_cache.json`` is enough for a live status view).
    """
    from yuxin_mea.dataset.cache import JsonCacheStore
    from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
    from .fingerprint import h5_fingerprint, h5_kwargs_for, params_hash
    from .sidecar import read_sidecar, sidecar_dir

    if full_hash and hash_mode is None:
        hash_mode = "full"

    recs = JsonCacheStore(analysis_root).load()
    pipe = JsonPipelineCacheStore(analysis_root).load()
    report = VerifyReport()
    fresh_h5: dict[str, dict] = {}
    cfg_cache: dict[str, str] = {}
    # Per recording: did we see a measured stamp / an adopted stamp?
    seen_measured: set[str] = set()
    seen_adopted: set[str] = set()

    for pkey, entry in sorted(pipe.items()):
        rkey = entry.recording_key
        if network_only and "/Network/" not in rkey:
            continue
        rec = recs.get(rkey)
        cur_rf = (rec.raw_fingerprint if rec else {}) or {}
        cur_meta = cur_rf.get("metadata")
        cur_h5 = cur_rf.get("h5")
        if hash_mode is not None and rec is not None:
            if rkey not in fresh_h5:
                try:
                    fresh_h5[rkey] = h5_fingerprint(data_root / rec.data_path,
                                                    **h5_kwargs_for(hash_mode))
                except Exception as exc:  # noqa: BLE001
                    fresh_h5[rkey] = {}
                    report.errors.append(f"{rkey}: {exc}")
            cur_h5 = fresh_h5[rkey] or cur_h5

        for tname, rec_task in entry.tasks.items():
            if rec_task.status != "complete":
                continue
            prov = rec_task.provenance
            if prov is None and use_sidecar and rec_task.output_path:
                prov = read_sidecar(sidecar_dir(rec_task.output_path))
            if prov is not None:
                (seen_adopted if prov.get("adopted") else seen_measured).add(rkey)
            if tname not in cfg_cache:
                cfg_cache[tname] = params_hash(cm.get_task_params(tname))
            issues = classify_task(prov, cur_h5, cur_meta, cfg_cache[tname])
            if not issues:
                report.counts["OK"] += 1
                report._bump(rkey, "OK")
                continue
            for i in issues:
                report.counts[i] += 1
                report._bump(rkey, i)
            # RAW/CONFIG are computational → actionable via --mark-stale.
            comp = [i for i in issues if i in ("RAW-CHANGED", "CONFIG-CHANGED")]
            if comp:
                report.drifted.append({
                    "recording_key": rkey, "well_id": rec_task_well_id(entry, pkey),
                    "task": tname, "labels": comp,
                })
            # UNKNOWN (stamp-less, pre-provenance) is counted but kept out of the
            # drift-detail list — it is not actionable drift and would otherwise
            # bury real findings on a cache full of old outputs.
            if issues != ["UNKNOWN"]:
                report.details.append(("+".join(issues), f"{rkey}/{_well_of(pkey)}/{tname}"))

    # A recording's provenance is "adopted-only" when it has ≥1 adopted stamp and
    # no measured stamp — i.e. its verification rests on the mtime-guarded baseline,
    # not on anything recorded during a real run.
    for rkey in seen_adopted:
        report.per_recording_adopted[rkey] = rkey not in seen_measured
    return report


def rec_task_well_id(entry, pipeline_key: str) -> str:
    """Well id for refresh() — prefer the entry's own field, else parse the key."""
    return getattr(entry, "well_id", None) or _well_of(pipeline_key)


def status_by_recording(cm, analysis_root: Path) -> dict[str, str]:
    """Cheap per-recording provenance status for dashboard badges.

    Compares each COMPLETE task's stamp against the **cached** raw fingerprint +
    current config (no NAS content re-hash, no NAS sidecar reads). Returns
    ``{recording_key: status}`` (worst-case per recording). Read-only.
    """
    return verify_provenance(cm, analysis_root, analysis_root,
                             full_hash=False, use_sidecar=False).per_recording


def status_and_adopted_by_recording(cm, analysis_root: Path) -> dict[str, dict]:
    """Like :func:`status_by_recording` but also flags adopted-baseline provenance.

    Returns ``{recording_key: {"status": <str>, "adopted": <bool>}}``. ``adopted``
    is True when the recording's provenance is an adopted baseline only (no
    measured stamp) — the dashboard badge annotates those so an assumed baseline
    is never mistaken for provenance measured during a real run.
    """
    rep = verify_provenance(cm, analysis_root, analysis_root,
                            full_hash=False, use_sidecar=False)
    return {
        rkey: {"status": status, "adopted": rep.per_recording_adopted.get(rkey, False)}
        for rkey, status in rep.per_recording.items()
    }
