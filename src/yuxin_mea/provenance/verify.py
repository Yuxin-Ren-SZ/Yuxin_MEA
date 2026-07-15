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
        return a.get("method") == b.get("method") and a["sha256"] == b["sha256"]
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


@dataclass
class VerifyReport:
    counts: dict[str, int] = field(default_factory=lambda: {s: 0 for s in STATUSES})
    details: list[tuple[str, str]] = field(default_factory=list)  # (labels, "rkey/well/task")
    errors: list[str] = field(default_factory=list)

    @property
    def computational_drift(self) -> int:
        return self.counts["RAW-CHANGED"] + self.counts["CONFIG-CHANGED"]


def _well_of(pipeline_key: str) -> str:
    parts = pipeline_key.split("/")
    return "/".join(parts[5:]) if len(parts) > 5 else pipeline_key


def verify_provenance(cm, data_root: Path, analysis_root: Path, *,
                      full_hash: bool = False, network_only: bool = False) -> VerifyReport:
    """Verify all COMPLETE tasks. ``cm`` is a loaded ConfigManager (current config).

    Without ``full_hash`` the current raw fingerprint comes from
    ``experiment_cache.json`` (cheap; reflects the last scan). With ``full_hash``
    each recording's h5 is re-fingerprinted from disk (content sha256; NAS-costly)
    and never persisted.
    """
    from yuxin_mea.dataset.cache import JsonCacheStore
    from yuxin_mea.pipeline.cache import JsonPipelineCacheStore
    from .fingerprint import h5_fingerprint, params_hash
    from .sidecar import read_sidecar

    recs = JsonCacheStore(analysis_root).load()
    pipe = JsonPipelineCacheStore(analysis_root).load()
    report = VerifyReport()
    fresh_h5: dict[str, dict] = {}
    cfg_cache: dict[str, str] = {}

    for pkey, entry in sorted(pipe.items()):
        rkey = entry.recording_key
        if network_only and "/Network/" not in rkey:
            continue
        rec = recs.get(rkey)
        cur_rf = (rec.raw_fingerprint if rec else {}) or {}
        cur_meta = cur_rf.get("metadata")
        cur_h5 = cur_rf.get("h5")
        if full_hash and rec is not None:
            if rkey not in fresh_h5:
                try:
                    fresh_h5[rkey] = h5_fingerprint(data_root / rec.data_path, full=True)
                except Exception as exc:  # noqa: BLE001
                    fresh_h5[rkey] = {}
                    report.errors.append(f"{rkey}: {exc}")
            cur_h5 = fresh_h5[rkey] or cur_h5

        for tname, rec_task in entry.tasks.items():
            if rec_task.status != "complete":
                continue
            prov = rec_task.provenance
            if prov is None and rec_task.output_path:
                prov = read_sidecar(Path(rec_task.output_path).parent)
            if tname not in cfg_cache:
                cfg_cache[tname] = params_hash(cm.get_task_params(tname))
            issues = classify_task(prov, cur_h5, cur_meta, cfg_cache[tname])
            if not issues:
                report.counts["OK"] += 1
                continue
            for i in issues:
                report.counts[i] += 1
            # UNKNOWN (stamp-less, pre-provenance) is counted but kept out of the
            # drift-detail list — it is not actionable drift and would otherwise
            # bury real findings on a cache full of old outputs.
            if issues != ["UNKNOWN"]:
                report.details.append(("+".join(issues), f"{rkey}/{_well_of(pkey)}/{tname}"))
    return report
