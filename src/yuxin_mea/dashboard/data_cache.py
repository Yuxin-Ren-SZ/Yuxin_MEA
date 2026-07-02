"""Signature-keyed in-process memoize for plate-viewer data (Caching guide Tier 1).

The dashboard serves a single SSH-forwarded user at a time, so a plain
in-memory cache is safe and simplest. We key each cached ``list[WellRecord]``
on a cheap **staleness signature** — ``(path, st_mtime_ns, st_size)`` per
source file, ``stat`` only, never opening the file. A pipeline re-run rewrites
the outputs, their mtime/size change, the signature changes, and the stale
entry is simply never hit again (guide §5: bust by signature, never by timer).

Only the manifest fast-path (exact per-well dirs recovered from
``pipeline_cache.json``) is cached: there we know the precise files to ``stat``
without walking the NAS. When no manifest is available we fall back to the
uncached legacy loader rather than glob the filesystem to build a key.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from yuxin_mea.analysis.plate_raster_synchrony import WellRecord, load_plate_data
from yuxin_mea.dashboard.cache import cache_get, cache_set, make_key

# Bounded LRU: a session views only a handful of recordings, but many display-
# setting tweaks re-render the same data. Keep the last few recordings' records
# so repeat Loads / Export are instant, without growing memory unbounded.
_MAX_ENTRIES = 12
_CACHE: "OrderedDict[tuple, list[WellRecord]]" = OrderedDict()


def _stat(path: Path) -> tuple[int, int]:
    """Return ``(st_mtime_ns, st_size)``; ``(-1, -1)`` when the file is absent.

    Missing files still produce a stable, distinct signature component so a
    file appearing/disappearing busts the key.
    """
    try:
        st = path.stat()
    except OSError:
        return (-1, -1)
    return (int(st.st_mtime_ns), int(st.st_size))


def data_sig(paths: Iterable[Path]) -> tuple:
    """Cheap staleness signature over ``paths`` — ``stat`` only, never opens."""
    return tuple((str(p), *_stat(Path(p))) for p in paths)


def _manifest_sig(
    burst_well_dirs: dict[str, Path] | None,
    curation_well_dirs: dict[str, Path] | None,
    experiment_cache_path: Path | None,
) -> tuple:
    """Signature over the exact files ``load_plate_data`` will read.

    Stats the two ``np.load`` targets per well (``plot_signals.npy`` and
    ``curated_spike_times.npy``) plus ``experiment_cache.json`` (well names /
    groups). Event ``.pkl`` tables are rewritten in the same task run as
    ``plot_signals.npy``, so the latter's mtime already covers them.
    """
    burst = burst_well_dirs or {}
    curation = curation_well_dirs or {}
    files: list[Path] = []
    for well_id in sorted(set(burst) | set(curation)):
        if well_id in burst:
            files.append(Path(burst[well_id]) / "plot_signals.npy")
        if well_id in curation:
            files.append(Path(curation[well_id]) / "curated_spike_times.npy")
    if experiment_cache_path is not None:
        files.append(Path(experiment_cache_path))
    return data_sig(files)


def manifest_signature(
    burst_well_dirs: dict[str, Path] | None,
    curation_well_dirs: dict[str, Path] | None,
    experiment_cache_path: Path | None,
) -> str:
    """Public string form of the staleness signature (used to stamp bundles)."""
    return str(_manifest_sig(burst_well_dirs, curation_well_dirs, experiment_cache_path))


def load_plate_data_cached(
    *,
    recording_key: str,
    source: str,
    burst_root: Path,
    curation_root: Path,
    burst_terminal: str,
    experiment_cache_path: Path | None,
    burst_well_dirs: dict[str, Path] | None,
    curation_well_dirs: dict[str, Path] | None,
    bundle_dir: Path | str | None = None,
) -> list[WellRecord]:
    """Load 24 ``WellRecord``s, memoized by manifest staleness signature.

    Repeat calls for the same recording/source with unchanged files return the
    cached records instantly (no NAS touch). Falls back to the uncached loader
    when no manifest is available.
    """
    have_manifest = bool(burst_well_dirs) or bool(curation_well_dirs)
    if not have_manifest:
        # Legacy path: without exact dirs we'd have to glob to build a key, which
        # is the very cost we're avoiding. Load directly, uncached.
        return load_plate_data(
            burst_detection_root=burst_root,
            curation_output_root=curation_root,
            recording_key=recording_key,
            rec_name="auto",
            experiment_cache_path=experiment_cache_path,
            burst_terminal=burst_terminal,
        )

    sig = _manifest_sig(burst_well_dirs, curation_well_dirs, experiment_cache_path)
    key = (str(recording_key), str(source), str(burst_terminal), sig)

    # L1: in-process (fastest; gives same-session object identity).
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit

    # L2: persistent FileSystemCache on local scratch (survives restarts).
    l2_key = make_key(*key)
    persisted = cache_get(l2_key)
    if persisted is not None:
        _remember(key, persisted)
        return persisted

    # Tier 3: prebuilt per-recording viewer bundle (one sequential file read),
    # accepted only when its stored signature still matches the live source files.
    if bundle_dir is not None:
        from yuxin_mea.analysis.viewer_bundle import bundle_path, read_viewer_bundle
        bundled = read_viewer_bundle(
            bundle_path(bundle_dir, recording_key, str(source)),
            expected_signature=str(sig),
        )
        if bundled is not None:
            _remember(key, bundled)
            cache_set(l2_key, bundled)
            return bundled

    records = load_plate_data(
        burst_detection_root=burst_root,
        curation_output_root=curation_root,
        recording_key=recording_key,
        rec_name="auto",
        experiment_cache_path=experiment_cache_path,
        burst_terminal=burst_terminal,
        burst_well_dirs=burst_well_dirs,
        curation_well_dirs=curation_well_dirs,
    )
    _remember(key, records)
    cache_set(l2_key, records)
    return records


def _remember(key: tuple, records: list[WellRecord]) -> None:
    """Store in the bounded in-process LRU."""
    _CACHE[key] = records
    while len(_CACHE) > _MAX_ENTRIES:
        _CACHE.popitem(last=False)


def clear_cache() -> None:
    """Drop all memoized records (used by tests)."""
    _CACHE.clear()
