"""Content fingerprints for reproducibility provenance.

Three fingerprint kinds, all sha256-based (same ``hashlib`` idiom as
``dashboard/cache.py`` / ``analysis/raster_image.py``):

- :func:`file_hash` — full stream hash of a small file (``mxassay.metadata``).
- :func:`params_hash` — hash of a resolved task-params dict (config provenance).
- :func:`h5_fingerprint` — **HDF5-structure-aware** hash of ``data.raw.h5``.

The h5 fingerprint is the subtle one. A MaxWell recording is 30–100 GB, almost
all of it the raw voltage array ``recordings/<rec>/<well>/groups/routed/raw``
``(n_channels, n_frames) uint16``. Hashing the whole file every scan is
prohibitive over the NAS, so we **partition datasets by size**:

- **Small datasets are hashed in full** — this is the analysis-critical part
  (``settings/{gain,lsb,hpf,sampling,spike_threshold,mapping}``, ``channels``,
  ``spikes``, ``events``, top-level ``version``/``assay``/… and every attr). A
  change to gain / electrode mapping / sampling rate is therefore caught exactly.
- **Large datasets are sampled** — a fixed number of fixed-size windows at
  deterministic *fractional* offsets along the longest axis, with the dataset's
  shape/dtype/chunks folded in. ``n_frames``/``n_channels`` vary per recording,
  so positions are fractional (reproducible per file) and any change to length
  or electrode count changes the shape and hence the fingerprint.

``--full`` promotes the large datasets to full streaming hashes. Everything
streams in fixed-size blocks, so memory stays constant regardless of file size.

The ``method`` id (``h5struct-v1`` / ``h5full-v1``) is stored alongside the hash
so the fingerprint method is itself versioned — a later method change is
detectable rather than silently producing incomparable hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

_READ_BLOCK = 8 << 20  # 8 MiB stream block
_H5_SMALL_THRESHOLD = 16 << 20  # datasets <= 16 MiB are hashed in full
_H5_WINDOWS = 32  # sample windows for large datasets
_H5_WINDOW_LEN = 256  # elements per window along the sampled axis
_METHOD_META = "h5meta-v1"      # small datasets hashed; large ones metadata-only
_METHOD_SAMPLED = "h5struct-v1"  # + large datasets sampled
_METHOD_FULL = "h5full-v1"       # + large datasets read in full


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def stat_sig(path: Path | str) -> dict[str, int]:
    """Cheap ``{size, mtime_ns}`` change-signal — ``stat`` only, never opens.

    Returns ``{"size": -1, "mtime_ns": -1}`` when the file is absent, so a
    file appearing/disappearing still changes the signal.
    """
    try:
        st = os.stat(path)
    except OSError:
        return {"size": -1, "mtime_ns": -1}
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def file_hash(path: Path | str, *, block: int = _READ_BLOCK) -> dict[str, Any]:
    """Full streaming sha256 of a (small) file, plus its stat signature.

    Used for ``mxassay.metadata`` (tiny, frequently overwritten). Constant
    memory. Raises ``OSError`` if the file cannot be read.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    sig = stat_sig(path)
    return {"size": sig["size"], "mtime_ns": sig["mtime_ns"], "sha256": h.hexdigest()}


def params_hash(params: dict) -> str:
    """sha256 of a resolved task-params dict — canonical, order-independent.

    ``default=str`` keeps it total over non-JSON scalars (e.g. ``Path``). This
    is the config-provenance fingerprint; it matches what a task actually ran
    with (``TaskRecord.config``), not the raw config file text.
    """
    payload = json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# HDF5 structure-aware fingerprint
# ---------------------------------------------------------------------------


def _feed(h: "hashlib._Hash", *fields: Any) -> None:
    """Feed length-prefixed fields into the digest (unambiguous framing)."""
    for f in fields:
        b = f if isinstance(f, (bytes, bytearray)) else str(f).encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)


def _array_bytes(arr: Any) -> bytes:
    """Bytes for a numpy array, tolerant of object/structured dtypes."""
    try:
        return arr.tobytes()
    except (ValueError, TypeError):
        # Object dtype (e.g. events' 'eventmessage') — fall back to a stable repr.
        return repr(arr.tolist()).encode("utf-8")


def _window_offsets(length: int, window: int, n_windows: int) -> list[int]:
    """Deterministic fractional start offsets for sampling a long axis."""
    if length <= window or n_windows <= 1:
        return [0]
    span = length - window
    return sorted({round(span * i / (n_windows - 1)) for i in range(n_windows)})


def _hash_dataset(h: "hashlib._Hash", dset: Any, *, threshold: int,
                  windows: int, window_len: int, full: bool,
                  sample_large: bool = True) -> None:
    import numpy as np  # local — numpy is a heavy import, keep it lazy

    shape = tuple(dset.shape)
    itemsize = dset.dtype.itemsize
    nbytes = itemsize * math.prod(shape) if shape else itemsize
    _feed(h, dset.name, str(shape), str(dset.dtype), str(dset.chunks),
          str(dset.compression))

    if not shape:  # scalar
        _feed(h, b"S", _array_bytes(dset[()]))
        return

    if not full and nbytes > threshold and not sample_large:
        # Large dataset, sampling disabled → metadata only (already fed above).
        # Keeps the analysis-critical small datasets fully covered while avoiding
        # the scattered chunk reads that dominate cost on a busy NAS. Still
        # detects re-acquisition (shape/dtype/chunking change).
        _feed(h, b"M")
        return

    if full or nbytes <= threshold:
        # Full content, streamed by blocks of the outermost axis so a large
        # dataset under --full never lands in memory whole.
        _feed(h, b"F")
        step = max(1, _READ_BLOCK // max(1, nbytes // shape[0])) if shape[0] else 1
        for i in range(0, shape[0], step):
            _feed(h, _array_bytes(dset[i:i + step]))
        return

    # Large → sample fixed windows at fractional offsets along the longest axis.
    axis = int(np.argmax(shape))
    offsets = _window_offsets(shape[axis], window_len, windows)
    _feed(h, b"W", str(axis), str(offsets))
    for off in offsets:
        sl = [slice(None)] * len(shape)
        sl[axis] = slice(off, off + window_len)
        _feed(h, _array_bytes(dset[tuple(sl)]))


def h5_fingerprint(
    path: Path | str,
    *,
    size_threshold: int = _H5_SMALL_THRESHOLD,
    windows: int = _H5_WINDOWS,
    window_len: int = _H5_WINDOW_LEN,
    full: bool = False,
    sample_large: bool = True,
) -> dict[str, Any]:
    """Structure-aware sha256 of a MaxWell ``data.raw.h5``.

    Small datasets + all attrs are always hashed in full — that is the
    analysis-critical part (gain/lsb/mapping/sampling/channels/spikes).

    Large datasets (the raw voltage array) are, in increasing cost:

    - ``sample_large=False``          → **metadata only** (``h5meta-v1``): shape/
      dtype/chunking folded in, no bulk reads. Cheapest; still catches a
      re-acquisition that changes length/electrode count, and any settings change.
    - ``sample_large=True`` (default) → sampled windows (``h5struct-v1``).
    - ``full=True``                   → read in full (``h5full-v1``).

    Sampling/full are I/O-latency heavy on a busy NAS (scattered chunk reads), so
    callers make them opt-in. Deterministic: objects visited in sorted name order.
    """
    import h5py

    method = _METHOD_FULL if full else (
        _METHOD_SAMPLED if sample_large else _METHOD_META
    )
    h = hashlib.sha256()
    sig = stat_sig(path)
    _feed(h, method, str(sig["size"]))

    with h5py.File(path, "r") as hf:
        # Collect every node name up front, then process in sorted order so the
        # digest is independent of HDF5 link iteration order.
        names: list[str] = ["/"]
        hf.visit(names.append)
        for name in sorted(set(names)):
            obj = hf[name]
            # Attrs (tiny) always folded in, sorted for determinism.
            for k in sorted(obj.attrs.keys()):
                _feed(h, "attr", name, k, repr(obj.attrs[k]))
            if isinstance(obj, h5py.Dataset):
                _hash_dataset(h, obj, threshold=size_threshold,
                              windows=windows, window_len=window_len, full=full,
                              sample_large=sample_large)

    return {
        "method": method,
        "file_size": sig["size"],
        "mtime_ns": sig["mtime_ns"],
        "sha256": h.hexdigest(),
        "size_threshold": size_threshold,
        "windows": windows,
        "window_len": window_len,
    }


# Fingerprint modes that actually hash the h5 (anything else = cheap stat only).
# Ordered cheapest → costliest. See h5_fingerprint for what each covers.
H5_HASH_MODES = ("struct", "content", "full")
_MODE_METHOD = {"struct": _METHOD_META, "content": _METHOD_SAMPLED, "full": _METHOD_FULL}
_MODE_KWARGS = {
    "struct":  {"full": False, "sample_large": False},
    "content": {"full": False, "sample_large": True},
    "full":    {"full": True,  "sample_large": True},
}


def h5_method_for(mode: str) -> str:
    """Method id a given fingerprint mode produces (for stat-drift reuse checks)."""
    return _MODE_METHOD[mode]


def h5_kwargs_for(mode: str) -> dict[str, bool]:
    """``h5_fingerprint`` kwargs for a given fingerprint mode."""
    return dict(_MODE_KWARGS[mode])


def raw_fingerprint(
    h5_path: Path | str,
    metadata_path: Path | str | None = None,
    *,
    full: bool = False,
) -> dict[str, Any]:
    """Combined raw-input fingerprint: ``{"h5": {...}, "metadata": {...}|None}``.

    ``metadata`` is ``None`` when ``mxassay.metadata`` is absent (it is optional
    lab annotation — see ``doc/caching.md``).
    """
    out: dict[str, Any] = {"h5": h5_fingerprint(h5_path, full=full)}
    if metadata_path is not None and Path(metadata_path).is_file():
        out["metadata"] = file_hash(metadata_path)
    else:
        out["metadata"] = None
    return out
