"""Persistent server-side cache on local SSD scratch (Caching guide Tier 2).

Backs the in-process memoize with ``flask-caching``'s ``FileSystemCache`` pointed
at ``cache_root`` (local scratch, never the NAS), so parsed plate records and
diagnostic batches survive dashboard restarts. Same discipline as everywhere
else: entries are keyed by a staleness signature and never time-expire — a
pipeline re-run changes the signature, so the stale entry is simply never hit.

Everything degrades gracefully: if ``flask-caching`` isn't installed, or
``cache_root`` is unset, or we're called outside a Flask app context (e.g. in a
unit test), the accessors become no-ops and callers fall back to their own
in-process layer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

try:
    from flask_caching import Cache
except ImportError:  # dependency optional — dashboard still runs without Tier 2
    Cache = None  # type: ignore[assignment]


cache = Cache() if Cache is not None else None
_initialized = False


def init_cache(server, cache_root: Path | str | None):
    """Attach a ``FileSystemCache`` on ``cache_root`` to the Flask ``server``.

    No-op (returns ``None``) when flask-caching is missing or ``cache_root`` is
    unset. Called once from :func:`build_app`.
    """
    global _initialized
    if cache is None or cache_root is None:
        _initialized = False
        return None
    cache_dir = Path(cache_root) / "flask_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.init_app(server, config={
            "CACHE_TYPE": "FileSystemCache",
            "CACHE_DIR": str(cache_dir),
            "CACHE_DEFAULT_TIMEOUT": 0,   # never expire; bust by signature
            "CACHE_THRESHOLD": 4000,
        })
    except Exception:  # noqa: BLE001 — a broken cache dir must not sink the app
        _initialized = False
        return None
    _initialized = True
    return cache


def is_active() -> bool:
    return _initialized and cache is not None


def make_key(*parts: Any) -> str:
    """Stable, filename-safe key from arbitrary hashable parts."""
    raw = "\x00".join(repr(p) for p in parts)
    return "ym_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def cache_get(key: str):
    """Return the cached value, or ``None`` (miss / no cache / no app context)."""
    if not is_active():
        return None
    try:
        return cache.get(key)
    except Exception:  # noqa: BLE001 — outside app ctx / backend error → miss
        return None


def cache_set(key: str, value: Any) -> None:
    """Persist ``value`` under ``key``; silently no-op when the cache is inactive."""
    if not is_active():
        return
    try:
        cache.set(key, value)
    except Exception:  # noqa: BLE001 — unpicklable value / backend error → skip L2
        pass
