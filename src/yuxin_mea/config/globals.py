"""Schema for `ConfigManager` global settings.

The dashboard's Settings page renders the Globals tab from this dict. Adding
a new global is a one-line edit here plus a consumer that calls
`cm.get_global("…")`. Order is preserved (Python 3.7+ dict ordering) so the
form fields render top-to-bottom in the declared order.
"""

from __future__ import annotations

from .schema import ParamSpec


GLOBALS_SCHEMA: dict[str, ParamSpec] = {
    "data_root": ParamSpec(
        "path", "",
        "Root directory holding raw MEA recordings (data.raw.h5 files).",
    ),
    "analysis_root": ParamSpec(
        "path", "",
        "Where analysis caches and per-task outputs live "
        "(experiment_cache.json, pipeline_cache.json, etc.).",
    ),
    "figure_root": ParamSpec(
        "path", "",
        "Where figure exports (HTML, PNG) are written.",
    ),
}
