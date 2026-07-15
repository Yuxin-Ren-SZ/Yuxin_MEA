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
    "cache_root": ParamSpec(
        "path", "",
        "Local (non-NAS) scratch dir for the dashboard's read cache and "
        "pre-rendered raster PNGs. Leave blank to default to "
        "<analysis_root>/../dashboard_cache.",
    ),
    "fingerprint_mode": ParamSpec(
        "str", "stat",
        "How much of each raw data.raw.h5 to fingerprint when scanning, for "
        "reproducibility provenance (see doc/caching.md). "
        "'stat' = size/mtime only (default, free). "
        "'struct' = also hash the small analysis-critical datasets "
        "(gain/lsb/mapping/settings) + raw shapes — no bulk reads. "
        "'content' = also sample the raw array (slow on a busy NAS). "
        "'full' = also read the raw array entirely (very slow on 30-100 GB). "
        "Hashes are reused while the h5 size/mtime is unchanged.",
        choices=["stat", "struct", "content", "full"],
    ),
}
