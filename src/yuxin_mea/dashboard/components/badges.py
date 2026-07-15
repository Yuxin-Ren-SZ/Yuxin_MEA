"""Provenance status badge — shared by Recordings + Plate viewer.

Renders the per-recording status from ``dashboard.data.recording_provenance``
(OK / RAW-CHANGED / CONFIG-CHANGED / METADATA-CHANGED / UNKNOWN). RAW/CONFIG are
computational drift (re-run); METADATA is label-only; UNKNOWN predates
provenance. See ``doc/caching.md``.
"""

from __future__ import annotations

from dash import html

# status → (short label, colour, tooltip)
_PROV = {
    "OK": ("✓ verified", "var(--ok, #2e7d32)",
           "Output matches the current raw fingerprint + config."),
    "RAW-CHANGED": ("⚠ raw changed", "var(--fail, #c62828)",
                    "The raw data.raw.h5 differs from what produced this output — re-run analysis."),
    "CONFIG-CHANGED": ("⚠ config changed", "#c77700",
                       "Task config changed since this output was produced — re-run analysis."),
    "METADATA-CHANGED": ("labels changed", "#1f5aa6",
                         "mxassay.metadata changed (groupname/labels only; no re-run needed)."),
    "UNKNOWN": ("unstamped", "var(--ink-3, #84807a)",
                "No provenance stamp — this output predates provenance tracking."),
}


def provenance_badge(status: str | None, *, show_ok: bool = True) -> html.Span | None:
    """Return a styled badge for a provenance status, or ``None`` to render nothing.

    ``None``/empty status → nothing (recording has no verified task). ``show_ok``
    False hides the green OK badge (keep list views quiet, only flag problems).
    """
    if not status or (status == "OK" and not show_ok):
        return None
    label, colour, tip = _PROV.get(status, ("unstamped", "var(--ink-3)", status))
    return html.Span(
        label,
        title=tip,
        style={
            "fontFamily": "var(--font-mono)", "fontSize": "10px", "fontWeight": "600",
            "color": colour, "border": f"1px solid {colour}", "borderRadius": "999px",
            "padding": "1px 8px", "whiteSpace": "nowrap",
        },
    )
