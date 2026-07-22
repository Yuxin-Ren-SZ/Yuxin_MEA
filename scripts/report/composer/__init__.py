"""Interactive figure composer for the journal report.

``python -m scripts.report.composer`` serves a single page where every report
panel is rendered, laid out on a 12-column grid and exported to a multi-page PDF
plus a ``figure_spec.json`` that restores the session next time.

The package splits cleanly in two: :mod:`spec` and :mod:`compose` are headless
and testable (spec in, PDF out), while :mod:`server` only translates HTTP into
calls on them.
"""
from .spec import Composition, FigureSpec, PanelPlacement, default_composition

__all__ = ["Composition", "FigureSpec", "PanelPlacement", "default_composition"]
