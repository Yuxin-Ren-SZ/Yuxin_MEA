"""Multipage Dash dashboard for yuxin_mea.

Read-only browser UI for browsing the dataset cache and pipeline status.
Phase 2 ships two data pages (Recordings, Pipeline status); the burst
diagnostic page is added in Phase 2b.
"""

from .app import build_app

__all__ = ["build_app"]
