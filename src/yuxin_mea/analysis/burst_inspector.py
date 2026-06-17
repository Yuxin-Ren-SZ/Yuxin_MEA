"""Per-well diagnostic library for burst detectors.

Pure-library figure builders used by
``yuxin_mea.dashboard.pages.burst_inspector``. No Dash imports — so the same
figures can be rendered in notebooks/HTML reports.

Supports two methods:
- **traditional** — standard BurstResults only (no debug trace)
- **ml** — standard BurstResults + optional MLBurstTrace

Public API:
- ``InspectorBundle`` — dataclass holding everything one panel needs.
- ``load_generic_bundle`` — disk loader for a detector's BurstResults output.
- ``fig_*`` — one Plotly figure per UI panel.
- ``summary_card`` — returns a dict of KV rows (rendered as HTML by the page).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Method → terminal directory name inside each well's output path.
METHOD_TERMINALS: dict[str, str] = {
    "traditional": "burst_detection",
    "ml": "ml_burst_detection",
}

# Method → pipeline config task name (for output_root lookup).
METHOD_TASK_NAMES: dict[str, str] = {
    "traditional": "burst_detection",
    "ml": "ml_burst_detection",
}

# Method → conventional subdir under analysis_root (fallback).
METHOD_SUBDIRS: dict[str, str] = {
    "traditional": "burst_detection_data",
    "ml": "ml_burst_data",
}


def output_root_from_cache(analysis_root: Path | str, method: str) -> Path | None:
    """Recover a method's burst output_root from ``pipeline_cache.json``.

    The pipeline cache is the source of truth for where task outputs live (it
    records the actual ``output_path`` the detector wrote to), so this stays
    correct even when ``output_root`` is changed between runs — unlike resolving
    from the config or a hardcoded convention subdir. Mirrors how
    ``scripts/inspect_ml_bursts.py`` discovers wells.

    Each completed entry stores ``output_path`` as
    ``<root>/<recording_key>/<rec_name>/<well_id>/<terminal>``; we strip that
    per-well suffix to recover the shared ``<root>``. Returns None when the cache
    is missing or has no completed entry for the method's task.
    """
    task = METHOD_TASK_NAMES.get(method)
    terminal = METHOD_TERMINALS.get(method)
    if not task or not terminal:
        return None
    pc_path = Path(analysis_root) / "pipeline_cache.json"
    if not pc_path.exists():
        return None
    try:
        with pc_path.open() as fh:
            cache = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", pc_path, exc)
        return None
    for entry in cache.values():
        t = (entry.get("tasks") or {}).get(task)
        if not t or t.get("status") != "complete":
            continue
        op = t.get("output_path")
        rec_key = entry.get("recording_key")
        compound = entry.get("well_id", "")
        if not op or not rec_key or "/" not in compound:
            continue
        rec_name, well_id = compound.split("/", 1)
        rel = Path(rec_key) / rec_name / well_id / terminal
        op_path = Path(op)
        # Only trust entries whose output_path actually ends with the expected
        # per-well suffix, then strip it to get the shared root.
        if op_path.parts[-len(rel.parts):] != rel.parts:
            continue
        root = op_path
        for _ in range(len(rel.parts)):
            root = root.parent
        return root
    return None


# ---------------------------------------------------------------------------
# Bundle + loader
# ---------------------------------------------------------------------------


@dataclass
class InspectorBundle:
    """Everything one well's diagnostic panels need.

    ``source`` lets the page show a "disk" vs "on-demand" status badge so the
    user knows whether they're looking at persisted task output or a live
    in-process re-run.

    ``method`` indicates which detector produced the data. ``results`` holds
    the standard ``BurstResults``.
    """

    spike_times: dict[str, np.ndarray]
    burstlets: pd.DataFrame
    recording_key: str
    rec_name: str
    well_id: str
    source: Literal["disk", "on_demand"]
    method: str = "ml"
    output_dir: Path | None = None
    results: Any = None  # BurstResults — stored for summary_card


def load_generic_bundle(
    output_root: Path,
    recording_key: str,
    rec_name: str,
    well_id: str,
    method: str,
) -> InspectorBundle:
    """Load a BurstResults bundle for any detector method.

    Used by the dashboard when the user picks traditional or ML. Reads the
    standard pickle output (burstlets.pkl, metrics.json, etc.) without
    requiring a debug trace.
    """
    from yuxin_mea.analysis.burst_output import PickleBurstOutputWriter

    terminal = METHOD_TERMINALS[method]
    output_dir = Path(output_root) / recording_key / rec_name / well_id / terminal
    if not output_dir.exists():
        raise FileNotFoundError(f"No output at {output_dir}")

    results = PickleBurstOutputWriter().read(output_dir)

    spike_times: dict[str, np.ndarray] = {}
    spikes_path = output_dir / "debug_spike_times.npy"
    if spikes_path.exists():
        spike_times = np.load(spikes_path, allow_pickle=True).item()

    return InspectorBundle(
        spike_times=spike_times,
        burstlets=results.burstlets,
        recording_key=recording_key,
        rec_name=rec_name,
        well_id=well_id,
        source="disk",
        method=method,
        output_dir=output_dir,
        results=results,
    )


def _empty_figure(message: str) -> go.Figure:
    """A single annotation centered in a blank figure — used for missing data states."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#888"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    return fig


def _spike_times_by_str_uid(
    spike_times: dict[Any, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Re-key spike_times to ``{str: array}`` so lookups tolerate int↔str uids."""
    if not spike_times:
        return {}
    return {str(k): v for k, v in spike_times.items()}


# ---------------------------------------------------------------------------
# summary_card — KV dict for the page, not a figure
# ---------------------------------------------------------------------------


def summary_card(bundle: InspectorBundle) -> dict[str, Any]:
    """Compact summary of detector state — rendered as a KV card by the page."""
    card: dict[str, Any] = {
        "well": f"{bundle.recording_key} / {bundle.rec_name} / {bundle.well_id}",
        "source": bundle.source,
        "method": bundle.method,
        "n_units": len(bundle.spike_times) if bundle.spike_times else "n/a",
        "n_burstlets": int(len(bundle.burstlets)),
    }
    if bundle.results is not None:
        card["n_network_bursts"] = int(len(bundle.results.network_bursts))
        card["n_superbursts"] = int(len(bundle.results.superbursts))
        for key in ("n_units", "adaptive_bin_ms", "burst_activity_detected"):
            val = bundle.results.diagnostics.get(key)
            if val is not None:
                card[key] = val
    return card


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_raster_basic(bundle: InspectorBundle, max_units: int = 64) -> go.Figure:
    """Spike raster with burstlet event overlays — no iteration trace needed."""
    spike_times = bundle.spike_times
    if not spike_times:
        return _empty_figure("No spike times available.")

    sp_by_uid = _spike_times_by_str_uid(spike_times)
    unit_ids = sorted(sp_by_uid.keys())
    if len(unit_ids) > max_units:
        step = len(unit_ids) // max_units
        unit_ids = unit_ids[::step][:max_units]

    fig = go.Figure()
    for row, uid in enumerate(unit_ids):
        spikes = np.asarray(sp_by_uid.get(uid, []), dtype=float)
        if spikes.size == 0:
            continue
        fig.add_trace(go.Scattergl(
            x=spikes, y=np.full(spikes.size, row),
            mode="markers",
            marker=dict(size=2.5, color="#222", line=dict(width=0)),
            hoverinfo="skip", showlegend=False, name=uid,
        ))

    for _, evt in bundle.burstlets.iterrows():
        fig.add_vrect(
            x0=float(evt["start"]), x1=float(evt["end"]),
            fillcolor="rgba(220, 80, 60, 0.12)", line_width=0, layer="below",
        )

    fig.update_layout(
        height=max(220, 12 * len(unit_ids)),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="time (s)", yaxis_title="unit", plot_bgcolor="white",
    )
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(len(unit_ids))),
        ticktext=unit_ids, autorange="reversed",
    )
    return fig


def fig_composite_basic(bundle: InspectorBundle) -> go.Figure:
    """Composite signal from plot_data with burstlet event overlays."""
    if bundle.results is None or not bundle.results.plot_data:
        return _empty_figure("No plot data available.")

    pd_data = bundle.results.plot_data
    t = np.asarray(pd_data.get("t", []), dtype=float)
    if t.size == 0:
        return _empty_figure("Empty time axis in plot data.")

    fig = go.Figure()
    for key, label, color in [
        ("participation_signal", "participation", "#1f77b4"),
        ("rate_signal", "PFR", "#ff7f0e"),
    ]:
        y = pd_data.get(key)
        if y is not None:
            fig.add_trace(go.Scattergl(
                x=t, y=np.asarray(y, dtype=float),
                mode="lines", name=label,
                line=dict(width=1.5, color=color),
            ))

    for _, evt in bundle.burstlets.iterrows():
        fig.add_vrect(
            x0=float(evt["start"]), x1=float(evt["end"]),
            fillcolor="rgba(220, 80, 60, 0.12)", line_width=0, layer="below",
        )

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="time (s)", yaxis_title="signal",
        plot_bgcolor="white", legend=dict(orientation="h", y=1.02),
    )
    return fig
