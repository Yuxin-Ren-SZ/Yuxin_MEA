"""Cluster-overlay aggregate tasks — the first users of the Scope system.

Two ways to compare the shared-axis bin-level UMAP/PCA embedding:

* :class:`ClusterOverlayByWellTask` — scope ``recording`` (the 24 wells of one
  Network recording); one shared fit, **color = well** (static overlay).
* :class:`ClusterOverlayByRecordingTask` — scope ``longitudinal_well`` (one
  physical well across its recordings); one shared fit, **color = recording**
  (time-lapse animation over DIV).

Both depend on the per-well ``ml_burst_detection`` task and read each member's
``debug_trace.pkl`` from its ``TaskRecord.output_path``. Output:
``<output_root>/__agg__/<task_name>/<scope_key>/overlay.html``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from yuxin_mea.analysis.cluster_overlay import (
    build_overlay_figure,
    load_series_trace,
    pool_and_fit,
)
from yuxin_mea.pipeline.aggregate_task import BaseAggregateTask
from yuxin_mea.pipeline.scope import SCOPE_RECORDING, SCOPE_WELL_UID
from yuxin_mea.pipeline.well_dims import Member

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "norm": "shared",          # shared | per-recording
    "palette": "dark24",
    "method": "umap",          # primary view; the other is also in the dropdown
    "max_bins_per_rec": 1500,
    "n_neighbors": 30,
    "min_dist": 0.1,
    "seed": 42,
    # output_root: falls back to the config's figure_root, injected by the CLI
    # aggregate drain when absent.
    "output_root": None,
}


def _render_overlay(
    task: BaseAggregateTask,
    scope_key: dict[str, str],
    members: list[Member],
    params: dict[str, Any],
    *,
    color_by: str,
    animate: bool,
    label_fn: Callable[[Member], str],
    sort_key: Callable[[Member], Any],
    title_label: str,
) -> Path:
    p = task.resolve_params(params)
    output_root = p.get("output_root")
    if not output_root:
        raise ValueError(
            f"{task.task_name}: no output_root in params (config or CLI must set it / "
            "provide a figure_root global)."
        )

    series = []
    feat: list[str] | None = None
    for m in sorted(members, key=sort_key):
        if m.upstream_output_path is None:
            logger.warning("%s: member %s has no upstream output path — skipping",
                           task.task_name, m.compound_well_id)
            continue
        st, feat = load_series_trace(
            Path(m.upstream_output_path), label_fn(m), int(p["max_bins_per_rec"]), feat
        )
        if st is not None:
            series.append(st)
        else:
            logger.warning("%s: no usable trace for %s/%s — skipping",
                           task.task_name, m.recording_key, m.compound_well_id)

    if not series:
        raise RuntimeError(
            f"{task.task_name}: no usable member traces for scope {scope_key} "
            f"(of {len(members)} complete members)."
        )

    method_coords = pool_and_fit(
        series, p["norm"], feat, p["method"],
        int(p["n_neighbors"]), float(p["min_dist"]), int(p["seed"]),
    )
    fig = build_overlay_figure(
        [(title_label, series, method_coords)],
        color_by=color_by, animate=animate, palette=p["palette"], norm=p["norm"],
    )
    out = task.build_output_path(output_root, task.task_name, scope_key) / "overlay.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="inline", full_html=True)
    logger.info("%s: wrote %s (%d series)", task.task_name, out, len(series))
    return out


class ClusterOverlayByWellTask(BaseAggregateTask):
    """One shared-axis overlay per recording; color = well (compare wells)."""

    task_name = "cluster_overlay_by_well"
    dependencies = ["ml_burst_detection"]
    scope = SCOPE_RECORDING

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return dict(_DEFAULT_PARAMS)

    def run(self, scope_key, members, params) -> Path:
        return _render_overlay(
            self, scope_key, members, params,
            color_by="well", animate=False,
            label_fn=lambda m: m.dims.well_id,
            sort_key=lambda m: m.dims.well_id,
            title_label=scope_key.get("recording_key", "recording"),
        )


class ClusterOverlayByRecordingTask(BaseAggregateTask):
    """One shared-axis overlay per physical well; color = recording (time-lapse)."""

    task_name = "cluster_overlay_by_recording"
    dependencies = ["ml_burst_detection"]
    scope = SCOPE_WELL_UID

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return dict(_DEFAULT_PARAMS)

    def run(self, scope_key, members, params) -> Path:
        return _render_overlay(
            self, scope_key, members, params,
            color_by="recording", animate=True,
            label_fn=lambda m: f"{m.dims.date} {m.dims.run_id}",
            sort_key=lambda m: (m.dims.date, m.dims.run_id, m.dims.rec_name),
            title_label=scope_key.get("well_uid", "well"),
        )
