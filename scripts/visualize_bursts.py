"""Render 24-well plate raster + burst-event HTML for each recording.

Produces one standalone Plotly HTML per (recording_key, detector). Each cell of
the 4x6 plate shows:
  - a raster of curated spikes (units ranked by firing rate; rich hover with
    unit id, time, firing rate, and bin info when available),
  - a marginal "burst events" lane with three sub-tracks: burstlets,
    network_bursts, superbursts. For the ML / iterative detector the tracks
    represent the three hierarchy levels; "all labels" = burstlets and
    "burst labels" = network_bursts.

Reads pipeline_cache.json (next to analysis_root) to find which wells have
completed outputs. Reads experiment_cache.json for groupname / well_name.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger("visualize_bursts")

DETECTORS = {
    "traditional": {
        "task_name": "burst_detection",
        "subdir": "burst_detection",
        "title": "Traditional burst detector",
    },
    "iterative": {
        "task_name": "iterative_burst_detection",
        "subdir": "iterative_burst_detection",
        "title": "ML / iterative burst detector",
    },
}

EVENT_TRACKS = [
    ("burstlets", "Burstlets (all labels)", "rgba(31, 119, 180, 0.55)"),
    ("network_bursts", "Network bursts (burst labels)", "rgba(255, 127, 14, 0.65)"),
    ("superbursts", "Superbursts", "rgba(148, 103, 189, 0.55)"),
]

RASTER_COLOR = "rgba(40, 40, 40, 0.85)"

PLATE_ROWS, PLATE_COLS = 4, 6
PLATE_WELLS = PLATE_ROWS * PLATE_COLS
MAX_RASTER_POINTS_PER_WELL = 12_000

# In the 8-row × 6-col grid, each well occupies a (marginal, raster) row pair.
MARGINAL_HEIGHT = 0.28
RASTER_HEIGHT = 0.72
N_EVENT_LANES = 3   # burstlets, network_bursts, superbursts


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class WellData:
    well_id: str          # "well000"
    well_name: str        # "A1"
    groupname: str        # "NPH"
    rec_name: str | None  # "rec0000" etc., None if unknown
    spike_times: dict[str, np.ndarray] | None      # unit_id -> seconds
    plot_signals: dict[str, Any] | None            # detector's plot_signals.npy
    event_intervals: dict[str, list[dict]]         # type -> list of {start, end, ...}


def _well_position(well_id: str) -> tuple[int, int]:
    n = int(well_id.replace("well", ""))
    return n // PLATE_COLS, n % PLATE_COLS


def _load_pipeline_cache(analysis_root: Path) -> dict[str, Any]:
    path = analysis_root / "pipeline_cache.json"
    if not path.exists():
        raise SystemExit(f"pipeline_cache.json not found at {path}")
    with path.open() as fh:
        return json.load(fh)


def _load_experiment_cache(analysis_root: Path) -> dict[str, Any]:
    path = analysis_root / "experiment_cache.json"
    if not path.exists():
        logger.warning("experiment_cache.json not found at %s; groupnames will be '?'", path)
        return {}
    with path.open() as fh:
        return json.load(fh)


def _read_event_table(pkl_path: Path) -> list[dict[str, Any]]:
    if not pkl_path.exists():
        return []
    try:
        df = pd.read_pickle(pkl_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", pkl_path, exc)
        return []
    if df is None or df.empty or "start" not in df or "end" not in df:
        return []
    rows: list[dict[str, Any]] = []
    for r in df.to_dict(orient="records"):
        try:
            s = float(r["start"])
            e = float(r["end"])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(s) and np.isfinite(e)) or e <= s:
            continue
        clean: dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, (np.generic,)):
                v = v.item()
            if isinstance(v, float) and not np.isfinite(v):
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
        clean["start"], clean["end"] = s, e
        rows.append(clean)
    return rows


def _collect_recording_wells(
    pipeline_cache: dict[str, Any],
    detector_task: str,
) -> dict[str, list[tuple[str, str]]]:
    """Return {recording_key: [(rec_name, well_id), ...]} for entries where
    auto_curation AND the requested detector are complete."""
    out: dict[str, list[tuple[str, str]]] = {}
    for entry in pipeline_cache.values():
        tasks = entry.get("tasks", {})
        if tasks.get("auto_curation", {}).get("status") != "complete":
            continue
        if tasks.get(detector_task, {}).get("status") != "complete":
            continue
        rec_key = entry["recording_key"]
        compound = entry["well_id"]
        if "/" not in compound:
            continue
        rec_name, well_id = compound.split("/", 1)
        out.setdefault(rec_key, []).append((rec_name, well_id))
    return out


def _make_resolver(search_roots: list[Path]):
    def _resolve(rel: str | None) -> Path | None:
        if not rel:
            return None
        p = Path(rel)
        if p.is_absolute():
            return p if p.exists() else None
        for base in search_roots:
            cand = (base / p).resolve()
            if cand.exists():
                return cand
        return None
    return _resolve


def _load_well_data(
    resolve: Any,
    detector_subdir: str,
    pipeline_entry: dict[str, Any],
    well_meta: dict[str, Any],
) -> WellData:
    tasks = pipeline_entry["tasks"]
    compound = pipeline_entry["well_id"]
    rec_name, well_id = compound.split("/", 1)

    cur_dir = resolve(tasks["auto_curation"].get("output_path"))
    cur_path = cur_dir / "curated_spike_times.npy" if cur_dir else None
    burst_dir = resolve(
        tasks.get(_detector_task_for(detector_subdir), {}).get("output_path")
    )

    spike_times = None
    if cur_path is not None and cur_path.exists():
        try:
            spike_times = np.load(cur_path, allow_pickle=True).item()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load spike times %s: %s", cur_path, exc)

    plot_signals = None
    events: dict[str, list[dict]] = {k: [] for k, *_ in EVENT_TRACKS}
    if burst_dir is not None and burst_dir.exists():
        ps_path = burst_dir / "plot_signals.npy"
        if ps_path.exists():
            try:
                plot_signals = np.load(ps_path, allow_pickle=True).item()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load plot_signals %s: %s", ps_path, exc)
        for key, *_ in EVENT_TRACKS:
            events[key] = _read_event_table(burst_dir / f"{key}.pkl")

    return WellData(
        well_id=well_id,
        well_name=well_meta.get("well_name", "?"),
        groupname=well_meta.get("groupname", "?"),
        rec_name=rec_name,
        spike_times=spike_times,
        plot_signals=plot_signals,
        event_intervals=events,
    )


def _detector_task_for(subdir: str) -> str:
    for cfg in DETECTORS.values():
        if cfg["subdir"] == subdir:
            return cfg["task_name"]
    raise KeyError(subdir)


# ---------------------------------------------------------------------------
# Figure building
# ---------------------------------------------------------------------------


def _firing_rate(spike_times: np.ndarray, t_end: float) -> float:
    if t_end <= 0 or spike_times.size == 0:
        return 0.0
    return float(spike_times.size / t_end)


def _bin_centers(plot_signals: dict[str, Any] | None) -> np.ndarray | None:
    if plot_signals is None:
        return None
    t = plot_signals.get("t")
    if t is None or len(t) < 2:
        return None
    return np.asarray(t, dtype=float)


def _build_raster_traces(
    well: WellData,
) -> tuple[list[go.Scattergl], int, float]:
    """Return (traces, n_units, t_end). Units sorted by firing rate desc; y=rank."""
    if not well.spike_times:
        return [], 0, 0.0

    items: list[tuple[str, np.ndarray]] = []
    for unit_id, st in well.spike_times.items():
        arr = np.asarray(st, dtype=float)
        arr = arr[np.isfinite(arr)]
        items.append((str(unit_id), arr))

    t_end = max((float(a.max()) if a.size else 0.0 for _, a in items), default=0.0)
    items.sort(key=lambda kv: _firing_rate(kv[1], t_end), reverse=True)

    bins = _bin_centers(well.plot_signals)
    p_signal = well.plot_signals.get("participation_signal") if well.plot_signals else None
    r_signal = well.plot_signals.get("rate_signal") if well.plot_signals else None
    p_arr = np.asarray(p_signal, dtype=float) if p_signal is not None else None
    r_arr = np.asarray(r_signal, dtype=float) if r_signal is not None else None

    traces: list[go.Scattergl] = []
    for rank, (uid, spikes) in enumerate(items):
        if spikes.size == 0:
            continue
        rate = _firing_rate(spikes, t_end)

        # stride-downsample so total points across all units stay reasonable
        n_units = len(items) or 1
        per_unit_cap = max(40, MAX_RASTER_POINTS_PER_WELL // n_units)
        if spikes.size > per_unit_cap:
            stride = int(np.ceil(spikes.size / per_unit_cap))
            spikes = spikes[::stride]

        # bin lookup
        if bins is not None:
            idx = np.clip(np.searchsorted(bins, spikes) - 1, 0, len(bins) - 1)
            bin_idx = idx.astype(int)
            bin_p = p_arr[idx] if p_arr is not None and p_arr.size == bins.size else np.full_like(spikes, np.nan)
            bin_r = r_arr[idx] if r_arr is not None and r_arr.size == bins.size else np.full_like(spikes, np.nan)
        else:
            bin_idx = np.full(spikes.size, -1, dtype=int)
            bin_p = np.full(spikes.size, np.nan)
            bin_r = np.full(spikes.size, np.nan)

        customdata = np.column_stack([
            np.full(spikes.size, uid, dtype=object),
            spikes.astype(float),
            np.full(spikes.size, rate, dtype=float),
            bin_idx.astype(float),
            bin_p.astype(float),
            bin_r.astype(float),
        ])

        traces.append(go.Scattergl(
            x=spikes,
            y=np.full(spikes.size, rank, dtype=float),
            mode="markers",
            marker=dict(
                size=4.0,
                color=RASTER_COLOR,
                symbol="line-ns-open",
                line=dict(width=0.6, color=RASTER_COLOR),
            ),
            customdata=customdata,
            hovertemplate=(
                "unit: %{customdata[0]}"
                "<br>t: %{customdata[1]:.4f}s"
                "<br>rate: %{customdata[2]:.2f} Hz"
                "<br>bin: #%{customdata[3]:.0f}"
                " (P=%{customdata[4]:.3f}, rate=%{customdata[5]:.2f})"
                "<extra></extra>"
            ),
            showlegend=False,
            name=uid,
        ))
    return traces, len(items), t_end


def _build_event_traces(well: WellData) -> list[go.Scattergl]:
    """One trace per event track. Drawn as filled rectangles inside the
    marginal panel (lanes stacked at y=0,1,2 covering ±0.4 around the lane
    center). Returns exactly len(EVENT_TRACKS) traces (in EVENT_TRACKS order),
    using empty placeholder traces when a track has no events — so trace
    indices remain stable across wells for visibility toggling.
    """
    traces: list[go.Scattergl] = []
    for lane_idx, (key, label, color) in enumerate(EVENT_TRACKS):
        events = (well.event_intervals or {}).get(key) or []
        if not events:
            traces.append(go.Scattergl(
                x=[None], y=[None], mode="lines",
                line=dict(width=0, color=color),
                name=label, legendgroup=key, showlegend=False, hoverinfo="skip",
            ))
            continue
        y_center = float(lane_idx)
        y_lo = y_center - 0.4
        y_hi = y_center + 0.4
        xs: list[float | None] = []
        ys: list[float | None] = []
        cd: list[list[Any]] = []
        for ev in events:
            s = ev["start"]
            e = ev["end"]
            xs.extend([s, e, e, s, s, None])
            ys.extend([y_lo, y_lo, y_hi, y_hi, y_lo, None])
            duration = ev.get("duration_s", e - s)
            peak = ev.get("peak_synchrony", ev.get("burst_peak", float("nan")))
            base = [label, s, e, duration, peak]
            cd.extend([base, base, base, base, base, [None] * 5])
        traces.append(go.Scattergl(
            x=xs, y=ys, mode="lines",
            fill="toself",
            line=dict(width=0.5, color=color),
            fillcolor=color,
            customdata=cd,
            hovertemplate=(
                "%{customdata[0]}"
                "<br>start: %{customdata[1]:.3f}s"
                "<br>end: %{customdata[2]:.3f}s"
                "<br>duration: %{customdata[3]:.3f}s"
                "<br>peak: %{customdata[4]:.3f}"
                "<extra></extra>"
            ),
            name=label,
            legendgroup=key,
            showlegend=False,
        ))
    return traces


def _axis_id(grid_row_1: int, col_1: int, n_cols: int) -> str:
    """Plotly subplot axis-id for (1-indexed) grid_row, col in an n_cols grid."""
    n = (grid_row_1 - 1) * n_cols + col_1
    return "x" if n == 1 else f"x{n}"


def build_recording_figure(
    recording_key: str,
    detector_title: str,
    wells: list[WellData],
) -> go.Figure:
    """4×6 plate; each well = (marginal events panel above, raster below) with
    x-axes linked per cell so zoom in the marginal drives the raster."""
    # Build a 24-slot list keyed by plate position; absent wells -> None
    slot: list[WellData | None] = [None] * PLATE_WELLS
    for w in wells:
        try:
            n = int(w.well_id.replace("well", ""))
        except ValueError:
            continue
        if 0 <= n < PLATE_WELLS:
            slot[n] = w

    # 8 grid rows: pairs of (marginal, raster) per well row. Titles go on the
    # marginal rows so they sit at the top of each well's pair.
    n_grid_rows = PLATE_ROWS * 2
    row_heights: list[float] = []
    for _ in range(PLATE_ROWS):
        row_heights.extend([MARGINAL_HEIGHT, RASTER_HEIGHT])

    titles: list[str] = []
    for grid_row in range(n_grid_rows):
        is_marginal = (grid_row % 2 == 0)
        for col in range(PLATE_COLS):
            if not is_marginal:
                titles.append("")
                continue
            well_row = grid_row // 2
            n = well_row * PLATE_COLS + col
            w = slot[n]
            titles.append(
                f"well{n:03d} / (missing)" if w is None
                else f"{w.well_name} / {w.groupname}"
            )

    fig = make_subplots(
        rows=n_grid_rows, cols=PLATE_COLS,
        row_heights=row_heights,
        vertical_spacing=0.012,
        horizontal_spacing=0.025,
        subplot_titles=titles,
    )

    event_trace_indices: dict[str, list[int]] = {k: [] for k, *_ in EVENT_TRACKS}

    for n, well in enumerate(slot):
        well_row = n // PLATE_COLS
        col_1 = n % PLATE_COLS + 1
        margin_grid_row = 2 * well_row + 1  # 1-indexed
        raster_grid_row = 2 * well_row + 2

        # Link the marginal x-axis to the raster x-axis for this cell.
        raster_axis = _axis_id(raster_grid_row, col_1, PLATE_COLS)
        fig.update_xaxes(matches=raster_axis,
                         showticklabels=False,
                         row=margin_grid_row, col=col_1)

        # Clean look on the marginal y-axis.
        fig.update_yaxes(
            range=[-0.6, N_EVENT_LANES - 0.4],
            showticklabels=False, showgrid=False, zeroline=False, ticks="",
            row=margin_grid_row, col=col_1,
        )

        if well is None or not well.spike_times:
            fig.add_annotation(
                text="no data", x=0.5, y=0.5, showarrow=False,
                xref=f"x{(raster_grid_row - 1) * PLATE_COLS + col_1} domain",
                yref=f"y{(raster_grid_row - 1) * PLATE_COLS + col_1} domain",
                font=dict(size=10, color="rgba(120,120,120,0.7)"),
                row=raster_grid_row, col=col_1,
            )
            continue

        raster_traces, n_units, t_end = _build_raster_traces(well)
        for tr in raster_traces:
            fig.add_trace(tr, row=raster_grid_row, col=col_1)

        event_traces = _build_event_traces(well)
        for (key, *_), tr in zip(EVENT_TRACKS, event_traces):
            fig.add_trace(tr, row=margin_grid_row, col=col_1)
            event_trace_indices[key].append(len(fig.data) - 1)

        if well_row == PLATE_ROWS - 1:
            fig.update_xaxes(title_text="time (s)", row=raster_grid_row, col=col_1)
        if col_1 == 1:
            fig.update_yaxes(title_text="unit rank", row=raster_grid_row, col=col_1)
        if t_end > 0:
            fig.update_xaxes(range=[0, t_end], row=raster_grid_row, col=col_1)

    # Updatemenu: toggle each event track
    n_traces = len(fig.data)
    base_visible = [True] * n_traces

    def _with(visible_keys: set[str]) -> list[bool]:
        v = list(base_visible)
        for key, *_ in EVENT_TRACKS:
            on = key in visible_keys
            for idx in event_trace_indices[key]:
                v[idx] = on
        return v

    fig.update_layout(
        title=dict(
            text=f"<b>{recording_key}</b> — {detector_title}",
            x=0.01, xanchor="left",
            y=0.985, yanchor="top",
        ),
        height=PLATE_ROWS * 280 + 120,
        width=PLATE_COLS * 340,
        margin=dict(l=40, r=20, t=70, b=90),
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=11),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5, xanchor="center",
            y=-0.02, yanchor="top",
            pad=dict(t=4, b=4),
            buttons=[
                dict(label="All tracks",
                     method="update",
                     args=[{"visible": _with({k for k, *_ in EVENT_TRACKS})}]),
                dict(label="Bursts only",
                     method="update",
                     args=[{"visible": _with({"network_bursts"})}]),
                dict(label="All labels (burstlets)",
                     method="update",
                     args=[{"visible": _with({"burstlets"})}]),
                dict(label="Hide events",
                     method="update",
                     args=[{"visible": _with(set())}]),
            ],
            showactive=False,
        )],
    )

    fig.update_xaxes(showgrid=False, zeroline=False, ticks="outside")
    fig.update_yaxes(showgrid=False, zeroline=False, ticks="outside")
    # Annotations (subplot titles) — shrink font so they don't crowd the
    # marginal panel.
    for ann in fig.layout.annotations:
        ann.font = dict(size=10)
    return fig


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="Path to pipeline_config.json (read for analysis_root + figure_root).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output directory. Defaults to global.figure_root.")
    p.add_argument("--detector", choices=["traditional", "iterative", "both"],
                   default="both")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s — %(message)s")

    with args.config.open() as fh:
        cfg = json.load(fh)
    analysis_root = Path(cfg["global"]["analysis_root"])
    figure_root = Path(args.output_dir or cfg["global"].get("figure_root") or analysis_root)
    figure_root.mkdir(parents=True, exist_ok=True)

    pipeline_cache = _load_pipeline_cache(analysis_root)
    experiment_cache = _load_experiment_cache(analysis_root)

    config_dir = args.config.resolve().parent
    resolve = _make_resolver([config_dir, analysis_root, Path.cwd().resolve()])

    selected = (
        list(DETECTORS.keys()) if args.detector == "both" else [args.detector]
    )

    n_written = 0
    for det_key in selected:
        det = DETECTORS[det_key]
        recordings = _collect_recording_wells(pipeline_cache, det["task_name"])
        if not recordings:
            logger.info("No complete %s outputs found; skipping detector.", det_key)
            continue

        for rec_key, pairs in sorted(recordings.items()):
            well_meta_all = (
                experiment_cache.get(rec_key, {}).get("wells", {}) or {}
            )

            wells_data: list[WellData] = []
            for rec_name, well_id in pairs:
                compound_key = f"{rec_key}/{rec_name}/{well_id}"
                entry = pipeline_cache.get(compound_key)
                if entry is None:
                    continue
                meta = well_meta_all.get(well_id, {}).get("metadata", {})
                wells_data.append(_load_well_data(resolve, det["subdir"], entry, meta))

            if not wells_data:
                continue

            fig = build_recording_figure(rec_key, det["title"], wells_data)
            out_path = figure_root / rec_key / f"burst_viewer_{det_key}.html"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
            n_written += 1
            logger.info("Wrote %s (%d wells)", out_path, len(wells_data))

    logger.info("Done. %d HTML file(s) written under %s", n_written, figure_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
