#!/usr/bin/env python3
"""Render duration-labeled raster HTML for wells where ML likely over-detects.

``compare_burst_methods.py`` shows ML calls many more bursts than the
traditional detector in a large slice of wells (e.g. traditional finds 0 while
ML finds hundreds). Those are over-detection *suspects* -- but only a human
looking at the raster can judge whether the extra ML events are real bursts or
noise fragments. This script produces that view: one standalone Plotly HTML per
suspect well, showing the curated-spike raster with each detector's bursts drawn
as shaded, **duration-labeled** spans (ML on top, traditional below) so the
extra ML events are immediately visible against the spikes.

Suspect selection (from per_well.csv):
    ratio = (ml_count + 1) / (trad_count + 1) >= --min-ratio
    AND ml_count >= --min-ml
ranked by excess = ml_count - trad_count, top --top wells.

Directories (curated spikes + both detectors' network_bursts.pkl) come from the
authoritative pipeline_cache.json, same as compare_burst_methods.py.

Run (after compare_burst_methods.py has written per_well.csv)::

    python scripts/render_overdetection_rasters.py --config pipeline_config.json

Outputs: <figure_root>/burst_method_comparison/overdetection_rasters/
         {recording}__{well_name}_{well_id}.html  + index.html
"""

from __future__ import annotations

import argparse
import html
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Reuse loaders from the comparison script (same scripts/ dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_burst_methods import (  # noqa: E402
    ML_TASK, TRAD_TASK, CUR_TASK,
    _load_config, _load_pipeline_cache, _load_experiment_cache,
    _group_lookup, _make_resolver, _read_event_table,
)

logger = logging.getLogger("render_overdetection_rasters")

MAX_RASTER_POINTS = 14_000
ML_COLOR = "rgba(214, 39, 40, 0.55)"      # red  -- the over-detecting method
TRAD_COLOR = "rgba(31, 119, 180, 0.55)"   # blue -- reference
RASTER_COLOR = "rgba(40, 40, 40, 0.85)"


# --------------------------------------------------------------------------- #
# Discovery: over-detection suspects + their directories
# --------------------------------------------------------------------------- #
def _dir_index(pipeline_cache: dict) -> dict[tuple[str, str], dict[str, str]]:
    """{(recording_key, bare_well_id): {trad, ml, cur output_path}} for
    both-complete wells."""
    out: dict[tuple[str, str], dict[str, str]] = {}
    for entry in pipeline_cache.values():
        tasks = entry.get("tasks", {})
        if tasks.get(TRAD_TASK, {}).get("status") != "complete":
            continue
        if tasks.get(ML_TASK, {}).get("status") != "complete":
            continue
        compound = entry.get("well_id", "")
        if "/" not in compound:
            continue
        _, well_id = compound.split("/", 1)
        out[(entry["recording_key"], well_id)] = {
            "trad": tasks[TRAD_TASK].get("output_path"),
            "ml": tasks[ML_TASK].get("output_path"),
            "cur": tasks.get(CUR_TASK, {}).get("output_path"),
        }
    return out


def select_suspects(per_well: pd.DataFrame, min_ml: int, min_ratio: float,
                    top: int) -> pd.DataFrame:
    d = per_well.copy()
    d["ratio"] = (d["ml_count"] + 1.0) / (d["trad_count"] + 1.0)
    d["excess"] = d["ml_count"] - d["trad_count"]
    mask = (d["ml_count"] >= min_ml) & (d["ratio"] >= min_ratio)
    sel = d[mask].sort_values("excess", ascending=False)
    logger.info("suspects: %d wells (ml>=%d, ratio>=%.1f); rendering top %d",
                len(sel), min_ml, min_ratio, min(top, len(sel)))
    return sel.head(top)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _load_spikes(cur_dir: Path | None) -> dict[Any, np.ndarray]:
    if cur_dir is None:
        return {}
    p = cur_dir / "curated_spike_times.npy"
    if not p.exists():
        return {}
    try:
        return np.load(p, allow_pickle=True).item()
    except Exception as exc:  # noqa: BLE001
        logger.warning("spike load failed %s: %s", p, exc)
        return {}


def _raster_traces(spikes: dict[Any, np.ndarray]) -> tuple[list[go.Scattergl], float]:
    items: list[tuple[str, np.ndarray]] = []
    for uid, st in spikes.items():
        arr = np.asarray(st, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            items.append((str(uid), arr))
    if not items:
        return [], 0.0
    t_end = max(float(a.max()) for _, a in items)
    items.sort(key=lambda kv: kv[1].size, reverse=True)  # busy units at bottom
    n_units = len(items)
    per_unit_cap = max(30, MAX_RASTER_POINTS // n_units)
    traces: list[go.Scattergl] = []
    for rank, (uid, sp) in enumerate(items):
        if sp.size > per_unit_cap:
            sp = sp[:: int(np.ceil(sp.size / per_unit_cap))]
        traces.append(go.Scattergl(
            x=sp, y=np.full(sp.size, rank, dtype=float),
            mode="markers",
            marker=dict(size=3.5, color=RASTER_COLOR, symbol="line-ns-open",
                        line=dict(width=0.5, color=RASTER_COLOR)),
            hoverinfo="skip", showlegend=False, name=f"u{uid}",
        ))
    return traces, t_end


def _burst_lane(events: list[dict], y_center: float, color: str, name: str,
                label: bool) -> list[go.Scatter]:
    """Shaded rectangles for each burst + a duration text label at the top."""
    if not events:
        return []
    y_lo, y_hi = y_center - 0.42, y_center + 0.42
    xs: list[float | None] = []
    ys: list[float | None] = []
    cd: list[list[Any]] = []
    tx: list[float] = []
    ty: list[float] = []
    txt: list[str] = []
    for ev in events:
        s, e = float(ev["start"]), float(ev["end"])
        dur = float(ev.get("duration_s", e - s))
        xs.extend([s, e, e, s, s, None])
        ys.extend([y_lo, y_lo, y_hi, y_hi, y_lo, None])
        row = [name, s, e, dur]
        cd.extend([row, row, row, row, row, [None] * 4])
        tx.append((s + e) / 2.0)
        ty.append(y_hi)
        txt.append(f"{dur:.2f}")
    traces: list[go.Scatter] = [go.Scatter(
        x=xs, y=ys, mode="lines", fill="toself",
        line=dict(width=0.5, color=color), fillcolor=color,
        customdata=cd,
        hovertemplate=("%{customdata[0]}"
                       "<br>start: %{customdata[1]:.3f}s"
                       "<br>end: %{customdata[2]:.3f}s"
                       "<br>duration: %{customdata[3]:.3f}s<extra></extra>"),
        name=name, showlegend=False,
    )]
    if label:
        traces.append(go.Scatter(
            x=tx, y=ty, mode="text", text=txt,
            textposition="top center", textfont=dict(size=7, color="#333"),
            hoverinfo="skip", showlegend=False, name=f"{name} dur",
        ))
    return traces


def build_well_figure(well_name: str, groupname: str, recording_key: str,
                      spikes: dict, ml_ev: list[dict], trad_ev: list[dict],
                      label: bool) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.14, 0.14, 0.72],
        subplot_titles=(f"ML bursts (n={len(ml_ev)})",
                        f"Traditional bursts (n={len(trad_ev)})",
                        "Curated-spike raster"),
    )
    for tr in _burst_lane(ml_ev, 0.0, ML_COLOR, "ML burst", label):
        fig.add_trace(tr, row=1, col=1)
    for tr in _burst_lane(trad_ev, 0.0, TRAD_COLOR, "Traditional burst", label):
        fig.add_trace(tr, row=2, col=1)
    raster, t_end = _raster_traces(spikes)
    for tr in raster:
        fig.add_trace(tr, row=3, col=1)

    for r in (1, 2):
        fig.update_yaxes(range=[-0.6, 0.8], showticklabels=False, showgrid=False,
                         zeroline=False, ticks="", row=r, col=1)
    fig.update_yaxes(title_text="unit (by rate)", row=3, col=1)
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    if t_end > 0:
        fig.update_xaxes(range=[0, t_end])
    fig.update_layout(
        title=(f"<b>{html.escape(well_name)}</b> / {html.escape(groupname)}"
               f" &nbsp;&mdash;&nbsp; {html.escape(recording_key)}"
               f" &nbsp;&mdash;&nbsp; ML={len(ml_ev)} vs traditional={len(trad_ev)}"
               " network bursts"),
        template="plotly_white", height=760, margin=dict(l=60, r=20, t=70, b=45),
        hovermode="closest",
    )
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--per-well-csv", type=Path, default=None,
                   help="Default <figure_root>/burst_method_comparison/per_well.csv")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default <figure_root>/burst_method_comparison/overdetection_rasters")
    p.add_argument("--min-ml", type=int, default=15,
                   help="Minimum ML burst count to be a suspect.")
    p.add_argument("--min-ratio", type=float, default=3.0,
                   help="Minimum (ml+1)/(trad+1) ratio to be a suspect.")
    p.add_argument("--top", type=int, default=24, help="Render this many worst wells.")
    p.add_argument("--no-labels", action="store_true",
                   help="Skip per-burst duration text (keep hover only).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    analysis_root, figure_root = _load_config(args.config)
    comp_dir = figure_root / "burst_method_comparison"
    csv_path = args.per_well_csv or (comp_dir / "per_well.csv")
    if not csv_path.exists():
        raise SystemExit(f"per_well.csv not found at {csv_path}; run compare_burst_methods.py first")
    per_well = pd.read_csv(csv_path)

    pipeline_cache = _load_pipeline_cache(analysis_root)
    dir_idx = _dir_index(pipeline_cache)
    resolve = _make_resolver([analysis_root])

    suspects = select_suspects(per_well, args.min_ml, args.min_ratio, args.top)
    if suspects.empty:
        raise SystemExit("no over-detection suspects matched the criteria")

    out_dir = args.output_dir or (comp_dir / "overdetection_rasters")
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[str] = []
    n_written = 0
    for _, r in suspects.iterrows():
        key = (r["recording_key"], r["well_id"])
        dirs = dir_idx.get(key)
        if dirs is None:
            logger.warning("no pipeline_cache dirs for %s / %s; skipping", *key)
            continue
        ml_dir = resolve(dirs["ml"])
        trad_dir = resolve(dirs["trad"])
        cur_dir = resolve(dirs["cur"])
        ml_ev = _read_event_table(ml_dir / "network_bursts.pkl") if ml_dir else []
        trad_ev = _read_event_table(trad_dir / "network_bursts.pkl") if trad_dir else []
        spikes = _load_spikes(cur_dir)
        if not spikes:
            logger.warning("no curated spikes for %s / %s; skipping", *key)
            continue

        fig = build_well_figure(str(r["well_name"]), str(r["groupname"]),
                                str(r["recording_key"]), spikes, ml_ev, trad_ev,
                                label=not args.no_labels)
        rec_slug = str(r["recording_key"]).replace("/", "-")
        fname = f"{rec_slug}__{r['well_name']}_{r['well_id']}.html"
        fig.write_html(out_dir / fname, include_plotlyjs="cdn", full_html=True)
        n_written += 1
        index_rows.append(
            f'<tr><td>{n_written}</td>'
            f'<td><a href="{html.escape(fname)}">{html.escape(str(r["well_name"]))}</a></td>'
            f'<td>{html.escape(str(r["groupname"]))}</td>'
            f'<td>{html.escape(str(r["recording_key"]))}</td>'
            f'<td>{int(r["trad_count"])}</td><td>{int(r["ml_count"])}</td>'
            f'<td>{r["ratio"]:.1f}</td></tr>')
        logger.info("wrote %s  (trad=%d ml=%d)", fname, int(r["trad_count"]), int(r["ml_count"]))

    index = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>ML over-detection suspects</title>"
        "<style>body{font-family:sans-serif;margin:2rem;max-width:900px}"
        "table{border-collapse:collapse;width:100%;font-size:.9rem}"
        "th,td{border:1px solid #ddd;padding:5px 9px;text-align:right}"
        "td:nth-child(2),td:nth-child(3),td:nth-child(4){text-align:left}"
        "th{background:#f4f4f4}</style></head><body>"
        f"<h1>ML over-detection suspects ({n_written} wells)</h1>"
        f"<p>Selected: ml_count &ge; {args.min_ml}, (ml+1)/(trad+1) &ge; {args.min_ratio}; "
        "ranked by ml&minus;trad excess. Each raster shows ML bursts (red) and "
        "traditional bursts (blue) with duration labels over the curated spikes.</p>"
        "<table><tr><th>#</th><th>well</th><th>group</th><th>recording</th>"
        "<th>trad</th><th>ML</th><th>ratio</th></tr>"
        + "".join(index_rows) + "</table></body></html>")
    (out_dir / "index.html").write_text(index)

    logger.info("wrote %d well HTMLs + index.html to %s", n_written, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
