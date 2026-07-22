#!/usr/bin/env python3
"""Render duration-labeled raster HTML for wells where ML and traditional
detection diverge significantly.

For each well ``compare_burst_methods.py`` writes per-metric ML-vs-traditional
log2 ratios (burst count, duration, spikes/burst, participation, peak
synchrony). Most wells agree; a tail diverges. Only a human looking at the
raster can judge whether the divergence is a real difference or an artifact.
This script produces that view: one standalone Plotly HTML per divergent well,
showing the curated-spike raster with each detector's bursts drawn as shaded,
**duration-labeled** spans (ML on top, traditional below).

Two selection modes (``--select``):

* ``divergent`` (default) -- flag wells that are distribution outliers on **any**
  metric, in **either** direction. For metric ``m`` the per-well score is a
  robust, zero-centered z::

      z_m = log2ratio_m / (1.4826 * MAD(log2ratio_m))   # MAD is the scale only

  Zero-centered (not median-centered) so the score measures "changed vs *no
  method difference*", not "atypical vs the *typical* well" -- otherwise a
  metric with a systematic shift (ML runs ~0.5s longer) would flag the
  no-change wells. Scale falls back to IQR/1.349 then std when MAD is 0.
  A well is flagged when ``|z_m| >= --z-thresh`` for any metric, subject to an
  eligibility gate ``max(trad_count, ml_count) >= --min-events`` that drops
  trivial small-count flips. Ranked by (# metrics fired, sum of z clipped at
  --clip) so multi-metric divergences surface above single extreme ratios.

* ``overdetect`` -- the original ML-over-detection selector::

      ratio = (ml_count + 1) / (trad_count + 1) >= --min-ratio  AND  ml_count >= --min-ml

  ranked by excess = ml_count - trad_count.

Directories (curated spikes + both detectors' network_bursts.pkl) come from the
authoritative pipeline_cache.json, same as compare_burst_methods.py.

Run (after compare_burst_methods.py has written per_well.csv)::

    python scripts/render_overdetection_rasters.py --config pipeline_config.json

Outputs: <figure_root>/burst_method_comparison/<mode>_rasters/
         {recording}__{well_name}_{well_id}.html  + index.html
         (+ divergence_summary.csv listing every flagged well in divergent mode)
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
# Discovery: significant-divergence wells (any metric, either direction)
# --------------------------------------------------------------------------- #
# rate_per_min is deliberately excluded: it is a duration-normalised count and
# fires on essentially the same wells as `count`, so including it would just
# double-weight the count axis.
DIVERGENCE_METRICS = ["count", "duration_s", "spikes_per_burst",
                      "participation", "peak_synchrony"]


def _robust_scale(x: np.ndarray) -> float:
    """Robust spread of a log2-ratio column, used only as the z-score scale.

    MAD (x1.4826 ~= sigma) first; falls back to IQR/1.349 then std when >=50%
    of wells share the median value (MAD collapses to 0 -- common for count and
    peak_synchrony where many wells agree exactly)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    mad = float(np.median(np.abs(x - np.median(x))) * 1.4826)
    if mad > 0:
        return mad
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float((q75 - q25) / 1.349)
    if iqr > 0:
        return iqr
    std = float(np.std(x))
    return std if std > 0 else np.nan


def select_divergent(per_well: pd.DataFrame, metrics: list[str], z_thresh: float,
                     min_events: float, clip: float, top: int
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag wells that are robust outliers on any metric's log2 ratio.

    Returns (rendered, flagged): `flagged` is every eligible well with at least
    one metric firing (for the summary CSV); `rendered` is the top-N slice to
    draw. Adds columns: z_<metric>, n_fired, fired (direction-tagged string),
    div_score (ranking key), plus `ratio`/`excess` for a uniform index.
    """
    d = per_well.copy()
    metrics = [m for m in metrics if f"log2ratio_{m}" in d.columns]
    eligible = np.maximum(d["trad_count"].to_numpy(float),
                          d["ml_count"].to_numpy(float)) >= min_events

    absz = np.zeros((len(metrics), len(d)))
    signs = np.zeros_like(absz)
    for i, m in enumerate(metrics):
        col = d[f"log2ratio_{m}"].to_numpy(float)
        finite = np.isfinite(col)
        scale = _robust_scale(col)
        z = np.where(finite & np.isfinite(scale), col / scale, 0.0)
        d[f"z_{m}"] = z
        absz[i] = np.abs(z)
        signs[i] = np.sign(col)

    fired = (absz >= z_thresh) & eligible          # metric x well
    n_fired = fired.sum(axis=0)
    clipped = np.where(fired, np.minimum(absz, clip), 0.0)
    d["n_fired"] = n_fired
    d["div_score"] = clipped.sum(axis=0)

    def _fired_str(j: int) -> str:
        parts = [f"{metrics[i]}{'+' if signs[i, j] >= 0 else '-'}"
                 for i in range(len(metrics)) if fired[i, j]]
        return ",".join(parts)

    d["fired"] = [_fired_str(j) for j in range(len(d))]
    # Uniform columns so the index/render loop is mode-agnostic.
    d["ratio"] = (d["ml_count"] + 1.0) / (d["trad_count"] + 1.0)
    d["excess"] = d["ml_count"] - d["trad_count"]

    flagged = d[n_fired >= 1].sort_values(
        ["n_fired", "div_score"], ascending=False)
    logger.info(
        "divergent: %d/%d eligible wells (max_count>=%g) flagged on >=1 of %s "
        "at |z|>=%.2f; rendering top %d",
        len(flagged), int(eligible.sum()), min_events, ",".join(metrics),
        z_thresh, len(flagged) if top <= 0 else min(top, len(flagged)))
    for i, m in enumerate(metrics):
        logger.info("  %-16s scale=%.3f  fired=%d",
                    m, _robust_scale(d[f"log2ratio_{m}"].to_numpy(float)),
                    int((fired[i] > 0).sum()))
    rendered = flagged if top <= 0 else flagged.head(top)
    return rendered, flagged


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
                      label: bool, note: str = "") -> go.Figure:
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
    note_html = (f"<br><span style='font-size:12px;color:#b00'>"
                 f"diverges on: {html.escape(note)}</span>") if note else ""
    fig.update_layout(
        title=(f"<b>{html.escape(well_name)}</b> / {html.escape(groupname)}"
               f" &nbsp;&mdash;&nbsp; {html.escape(recording_key)}"
               f" &nbsp;&mdash;&nbsp; ML={len(ml_ev)} vs traditional={len(trad_ev)}"
               " network bursts" + note_html),
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
    p.add_argument("--select", choices=["divergent", "overdetect"],
                   default="divergent",
                   help="divergent: significant +/- change on any metric (default). "
                        "overdetect: legacy ML-over-detection ratio selector.")
    p.add_argument("--per-well-csv", type=Path, default=None,
                   help="Default <figure_root>/burst_method_comparison/per_well.csv")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default <figure_root>/burst_method_comparison/<mode>_rasters")
    # divergent-mode knobs
    p.add_argument("--metrics", default=",".join(DIVERGENCE_METRICS),
                   help="Comma-separated metrics to test (divergent mode). "
                        f"Default: {','.join(DIVERGENCE_METRICS)}")
    p.add_argument("--z-thresh", type=float, default=3.5,
                   help="Robust |z| outlier cutoff on a metric's log2 ratio (divergent).")
    p.add_argument("--min-events", type=float, default=5.0,
                   help="Eligibility gate: max(trad_count, ml_count) >= this (divergent).")
    p.add_argument("--clip", type=float, default=10.0,
                   help="Clip per-metric z at this value when ranking (divergent).")
    # overdetect-mode knobs
    p.add_argument("--min-ml", type=int, default=15,
                   help="Minimum ML burst count to be a suspect (overdetect).")
    p.add_argument("--min-ratio", type=float, default=3.0,
                   help="Minimum (ml+1)/(trad+1) ratio to be a suspect (overdetect).")
    p.add_argument("--top", type=int, default=40,
                   help="Render this many top-ranked wells; <=0 renders all flagged.")
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

    divergent = args.select == "divergent"
    if divergent:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        selected, flagged = select_divergent(
            per_well, metrics, args.z_thresh, args.min_events, args.clip, args.top)
        if selected.empty:
            raise SystemExit("no wells crossed the divergence threshold")
        default_dirname = "divergent_rasters"
    else:
        selected = select_suspects(per_well, args.min_ml, args.min_ratio, args.top)
        flagged = selected
        if selected.empty:
            raise SystemExit("no over-detection suspects matched the criteria")
        default_dirname = "overdetection_rasters"

    out_dir = args.output_dir or (comp_dir / default_dirname)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale HTMLs from a prior run so the deliverable is exactly this run.
    for old in out_dir.glob("*.html"):
        old.unlink()

    if divergent:
        summary_cols = (["recording_key", "well_id", "well_name", "groupname",
                         "trad_count", "ml_count", "n_fired", "fired", "div_score"]
                        + [f"z_{m}" for m in metrics if f"z_{m}" in flagged.columns])
        summary_path = out_dir / "divergence_summary.csv"
        flagged[summary_cols].to_csv(summary_path, index=False)
        logger.info("wrote %s (%d flagged wells)", summary_path, len(flagged))

    index_rows: list[str] = []
    n_written = 0
    for _, r in selected.iterrows():
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

        note = str(r["fired"]) if divergent else ""
        fig = build_well_figure(str(r["well_name"]), str(r["groupname"]),
                                str(r["recording_key"]), spikes, ml_ev, trad_ev,
                                label=not args.no_labels, note=note)
        rec_slug = str(r["recording_key"]).replace("/", "-")
        fname = f"{rec_slug}__{r['well_name']}_{r['well_id']}.html"
        fig.write_html(out_dir / fname, include_plotlyjs="cdn", full_html=True)
        n_written += 1
        if divergent:
            last_col = (f'<td>{int(r["n_fired"])}</td>'
                        f'<td style="text-align:left">{html.escape(str(r["fired"]))}</td>')
        else:
            last_col = f'<td>{r["ratio"]:.1f}</td>'
        index_rows.append(
            f'<tr><td>{n_written}</td>'
            f'<td><a href="{html.escape(fname)}">{html.escape(str(r["well_name"]))}</a></td>'
            f'<td>{html.escape(str(r["groupname"]))}</td>'
            f'<td>{html.escape(str(r["recording_key"]))}</td>'
            f'<td>{int(r["trad_count"])}</td><td>{int(r["ml_count"])}</td>'
            f'{last_col}</tr>')
        logger.info("wrote %s  (trad=%d ml=%d %s)", fname,
                    int(r["trad_count"]), int(r["ml_count"]),
                    str(r["fired"]) if divergent else f"ratio={r['ratio']:.1f}")

    if divergent:
        title = f"ML-vs-traditional significant divergence ({n_written} wells)"
        blurb = (f"Selected: robust |z| &ge; {args.z_thresh} on any of "
                 f"{html.escape(args.metrics)} (zero-centered log2 ratio, MAD scale), "
                 f"eligibility max(trad,ML) &ge; {args.min_events:g}; ranked by "
                 "#metrics&nbsp;fired then clipped-z. Each raster shows ML bursts "
                 "(red) and traditional bursts (blue) with duration labels over the "
                 "curated spikes. Sign convention: <b>metric+</b> = ML higher than "
                 "traditional, <b>metric&minus;</b> = ML lower. Metrics are correlated "
                 "(ML merging fragments moves count/duration/spikes/participation "
                 "together), so #fired counts co-moving axes, not independent signals. "
                 "Full flagged list (490 wells) in divergence_summary.csv; "
                 "re-run with --top 0 to render every flagged well.")
        last_th = "<th>#fired</th><th>diverges&nbsp;on</th>"
    else:
        title = f"ML over-detection suspects ({n_written} wells)"
        blurb = (f"Selected: ml_count &ge; {args.min_ml}, (ml+1)/(trad+1) &ge; "
                 f"{args.min_ratio}; ranked by ml&minus;trad excess. Each raster shows "
                 "ML bursts (red) and traditional bursts (blue) with duration labels "
                 "over the curated spikes.")
        last_th = "<th>ratio</th>"

    index = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:sans-serif;margin:2rem;max-width:1000px}"
        "table{border-collapse:collapse;width:100%;font-size:.9rem}"
        "th,td{border:1px solid #ddd;padding:5px 9px;text-align:right}"
        "td:nth-child(2),td:nth-child(3),td:nth-child(4){text-align:left}"
        "th{background:#f4f4f4}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p>{blurb}</p>"
        "<table><tr><th>#</th><th>well</th><th>group</th><th>recording</th>"
        f"<th>trad</th><th>ML</th>{last_th}</tr>"
        + "".join(index_rows) + "</table></body></html>")
    (out_dir / "index.html").write_text(index)

    logger.info("wrote %d well HTMLs + index.html to %s", n_written, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
