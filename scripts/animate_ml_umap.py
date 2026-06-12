"""Per-well MP4 animation of the ML-burst UMAP trajectory.

Companion to ``inspect_ml_bursts.py``: instead of a static UMAP scatter of every
time bin, this renders *how a well moves through that UMAP space over time*. A
filled "ball" marks the current location; a fading history trail (last
``--history`` seconds of recording time) of points connected by faint grey lines
shows where it has just been. Points are colored by HDBSCAN cluster, matching
the inspect HTMLs (burst cluster red).

Discovery, file loading, and the UMAP embedding are reused verbatim from
``inspect_ml_bursts`` so the embedding matches the static HTML for the same well.

Output: {output_dir}/{recording_key}/ml_burst_umap_anim/{well_name}_{rec_name}_{well_id}.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

# Reuse discovery / loading / UMAP helpers from the inspect script (same dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspect_ml_bursts as iml  # noqa: E402

logger = logging.getLogger("animate_ml_umap")

NOISE_RGBA = to_rgba("#b4b4b4")
BACKGROUND_ALPHA = 0.12
TRAIL_LINE_ALPHA = 0.4
BALL_SIZE = 220


def _cluster_rgba_map(labels: np.ndarray, burst_label: int | None) -> dict[int, tuple]:
    """Map each HDBSCAN label to a matplotlib RGBA tuple.

    Mirrors inspect_ml_bursts colors (burst = red, noise = grey, others cycle
    the palette) but as RGBA tuples so per-point alpha is trivial to apply.
    """
    out: dict[int, tuple] = {}
    palette_i = 0
    for lbl in sorted(set(int(x) for x in labels)):
        if lbl == -1:
            out[lbl] = NOISE_RGBA
        elif burst_label is not None and lbl == int(burst_label):
            out[lbl] = to_rgba(iml.BURST_COLOR)
        else:
            out[lbl] = to_rgba(iml.CLUSTER_PALETTE[palette_i % len(iml.CLUSTER_PALETTE)])
            palette_i += 1
    return out


def _ranked_raster(
    spike_times: dict[str, np.ndarray], cap: int = 12000
) -> tuple[list[np.ndarray], float]:
    """Return (per-unit spike-time arrays ranked by firing rate desc, t_end).

    Mirrors inspect_ml_bursts' raster: highest-firing units at the top, with a
    per-unit point cap so dense wells stay light to draw.
    """
    items: list[tuple[str, np.ndarray]] = []
    for uid, st in spike_times.items():
        a = np.asarray(st, dtype=float)
        a = a[np.isfinite(a)]
        items.append((str(uid), a))
    t_end = max((float(a.max()) if a.size else 0.0 for _, a in items), default=0.0)
    items.sort(
        key=lambda kv: (kv[1].size / t_end if t_end > 0 and kv[1].size else 0.0),
        reverse=True,
    )
    n_units = len(items) or 1
    per_cap = max(40, cap // n_units)
    positions: list[np.ndarray] = []
    for _, sp in items:
        if sp.size > per_cap:
            sp = sp[:: int(np.ceil(sp.size / per_cap))]
        positions.append(sp)
    return positions, t_end


def _animate_well(
    bundle: "iml.WellBundle",
    out_path: Path,
    duration: float,
    history: float,
    fps_override: int | None,
    dpi: int,
) -> bool:
    """Render one well's UMAP trajectory to ``out_path``. Returns True on write."""
    trace = bundle.trace
    if trace is None:
        logger.info("Skip %s: no debug_trace.pkl", bundle.well_name)
        return False
    res = iml._compute_umap_axes(trace)
    if res is None:
        logger.info("Skip %s: UMAP unavailable (umap-learn missing or <10 bins)", bundle.well_name)
        return False
    coords, xlabel, ylabel, kept = res

    t = np.asarray(trace.t_centers, dtype=float)[kept]
    labels_full = getattr(trace, "hdbscan_labels", None)
    if labels_full is not None and np.asarray(labels_full).size >= int(kept.max()) + 1:
        labels = np.asarray(labels_full)[kept].astype(int)
    else:
        labels = np.full(coords.shape[0], -1, dtype=int)
    burst_label = getattr(trace, "burst_label", None)

    n = coords.shape[0]
    rgba_map = _cluster_rgba_map(labels, burst_label)
    point_rgba = np.array([rgba_map[int(l)] for l in labels], dtype=float)  # (n, 4)

    # Derive fps from target duration, but cap at 60 (players choke above that);
    # a short --duration on a long recording just yields a slightly longer video.
    fps = fps_override or min(60, max(1, round(n / max(duration, 1e-6))))

    # Raster (top) over UMAP (bottom). Skip the raster panel if no spikes.
    raster = None
    if bundle.spike_times:
        raster, t_end_r = _ranked_raster(bundle.spike_times)
    t_max = max(float(t.max()), t_end_r if raster else 0.0)

    if raster:
        fig = plt.figure(figsize=(8, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.22)
        ax_r = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax_r = None
    fig.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.06)

    # Top panel: static spike raster + a moving shaded window for the trail span.
    span = None
    if ax_r is not None:
        ax_r.eventplot(raster, colors="#282828", linelengths=0.85, linewidths=0.4)
        ax_r.set_xlim(0, t_max)
        ax_r.set_ylim(-0.5, len(raster) - 0.5)
        ax_r.set_xlabel("time (s)")
        ax_r.set_ylabel("unit (FR rank)")
        ax_r.tick_params(labelsize=9)
        # x in data coords, y spans the full axes height (xaxis_transform).
        span = Rectangle((0, 0), history, 1, transform=ax_r.get_xaxis_transform(),
                         facecolor="#ffb000", alpha=0.25, edgecolor="#ff8c00",
                         lw=1.0, zorder=5)
        ax_r.add_patch(span)

    # Static background cloud: cluster-colored, faint.
    bg_rgba = point_rgba.copy()
    bg_rgba[:, 3] = BACKGROUND_ALPHA
    ax.scatter(coords[:, 0], coords[:, 1], s=6, c=bg_rgba, linewidths=0, zorder=1)

    # Fixed limits with a small margin.
    xpad = 0.05 * (np.ptp(coords[:, 0]) or 1.0)
    ypad = 0.05 * (np.ptp(coords[:, 1]) or 1.0)
    ax.set_xlim(coords[:, 0].min() - xpad, coords[:, 0].max() + xpad)
    ax.set_ylim(coords[:, 1].min() - ypad, coords[:, 1].max() + ypad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")

    # Dynamic artists.
    (trail_line,) = ax.plot([], [], color="#808080", alpha=TRAIL_LINE_ALPHA,
                            lw=1.2, zorder=2)
    trail_pts = ax.scatter([], [], s=22, linewidths=0, zorder=3)
    ball = ax.scatter([], [], s=BALL_SIZE, edgecolors="black", linewidths=1.2,
                      zorder=4)
    title = fig.suptitle(f"{bundle.well_name}  ·  {bundle.groupname}  ·  t = 0.0 s",
                         x=0.1, ha="left", fontsize=12)

    def update(ci: int):
        t_now = t[ci]
        win = (t >= t_now - history) & (t <= t_now)
        wx, wy = coords[win, 0], coords[win, 1]
        wt = t[win]
        trail_line.set_data(wx, wy)

        # Per-point alpha fades with age across the history window.
        age_alpha = np.clip((wt - (t_now - history)) / max(history, 1e-6), 0.05, 0.9)
        fc = point_rgba[win].copy()
        fc[:, 3] = age_alpha
        trail_pts.set_offsets(np.column_stack([wx, wy]))
        trail_pts.set_facecolors(fc)

        ball.set_offsets([[coords[ci, 0], coords[ci, 1]]])
        ball.set_facecolors([point_rgba[ci]])

        # Move the raster window to the trail span (clamp left edge to 0).
        if span is not None:
            x0 = max(0.0, t_now - history)
            span.set_x(x0)
            span.set_width(t_now - x0)

        title.set_text(f"{bundle.well_name}  ·  {bundle.groupname}  ·  t = {t_now:.1f} s")
        return trail_line, trail_pts, ball, title

    anim = FuncAnimation(fig, update, frames=n, interval=1000.0 / fps, blit=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(str(out_path), writer=writer, dpi=dpi)
    except Exception as exc:  # noqa: BLE001 — ffmpeg missing or failed
        gif_path = out_path.with_suffix(".gif")
        logger.warning("ffmpeg failed (%s); falling back to GIF %s", exc, gif_path.name)
        anim.save(str(gif_path), writer=PillowWriter(fps=fps), dpi=dpi)
        out_path = gif_path
    finally:
        plt.close(fig)

    logger.info("Wrote %s (%d frames @ %d fps)", out_path, n, fps)
    return True


def _want_well(bundle: "iml.WellBundle", filters: list[str] | None) -> bool:
    if not filters:
        return True
    keys = {bundle.well_name.lower(), bundle.well_id.lower()}
    return any(f.lower() in keys for f in filters)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="Path to pipeline_config.json (read for analysis_root + figure_root).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output directory. Default: {figure_root}.")
    p.add_argument("--duration", type=float, default=60.0,
                   help="Target video length in seconds (sets fps from frame count).")
    p.add_argument("--history", type=float, default=5.0,
                   help="Trail length in seconds of recording time.")
    p.add_argument("--fps", type=int, default=None, help="Override fps (ignores --duration).")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--well", action="append", default=None,
                   help="Render only matching well_name or well_id (repeatable).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s — %(message)s")

    with args.config.open() as fh:
        cfg = json.load(fh)
    analysis_root = Path(cfg["global"]["analysis_root"])
    figure_root = Path(args.output_dir or cfg["global"].get("figure_root") or analysis_root)
    figure_root.mkdir(parents=True, exist_ok=True)

    pipeline_cache = iml._load_pipeline_cache(analysis_root)
    experiment_cache = iml._load_experiment_cache(analysis_root)
    config_dir = args.config.resolve().parent
    resolve = iml._make_resolver([config_dir, analysis_root, Path.cwd().resolve()])

    wells = iml._collect_ml_wells(pipeline_cache)
    if not wells:
        logger.warning("No wells with completed ml_burst_detection found in %s/pipeline_cache.json",
                       analysis_root)
        return 0
    logger.info("Found %d wells with completed ml_burst_detection", len(wells))

    n_written = 0
    for rec_key, rec_name, well_id, entry in wells:
        well_meta = (
            experiment_cache.get(rec_key, {})
            .get("wells", {}).get(well_id, {}).get("metadata", {})
        )
        bundle = iml._load_well_bundle(rec_key, rec_name, well_id, entry, well_meta, resolve)
        if bundle is None or not _want_well(bundle, args.well):
            continue
        out_path = (figure_root / rec_key / "ml_burst_umap_anim"
                    / f"{bundle.well_name}_{rec_name}_{well_id}.mp4")
        try:
            if _animate_well(bundle, out_path, args.duration, args.history, args.fps, args.dpi):
                n_written += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to animate %s/%s/%s: %s", rec_key, rec_name, well_id, exc)

    logger.info("Done. %d animation(s) written under %s", n_written, figure_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
