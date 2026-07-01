#!/usr/bin/env python3
"""Render a high-res PNG per well per UMAP-dim and assemble an <img> grid.

Each PNG has three stacked panels — spike raster, burst-events lane, UMAP(2)
scatter — drawn with matplotlib at dpi=300 so all data reads clearly. The grid
HTML (rows = wells, cols = dims) references the PNGs via lazy <img>, so it loads
far lighter than the iframe/plotly version.

Reuses `inspect_ml_bursts` loaders + the same colors/burst-label semantics.

Run:
    python scripts/render_slim_png.py \
        --config pipeline_config.json \
        --sweep-root /mnt/benshalom-nas/analysis/Sadegh/new/CX138/umap_dim_sweep \
        --dims 2,3,5,10
Writes <sweep-root>/png_umap{D}d/*.png + compare_dims_png.html.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspect_ml_bursts as iml  # noqa: E402

logger = logging.getLogger("render_slim_png")


def _mpl_color(c: str):
    """Convert a plotly color string ('rgba(r,g,b,a)' or '#hex') to matplotlib."""
    if c.startswith("#"):
        return c
    m = re.match(r"rgba?\(([^)]+)\)", c)
    if not m:
        return c
    parts = [float(x) for x in m.group(1).split(",")]
    r, g, b = parts[0] / 255, parts[1] / 255, parts[2] / 255
    a = parts[3] if len(parts) > 3 else 1.0
    return (r, g, b, a)


def _render_png(well, out_path: Path, dpi: int) -> None:
    has_trace = well.trace is not None
    umap_result = iml._compute_umap_axes(well.trace) if has_trace else None
    has_umap = umap_result is not None

    fig = plt.figure(figsize=(11, 12))
    if has_umap:
        gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 0.7, 2.4], hspace=0.28)
        ax_r, ax_e, ax_u = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 0.7], hspace=0.28)
        ax_r, ax_e = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        ax_u = None

    # ----- Raster (units ranked by firing rate, subsampled like the inspector) -----
    t_end = 0.0
    if well.spike_times:
        items = []
        for uid, st in well.spike_times.items():
            arr = np.asarray(st, dtype=float)
            arr = arr[np.isfinite(arr)]
            items.append((str(uid), arr))
        t_end = max((float(a.max()) if a.size else 0.0 for _, a in items), default=0.0)
        items.sort(key=lambda kv: iml._firing_rate(kv[1], t_end), reverse=True)
        n_units = len(items) or 1
        per_unit_cap = max(40, iml.MAX_RASTER_POINTS_PER_WELL // n_units)
        offsets = []
        for _uid, spikes in items:
            if spikes.size > per_unit_cap:
                stride = int(np.ceil(spikes.size / per_unit_cap))
                spikes = spikes[::stride]
            offsets.append(spikes)
        ax_r.eventplot(offsets, colors=[_mpl_color(iml.RASTER_COLOR)],
                       linewidths=0.5, linelengths=0.8)
        ax_r.set_ylim(-0.5, len(offsets) - 0.5)
        ax_r.set_ylabel("unit rank (by FR)")
    else:
        ax_r.text(0.5, 0.5, "no spike times", ha="center", va="center", transform=ax_r.transAxes)
    ax_r.set_title(
        f"{well.well_name} ({well.well_id}) — "
        f"clusters={(well.diagnostics or {}).get('cluster_n_clusters')} "
        f"burst_labels={sorted(iml._burst_label_set(well.diagnostics))}",
        fontsize=11)

    # ----- Burst events lane -----
    bursts = well.events.get("network_bursts") or []
    burst_color = _mpl_color(EVENT_COLOR)
    for ev in bursts:
        s, e = float(ev["start"]), float(ev["end"])
        ax_e.axvspan(s, e, ymin=0.15, ymax=0.85, color=burst_color, lw=0)
    ax_e.set_yticks([])
    ax_e.set_ylabel(f"bursts (n={len(bursts)})", fontsize=9)
    ax_e.set_xlabel("time (s)")
    if t_end > 0:
        ax_r.set_xlim(0, t_end)
        ax_e.set_xlim(0, t_end)

    # ----- UMAP -----
    if ax_u is not None:
        coords, xl, yl, kept = umap_result
        labels = np.asarray(well.trace.hdbscan_labels)[kept]
        burst_set = iml._burst_label_set(well.diagnostics)
        color_by, name_by = iml._cluster_color_map(labels, burst_set)
        for lbl in sorted(color_by):
            mask = labels == lbl
            if not mask.any():
                continue
            is_burst = lbl in burst_set
            # Shape encodes burst-ness: diamond = burst cluster, circle = non-burst.
            marker = "D" if is_burst else "o"
            ax_u.scatter(coords[mask, 0], coords[mask, 1],
                         s=(18 if is_burst else 14) if lbl != -1 else 6,
                         marker=marker,
                         color=_mpl_color(color_by[lbl]),
                         alpha=0.85 if lbl != -1 else 0.35,
                         linewidths=0, label=f"{name_by[lbl]} (n={int(mask.sum())})")
        ax_u.set_xlabel(xl)
        ax_u.set_ylabel(yl)
        ax_u.set_title("UMAP(2) — color = cluster id, ◆ = burst", fontsize=10)
        ax_u.legend(fontsize=7, markerscale=1.2, loc="best", framealpha=0.6)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


EVENT_COLOR = iml.EVENT_TRACKS[0][2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--dims", default="2,3,5,10")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--thumb-width", type=int, default=520,
                    help="Displayed column width (px) in the grid; click opens full PNG.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s — %(message)s")

    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    sweep_root: Path = args.sweep_root

    cfg = json.loads(args.config.read_text())
    analysis_root = Path(cfg["global"]["analysis_root"])
    pipeline_cache = iml._load_pipeline_cache(analysis_root)
    experiment_cache = iml._load_experiment_cache(analysis_root)
    resolve = iml._make_resolver([args.config.resolve().parent, analysis_root, Path.cwd().resolve()])

    wells = iml._collect_ml_wells(pipeline_cache)
    if not wells:
        raise SystemExit("No completed ml_burst_detection wells in pipeline_cache.")
    logger.info("Rendering %d wells × %d dims @ dpi=%d", len(wells), len(dims), args.dpi)

    grid: dict[str, tuple[str, dict[int, str]]] = {}
    for rec_key, rec_name, well_id, entry in wells:
        well_meta = (experiment_cache.get(rec_key, {})
                     .get("wells", {}).get(well_id, {}).get("metadata", {}))
        per_dim: dict[int, str] = {}
        label = None
        for d in dims:
            ml_dir = (sweep_root / f"data_umap{d}d" / rec_key / rec_name / well_id
                      / "ml_burst_detection")
            if not (ml_dir / "metrics.json").exists():
                logger.warning("missing dim%d data for %s/%s", d, rec_name, well_id)
                continue
            e = copy.deepcopy(entry)
            e.setdefault("tasks", {}).setdefault(iml.ML_TASK, {})["output_path"] = str(ml_dir)
            bundle = iml._load_well_bundle(rec_key, rec_name, well_id, e, well_meta, resolve)
            if bundle is None:
                continue
            label = f"{bundle.well_name} {well_id}"
            out_dir = sweep_root / f"png_umap{d}d"
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{bundle.well_name}_{rec_name}_{well_id}.png"
            try:
                _render_png(bundle, out_dir / fname, args.dpi)
            except Exception:  # noqa: BLE001
                logger.exception("png failed dim%d %s/%s", d, rec_name, well_id)
                continue
            per_dim[d] = f"png_umap{d}d/{fname}"
        grid[f"{rec_name}/{well_id}"] = (label or well_id, per_dim)
        logger.info("done %s (%d dims)", well_id, len(per_dim))

    _write_grid(sweep_root, dims, grid, args.thumb_width)


def _write_grid(sweep_root, dims, grid, thumb_w):
    rows = []
    for key in sorted(grid):
        label, per_dim = grid[key]
        cells = []
        for d in dims:
            rel = per_dim.get(d)
            if rel:
                cells.append(f'<td><a href="{html.escape(rel)}" target="_blank">'
                             f'<img loading="lazy" src="{html.escape(rel)}"></a></td>')
            else:
                cells.append('<td class="missing">missing</td>')
        rows.append(f'<tr><th class="rowhdr">{html.escape(label)}</th>{"".join(cells)}</tr>')

    col_hdrs = "".join(f"<th>UMAP {d}-d</th>" for d in dims)
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>UMAP dim comparison (PNG) — assay 000012</title>
<style>
  :root {{ --thumb-w: {thumb_w}px; --hdr-w: 60px; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 10px; }}
  h1 {{ font-size: 17px; margin: 0 0 6px; }}
  .toolbar {{ position: sticky; top: 0; z-index: 5; background:#fff; padding:6px 0;
              display:flex; gap:12px; align-items:center; font-size:13px;
              border-bottom:1px solid #ddd; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ccc; vertical-align: top; padding: 2px; text-align:center; }}
  thead th {{ position: sticky; top: 38px; background:#f4f4f4; z-index:3;
             padding:6px 4px; font-size:13px; }}
  .rowhdr {{ position: sticky; left: 0; background:#f4f4f4; z-index:1;
             padding:6px 4px; font-size:11px; white-space:nowrap;
             writing-mode: vertical-rl; transform: rotate(180deg); }}
  img {{ width: var(--thumb-w); height: auto; display:block; }}
  td.missing {{ color:#b00; padding:20px; }}
</style></head>
<body>
<h1>UMAP clustering-dim comparison (PNG dpi=300: raster · burst events · UMAP) — assay 000012</h1>
<div class="toolbar">
  <label>column width <input id="w" type="range" min="280" max="1100" value="{thumb_w}" step="20"></label>
  <span id="wl">{thumb_w}px</span>
  <span style="color:#666">Click any panel to open the full-resolution PNG.</span>
</div>
<table>
<thead><tr><th class="rowhdr">well</th>{col_hdrs}</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
<script>
  const w = document.getElementById('w'), root = document.documentElement;
  w.addEventListener('input', () => {{
    root.style.setProperty('--thumb-w', w.value + 'px');
    document.getElementById('wl').textContent = w.value + 'px';
  }});
</script>
</body></html>
"""
    out = sweep_root / "compare_dims_png.html"
    out.write_text(doc)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
