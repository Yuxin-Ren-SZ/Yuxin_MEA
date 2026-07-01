#!/usr/bin/env python3
"""Slim per-well comparison grid across UMAP clustering dims.

Renders a compact 3-panel figure (raster + burst-events + UMAP) for every well
at every dim, then assembles one HTML: rows = wells, columns = dims. Reuses
`inspect_ml_bursts.build_well_figure_slim` and its loaders so colors / burst
semantics match the full inspector.

Run:
    python scripts/build_slim_compare.py \
        --config pipeline_config.json \
        --sweep-root /mnt/benshalom-nas/analysis/Sadegh/new/CX138/umap_dim_sweep \
        --dims 2,3,5,10
Writes <sweep-root>/compare_dims_slim.html + per-cell HTMLs under slim_umap{D}d/.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspect_ml_bursts as iml  # noqa: E402

logger = logging.getLogger("build_slim_compare")
ML_TASK = iml.ML_TASK


def _per_dim_ml_dir(sweep_root: Path, dim: int, rec_key: str, rec_name: str, well_id: str) -> Path:
    return (sweep_root / f"data_umap{dim}d" / rec_key / rec_name / well_id
            / "ml_burst_detection")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--dims", default="2,3,5,10")
    ap.add_argument("--fig-width", type=int, default=760)
    ap.add_argument("--cell-height", type=int, default=860)
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
    logger.info("Rendering %d wells × %d dims", len(wells), len(dims))

    # row_key -> (label, {dim: relative_html_path})
    grid: dict[str, tuple[str, dict[int, str]]] = {}
    for rec_key, rec_name, well_id, entry in wells:
        well_meta = (experiment_cache.get(rec_key, {})
                     .get("wells", {}).get(well_id, {}).get("metadata", {}))
        row_label = f"{iml._fallback_well_name(well_id)} {well_id}"
        per_dim: dict[int, str] = {}
        for d in dims:
            ml_dir = _per_dim_ml_dir(sweep_root, d, rec_key, rec_name, well_id)
            if not (ml_dir / "metrics.json").exists():
                logger.warning("missing dim%d data for %s/%s", d, rec_name, well_id)
                continue
            e = copy.deepcopy(entry)
            e.setdefault("tasks", {}).setdefault(ML_TASK, {})["output_path"] = str(ml_dir)
            bundle = iml._load_well_bundle(rec_key, rec_name, well_id, e, well_meta, resolve)
            if bundle is None:
                continue
            try:
                fig = iml.build_well_figure_slim(bundle)
            except Exception:  # noqa: BLE001
                logger.exception("slim fig failed dim%d %s/%s", d, rec_name, well_id)
                continue
            out_dir = sweep_root / f"slim_umap{d}d"
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{bundle.well_name}_{rec_name}_{well_id}.html"
            (out_dir / fname).write_text("")  # touch (overwrite below)
            fig.write_html(out_dir / fname, include_plotlyjs="cdn", full_html=True)
            per_dim[d] = f"slim_umap{d}d/{fname}"
        grid[f"{rec_name}/{well_id}"] = (row_label, per_dim)
        logger.info("done well %s (%d dims)", well_id, len(per_dim))

    _write_grid(sweep_root, dims, grid, args.fig_width, args.cell_height)


def _write_grid(sweep_root, dims, grid, fig_w, cell_h):
    rows = []
    for key in sorted(grid):
        label, per_dim = grid[key]
        cells = []
        for d in dims:
            rel = per_dim.get(d)
            if rel:
                cells.append(f'<td><div class="frame-wrap"><iframe loading="lazy" '
                             f'src="{html.escape(rel)}"></iframe></div></td>')
            else:
                cells.append('<td class="missing">missing</td>')
        rows.append(f'<tr><th class="rowhdr">{html.escape(label)}</th>{"".join(cells)}</tr>')

    ncols = len(dims)
    col_hdrs = "".join(f"<th>UMAP {d}-d</th>" for d in dims)
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>UMAP dim comparison (slim) — assay 000012</title>
<style>
  :root {{ --fig-w: {fig_w}px; --cell-h: {cell_h}px; --hdr-w: 64px; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 10px; }}
  h1 {{ font-size: 17px; margin: 0 0 6px; }}
  .toolbar {{ position: sticky; top: 0; z-index: 5; background:#fff; padding:6px 0;
              display:flex; gap:12px; align-items:center; font-size:13px;
              border-bottom:1px solid #ddd; }}
  table {{ border-collapse: collapse; table-layout: fixed; width: 100%; }}
  col.hdrcol {{ width: var(--hdr-w); }}
  th, td {{ border: 1px solid #ccc; vertical-align: top; padding: 0; }}
  thead th {{ position: sticky; top: 38px; background:#f4f4f4; z-index:3;
             padding:6px 4px; font-size:13px; text-align:center; }}
  .rowhdr {{ position: sticky; left: 0; background:#f4f4f4; z-index:1;
             padding:6px 4px; font-size:11px; white-space:nowrap;
             writing-mode: vertical-rl; transform: rotate(180deg); text-align:center; }}
  .frame-wrap {{ height: var(--cell-h); overflow: hidden; }}
  .frame-wrap iframe {{ width: var(--fig-w); border:0; display:block;
     height: calc(var(--cell-h) / var(--scale));
     transform: scale(var(--scale)); transform-origin: top left; }}
  td.missing {{ text-align:center; color:#b00; padding:20px; }}
</style></head>
<body>
<h1>UMAP clustering-dim comparison (slim: raster · burst events · UMAP) — assay 000012</h1>
<div class="toolbar">
  <label>zoom <input id="zoom" type="range" min="20" max="100" value="0" step="5"></label>
  <span id="zlbl"></span>
  <span style="color:#666">Default fits all {ncols} columns to window. Increase to enlarge (adds horizontal scroll).</span>
</div>
<table>
<colgroup><col class="hdrcol">{"".join("<col>" for _ in dims)}</colgroup>
<thead><tr><th class="rowhdr">well</th>{col_hdrs}</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
<script>
  const FIG_W = {fig_w}, NCOLS = {ncols}, HDR_W = 64, root = document.documentElement;
  const fitScale = () => Math.max(0.12, (window.innerWidth - HDR_W - 24) / NCOLS / FIG_W);
  function applyScale(s) {{ root.style.setProperty('--scale', s.toFixed(4));
    document.getElementById('zlbl').textContent = Math.round(s*100)+'%'; }}
  const slider = document.getElementById('zoom');
  const fromSlider = () => applyScale(+slider.value === 0 ? fitScale() : +slider.value/100);
  slider.addEventListener('input', fromSlider);
  window.addEventListener('resize', () => {{ if (+slider.value === 0) applyScale(fitScale()); }});
  applyScale(fitScale());
</script>
</body></html>
"""
    out = sweep_root / "compare_dims_slim.html"
    out.write_text(doc)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
