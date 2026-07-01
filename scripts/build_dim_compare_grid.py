#!/usr/bin/env python3
"""Build one HTML grid comparing per-well inspection figures across UMAP dims.

Layout: rows = wells, columns = clustering dims. Each cell is an <iframe> onto the
existing per-well inspection HTML (lazy-loaded; the per-dim HTMLs are ~2.5 MB
plotly each, so inlining all 96 is infeasible — iframes keep the page light).

Run:
    python scripts/build_dim_compare_grid.py \
        --sweep-root /mnt/benshalom-nas/analysis/Sadegh/new/CX138/umap_dim_sweep \
        --dims 2,3,5,10
Writes <sweep-root>/compare_dims_grid.html.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

REC_SUBPATH = "CX138/260325/T003346/Network/000012/ml_burst_inspect"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--dims", default="2,3,5,10")
    ap.add_argument("--fig-width", type=int, default=1500,
                    help="Native render width of each inspection figure (px).")
    ap.add_argument("--cell-height", type=int, default=1100,
                    help="Scaled iframe viewport height in px (figures scroll inside).")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    root: Path = args.sweep_root

    # well key -> {dim: relative html path}. Use the inspect dir of the first dim
    # to enumerate wells; filenames are identical across dims.
    first_dir = root / f"inspect_umap{dims[0]}d" / REC_SUBPATH
    files = sorted(first_dir.glob("*.html"), key=lambda p: p.name)
    if not files:
        raise SystemExit(f"No inspection HTMLs under {first_dir}")

    rows = []
    for f in files:
        name = f.name                      # e.g. A1_rec0000_well000.html
        label = name[:-5]                  # drop .html
        cells = []
        for d in dims:
            rel = Path(f"inspect_umap{d}d") / REC_SUBPATH / name
            target = root / rel
            if target.exists():
                cells.append(
                    f'<td><div class="frame-wrap"><iframe loading="lazy" '
                    f'src="{html.escape(str(rel))}"></iframe></div></td>'
                )
            else:
                cells.append('<td class="missing">missing</td>')
        rows.append(f'<tr><th class="rowhdr">{html.escape(label)}</th>{"".join(cells)}</tr>')

    col_hdrs = "".join(f'<th>UMAP {d}-d</th>' for d in dims)
    ncols = len(dims)
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>UMAP clustering-dim comparison — assay 000012</title>
<style>
  :root {{
    --fig-w: {args.fig_width}px;     /* native figure render width */
    --cell-h: {args.cell_height}px;  /* scaled cell viewport height */
    --ncols: {ncols};
    --hdr-w: 70px;                   /* left well-label column width */
    --gap: 6px;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 10px; }}
  h1 {{ font-size: 17px; margin: 0 0 6px; }}
  .toolbar {{ position: sticky; top: 0; z-index: 5; background: #fff;
              padding: 6px 0; display: flex; gap: 12px; align-items: center;
              font-size: 13px; border-bottom: 1px solid #ddd; }}
  table {{ border-collapse: collapse; table-layout: fixed; width: 100%; }}
  col.hdrcol {{ width: var(--hdr-w); }}
  th, td {{ border: 1px solid #ccc; vertical-align: top; padding: 0; }}
  thead th {{ position: sticky; top: 38px; background: #f4f4f4; z-index: 3;
             padding: 6px 4px; font-size: 13px; text-align: center; }}
  .rowhdr {{ position: sticky; left: 0; background: #f4f4f4; z-index: 1;
             padding: 6px 4px; font-size: 11px; white-space: nowrap;
             writing-mode: vertical-rl; transform: rotate(180deg); text-align: center; }}
  /* Each cell viewport is 1/ncols of the available width; the figure is rendered
     at native --fig-w then scaled down to fit via CSS transform. */
  .frame-wrap {{ height: var(--cell-h); overflow: hidden; }}
  .frame-wrap iframe {{
     width: var(--fig-w); border: 0; display: block;
     height: calc(var(--cell-h) / var(--scale));
     transform: scale(var(--scale)); transform-origin: top left;
  }}
  td.missing {{ text-align: center; color: #b00; padding: 20px; }}
</style></head>
<body>
<h1>UMAP clustering-dim comparison — assay 000012 (rows = wells, cols = dims fed to HDBSCAN)</h1>
<div class="toolbar">
  <label>zoom <input id="zoom" type="range" min="20" max="100" value="0" step="5"></label>
  <span id="zlbl"></span>
  <span style="color:#666">Default fits all {ncols} columns to window width. Increase to read detail (adds horizontal scroll). Scroll inside a cell for the full figure.</span>
</div>
<table>
<colgroup><col class="hdrcol">{"".join("<col>" for _ in dims)}</colgroup>
<thead><tr><th class="rowhdr">well</th>{col_hdrs}</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
<script>
  const FIG_W = {args.fig_width}, NCOLS = {ncols}, HDR_W = 70, root = document.documentElement;
  function fitScale() {{
    const avail = (window.innerWidth - HDR_W - 24) / NCOLS;   // px per column
    return Math.max(0.12, avail / FIG_W);
  }}
  function applyScale(s) {{
    root.style.setProperty('--scale', s.toFixed(4));
    document.getElementById('zlbl').textContent = Math.round(s * 100) + '%';
  }}
  const slider = document.getElementById('zoom');
  function fromSlider() {{
    const v = +slider.value;
    applyScale(v === 0 ? fitScale() : v / 100);
  }}
  slider.addEventListener('input', fromSlider);
  window.addEventListener('resize', () => {{ if (+slider.value === 0) applyScale(fitScale()); }});
  applyScale(fitScale());
</script>
</body></html>
"""
    out = root / "compare_dims_grid.html"
    out.write_text(doc)
    print(f"wrote {out} — {len(files)} wells x {len(dims)} dims")


if __name__ == "__main__":
    main()
