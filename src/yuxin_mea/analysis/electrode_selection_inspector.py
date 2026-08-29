"""Viewer library for the electrode-selection QC page and its exported figures.

Pure loaders + Plotly builders + a matplotlib rasterizer over what
:func:`yuxin_mea.analysis.electrode_selection_qc.write_well_qc` wrote. Imports no
Dash, no h5py, no SpikeInterface — everything here is ``json`` / ``np.load``, so
the dashboard never drags in the extraction path.

On-disk layout (note: no ``rec_name`` level — a well is assembled *across*
recordings, so a per-recording level would be meaningless)::

    electrode_selection_qc/
        selection_index.parquet   scans.parquet
        stability_summary.csv     index.md
        <sample_id>/<plate_id>/<well_id>/
            stability.json  counts.npz  pairs.csv
            selection_count_map.png/.pdf   selection_history.mp4   report.md

**Two normalisations, deliberately different.** The full-size map (Vis 1) shows
*raw selection counts*, which is what was asked for and what the colourbar
labels. The plate-grid thumbnails show the *fraction of that well's scans* on a
fixed 0–1 scale, because wells in one plate can have different in-window scan
counts and a raw-count scale would make a well with more scans look
systematically "greener" than its neighbour.

**Never-selected reads as white, not as a low value.** Only ~20 % of the array is
ever routed, so an unselected electrode is *absent data*, not a small number.
Every renderer here masks count 0 and paints an opaque white background — unlike
the activity-scan thumbnails, which are transparent on purpose so the page's
warm paper shows through.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from yuxin_mea.analysis.activity_scan import GRID_COLS, GRID_ROWS, PITCH_UM
from yuxin_mea.analysis.electrode_selection_qc import (  # noqa: F401 — re-exported
    DEFAULT_OUTPUT_SUBDIR,
    STABILITY_FILE,
    load_well_qc,
    make_well_uid,
    output_root,
    well_dir,
)

logger = logging.getLogger(__name__)

#: Requested ramp: red (selected once) -> green (selected in every scan).
#: Red-green is not colour-blind safe; ``--cmap`` swaps it at the CLI and
#: ``CVD_SAFE_CMAP`` is the drop-in alternative.
DEFAULT_CMAP = "RdYlGn"
CVD_SAFE_CMAP = "viridis"

WHITE = "#ffffff"
_INK = "#1c1a15"
_INK3 = "#84807a"

_RASTER_SUBDIR = "electrode_sel_png"

CHIP_W_UM = GRID_COLS * PITCH_UM      # 3850.0
CHIP_H_UM = GRID_ROWS * PITCH_UM      # 2100.0


# --------------------------------------------------------------------------- #
# Discovery / loaders
# --------------------------------------------------------------------------- #
def discover_wells(root: str | Path) -> list[tuple[str, str, str]]:
    """``(sample_id, plate_id, well_id)`` triples that have a written QC."""
    root = Path(root)
    if not root.exists():
        return []
    found = []
    for stats in root.glob("*/*/well*/" + STABILITY_FILE):
        rel = stats.relative_to(root)
        found.append((rel.parts[0], rel.parts[1], rel.parts[2]))
    return sorted(set(found))


def samples(root: str | Path) -> list[str]:
    return sorted({s for s, _p, _w in discover_wells(root)})


def plates(root: str | Path, sample_id: str) -> list[str]:
    return sorted({p for s, p, _w in discover_wells(root) if s == sample_id})


def wells(root: str | Path, sample_id: str, plate_id: str) -> list[str]:
    return sorted(w for s, p, w in discover_wells(root)
                  if s == sample_id and p == plate_id)


def load_well(root: str | Path, sample_id: str, plate_id: str,
              well_id: str) -> dict | None:
    """One well's QC payload (with ``counts``), or None when not computed."""
    try:
        return load_well_qc(well_dir(root, sample_id, plate_id, well_id))
    except Exception as exc:  # noqa: BLE001 — a corrupt well must not break the page
        logger.warning("could not load %s/%s/%s: %s", sample_id, plate_id, well_id, exc)
        return None


def well_index_to_position(well_id: str) -> tuple[int, int]:
    """``well000`` -> (row 0, col 0); 24 wells in a 4x6 row-major plate."""
    n = int(str(well_id).replace("well", ""))
    return n // 6, n % 6


def well_name(well_id: str) -> str:
    """Plate label ``A1`` … ``D6``."""
    row, col = well_index_to_position(well_id)
    return f"{chr(ord('A') + row)}{col + 1}"


# --------------------------------------------------------------------------- #
# Grids
# --------------------------------------------------------------------------- #
def n_scans(payload: dict | None) -> int:
    if not payload:
        return 0
    return int((payload.get("stability") or {}).get("n_scans") or 0)


def count_grid(payload: dict | None) -> np.ndarray:
    """``(120, 220)`` selection counts, **NaN** where never selected."""
    if not payload or "counts" not in payload:
        return np.full((GRID_ROWS, GRID_COLS), np.nan)
    grid = np.asarray(payload["counts"], dtype=float)
    grid[grid <= 0] = np.nan
    return grid


def fraction_grid(payload: dict | None) -> np.ndarray:
    """Counts as a fraction of the well's in-window scans, NaN off coverage."""
    grid = count_grid(payload)
    n = n_scans(payload)
    return grid / n if n > 0 else grid


def window_label(payload: dict | None) -> str:
    """Human phrase for the window in force, naming how it was anchored."""
    if not payload:
        return "—"
    w = payload.get("window") or {}
    kind, lo, hi = w.get("kind"), w.get("lo"), w.get("hi")
    if kind == "all" or lo is None or hi is None:
        base = "all scans"
    elif kind == "tau":
        base = f"D{lo:g}–D{hi:g} after treatment"
    elif kind == "div":
        base = f"DIV {lo:g}–{hi:g}"
    else:
        base = f"{int(lo)}–{int(hi)}"
    anchor = w.get("anchor")
    if anchor in ("plate", "unanchored"):
        base += f" ({anchor} anchor)"
    return base


# --------------------------------------------------------------------------- #
# Plotly (dashboard)
# --------------------------------------------------------------------------- #
def _empty_figure(message: str, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(color=_INK3, size=13), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8))
    return fig


def fig_count_map(payload: dict | None, cmap: str = DEFAULT_CMAP,
                  height: int = 420) -> go.Figure:
    """Interactive count map on true chip geometry (3850 x 2100 µm)."""
    if not payload:
        return _empty_figure("no electrode-selection QC for this well", height)
    grid = count_grid(payload)
    if not np.isfinite(grid).any():
        return _empty_figure("no electrodes were ever routed in this window", height)

    n = n_scans(payload)
    x = np.arange(GRID_COLS) * PITCH_UM
    y = np.arange(GRID_ROWS) * PITCH_UM
    fig = go.Figure(go.Heatmap(
        z=grid, x=x, y=y, colorscale=cmap,
        # zmin=1 so a single selection reads at the ramp's red end rather than
        # washing out next to the white "never selected" background.
        zmin=1, zmax=max(2, n),
        colorbar=dict(title=f"scans<br>(of {n})", thickness=12, len=0.75),
        hovertemplate=("x %{x:.0f} µm · y %{y:.0f} µm<br>"
                       "selected in %{z:.0f} scan(s)<extra></extra>"),
        hoverongaps=False,
    ))
    fig.update_layout(
        height=height, margin=dict(l=52, r=16, t=30, b=42),
        title=dict(text=f"{payload.get('well_uid', '')} · {window_label(payload)} "
                        f"· {n} scan(s)", font=dict(size=12)),
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    )
    fig.update_xaxes(title="x (µm)", constrain="domain", range=[0, CHIP_W_UM])
    fig.update_yaxes(title="y (µm)", scaleanchor="x", scaleratio=1,
                     range=[0, CHIP_H_UM])
    return fig


def fig_lag_decay(payload: dict | None, height: int = 280) -> go.Figure:
    """Overlap vs temporal distance — de-duplicated curve, with the raw one behind."""
    if not payload:
        return _empty_figure("no data", height)
    stab = payload.get("stability") or {}
    dedup = stab.get("lag_decay_dedup") or {}
    raw = stab.get("lag_decay_raw") or {}
    if not dedup.get("lag_days"):
        return _empty_figure("too few scans for a decay curve", height)

    fig = go.Figure()
    if raw.get("lag_days"):
        fig.add_trace(go.Scatter(
            x=raw["lag_days"], y=raw["mean_jaccard"], mode="lines",
            name="raw", line=dict(color=_INK3, width=1, dash="dot"),
            hovertemplate="lag %{x} d · J %{y:.3f} (raw)<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=dedup["lag_days"], y=dedup["mean_jaccard"], mode="lines+markers",
        name="config-reuse removed", line=dict(color="#b2182b", width=2),
        marker=dict(size=5),
        hovertemplate="lag %{x} d · J %{y:.3f}<extra></extra>"))

    hl = dedup.get("half_life_days")
    title = "Overlap vs temporal distance"
    if hl is not None and np.isfinite(hl):
        title += f" · half-life {hl:.1f} d"
    fig.update_layout(
        height=height, margin=dict(l=52, r=16, t=30, b=42),
        title=dict(text=title, font=dict(size=12)),
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
    )
    fig.update_xaxes(title="days between scans", gridcolor="#e8e4da")
    fig.update_yaxes(title="mean Jaccard", rangemode="tozero", gridcolor="#e8e4da")
    return fig


# --------------------------------------------------------------------------- #
# Matplotlib rasterizers
# --------------------------------------------------------------------------- #
def _cmap_white_bad(name: str):
    from matplotlib import colormaps

    cmap = colormaps[name].copy()
    cmap.set_bad(WHITE)          # never selected == absent data, painted white
    return cmap


def render_count_map(payload: dict, out_path: str | Path, *,
                     cmap: str = DEFAULT_CMAP, formats: tuple[str, ...] = ("png",),
                     dpi: int = 200) -> list[Path]:
    """Vis 1 — full-size cumulative selection map with axes and a colourbar.

    Uses the matplotlib object API (never pyplot): pyplot's global figure
    registry is unsafe under a Dash server's concurrent callbacks, and this
    function is shared with the page.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grid = count_grid(payload)
    masked = np.ma.masked_invalid(grid)
    n = n_scans(payload)

    fig = Figure(figsize=(8.2, 5.0), dpi=dpi, facecolor=WHITE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, facecolor=WHITE)
    im = ax.imshow(masked, origin="lower", cmap=_cmap_white_bad(cmap),
                   vmin=1, vmax=max(2, n), interpolation="nearest",
                   aspect="equal", extent=[0, CHIP_W_UM, 0, CHIP_H_UM])
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

    stab = payload.get("stability") or {}
    union = stab.get("union_electrodes")
    frac = stab.get("union_fraction")
    sub = f"{n} scan(s) · {window_label(payload)}"
    if union is not None and frac is not None:
        sub += f" · union {union:,} electrodes ({frac * 100:.1f}% of array)"
    ax.set_title(f"{payload.get('well_uid', '')}\n{sub}", fontsize=10, color=_INK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label(f"scans selected (of {n})", fontsize=9)
    fig.text(0.01, 0.01, "white = never selected", fontsize=7, color=_INK3)
    fig.tight_layout()

    written = []
    for ext in formats:
        target = out_path.with_suffix("." + ext.lstrip("."))
        fig.savefig(target, dpi=dpi, facecolor=WHITE, bbox_inches="tight",
                    pad_inches=0.03)
        written.append(target)
    return written


def render_count_thumbnail(payload: dict, out_path: str | Path, *,
                           cmap: str = DEFAULT_CMAP,
                           w_px: int = 220, h_px: int = 120) -> Path:
    """Plate-grid thumbnail: fraction of scans on a fixed 0-1 scale, white ground.

    Fraction rather than raw counts so wells with different in-window scan
    counts stay comparable across the plate; opaque white so never-selected
    electrodes read as blank rather than borrowing the page background.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    masked = np.ma.masked_invalid(fraction_grid(payload))
    fig = Figure(figsize=(w_px / 100, h_px / 100), dpi=100, facecolor=WHITE)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=WHITE)
    ax.axis("off")
    if masked.count():
        ax.imshow(masked, origin="lower", aspect="auto",
                  cmap=_cmap_white_bad(cmap), vmin=0.0, vmax=1.0,
                  interpolation="nearest")
    fig.savefig(out_path, dpi=100, facecolor=WHITE)
    return out_path


def _thumbnail_path(cache_dir: Path, results_dir: Path, cmap: str) -> Path:
    """Content-keyed cache path: re-renders only when the QC output changed."""
    marker = Path(results_dir) / "counts.npz"
    try:
        sig = marker.stat()
        base = f"{results_dir}|{cmap}|{sig.st_mtime_ns}|{sig.st_size}"
    except OSError:
        base = f"{results_dir}|{cmap}"
    name = hashlib.sha1(base.encode()).hexdigest()[:16]
    return Path(cache_dir) / _RASTER_SUBDIR / f"{name}.png"


def count_png_data_uri(payload: dict | None, cache_dir: Path | str,
                       results_dir: Path | str,
                       cmap: str = DEFAULT_CMAP) -> str:
    """Cached base64 thumbnail; empty string when there is nothing to draw."""
    if not payload:
        return ""
    path = _thumbnail_path(Path(cache_dir), Path(results_dir), cmap)
    if not path.exists():
        try:
            render_count_thumbnail(payload, path, cmap=cmap)
        except Exception as exc:  # noqa: BLE001 — a bad thumbnail must not break the grid
            logger.warning("thumbnail render failed for %s: %s", results_dir, exc)
            return ""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


# --------------------------------------------------------------------------- #
# Tabular views
# --------------------------------------------------------------------------- #
def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "—"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return str(v)


def scan_table(payload: dict | None) -> list[dict[str, str]]:
    """One row per in-window scan, with overlap against the previous scan."""
    if not payload:
        return []
    scans = payload.get("scans") or []
    stab = payload.get("stability") or {}
    prev_j = {p["j"]: p for p in _pairs_from(stab) if p["j"] == p["i"] + 1}
    rows = []
    for i, s in enumerate(scans):
        p = prev_j.get(i)
        rows.append({
            "date": str(s.get("date") or ""),
            "run": str(s.get("run_id") or ""),
            "tau": _fmt(s.get("tau"), 0),
            "DIV": _fmt(s.get("DIV"), 0),
            "routed": str(s.get("n_routed") or ""),
            "J vs prev": _fmt(p["jaccard"]) if p else "—",
            "base": str(s.get("base_run") or ""),
        })
    return rows


def _pairs_from(stability: dict) -> list[dict]:
    """Adjacent-pair info is not stored in stability.json; recover it if present."""
    return stability.get("adjacent_pairs") or []


def stability_rows(payload: dict | None) -> list[tuple[str, str]]:
    """``(label, value)`` pairs for the stability block on the page and report."""
    if not payload:
        return []
    s = payload.get("stability") or {}
    null = s.get("shift_null") or {}
    dil = s.get("dilated") or {}
    dedup = s.get("lag_decay_dedup") or {}
    raw = s.get("lag_decay_raw") or {}
    core = s.get("core_counts") or {}
    corefrac = s.get("core_fraction") or {}

    def _core(level: str) -> str:
        return f"{core.get(level, 0):,} ({_fmt(corefrac.get(level))} of median routed)"

    rows = [
        ("Scans in window", f"{s.get('n_scans')} of {s.get('n_scans_total')} total"),
        ("Routed per scan", f"{_fmt(s.get('n_routed_min'), 0)} – "
                            f"{_fmt(s.get('n_routed_max'), 0)} "
                            f"(median {_fmt(s.get('n_routed_median'), 0)})"),
        ("Union of selections", f"{s.get('union_electrodes', 0):,} electrodes "
                                f"({_fmt((s.get('union_fraction') or 0) * 100, 1)}% of 26,400)"),
        ("Never selected", f"{s.get('n_never_selected', 0):,}"),
        ("Core ≥50% of scans", _core("0.5")),
        ("Core ≥80% of scans", _core("0.8")),
        ("Core in every scan", _core("1")),
        ("Gini of counts", _fmt(s.get("gini"))),
        ("Mean Jaccard (reuse removed)", _fmt(s.get("mean_jaccard_dedup"))),
        ("Mean Jaccard (raw)", _fmt(s.get("mean_jaccard_raw"))),
        ("Adjacent-scan Jaccard", _fmt(s.get("adjacent_jaccard_dedup"))),
        ("Shift-null enrichment", f"{_fmt(null.get('median_enrichment'), 2)}x "
                                  f"(median p {_fmt(null.get('median_p'), 5)})"),
        ("Pairs with p < 0.01", _fmt(null.get("frac_p_below_0.01"), 2)),
        ("Half-life (reuse removed)", f"{_fmt(dedup.get('half_life_days'), 1)} d"),
        ("Half-life (raw)", f"{_fmt(raw.get('half_life_days'), 1)} d"),
        ("Config reuse", f"{s.get('n_identical_pairs', 0)} identical pair(s), "
                         f"{s.get('n_declared_base_pairs', 0)} declared-base pair(s) "
                         f"of {s.get('n_pairs', 0)}"),
        ("Centroid drift", f"{_fmt(s.get('centroid_drift_um'), 0)} µm"),
    ]
    for k in sorted(dil):
        d = dil[k] or {}
        rows.append((f"Dilated overlap ({k})",
                     f"J {_fmt(d.get('mean_jaccard'))} · "
                     f"{_fmt(d.get('median_enrichment'), 2)}x enriched"))
    return rows


# --------------------------------------------------------------------------- #
# Text report
# --------------------------------------------------------------------------- #
def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(none)_\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def report_markdown(payload: dict | None, *, map_rel: str | None = None,
                    anim_rel: str | None = None,
                    anim_hint: str | None = None) -> str:
    """The per-well text report.

    Written as a lines list joined at the end (the same shape as
    ``scripts/sweep_umap_dims``' report), with relative image embeds so the
    folder stays portable.
    """
    if not payload:
        return "# Electrode-selection QC\n\n_No output for this well._\n"

    s = payload.get("stability") or {}
    w = payload.get("window") or {}
    scans = payload.get("scans") or []
    uid = payload.get("well_uid", "")
    # The group is taken from the LAST scan, not the first: wells are relabelled
    # from a placeholder at treatment, so the earliest name can be stale.
    group, seen = ("", [])
    if scans:
        from yuxin_mea.analysis.electrode_selection_qc import well_groupname
        group, seen = well_groupname(scans)

    lines: list[str] = [
        f"# Electrode-selection QC — {uid}",
        "",
        f"- **Well**: {payload.get('well_id')} "
        f"({well_name(str(payload.get('well_id') or 'well000'))}) on chip "
        f"{payload.get('plate_id')}, sample {payload.get('sample_id')}",
        f"- **Group**: {group or '—'}"
        + (f" (relabelled during the run; also seen as "
           f"{', '.join(repr(g) for g in seen if g != group)})"
           if len(seen) > 1 else ""),
        f"- **Window**: {window_label(payload)}",
    ]

    anchor = w.get("anchor")
    anchor_note = {
        "exact": "treatment date read directly from `treatment_map.csv` for this "
                 "sample/plate/group.",
        "plate": "no exact treatment-map entry for this group — anchored on the "
                 "earliest treatment date of this sample/plate.",
        "unanchored": "no treatment or plating anchor available — the window "
                      "could not be applied, so **all scans** are used.",
        "n/a": "no treatment anchor was needed for this window.",
    }.get(str(anchor), "")
    if anchor_note:
        lines.append(f"- **Window anchor**: `{anchor}` — {anchor_note}")
    if w.get("treatment_date"):
        lines.append(f"- **Treatment date (D0)**: {w['treatment_date']}")
    if w.get("plating_date"):
        lines.append(f"- **Plating date (DIV 0)**: {w['plating_date']}")
    if w.get("note"):
        lines.append(f"- **Note**: {w['note']}")

    lines += ["", "## Verdict", "",
              f"**{str(s.get('verdict', '—')).upper()}** — {s.get('verdict_text', '')}",
              ""]

    if s.get("n_scans", 0) and s.get("n_scans", 0) < s.get("min_scans", 3):
        lines += [
            "> Below the minimum scan count, the lag-decay fit and the core-set "
            "fractions are dominated by noise, so no stability band is claimed "
            "here — only the raw counts above.",
            "",
        ]

    lines += ["## Scans", ""]
    rows = [[r["date"], r["run"], r["tau"], r["DIV"], r["routed"],
             r["J vs prev"], r["base"] or "—"] for r in scan_table(payload)]
    lines.append(_md_table(
        ["date", "run", "τ (d)", "DIV", "routed", "J vs prev", "base run"], rows))

    lines += ["", "## Stability", ""]
    lines.append(_md_table(["metric", "value"],
                           [[k, v] for k, v in stability_rows(payload)]))

    lines += [
        "",
        "### How to read these numbers",
        "",
        "- **Jaccard** is *identity* overlap: the fraction of the two scans' "
        "combined electrodes that both scans routed. It is not a similarity of "
        "position — two adjacent-but-different electrodes score 0.",
        "- **Shift-null enrichment** compares that overlap against sliding one "
        "scan's selection over the array as a torus, which preserves both its "
        "electrode count and its spatial clustering. Enrichment well above 1 "
        "means scans revisit the same *regions* even when identity overlap is "
        "low.",
        "- **Config reuse** matters because a run copied from an earlier one "
        "reproduces its selection exactly and would otherwise inflate stability. "
        "Primary statistics exclude those pairs; the raw values are kept beside "
        "them.",
        "- **Dilated overlap** is reported only as a secondary line: allowing a "
        "1–2 electrode tolerance raises raw overlap but raises the matched "
        "chance level at least as fast, so enrichment falls rather than rises.",
        "",
    ]

    figs = []
    if map_rel:
        figs.append(f"![cumulative selection map]({map_rel})")
    if anim_rel:
        figs.append(f"[Selection history animation]({anim_rel})")
    elif anim_hint:
        figs.append(f"Animation not rendered. To produce it:\n\n```\n{anim_hint}\n```")
    if figs:
        lines += ["## Figures", ""] + figs + [""]

    return "\n".join(lines)
