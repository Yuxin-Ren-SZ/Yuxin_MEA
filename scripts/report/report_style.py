"""Single source of truth for journal-figure styling and export.

Every report figure imports :func:`apply_style`, the :data:`GROUP_PALETTE` /
:data:`GROUP_ORDER` constants, and :func:`save_fig` so that colours, fonts and
export formats stay identical across F1-S3. Vector output (PDF/SVG) is new to
this repo — nothing in ``scripts/`` emitted vector before now.

Design notes
------------
* Palette is colour-blind safe (Okabe-Ito), paired by biological family:
  IVH = warm (orange/vermillion), H2O2 = blue (sky/blue), Control = neutral.
* ``pdf.fonttype = 42`` / ``svg.fonttype = 'none'`` keep text as editable
  glyphs so the figures can be tweaked in Illustrator/Inkscape post-export.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Roots / config
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _REPO_ROOT / "pipeline_config_local.json"


def resolve_roots(config_path: Path | str | None = None) -> tuple[Path, Path]:
    """Return ``(analysis_root, figure_root)`` from a pipeline config JSON.

    Mirrors ``scripts/compare_treatment_groups._load_config`` so the report
    package stays import-light (no heavy pipeline imports at module load).
    """
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG
    with cfg_path.open() as fh:
        cfg = json.load(fh)
    g = cfg.get("global", {})
    analysis_root = Path(g["analysis_root"])
    figure_root = Path(g.get("figure_root") or analysis_root)
    return analysis_root, figure_root


def figures_base(figure_root: Path) -> Path:
    """Directory that actually holds the rolled-up export dirs.

    The current config sets ``figure_root`` to the analysis root, but existing
    exports (``treatment_comparison/``, ``burst_method_comparison/``) live under
    a ``figures/`` subdir. Prefer whichever base already contains them; else
    default to ``figures/`` so new output co-locates with prior exports.
    """
    fr = Path(figure_root)
    for cand in (fr, fr / "figures"):
        if (cand / "treatment_comparison").is_dir():
            return cand
    return fr / "figures" if (fr / "figures").is_dir() else fr


def report_dir(figure_root: Path, subdir: str = "") -> Path:
    """``<figures_base>/report[/<subdir>]``, created on demand."""
    out = figures_base(figure_root) / "report"
    if subdir:
        out = out / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
# Okabe-Ito colour-blind-safe base.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#808080",
}

# canonical_group -> colour (values match tidy_long.csv `canonical_group`).
GROUP_PALETTE: dict[str, str] = {
    "Control": OKABE_ITO["black"],
    "IVH_Early": OKABE_ITO["orange"],
    "IVH_Late": OKABE_ITO["vermillion"],
    "NPH": OKABE_ITO["green"],
    "AraC": OKABE_ITO["purple"],
    "H2O2_10uM": OKABE_ITO["sky"],
    "H2O2_20uM": OKABE_ITO["blue"],
}

# Canonical x-order for grouped plots: Control first, then injury families.
GROUP_ORDER: list[str] = [
    "Control",
    "IVH_Early",
    "IVH_Late",
    "NPH",
    "AraC",
    "H2O2_10uM",
    "H2O2_20uM",
]

# Human-readable metric labels (keys = tidy_long / METRIC_SPECS names).
METRIC_LABELS: dict[str, str] = {
    "nb_rate": "Network burst rate (Hz)",
    "nb_count": "Network burst count",
    "nb_duration_mean": "Burst duration (s)",
    "nb_spikes_per_burst_mean": "Spikes per burst",
    "nb_ibi_mean": "Inter-burst interval (s)",
    "median_firing_rate": "Median firing rate (Hz)",
    "n_curated": "Curated units (n)",
    "burst_modulation_index": "Burst modulation index",
    "burst_type_k": "Burst types (k)",
    "cluster_n_clusters": "Bin clusters (n)",
    # Spatial activity map
    "activity_gini": "Activity concentration (Gini)",
    "mean_prop_speed_um_ms": "Propagation speed (µm/ms)",
    # Functional connectivity (STTC)
    "mean_sttc": "Mean STTC",
    "edge_density": "Edge density",
    "modularity": "Modularity",
    "small_worldness": "Small-worldness",
}


def group_color(group: str) -> str:
    """Colour for a ``canonical_group``; grey fallback for unknowns."""
    return GROUP_PALETTE.get(group, OKABE_ITO["grey"])


def ordered_groups(present: Iterable[str]) -> list[str]:
    """Filter+sort an iterable of groups into canonical order."""
    present = set(present)
    ordered = [g for g in GROUP_ORDER if g in present]
    ordered += sorted(present - set(ordered))  # any unexpected groups last
    return ordered


# --------------------------------------------------------------------------- #
# Matplotlib style
# --------------------------------------------------------------------------- #
def apply_style() -> None:
    """Apply journal rcParams. Safe to call more than once."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        # Editable vector text (Illustrator/Inkscape friendly).
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Fonts — Arial/Helvetica if installed, else DejaVu (always present).
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
    })


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def save_fig(
    fig,
    name: str,
    figure_root: Path,
    subdir: str = "main",
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """Save a matplotlib figure to ``<figure_root>/report/<subdir>/`` in each
    requested format. Returns the written paths.
    """
    out_dir = report_dir(figure_root, subdir)
    written: list[Path] = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        written.append(p)
    return written


def save_plotly(
    fig,
    name: str,
    figure_root: Path,
    subdir: str = "main",
    formats: Sequence[str] = ("pdf", "png"),
    width: int = 1600,
    height: int = 1200,
    scale: float = 2.0,
) -> list[Path]:
    """Export a Plotly figure to static vector/raster via kaleido.

    Falls back to an HTML dump (and a clear message) if kaleido is missing, so
    a plate-raster panel never hard-fails the whole report build.
    """
    out_dir = report_dir(figure_root, subdir)
    written: list[Path] = []
    try:
        for ext in formats:
            p = out_dir / f"{name}.{ext}"
            fig.write_image(str(p), width=width, height=height, scale=scale)
            written.append(p)
    except Exception as exc:  # kaleido missing / render failure
        html_path = out_dir / f"{name}.html"
        fig.write_html(str(html_path))
        written.append(html_path)
        print(
            f"[report_style] static export failed for {name!r} ({exc}); "
            f"wrote interactive HTML instead -> {html_path}"
        )
    return written
