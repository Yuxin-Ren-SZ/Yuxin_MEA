"""Compose the rendered panel PNGs into one sheet per figure.

    python -m scripts.report.assemble_figure --figures f4 f5 f7

The panels are the deliverable — each is a clean, caption-free image meant to be
placed by hand in a manuscript, a slide or a poster. This driver only *arranges*
already-rendered panels so the whole figure can be reviewed at a glance; it never
re-renders, so assembling costs nothing and can never disagree with the panels.

Layouts are a list of rows, each row a list of panel names. Panels are scaled to
a common row height and lettered A, B, C… in reading order.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import panel_data as D
from .panel_base import captions_dir, panels_dir
from .report_style import report_dir

logger = logging.getLogger("report.assemble")

#: figure id -> rows of panel names. Panels not on disk are skipped with a note,
#: so a partially-rendered figure still assembles.
LAYOUTS: dict[str, list[list[str]]] = {
    "f2": [["F2a_schedule"]],
    "f3": [["F3a_qpcr_heatmap"],
           ["F3b_qpcr_trajectories"],
           ["F3c_icc"]],
    "f4": [["F4a_raster_pre", "F4b_raster_post"],
           ["F4c_firing_rate", "F4d_nb_rate"],
           ["F4e_ibi_cv", "F4f_nb_duration"]],
    "f5": [["F5a_hmm_posteriors"],
           ["F5b_did_nb_rate", "F5c_did_nb_duration", "F5d_did_spikes_per_burst"],
           ["F5e_did_nb_ibi", "F5f_did_firing_rate"],
           ["F5g_did_timecourse"]],
    "f6": [["F6a_umap_state", "F6b_umap_migration"],
           ["F6c_archetype_composition", "F6d_archetype_profiles"],
           ["F6e_archetype_rasters"]],
    "f7": [["F7a_ccg_control", "F7b_ccg_ivh_early", "F7c_ccg_ivh_late"],
           ["F7d_sttc_graphs"],
           ["F7e_activity_fields"],
           ["F7f_sttc_distance", "F7n_burst_modulation", "F7o_did_burst_modulation"],
           ["F7g_did_mean_sttc", "F7h_did_edge_density", "F7i_did_clustering"],
           ["F7j_did_modularity", "F7k_did_global_efficiency",
            "F7l_did_small_worldness"],
           ["F7m_did_timecourse"],
           ["F7p_node_cartography"],
           ["F7q_te_graph"],
           ["F7r_node_metrics_did", "F7s_directed_metrics_did"]],
    "f8": [["F8a_branching_trajectory", "F8b_dcc_trajectory"],
           ["F8c_did_branching_ratio", "F8d_did_dcc", "F8e_did_aval_tau",
            "F8f_did_aval_alpha"],
           ["F8g_did_timecourse"],
           ["F8h_avalanche_pdf", "F8i_crackling"]],
}

#: Figure titles, used by the review sheets and by the PowerPoint export so the
#: two never disagree. Kept short — the detail belongs in the panel captions.
FIGURE_TITLES: dict[str, str] = {
    "f2": "Experimental schedule",
    "f3": "Characterisation of the cultures",
    "f4": "HD-MEA recording schema and network maturation",
    "f5": "Network-burst phenotype after intraventricular-haemorrhage CSF",
    "f6": "Network-state embeddings and cohort-wide burst archetypes",
    "f7": "Functional connectivity, from single-pair correlograms to network metrics",
    "f8": "Neuronal-avalanche criticality",
}

_PAD = 18          # px between panels
_LETTER = 34       # px reserved above each panel for its letter


def _letters(n: int):
    from string import ascii_uppercase
    return [ascii_uppercase[i % 26] * (1 + i // 26) for i in range(n)]


def assemble(fig_id: str, src: Path, out_dir: Path, *, row_height: int = 900):
    """Compose one figure sheet; returns the written path (or None if empty)."""
    from PIL import Image, ImageDraw

    rows = LAYOUTS.get(fig_id)
    if not rows:
        logger.warning("[%s] no layout defined", fig_id)
        return None

    loaded, missing = [], []
    for row in rows:
        imgs = []
        for name in row:
            p = src / f"{name}.png"
            if p.exists():
                imgs.append((name, Image.open(p).convert("RGB")))
            else:
                missing.append(name)
        if imgs:
            loaded.append(imgs)
    if not loaded:
        logger.warning("[%s] no rendered panels found in %s", fig_id, src)
        return None

    # Scale every panel in a row to the same height, then rows to a common width.
    scaled_rows = []
    for imgs in loaded:
        h = row_height
        row_imgs = [(n, im.resize((max(1, round(im.width * h / im.height)), h),
                                  Image.LANCZOS)) for n, im in imgs]
        w = sum(im.width for _, im in row_imgs) + _PAD * (len(row_imgs) - 1)
        scaled_rows.append((row_imgs, w))
    sheet_w = max(w for _, w in scaled_rows)
    # A row narrower than the sheet keeps its own scale rather than being blown
    # up: a single wide panel should not force the others to giant proportions.
    total_h = sum(_LETTER + imgs[0][1].height + _PAD for imgs, _ in scaled_rows)

    sheet = Image.new("RGB", (sheet_w, total_h), "white")
    draw = ImageDraw.Draw(sheet)
    letters = _letters(sum(len(imgs) for imgs, _ in scaled_rows))
    k, y = 0, 0
    for imgs, w in scaled_rows:
        x = 0
        for _name, im in imgs:
            draw.text((x + 4, y + 6), letters[k], fill="black")
            sheet.paste(im, (x, y + _LETTER))
            x += im.width + _PAD
            k += 1
        y += _LETTER + imgs[0][1].height + _PAD

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{fig_id.upper()}_sheet.png"
    sheet.save(path)
    if missing:
        logger.warning("[%s] missing panels: %s", fig_id, ", ".join(missing))
    return path


def write_caption_bundle(fig_id: str, cap_dir: Path, out_dir: Path):
    """One ``<FIG>_captions.txt`` collecting the figure's panel captions in order."""
    rows = LAYOUTS.get(fig_id) or []
    names = [n for row in rows for n in row]
    letters = _letters(len(names))
    lines = []
    for letter, name in zip(letters, names):
        p = cap_dir / f"{name}.txt"
        if p.exists():
            lines.append(f"({letter}) {p.read_text().strip()}")
    if not lines:
        return None
    path = out_dir / f"{fig_id.upper()}_captions.txt"
    path.write_text("\n\n".join(lines) + "\n")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--figures", nargs="*", default=None)
    p.add_argument("--row-height", type=int, default=900)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ctx = D.context(args.config)
    src = panels_dir(ctx.figure_root)
    caps = captions_dir(ctx.figure_root)
    out = report_dir(ctx.figure_root, "figures")
    for fid in (args.figures or sorted(LAYOUTS)):
        sheet = assemble(fid.lower(), src, out, row_height=args.row_height)
        bundle = write_caption_bundle(fid.lower(), caps, out)
        if sheet:
            print(sheet)
        if bundle:
            print(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
