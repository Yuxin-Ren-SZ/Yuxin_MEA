"""The composition spec — layout as data, and the file that makes it resumable.

A spec records everything needed to rebuild a set of figures: which panels, where
on a 12-column grid, with which parameters, for which treatment groups, against
which data. Exporting writes it next to the PDF, and loading it restores the
session — so a figure that took an afternoon of nudging can be reopened, tweaked
and re-exported months later.

Layout lives on an integer grid rather than in inches. The browser lays panels
out on a 12-column GridStack and the exporter maps the *same* integers onto a
matplotlib ``GridSpec``, so what the canvas shows and what the PDF contains agree
by construction instead of by eye.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SPEC_VERSION = 1
GENERATOR = "yuxin_mea.report.composer/1.0"

DEFAULT_GRID_COLS = 12
DEFAULT_ROW_HEIGHT_IN = 0.55
DEFAULT_GROUPS = ["Control", "IVH_Early", "IVH_Late", "H2O2_20uM"]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str, fallback: str = "composition") -> str:
    """Filesystem-safe spec name — specs are saved by name into a shared dir."""
    cleaned = _SAFE_NAME.sub("_", str(name).strip()).strip("._-")
    return cleaned or fallback


class SpecError(ValueError):
    """Raised when a spec is structurally invalid (bad grid, unknown panel)."""


@dataclass
class PanelPlacement:
    uid: str
    panel: str
    x: int = 0
    y: int = 0
    w: int = 6
    h: int = 4
    label: str = ""                  # "" -> auto-lettered by reading order
    params: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "PanelPlacement":
        return cls(
            uid=str(d.get("uid") or d.get("panel", "p")),
            panel=str(d["panel"]),
            x=int(d.get("x", 0)), y=int(d.get("y", 0)),
            w=int(d.get("w", 6)), h=int(d.get("h", 4)),
            label=str(d.get("label", "")),
            params=dict(d.get("params") or {}),
        )


@dataclass
class FigureSpec:
    id: str
    title: str = ""
    width_in: float = 9.0
    rows: int = 8
    show_suptitle: bool = True
    panels: list[PanelPlacement] = field(default_factory=list)

    @property
    def height_in(self) -> float:
        return round(self.rows * DEFAULT_ROW_HEIGHT_IN, 4)

    def to_json(self) -> dict:
        return {"id": self.id, "title": self.title, "width_in": self.width_in,
                "rows": self.rows, "show_suptitle": self.show_suptitle,
                "panels": [p.to_json() for p in self.panels]}

    @classmethod
    def from_json(cls, d: dict) -> "FigureSpec":
        return cls(
            id=str(d["id"]), title=str(d.get("title", "")),
            width_in=float(d.get("width_in", 9.0)), rows=int(d.get("rows", 8)),
            show_suptitle=bool(d.get("show_suptitle", True)),
            panels=[PanelPlacement.from_json(p) for p in d.get("panels", [])],
        )

    def ordered_panels(self) -> list[PanelPlacement]:
        """Reading order: top-to-bottom, then left-to-right. Drives auto-labels."""
        return sorted(self.panels, key=lambda p: (p.y, p.x))

    def auto_labels(self) -> dict[str, str]:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        out = {}
        for i, p in enumerate(self.ordered_panels()):
            out[p.uid] = p.label or (letters[i] if i < len(letters) else f"P{i + 1}")
        return out


@dataclass
class Composition:
    """A whole session: global settings, the figures, and the data provenance."""

    name: str = "composition"
    groups: list[str] = field(default_factory=lambda: list(DEFAULT_GROUPS))
    grid_cols: int = DEFAULT_GRID_COLS
    row_height_in: float = DEFAULT_ROW_HEIGHT_IN
    dpi: int = 300
    # Gap around every cell, **in inches**. Pinned rects give matplotlib no
    # chance to reserve room for decorations, so the gutter *is* the space that
    # axis labels, tick labels and titles live in. Inches rather than figure
    # fractions because that space is set by font size, not by figure size: a
    # short figure needs the same 0.16" for an x-label as a tall one. Larger
    # vertically because a panel's x-label and the next panel's title share it.
    gutter_x_in: float = 0.12
    gutter_y_in: float = 0.20
    seed: int = 0
    figures: list[FigureSpec] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    created: str = ""

    # -- serialisation ------------------------------------------------------ #
    def to_json(self) -> dict:
        return {
            "version": SPEC_VERSION,
            "generator": GENERATOR,
            "name": self.name,
            "created": self.created,
            "data": dict(self.data),
            "globals": {
                "groups": list(self.groups),
                "grid_cols": self.grid_cols,
                "row_height_in": self.row_height_in,
                "dpi": self.dpi,
                "gutter_x_in": self.gutter_x_in,
                "gutter_y_in": self.gutter_y_in,
                "seed": self.seed,
            },
            "figures": [f.to_json() for f in self.figures],
        }

    @classmethod
    def from_json(cls, d: dict) -> "Composition":
        version = int(d.get("version", SPEC_VERSION))
        if version > SPEC_VERSION:
            raise SpecError(
                f"spec version {version} is newer than this composer "
                f"({SPEC_VERSION}); upgrade before loading")
        g = d.get("globals") or {}
        return cls(
            name=str(d.get("name", "composition")),
            groups=list(g.get("groups") or DEFAULT_GROUPS),
            grid_cols=int(g.get("grid_cols", DEFAULT_GRID_COLS)),
            row_height_in=float(g.get("row_height_in", DEFAULT_ROW_HEIGHT_IN)),
            dpi=int(g.get("dpi", 300)),
            gutter_x_in=float(g.get("gutter_x_in", 0.12)),
            gutter_y_in=float(g.get("gutter_y_in", 0.20)),
            seed=int(g.get("seed", 0)),
            figures=[FigureSpec.from_json(f) for f in d.get("figures", [])],
            data=dict(d.get("data") or {}),
            created=str(d.get("created", "")),
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "Composition":
        return cls.from_json(json.loads(Path(path).read_text()))

    # -- validation --------------------------------------------------------- #
    def validate(self, known_panels: set[str] | None = None) -> list[str]:
        """Return human-readable problems. Structural errors raise; the rest warn."""
        problems: list[str] = []
        if self.grid_cols < 1:
            raise SpecError("grid_cols must be >= 1")
        if not self.figures:
            problems.append("composition has no figures")
        seen_fig_ids: set[str] = set()
        for f in self.figures:
            if f.id in seen_fig_ids:
                raise SpecError(f"duplicate figure id {f.id!r}")
            seen_fig_ids.add(f.id)
            if f.rows < 1:
                raise SpecError(f"figure {f.id!r}: rows must be >= 1")
            if f.width_in <= 0:
                raise SpecError(f"figure {f.id!r}: width_in must be > 0")
            seen_uids: set[str] = set()
            for p in f.panels:
                if p.uid in seen_uids:
                    raise SpecError(f"figure {f.id!r}: duplicate panel uid {p.uid!r}")
                seen_uids.add(p.uid)
                if p.w < 1 or p.h < 1:
                    raise SpecError(f"{f.id}/{p.uid}: w and h must be >= 1")
                if p.x < 0 or p.y < 0:
                    raise SpecError(f"{f.id}/{p.uid}: x and y must be >= 0")
                if p.x + p.w > self.grid_cols:
                    raise SpecError(
                        f"{f.id}/{p.uid}: spans past the grid "
                        f"(x={p.x} w={p.w} > {self.grid_cols} columns)")
                if p.y + p.h > f.rows:
                    raise SpecError(
                        f"{f.id}/{p.uid}: spans past the figure "
                        f"(y={p.y} h={p.h} > {f.rows} rows)")
                if known_panels is not None and p.panel not in known_panels:
                    problems.append(f"{f.id}/{p.uid}: unknown panel {p.panel!r}")
            problems.extend(f"{f.id}: {m}" for m in _overlaps(f))
        return problems

    # -- editing helpers used by the server --------------------------------- #
    def figure(self, fig_id: str) -> FigureSpec:
        for f in self.figures:
            if f.id == fig_id:
                return f
        raise SpecError(f"no figure {fig_id!r}")

    def new_figure_id(self, stem: str = "F") -> str:
        used = {f.id for f in self.figures}
        i = 1
        while f"{stem}{i}" in used:
            i += 1
        return f"{stem}{i}"

    def new_uid(self) -> str:
        used = {p.uid for f in self.figures for p in f.panels}
        i = 1
        while f"p{i}" in used:
            i += 1
        return f"p{i}"


def _overlaps(f: FigureSpec) -> list[str]:
    """Overlapping cells are a warning, not an error — matplotlib will draw both."""
    msgs = []
    for i, a in enumerate(f.panels):
        for b in f.panels[i + 1:]:
            if (a.x < b.x + b.w and b.x < a.x + a.w
                    and a.y < b.y + b.h and b.y < a.y + a.h):
                msgs.append(f"{a.uid} and {b.uid} overlap")
    return msgs


# --------------------------------------------------------------------------- #
# Layout operations — split / merge / reflow
# --------------------------------------------------------------------------- #
def reflow(f: FigureSpec, grid_cols: int = DEFAULT_GRID_COLS,
           order: str = "reading") -> FigureSpec:
    """Re-pack panels into a tidy grid, preserving cell sizes.

    ``order="reading"`` re-packs by current position, which is what you want
    after a split. ``order="list"`` keeps the panel list's order — used after a
    merge, where sorting by position would interleave the two source figures'
    panels and scramble the A/B/C lettering.
    """
    x = y = row_h = 0
    seq = f.panels if order == "list" else f.ordered_panels()
    for p in seq:
        w = min(p.w, grid_cols)
        if x + w > grid_cols:
            x, y, row_h = 0, y + row_h, 0
        p.x, p.y = x, y
        p.w = w
        x += w
        row_h = max(row_h, p.h)
    f.rows = max(1, y + row_h)
    return f


def split_figure(comp: Composition, fig_id: str, uids: list[str],
                 new_id: str | None = None, new_title: str = "") -> FigureSpec:
    """Move ``uids`` out of a figure into a new one placed directly after it."""
    src = comp.figure(fig_id)
    moving = [p for p in src.panels if p.uid in set(uids)]
    if not moving:
        raise SpecError("split needs at least one panel")
    if len(moving) == len(src.panels):
        raise SpecError("split would empty the source figure")
    src.panels = [p for p in src.panels if p.uid not in set(uids)]
    dst = FigureSpec(id=new_id or comp.new_figure_id(f"{fig_id}_"),
                     title=new_title or f"{src.title} (split)",
                     width_in=src.width_in, rows=src.rows, panels=moving)
    reflow(src, comp.grid_cols)
    reflow(dst, comp.grid_cols)
    comp.figures.insert(comp.figures.index(src) + 1, dst)
    return dst


def merge_figures(comp: Composition, fig_ids: list[str],
                  new_title: str = "") -> FigureSpec:
    """Concatenate figures into the first one, then re-flow to remove collisions."""
    if len(fig_ids) < 2:
        raise SpecError("merge needs at least two figures")
    targets = [comp.figure(i) for i in fig_ids]
    head = targets[0]
    for other in targets[1:]:
        head.panels.extend(other.panels)
        comp.figures.remove(other)
    head.width_in = max(t.width_in for t in targets)
    if new_title:
        head.title = new_title
    reflow(head, comp.grid_cols, order="list")
    return head


# --------------------------------------------------------------------------- #
# Data provenance
# --------------------------------------------------------------------------- #
def data_fingerprint(tidy, analysis_root: Path, config_path: Path | str | None) -> dict:
    """Identify the dataset a composition was built against.

    Hashing the tidy table's shape, columns and content is enough to notice that
    a rollup was re-run underneath a saved spec, which is exactly when a restored
    layout would silently mean something different.
    """
    h = hashlib.sha256()
    h.update(",".join(map(str, tidy.columns)).encode())
    h.update(str(tidy.shape).encode())
    try:
        h.update(pd_hash(tidy))
    except Exception:  # noqa: BLE001 - fingerprinting must never break an export
        pass
    return {
        "config": str(config_path) if config_path else None,
        "analysis_root": str(analysis_root),
        "tidy_rows": int(len(tidy)),
        "tidy_cols": int(tidy.shape[1]),
        "tidy_sha256": h.hexdigest(),
    }


def pd_hash(df) -> bytes:
    import pandas as pd

    return pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()


def fingerprint_delta(saved: dict, current: dict) -> list[str]:
    """Human-readable differences between a spec's data and the data now loaded."""
    if not saved:
        return []
    out = []
    if saved.get("tidy_sha256") and saved["tidy_sha256"] != current.get("tidy_sha256"):
        out.append("tidy_long.csv content changed since this spec was saved")
    if saved.get("tidy_rows") != current.get("tidy_rows"):
        out.append(f"row count {saved.get('tidy_rows')} -> {current.get('tidy_rows')}")
    if saved.get("tidy_cols") != current.get("tidy_cols"):
        out.append(f"column count {saved.get('tidy_cols')} -> {current.get('tidy_cols')}")
    if saved.get("analysis_root") != current.get("analysis_root"):
        out.append(f"analysis_root {saved.get('analysis_root')} -> "
                   f"{current.get('analysis_root')}")
    return out


# --------------------------------------------------------------------------- #
# Default composition — the current report, expressed as a spec
# --------------------------------------------------------------------------- #
def default_composition(groups: list[str] | None = None) -> Composition:
    """The batch report's figures as a starting point for editing."""
    g = list(groups or DEFAULT_GROUPS)

    def panels(*items) -> list[PanelPlacement]:
        return [PanelPlacement(uid=f"p{i + 1}", panel=pid, x=x, y=y, w=w, h=h,
                               params=dict(prm or {}))
                for i, (pid, x, y, w, h, prm) in enumerate(items)]

    return Composition(name="report", groups=g, figures=[
        FigureSpec(id="F4", title="Network activity & development", width_in=9.0,
                   rows=8, panels=panels(
                       ("raster.well", 0, 0, 6, 3, {"group": "Control"}),
                       ("raster.well", 6, 0, 6, 3, {"group": "IVH_Late"}),
                       ("trajectory.metric", 0, 3, 6, 5,
                        {"metric": "median_firing_rate"}),
                       ("trajectory.metric", 6, 3, 6, 5, {"metric": "nb_rate"}))),
        FigureSpec(id="F5", title="Burst phenotype vs Control maturation",
                   width_in=8.6, rows=8, panels=panels(
                       ("forest.burst", 0, 0, 8, 8, None),
                       ("ml.modulation", 8, 0, 4, 8, None))),
        FigureSpec(id="F6", title="ML burst characterization", width_in=7.6,
                   rows=10, panels=panels(
                       ("ml.posterior", 0, 0, 12, 5, None),
                       ("ml.umap", 3, 5, 6, 5, None))),
        FigureSpec(id="F6b", title="Post-treatment network-state dynamics",
                   width_in=12.0, rows=5, panels=panels(
                       ("manifold.state", 0, 0, 3, 5,
                        {"arm": "IVH_Late", "stage": 0}),
                       ("manifold.state", 3, 0, 3, 5,
                        {"arm": "IVH_Late", "stage": 1}),
                       ("manifold.state", 6, 0, 3, 5,
                        {"arm": "IVH_Late", "stage": 2}),
                       ("manifold.state", 9, 0, 3, 5,
                        {"arm": "IVH_Late", "stage": 3}))),
        FigureSpec(id="F6c", title="Cohort-wide burst archetypes", width_in=8.4,
                   rows=10, panels=panels(
                       ("archetypes.space", 0, 0, 6, 5, {"color_by": "group"}),
                       ("archetypes.space", 6, 0, 6, 5, {"color_by": "archetype"}),
                       ("archetypes.composition", 0, 5, 6, 5, None),
                       ("archetypes.profiles", 6, 5, 6, 5, None))),
        FigureSpec(id="F7", title="Spatial activity & functional connectivity",
                   width_in=9.0, rows=10, panels=panels(
                       ("spatial.field", 0, 0, 4, 5, None),
                       ("spatial.propagation", 4, 0, 4, 5, None),
                       ("connectivity.graph", 8, 0, 4, 5, None),
                       ("connectivity.sttc_distance", 0, 5, 6, 5, None),
                       ("forest.connectivity", 6, 5, 6, 5, None))),
        FigureSpec(id="F7b", title="Node-level & directed connectivity",
                   width_in=9.0, rows=10, panels=panels(
                       ("node.cartography", 0, 0, 6, 5, None),
                       ("directed.graph", 6, 0, 6, 5, None),
                       ("forest.node", 0, 5, 6, 5, None),
                       ("forest.directed", 6, 5, 6, 5, None))),
        FigureSpec(id="F8", title="Neuronal-avalanche criticality", width_in=8.6,
                   rows=10, panels=panels(
                       ("criticality.distributions", 0, 0, 6, 5, None),
                       ("criticality.crackling", 6, 0, 6, 5, None),
                       ("criticality.branching", 0, 5, 6, 5, None),
                       ("forest.criticality", 6, 5, 6, 5, None))),
        FigureSpec(id="S1", title="ML vs traditional burst detection", width_in=9.0,
                   rows=5, panels=panels(
                       ("method.compare", 0, 0, 4, 5, {"view": "counts"}),
                       ("method.compare", 4, 0, 4, 5, {"view": "bland_altman"}),
                       ("method.compare", 8, 0, 4, 5, {"view": "iou"}))),
        FigureSpec(id="S2", title="Single-unit QC & waveforms", width_in=9.0,
                   rows=10, panels=panels(
                       ("units.qc_hist", 0, 0, 4, 5, {"metric": "presence_ratio"}),
                       ("units.qc_hist", 4, 0, 4, 5, {"metric": "rp_contamination"}),
                       ("units.qc_hist", 8, 0, 4, 5, {"metric": "amplitude_median"}),
                       ("units.qc_hist", 0, 5, 4, 5,
                        {"metric": "firing_rate", "log": True}),
                       ("units.waveforms", 4, 5, 4, 5, None),
                       ("units.celltype", 8, 5, 4, 5, None))),
    ])
