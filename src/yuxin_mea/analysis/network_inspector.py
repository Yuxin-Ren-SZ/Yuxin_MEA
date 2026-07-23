"""Per-well network-analysis inspector library.

Pure loaders + Plotly figure builders for the dashboard ``Network inspector``
page (:mod:`yuxin_mea.dashboard.pages.network_inspector`). No Dash imports and —
like :mod:`yuxin_mea.analysis.unit_inspector` — **no SpikeInterface / torch**:
everything is ``np.load`` / ``pd.read_parquet`` / ``json`` straight off what the
four network stages wrote.

Stages and their on-disk layout (per well)::

    connectivity_data/<rk>/<rec>/<well>/connectivity/
        sttc_matrix.npy  sttc_sweep.npz  edges.parquet  graph_metrics.json
        diagnostics.json   node_metrics.parquet  node_scalars.json   (node pair
                                                   is optional — see below)
    spatial_map_data/<rk>/<rec>/<well>/spatial_map/
        activity_field.npy  burst_propagation.parquet  spatial_metrics.json
        diagnostics.json
    criticality_data/<rk>/<rec>/<well>/criticality/
        avalanches.parquet  criticality_metrics.json  powerlaw_fits.json
        diagnostics.json
    directed_connectivity_data/<rk>/<rec>/<well>/directed/
        te_matrix.npy  unit_ids.npy  directed_edges.parquet
        directed_metrics.json  diagnostics.json

Two facts about the live data shape this module (both are *read* problems, not
things this module fixes):

1. ``criticality`` and ``directed_connectivity`` are **absent from
   pipeline_cache.json** — they were run out of band. The manifest resolver every
   other viewer leans on therefore returns nothing for them, so
   :func:`stage_well_dirs` falls back to reconstructing the convention path from
   each cache entry's compound ``rec_name/well_id`` via the task class's own
   ``build_output_path``. Verified against the live cache + NAS: the
   reconstruction resolves exactly the 1585 criticality / 1599 directed wells
   that exist on disk.
2. ``node_metrics.parquet`` / ``node_scalars.json`` have **no committed
   producer** — :func:`yuxin_mea.analysis.graph_nodes.compute_node_metrics`
   exists but nothing writes its output, so the files are present for only a
   subset of wells. Treated as optional throughout.

Unit coordinates are a third, independent absence: ``ConnectivityTask`` only
records positions when ``quality_metrics.pkl`` actually carries ``loc_x``/
``loc_y`` (see :mod:`yuxin_mea.analysis.connectivity`), so a well can have a
perfectly good STTC graph and no coordinates at all. The three spatial figures
render their own "no unit positions" empty state rather than raising.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from yuxin_mea.tasks.connectivity import ConnectivityTask
from yuxin_mea.tasks.criticality import CriticalityTask
from yuxin_mea.tasks.directed_connectivity import DirectedConnectivityTask
from yuxin_mea.tasks.spatial_map import SpatialMapTask

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
# Categorical hues for the four node roles. The dashboard's own `mea_paper`
# colorway (theme.py) was validated as a 4-way categorical set and FAILED — the
# sage/info pair sits at normal-vision ΔE 13.2, below the 15 floor, and two
# slots fall under the chroma floor. These three are the reference palette's
# first three slots, which pass every check on the *all-pairs* list (the right
# list for a scatter, where any two roles can land side by side): worst CVD ΔE
# 9.2, worst normal-vision ΔE 24.0 on this surface.
#
# `peripheral` is deliberately the muted "Other" slot rather than a fourth hue —
# it is the majority role (the un-notable middle of the cartography plane), and a
# 4th chromatic slot cannot clear the all-pairs floors. Role is also encoded by
# position in fig_cartography (the z=2.5 / P=0.62 guides partition the plane), so
# colour there is a secondary encoding.
#
# `#1baf7a` sits at 2.67:1 on the paper surface, so the relief rule applies: the
# cartography figure always ships with a legend and the Nodes tab carries the
# node table view.
ROLE_COLORS = {
    "connector_hub": "#2a78d6",
    "provincial_hub": "#eb6834",
    "leaf": "#1baf7a",
    "peripheral": "#84807a",
}
ROLE_ORDER = ("connector_hub", "provincial_hub", "peripheral", "leaf")

# Guide lines from graph_nodes._role (Guimera-Amaral / MEA-NAP convention).
Z_HUB = 2.5
P_CONNECTOR = 0.62

_INK = "#1c1a15"
_INK3 = "#84807a"
_LINE = "#d9d3c5"
_SURFACE = "#fbf9f3"
_ACCENT = "#2a78d6"

# Sequential ramp: one hue, light to dark (paper surface -> deep blue).
SEQ_SCALE = [
    [0.0, "#fbf9f3"], [0.25, "#b9d0ea"], [0.5, "#79a8dc"],
    [0.75, "#3b7ec4"], [1.0, "#17427a"],
]
# Diverging ramp: two hues either side of a neutral grey midpoint (never a hue
# at the midpoint). Used for metrics with a meaningful centre (branching ratio
# about 1, DCC about 0).
DIV_SCALE = [
    [0.0, "#8c3410"], [0.25, "#eb6834"], [0.5, "#e6e1d2"],
    [0.75, "#2a78d6"], [1.0, "#123f77"],
]

# Metrics that read as "distance from a critical/neutral point" -> diverging.
DIVERGING_METRICS = {"branching_ratio_mr", "branching_ratio_naive", "dcc",
                     "assortativity", "degree_skew", "small_worldness"}
# The centre value each diverging metric pivots on.
METRIC_CENTER = {"branching_ratio_mr": 1.0, "branching_ratio_naive": 1.0,
                 "dcc": 0.0, "assortativity": 0.0, "degree_skew": 0.0,
                 "small_worldness": 1.0}


# --------------------------------------------------------------------------- #
# Stage registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageSpec:
    """How one network stage is found on disk."""

    key: str
    dirname: str      # default output-root basename under analysis_root
    task_name: str    # name used in pipeline_cache.json
    task: Any         # task class exposing build_output_path()
    marker: str       # file that must exist for the well to count as present


STAGES: dict[str, StageSpec] = {
    "connectivity": StageSpec(
        "connectivity", "connectivity_data", "connectivity",
        ConnectivityTask, "graph_metrics.json"),
    "spatial_map": StageSpec(
        "spatial_map", "spatial_map_data", "spatial_map",
        SpatialMapTask, "spatial_metrics.json"),
    "criticality": StageSpec(
        "criticality", "criticality_data", "criticality",
        CriticalityTask, "criticality_metrics.json"),
    "directed": StageSpec(
        "directed", "directed_connectivity_data", "directed_connectivity",
        DirectedConnectivityTask, "directed_metrics.json"),
}

STAGE_ORDER = ("connectivity", "spatial_map", "criticality", "directed")

# Well-level scalars offered to the plate grid, in menu order, mapped to the
# stage whose JSON carries them.
PLATE_METRICS: dict[str, str] = {
    "mean_sttc": "connectivity", "edge_density": "connectivity",
    "mean_degree": "connectivity", "clustering_coeff": "connectivity",
    "modularity": "connectivity", "global_efficiency": "connectivity",
    "small_worldness": "connectivity", "n_nodes": "connectivity",
    "n_edges": "connectivity",
    "hub_fraction": "connectivity", "leaf_fraction": "connectivity",
    "participation_mean": "connectivity", "mean_betweenness": "connectivity",
    "assortativity": "connectivity", "rich_club": "connectivity",
    "degree_cv": "connectivity", "degree_skew": "connectivity",
    "n_ccg_sig_pairs": "connectivity", "frac_ccg_sig": "connectivity",
    "mean_abs_ccg_lag_ms": "connectivity", "ccg_flow_asymmetry": "connectivity",
    "branching_ratio_mr": "criticality", "branching_ratio_naive": "criticality",
    "dcc": "criticality", "aval_tau": "criticality", "aval_alpha": "criticality",
    "gamma_fit": "criticality", "n_avalanches": "criticality",
    "mean_aval_size": "criticality",
    "mean_te": "directed", "te_edge_density": "directed",
    "degree_asymmetry": "directed", "reciprocity": "directed",
    "flow_hierarchy": "directed", "causal_density": "directed",
    "activity_gini": "spatial_map", "active_area_frac": "spatial_map",
    "mean_prop_speed_um_ms": "spatial_map", "prop_planarity": "spatial_map",
    "mean_prop_delay_ms": "spatial_map",
}

METRIC_LABELS = {
    "mean_sttc": "mean STTC", "edge_density": "edge density",
    "mean_degree": "mean degree", "clustering_coeff": "clustering",
    "modularity": "modularity", "global_efficiency": "global efficiency",
    "small_worldness": "small-worldness", "hub_fraction": "hub fraction",
    "leaf_fraction": "leaf fraction", "participation_mean": "mean participation",
    "mean_betweenness": "mean betweenness", "assortativity": "assortativity",
    "rich_club": "rich club", "degree_cv": "degree CV",
    "degree_skew": "degree skew", "branching_ratio_mr": "branching ratio (MR)",
    "branching_ratio_naive": "branching ratio (naive)",
    "dcc": "DCC (distance to criticality)", "aval_tau": "avalanche tau (size)",
    "aval_alpha": "avalanche alpha (duration)", "gamma_fit": "gamma (crackling)",
    "n_avalanches": "n avalanches", "mean_aval_size": "mean avalanche size",
    "mean_te": "mean TE", "te_edge_density": "TE edge density",
    "degree_asymmetry": "degree asymmetry", "reciprocity": "reciprocity",
    "flow_hierarchy": "flow hierarchy", "causal_density": "causal density",
    "activity_gini": "activity Gini", "active_area_frac": "active area frac",
    "mean_prop_speed_um_ms": "propagation speed (um/ms)",
    "prop_planarity": "propagation planarity",
    "mean_prop_delay_ms": "propagation delay (ms)",
    "n_nodes": "n units", "n_edges": "n edges",
    "n_ccg_sig_pairs": "n CCG-significant pairs",
    "frac_ccg_sig": "frac CCG-significant",
    "mean_abs_ccg_lag_ms": "mean |CCG lag| (ms)",
    "ccg_flow_asymmetry": "CCG asymmetry (mean |.|)",
    "ccg_enabled": "CCG enabled", "ccg_bin_ms": "CCG bin (ms)",
    "ccg_window_ms": "CCG window (ms)", "ccg_n_pairs": "CCG pairs computed",
    "ccg_pairs_capped": "CCG pairs skipped (cap)",
}


def stage_output_root(
    analysis_root: Path | str,
    stage: str,
    config_roots: dict[str, str] | None = None,
) -> Path:
    """Resolve one stage's output root.

    An explicit override (from config or the page's root inputs) wins; otherwise
    default to ``<analysis_root>/<dirname>``, which matches both the task
    defaults and what is actually on disk. ``criticality`` and
    ``directed_connectivity`` are not in the shipped config, so they always take
    the convention branch unless the user overrides them on the page.
    """
    override = (config_roots or {}).get(stage)
    if override:
        p = Path(override)
        return p if p.is_absolute() else Path(analysis_root) / p
    return Path(analysis_root) / STAGES[stage].dirname


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def list_network_recordings(analysis_root: Path | str) -> list[str]:
    """Recording keys with a completed ``connectivity`` task."""
    from yuxin_mea.analysis.burst_inspector import _load_pipeline_cache

    cache = _load_pipeline_cache(analysis_root)
    recs: set[str] = set()
    for entry in cache.values():
        if not isinstance(entry, dict):
            continue
        task = (entry.get("tasks") or {}).get("connectivity")
        rk = entry.get("recording_key")
        if task and task.get("status") == "complete" and rk:
            recs.add(rk)
    return sorted(recs)


def wells_in_recording(
    analysis_root: Path | str, recording_key: str
) -> list[tuple[str, str]]:
    """Return sorted ``(rec_name, well_id)`` pairs for one recording.

    Reads the pipeline cache entries directly rather than going through
    :func:`~yuxin_mea.analysis.burst_inspector.well_output_dirs_from_cache`,
    which keys by ``well_id`` alone and so **discards the ``rec_name``** that
    ``build_output_path`` needs to rebuild a convention path.
    """
    from yuxin_mea.analysis.burst_inspector import _load_pipeline_cache

    cache = _load_pipeline_cache(analysis_root)
    out: set[tuple[str, str]] = set()
    for entry in cache.values():
        if not isinstance(entry, dict) or entry.get("recording_key") != recording_key:
            continue
        compound = entry.get("well_id", "")
        if "/" not in compound:
            continue
        rec_name, well_id = compound.split("/", 1)
        out.add((rec_name, well_id))
    return sorted(out)


def stage_well_dirs(
    analysis_root: Path | str,
    recording_key: str,
    stage: str,
    output_root: Path | str | None = None,
) -> dict[str, Path]:
    """Return ``{well_id: <stage output dir>}`` for one recording + stage.

    Manifest first (:func:`well_output_dirs_from_cache`), which resolves
    ``connectivity`` / ``spatial_map``. Then fill any well the manifest didn't
    cover by rebuilding the convention path with the task class's own
    ``build_output_path`` and ``stat``-checking the stage's marker file — this is
    the only branch that resolves ``criticality`` / ``directed``, which no
    manifest knows about. 24 stats per recording; never walks the tree.
    """
    from yuxin_mea.analysis.burst_inspector import well_output_dirs_from_cache

    spec = STAGES[stage]
    root = Path(output_root) if output_root is not None else \
        Path(analysis_root) / spec.dirname

    out: dict[str, Path] = {}
    for well_id, path in well_output_dirs_from_cache(
        analysis_root, recording_key, spec.task_name
    ).items():
        if (Path(path) / spec.marker).exists():
            out[well_id] = Path(path)

    for rec_name, well_id in wells_in_recording(analysis_root, recording_key):
        if well_id in out:
            continue
        path = spec.task.build_output_path(root, recording_key, rec_name, well_id)
        if (path / spec.marker).exists():
            out[well_id] = path
    return out


def curation_well_dirs(
    analysis_root: Path | str, recording_key: str
) -> dict[str, Path]:
    """``{well_id: <auto_curation dir>}`` — source of the unit coordinates."""
    from yuxin_mea.analysis.burst_inspector import well_output_dirs_from_cache

    return {w: Path(p) for w, p in well_output_dirs_from_cache(
        analysis_root, recording_key, "auto_curation").items()}


def all_stage_dirs(
    analysis_root: Path | str,
    recording_key: str,
    config_roots: dict[str, str] | None = None,
) -> dict[str, dict[str, Path]]:
    """``{stage: {well_id: dir}}`` for all four stages plus ``auto_curation``."""
    out = {
        stage: stage_well_dirs(
            analysis_root, recording_key, stage,
            stage_output_root(analysis_root, stage, config_roots),
        )
        for stage in STAGE_ORDER
    }
    out["auto_curation"] = curation_well_dirs(analysis_root, recording_key)
    return out


# --------------------------------------------------------------------------- #
# Small readers (missing file -> None, never raises)
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict | None:
    try:
        with Path(path).open() as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_npy(path: Path) -> np.ndarray | None:
    try:
        return np.load(path)
    except (OSError, ValueError):
        return None


def _read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError, ImportError):
        return None


def _read_npz(path: Path) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError):
        return None


def _read_positions(curation_dir: Path | None) -> dict[int, tuple[float, float]] | None:
    """Unit ``{unit_id: (x, y)}`` from ``quality_metrics.pkl``.

    Returns ``None`` when the file is missing *or* carries no ``loc_x``/``loc_y``
    columns — the same condition under which ``ConnectivityTask`` passes
    ``positions=None`` and every ``dist_um`` comes out NaN.
    """
    if curation_dir is None:
        return None
    path = Path(curation_dir) / "quality_metrics.pkl"
    try:
        qm = pd.read_pickle(path)
    except (OSError, ValueError, ImportError, AttributeError):
        return None
    if not isinstance(qm, pd.DataFrame) or "loc_x" not in qm.columns \
            or "loc_y" not in qm.columns:
        return None
    out: dict[int, tuple[float, float]] = {}
    for uid, row in qm.iterrows():
        try:
            out[int(uid)] = (float(row["loc_x"]), float(row["loc_y"]))
        except (TypeError, ValueError):
            continue
    return out or None


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
@dataclass
class NetworkBundle:
    """Everything the Network inspector's five tabs need for one well.

    Any stage that did not run for this well leaves its fields ``None``; the
    figure builders render an empty state rather than raising. ``positions`` is
    ``None`` independently of stage presence (see module docstring).
    """

    recording_key: str = ""
    rec_name: str = ""
    well_id: str = ""
    dirs: dict[str, Path] = field(default_factory=dict)

    # connectivity
    graph_metrics: dict | None = None
    conn_diagnostics: dict | None = None
    sttc_matrix: np.ndarray | None = None
    sttc_sweep: dict[str, np.ndarray] | None = None
    unit_ids: np.ndarray | None = None
    edges: pd.DataFrame | None = None
    # node metrics (optional even when connectivity is present)
    node_metrics: pd.DataFrame | None = None
    node_scalars: dict | None = None
    # correlograms (present only for wells computed/backfilled after CCGs landed)
    ccg_counts: np.ndarray | None = None
    ccg_baseline: np.ndarray | None = None
    ccg_lags_ms: np.ndarray | None = None
    ccg_pairs: np.ndarray | None = None
    ccg_meta: dict | None = None
    # criticality
    criticality_metrics: dict | None = None
    powerlaw_fits: dict | None = None
    crit_diagnostics: dict | None = None
    avalanches: pd.DataFrame | None = None
    # directed
    directed_metrics: dict | None = None
    dir_diagnostics: dict | None = None
    te_matrix: np.ndarray | None = None
    te_unit_ids: np.ndarray | None = None
    directed_edges: pd.DataFrame | None = None
    # spatial
    spatial_metrics: dict | None = None
    spatial_diagnostics: dict | None = None
    activity_field: np.ndarray | None = None
    burst_propagation: pd.DataFrame | None = None
    # shared
    positions: dict[int, tuple[float, float]] | None = None

    def has(self, stage: str) -> bool:
        """True when ``stage``'s scalar metrics loaded."""
        return {
            "connectivity": self.graph_metrics,
            "spatial_map": self.spatial_metrics,
            "criticality": self.criticality_metrics,
            "directed": self.directed_metrics,
        }.get(stage) is not None

    @property
    def stages_present(self) -> list[str]:
        return [s for s in STAGE_ORDER if self.has(s)]


def load_network_bundle(
    dirs: dict[str, Path | None],
    *,
    recording_key: str = "",
    rec_name: str = "",
    well_id: str = "",
) -> NetworkBundle:
    """Read every present stage for one well.

    ``dirs`` maps stage key (plus optional ``"auto_curation"``) to that well's
    output directory; a missing/``None`` entry simply leaves the stage empty.
    """
    b = NetworkBundle(
        recording_key=recording_key, rec_name=rec_name, well_id=well_id,
        dirs={k: Path(v) for k, v in dirs.items() if v is not None},
    )

    conn = b.dirs.get("connectivity")
    if conn is not None:
        b.graph_metrics = _read_json(conn / "graph_metrics.json")
        b.conn_diagnostics = _read_json(conn / "diagnostics.json")
        b.sttc_matrix = _read_npy(conn / "sttc_matrix.npy")
        sweep = _read_npz(conn / "sttc_sweep.npz")
        if sweep is not None:
            b.unit_ids = sweep.pop("unit_ids", None)
            b.sttc_sweep = sweep
        b.edges = _read_parquet(conn / "edges.parquet")
        # Optional: no committed producer (see module docstring).
        b.node_metrics = _read_parquet(conn / "node_metrics.parquet")
        b.node_scalars = _read_json(conn / "node_scalars.json")
        # Optional: wells computed before CCGs landed have no ccg.npz.
        ccg = _read_npz(conn / "ccg.npz")
        if ccg is not None and ccg.get("counts") is not None:
            b.ccg_counts = ccg.get("counts")
            b.ccg_baseline = ccg.get("baseline")
            b.ccg_lags_ms = ccg.get("lags_ms")
            b.ccg_pairs = ccg.get("pairs")
            # npz scalars come back as 0-d arrays.
            b.ccg_meta = {
                k: float(ccg[k].item())
                for k in ("bin_ms", "window_ms", "syn_lo_ms", "syn_hi_ms")
                if k in ccg
            }

    crit = b.dirs.get("criticality")
    if crit is not None:
        b.criticality_metrics = _read_json(crit / "criticality_metrics.json")
        b.powerlaw_fits = _read_json(crit / "powerlaw_fits.json")
        b.crit_diagnostics = _read_json(crit / "diagnostics.json")
        b.avalanches = _read_parquet(crit / "avalanches.parquet")

    dirn = b.dirs.get("directed")
    if dirn is not None:
        b.directed_metrics = _read_json(dirn / "directed_metrics.json")
        b.dir_diagnostics = _read_json(dirn / "diagnostics.json")
        b.te_matrix = _read_npy(dirn / "te_matrix.npy")
        b.te_unit_ids = _read_npy(dirn / "unit_ids.npy")
        b.directed_edges = _read_parquet(dirn / "directed_edges.parquet")

    spat = b.dirs.get("spatial_map")
    if spat is not None:
        b.spatial_metrics = _read_json(spat / "spatial_metrics.json")
        b.spatial_diagnostics = _read_json(spat / "diagnostics.json")
        b.activity_field = _read_npy(spat / "activity_field.npy")
        b.burst_propagation = _read_parquet(spat / "burst_propagation.parquet")

    b.positions = _read_positions(b.dirs.get("auto_curation"))
    return b


_BUNDLE_MEMO: dict[tuple, NetworkBundle] = {}
_MEMO_MAX = 24

_SIG_FILES = {
    "connectivity": ("graph_metrics.json", "sttc_matrix.npy", "node_metrics.parquet",
                     "ccg.npz"),
    "criticality": ("criticality_metrics.json", "avalanches.parquet"),
    "directed": ("directed_metrics.json", "te_matrix.npy"),
    "spatial_map": ("spatial_metrics.json", "activity_field.npy"),
    "auto_curation": ("quality_metrics.pkl",),
}


def _sig(dirs: dict[str, Path | None]) -> tuple:
    """Cheap staleness signature — ``stat`` only, never opens a file."""
    out = []
    for stage in sorted(_SIG_FILES):
        base = dirs.get(stage)
        for name in _SIG_FILES[stage]:
            if base is None:
                out.append((stage, name, -1, -1))
                continue
            try:
                st = (Path(base) / name).stat()
                out.append((stage, name, st.st_mtime_ns, st.st_size))
            except OSError:
                out.append((stage, name, -1, -1))
    return tuple(out)


def load_network_bundle_cached(
    dirs: dict[str, Path | None],
    *,
    recording_key: str = "",
    rec_name: str = "",
    well_id: str = "",
) -> NetworkBundle:
    """Memoized :func:`load_network_bundle`, keyed by source-file signature."""
    key = (recording_key, rec_name, well_id, _sig(dirs))
    hit = _BUNDLE_MEMO.get(key)
    if hit is not None:
        return hit
    bundle = load_network_bundle(
        dirs, recording_key=recording_key, rec_name=rec_name, well_id=well_id,
    )
    _BUNDLE_MEMO[key] = bundle
    while len(_BUNDLE_MEMO) > _MEMO_MAX:
        _BUNDLE_MEMO.pop(next(iter(_BUNDLE_MEMO)))
    return bundle


def clear_cache() -> None:
    """Drop memoized bundles (rescan / tests)."""
    _BUNDLE_MEMO.clear()
    _PLATE_MEMO.clear()


# --------------------------------------------------------------------------- #
# Plate roll-up
# --------------------------------------------------------------------------- #
_PLATE_MEMO: dict[tuple, dict[str, dict[str, float]]] = {}
_PLATE_SCALAR_FILES = {
    "connectivity": ("graph_metrics.json", "node_scalars.json"),
    "criticality": ("criticality_metrics.json",),
    "directed": ("directed_metrics.json",),
    "spatial_map": ("spatial_metrics.json",),
}


def plate_scalars(
    stage_dirs: dict[str, dict[str, Path]],
) -> dict[str, dict[str, float]]:
    """``{well_id: {metric: value}}`` for a whole recording.

    Reads only the small per-well JSONs — the arrays are never touched, so
    painting the 24-well grid costs a few dozen tiny reads rather than loading
    every well's STTC matrix. Memoized on the same stat-signature scheme as the
    bundles, so re-picking a metric is free and a pipeline re-run busts it.
    """
    sig_parts: list[tuple] = []
    for stage in sorted(_PLATE_SCALAR_FILES):
        for well_id, base in sorted((stage_dirs.get(stage) or {}).items()):
            for name in _PLATE_SCALAR_FILES[stage]:
                try:
                    st = (Path(base) / name).stat()
                    sig_parts.append((stage, well_id, name, st.st_mtime_ns, st.st_size))
                except OSError:
                    sig_parts.append((stage, well_id, name, -1, -1))
    key = tuple(sig_parts)
    hit = _PLATE_MEMO.get(key)
    if hit is not None:
        return hit

    out: dict[str, dict[str, float]] = {}
    for stage, names in _PLATE_SCALAR_FILES.items():
        for well_id, base in (stage_dirs.get(stage) or {}).items():
            slot = out.setdefault(well_id, {})
            for name in names:
                data = _read_json(Path(base) / name)
                if not data:
                    continue
                for k, v in data.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        slot[k] = float(v)

    _PLATE_MEMO[key] = out
    while len(_PLATE_MEMO) > 8:
        _PLATE_MEMO.pop(next(iter(_PLATE_MEMO)))
    return out


# --------------------------------------------------------------------------- #
# Plate-grid colouring
# --------------------------------------------------------------------------- #
def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _sample_scale(t: float, scale: list) -> str:
    """Linearly interpolate a Plotly-style ``[[pos, hex], ...]`` scale at ``t``."""
    t = min(1.0, max(0.0, float(t)))
    for (p0, c0), (p1, c1) in zip(scale, scale[1:]):
        if t <= p1:
            span = (p1 - p0) or 1.0
            f = (t - p0) / span
            a, b = _hex_to_rgb(c0), _hex_to_rgb(c1)
            return "#%02x%02x%02x" % tuple(
                int(round(a[i] + (b[i] - a[i]) * f)) for i in range(3)
            )
    return scale[-1][1]


def _ink_for(bg: str) -> str:
    """Pick primary or inverted ink so the value stays legible on ``bg``."""
    r, g, b = (c / 255 for c in _hex_to_rgb(bg))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return _INK if lum > 0.55 else "#fbf9f3"


def plate_colors(
    values: dict[str, float], metric: str
) -> tuple[dict[str, tuple[str, str]], float, float]:
    """Map ``{well: value}`` to ``{well: (background, ink)}`` plus the scale range.

    Magnitude metrics take the single-hue sequential ramp. Metrics with a
    meaningful centre (branching ratio about 1, DCC about 0) take the diverging
    ramp, symmetric about that centre so equal deviations either side read as
    equally far — never a rainbow, never a hue at the midpoint.
    """
    finite = {w: float(v) for w, v in values.items()
              if v is not None and np.isfinite(v)}
    if not finite:
        return {}, float("nan"), float("nan")

    vals = np.array(list(finite.values()), float)
    diverging = metric in DIVERGING_METRICS
    if diverging:
        center = METRIC_CENTER.get(metric, 0.0)
        half = float(np.max(np.abs(vals - center))) or 1.0
        vmin, vmax, scale = center - half, center + half, DIV_SCALE
    else:
        vmin, vmax = float(vals.min()), float(vals.max())
        scale = SEQ_SCALE
    span = (vmax - vmin) or 1.0

    out: dict[str, tuple[str, str]] = {}
    for well, v in finite.items():
        bg = _sample_scale((v - vmin) / span, scale)
        out[well] = (bg, _ink_for(bg))
    return out, vmin, vmax


# --------------------------------------------------------------------------- #
# Figure helpers
# --------------------------------------------------------------------------- #
def _empty_figure(message: str, height: int = 320) -> go.Figure:
    return go.Figure().update_layout(
        annotations=[{"text": message, "xref": "paper", "yref": "paper",
                      "x": 0.5, "y": 0.5, "showarrow": False,
                      "font": {"size": 13, "color": _INK3}}],
        xaxis={"visible": False}, yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 30, "b": 20}, height=height,
    )


_NO_POSITIONS = ("no unit positions for this well<br>"
                 "<span style='font-size:11px'>quality_metrics.pkl has no "
                 "loc_x / loc_y</span>")


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "—"
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


def metrics_kv(bundle: NetworkBundle, stage: str) -> dict[str, str]:
    """Flat ``{label: formatted value}`` for one tab's metrics card."""
    sources: list[dict | None]
    if stage == "connectivity":
        sources = [bundle.graph_metrics, bundle.conn_diagnostics]
    elif stage == "nodes":
        sources = [bundle.node_scalars]
    elif stage == "criticality":
        sources = [bundle.criticality_metrics, bundle.crit_diagnostics]
    elif stage == "directed":
        sources = [bundle.directed_metrics, bundle.dir_diagnostics]
    elif stage == "spatial_map":
        sources = [bundle.spatial_metrics, bundle.spatial_diagnostics]
    else:
        sources = []
    out: dict[str, str] = {}
    for src in sources:
        for k, v in (src or {}).items():
            out[METRIC_LABELS.get(k, k)] = _fmt(v)
    return out


def _ordered_index(bundle: NetworkBundle) -> np.ndarray | None:
    """Row order for the STTC matrix: by module, then descending degree.

    Falls back to degree alone (from ``edges``) when node metrics are absent,
    and to identity order when there is nothing to sort on — a block-diagonal
    read of the matrix is the whole point of the heatmap.
    """
    if bundle.sttc_matrix is None or bundle.sttc_matrix.size == 0:
        return None
    n = bundle.sttc_matrix.shape[0]
    uids = bundle.unit_ids
    if uids is None or len(uids) != n:
        return np.arange(n)

    nm = bundle.node_metrics
    if nm is not None and len(nm) and {"module", "degree"} <= set(nm.columns):
        module, degree = [], []
        for uid in uids:
            try:
                row = nm.loc[int(uid)]
            except (KeyError, TypeError, ValueError):
                module.append(10**6); degree.append(-1.0); continue
            module.append(int(row["module"])); degree.append(float(row["degree"]))
        return np.lexsort((-np.asarray(degree), np.asarray(module)))

    W = np.nan_to_num(bundle.sttc_matrix, nan=0.0)
    return np.argsort(-np.abs(W).sum(axis=1))


def _xy_for(uids: np.ndarray, positions: dict[int, tuple[float, float]]) -> np.ndarray:
    return np.array([positions.get(int(u), (np.nan, np.nan)) for u in uids], float)


# --------------------------------------------------------------------------- #
# Connectivity figures
# --------------------------------------------------------------------------- #
def fig_sttc_matrix(bundle: NetworkBundle, height: int = 380) -> go.Figure:
    """STTC weight matrix, rows/cols ordered by module then degree."""
    if bundle.sttc_matrix is None or bundle.sttc_matrix.size == 0:
        return _empty_figure("(no connectivity data)", height)
    order = _ordered_index(bundle)
    W = np.asarray(bundle.sttc_matrix, float)[np.ix_(order, order)]
    labels = ([str(int(u)) for u in np.asarray(bundle.unit_ids)[order]]
              if bundle.unit_ids is not None and len(bundle.unit_ids) == len(order)
              else [str(i) for i in order])
    fig = go.Figure(go.Heatmap(
        z=W, x=labels, y=labels, colorscale=SEQ_SCALE, zmin=0.0,
        colorbar={"title": {"text": "STTC", "font": {"size": 11}}, "thickness": 12},
        hovertemplate="unit %{y} · unit %{x}<br>STTC %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin={"l": 56, "r": 16, "t": 34, "b": 48},
        title={"text": "STTC matrix (ordered by module, then degree)",
               "font": {"size": 12}},
        xaxis={"title": {"text": "unit", "font": {"size": 11}},
               "showgrid": False, "constrain": "domain"},
        yaxis={"title": {"text": "unit", "font": {"size": 11}},
               "showgrid": False, "autorange": "reversed",
               "scaleanchor": "x", "constrain": "domain"},
    )
    return fig


def fig_sttc_graph(
    bundle: NetworkBundle, top_frac: float = 0.3, height: int = 380
) -> go.Figure:
    """Significant STTC edges drawn on the electrode plane."""
    if bundle.edges is None or bundle.unit_ids is None:
        return _empty_figure("(no connectivity data)", height)
    if not bundle.positions:
        return _empty_figure(_NO_POSITIONS, height)

    uids = np.asarray(bundle.unit_ids)
    xy = _xy_for(uids, bundle.positions)
    ok = np.isfinite(xy[:, 0])
    if not ok.any():
        return _empty_figure(_NO_POSITIONS, height)

    edges = bundle.edges
    traces: list[go.Scattergl] = []
    n_shown = 0
    if len(edges):
        e = edges.sort_values("sttc", ascending=False)
        keep = max(1, int(round(len(e) * float(top_frac))))
        e = e.head(keep)
        pos = bundle.positions
        # One trace for every edge, `None`-separated: N segments at the cost of
        # a single trace (drawing one trace per edge would stall the browser).
        ex: list[float | None] = []
        ey: list[float | None] = []
        for u, v in zip(e["u"].to_numpy(), e["v"].to_numpy()):
            pu, pv = pos.get(int(u)), pos.get(int(v))
            if pu is None or pv is None:
                continue
            ex += [pu[0], pv[0], None]
            ey += [pu[1], pv[1], None]
            n_shown += 1
        if ex:
            traces.append(go.Scattergl(
                x=ex, y=ey, mode="lines", hoverinfo="skip",
                line={"color": "rgba(42,120,214,0.28)", "width": 1},
                name=f"top {int(top_frac * 100)}% edges", showlegend=False,
            ))

    strength = np.zeros(len(uids), float)
    if len(edges):
        by_unit = (pd.concat([
            edges[["u", "sttc"]].rename(columns={"u": "unit"}),
            edges[["v", "sttc"]].rename(columns={"v": "unit"}),
        ]).groupby("unit")["sttc"].sum())
        strength = np.array([float(by_unit.get(int(u), 0.0)) for u in uids])

    traces.append(go.Scattergl(
        x=xy[ok, 0], y=xy[ok, 1], mode="markers",
        marker={"size": 9, "color": strength[ok], "colorscale": SEQ_SCALE,
                "line": {"color": _SURFACE, "width": 2},
                "colorbar": {"title": {"text": "strength", "font": {"size": 11}},
                             "thickness": 12}},
        text=[f"unit {int(u)}" for u in uids[ok]],
        hovertemplate="%{text}<br>strength %{marker.color:.3f}<extra></extra>",
        showlegend=False,
    ))

    fig = go.Figure(traces)
    fig.update_layout(
        height=height, margin={"l": 56, "r": 16, "t": 34, "b": 48},
        title={"text": f"STTC graph on the electrode plane ({n_shown} edges shown)",
               "font": {"size": 12}},
        xaxis={"title": {"text": "x (um)", "font": {"size": 11}}},
        yaxis={"title": {"text": "y (um)", "font": {"size": 11}},
               "scaleanchor": "x", "scaleratio": 1},
    )
    return fig


def fig_sttc_distance(bundle: NetworkBundle, height: int = 320) -> go.Figure:
    """Edge STTC against inter-unit distance, with a binned median overlay."""
    if bundle.edges is None or not len(bundle.edges):
        return _empty_figure("(no connectivity data)", height)
    e = bundle.edges
    if "dist_um" not in e.columns or not np.isfinite(e["dist_um"]).any():
        return _empty_figure(_NO_POSITIONS, height)

    d = e[np.isfinite(e["dist_um"]) & np.isfinite(e["sttc"])]
    if not len(d):
        return _empty_figure(_NO_POSITIONS, height)

    fig = go.Figure(go.Scattergl(
        x=d["dist_um"], y=d["sttc"], mode="markers",
        marker={"size": 4, "color": "rgba(42,120,214,0.35)"},
        hovertemplate="%{x:.0f} um<br>STTC %{y:.3f}<extra></extra>",
        showlegend=False,
    ))
    # Binned median: the trend a cloud of thousands of points hides.
    dist = d["dist_um"].to_numpy(float)
    if len(dist) > 20:
        edges_b = np.linspace(dist.min(), dist.max(), 13)
        idx = np.clip(np.digitize(dist, edges_b) - 1, 0, len(edges_b) - 2)
        cx, cy = [], []
        for b in range(len(edges_b) - 1):
            sel = d["sttc"].to_numpy(float)[idx == b]
            if len(sel) >= 3:
                cx.append((edges_b[b] + edges_b[b + 1]) / 2)
                cy.append(float(np.median(sel)))
        if cx:
            fig.add_trace(go.Scatter(
                x=cx, y=cy, mode="lines+markers", name="binned median",
                line={"color": _ACCENT, "width": 2},
                marker={"size": 8, "line": {"color": _SURFACE, "width": 2}},
                hovertemplate="%{x:.0f} um<br>median STTC %{y:.3f}<extra></extra>",
            ))
    fig.update_layout(
        height=height, margin={"l": 56, "r": 16, "t": 34, "b": 44},
        title={"text": "STTC vs inter-unit distance", "font": {"size": 12}},
        xaxis={"title": {"text": "distance (um)", "font": {"size": 11}}},
        yaxis={"title": {"text": "STTC", "font": {"size": 11}}},
        showlegend=True, legend={"x": 1, "y": 1, "xanchor": "right"},
    )
    return fig


def fig_dt_sweep(bundle: NetworkBundle, height: int = 320) -> go.Figure:
    """Mean off-diagonal STTC as a function of the tiling window dt."""
    if not bundle.sttc_sweep:
        return _empty_figure("(no dt sweep)", height)
    xs, ys = [], []
    for name, M in bundle.sttc_sweep.items():
        if not name.startswith("dt_"):
            continue
        try:
            dt_ms = float(name[3:].replace("ms", ""))
        except ValueError:
            continue
        M = np.asarray(M, float)
        if M.ndim != 2 or M.size == 0:
            continue
        off = ~np.eye(M.shape[0], dtype=bool)
        vals = M[off]
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            continue
        xs.append(dt_ms); ys.append(float(vals.mean()))
    if not xs:
        return _empty_figure("(no dt sweep)", height)
    order = np.argsort(xs)
    xs = np.asarray(xs)[order]; ys = np.asarray(ys)[order]

    primary = None
    if bundle.conn_diagnostics:
        p = bundle.conn_diagnostics.get("dt_primary")
        primary = float(p) * 1000 if p is not None else None

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="lines+markers", line={"color": _ACCENT, "width": 2},
        marker={"size": 9, "line": {"color": _SURFACE, "width": 2}},
        hovertemplate="dt %{x:.0f} ms<br>mean STTC %{y:.3f}<extra></extra>",
        showlegend=False,
    ))
    if primary is not None:
        fig.add_vline(x=primary, line={"color": _INK3, "width": 1, "dash": "dash"},
                      annotation={"text": f"primary dt {primary:.0f} ms",
                                  "font": {"size": 10, "color": _INK3}})
    fig.update_layout(
        height=height, margin={"l": 56, "r": 16, "t": 34, "b": 44},
        title={"text": "Mean STTC vs tiling window", "font": {"size": 12}},
        xaxis={"title": {"text": "dt (ms)", "font": {"size": 11}}},
        yaxis={"title": {"text": "mean off-diagonal STTC", "font": {"size": 11}}},
    )
    return fig


# --------------------------------------------------------------------------- #
# Correlogram figures
# --------------------------------------------------------------------------- #
_NO_CCG = ("no correlograms for this well<br>"
           "<span style='font-size:11px'>ccg.npz absent — re-run the connectivity "
           "stage or scripts/backfill_ccg.py</span>")


def _ccg_edge_stats(bundle: NetworkBundle) -> pd.DataFrame | None:
    """Per-row CCG stats aligned to ``bundle.ccg_counts`` **by (u, v)**.

    ``edges.parquet`` can be longer than ``ccg.npz`` (the ``ccg_max_pairs`` cap),
    so never zip the two by position — join on the pair key.
    """
    if bundle.ccg_counts is None or bundle.ccg_pairs is None:
        return None
    pairs = np.asarray(bundle.ccg_pairs)
    if pairs.ndim != 2 or pairs.shape[0] != bundle.ccg_counts.shape[0]:
        return None
    out = pd.DataFrame({"row": np.arange(pairs.shape[0]),
                        "u": pairs[:, 0], "v": pairs[:, 1]})
    e = bundle.edges
    cols = [c for c in ("ccg_peak_lag_ms", "ccg_peak_z", "ccg_sig", "ccg_syn_dir",
                        "sttc") if e is not None and c in e.columns]
    if cols:
        out = out.merge(e[["u", "v", *cols]], on=["u", "v"], how="left")
    return out


def _ccg_z(counts: np.ndarray, baseline: np.ndarray | None) -> np.ndarray:
    """Per-bin z of counts against the hollow-Gaussian baseline (floored)."""
    counts = np.asarray(counts, float)
    if baseline is None or np.shape(baseline) != np.shape(counts):
        return counts
    lam = np.maximum(np.asarray(baseline, float), 1e-6)
    return (counts - lam) / np.sqrt(lam)


def _ccg_interest(stats: pd.DataFrame) -> np.ndarray:
    """Row order for display: significant edges first, then by peak z."""
    z = (stats["ccg_peak_z"].to_numpy(dtype=float)
         if "ccg_peak_z" in stats.columns else np.zeros(len(stats)))
    z = np.nan_to_num(z, nan=-np.inf)
    if "ccg_sig" in stats.columns:
        sig = stats["ccg_sig"].fillna(False).to_numpy(dtype=bool)
    else:
        sig = np.zeros(len(stats), dtype=bool)
    return np.lexsort((-z, ~sig))


def fig_ccg_heatmap(
    bundle: NetworkBundle, max_rows: int = 400, height: int = 380
) -> go.Figure:
    """Every edge's correlogram as one row, z-scored, sorted by peak lag.

    Reference = ``u``, target = ``v``: a band right of zero means ``u`` leads.
    A dense well has thousands of edges, which would ship tens of MB of JSON to
    the browser — above ``max_rows`` the significant edges (then the strongest
    peaks) are kept and the title says how many were dropped.
    """
    stats = _ccg_edge_stats(bundle)
    if stats is None or not len(stats):
        return _empty_figure(_NO_CCG, height)

    n_total = len(stats)
    dropped = 0
    if n_total > max_rows:
        stats = stats.iloc[_ccg_interest(stats)[:max_rows]]
        dropped = n_total - len(stats)

    Z = _ccg_z(bundle.ccg_counts, bundle.ccg_baseline)
    lags = np.asarray(bundle.ccg_lags_ms, float)
    lag = stats["ccg_peak_lag_ms"].to_numpy(dtype=float) \
        if "ccg_peak_lag_ms" in stats.columns else np.zeros(len(stats))
    order = np.argsort(np.nan_to_num(lag, nan=np.inf), kind="mergesort")
    rows = stats["row"].to_numpy()[order]
    Z = Z[rows]
    labels = [f"{int(u)}→{int(v)}" for u, v in
              zip(stats["u"].to_numpy()[order], stats["v"].to_numpy()[order])]

    lim = float(np.nanpercentile(np.abs(Z), 99)) if Z.size else 1.0
    lim = lim if np.isfinite(lim) and lim > 0 else 1.0
    fig = go.Figure(go.Heatmap(
        z=Z, x=lags, y=labels, colorscale=DIV_SCALE, zmid=0.0,
        zmin=-lim, zmax=lim,
        colorbar={"title": {"text": "z", "font": {"size": 11}}, "thickness": 12},
        hovertemplate="%{y}<br>lag %{x:.0f} ms<br>z %{z:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0.0, line={"color": _INK3, "width": 1, "dash": "dot"})
    title = f"CCG z-map · {len(labels)} edges, sorted by peak lag"
    if dropped:
        title += f" ({dropped} weaker edges not shown of {n_total})"
    fig.update_layout(
        height=height, margin={"l": 76, "r": 16, "t": 34, "b": 44},
        title={"text": title, "font": {"size": 12}},
        xaxis={"title": {"text": "lag (ms) — positive = u leads v",
                         "font": {"size": 11}}, "showgrid": False},
        yaxis={"title": {"text": "edge u→v", "font": {"size": 11}},
               "showgrid": False,
               "showticklabels": len(labels) <= 40},
    )
    return fig


def fig_ccg_small_multiples(
    bundle: NetworkBundle, n: int = 12, height: int = 380
) -> go.Figure:
    """Top-``n`` correlograms — significant edges first — with the baseline drawn.

    The shaded band is the causal window that decides ``ccg_sig``/``ccg_syn_dir``;
    everything outside it (especially the bump at zero) is co-activation, not
    evidence of a directed interaction.
    """
    from plotly.subplots import make_subplots

    stats = _ccg_edge_stats(bundle)
    if stats is None or not len(stats):
        return _empty_figure(_NO_CCG, height)

    # A dense well's largest z values sit at ±70 ms — burst structure, not
    # coupling. Rank the causally-significant edges first so the panel opens on
    # what ccg_sig actually flagged.
    pick = stats.iloc[_ccg_interest(stats)[: max(1, int(n))]]

    counts = np.asarray(bundle.ccg_counts, float)
    base = bundle.ccg_baseline
    lags = np.asarray(bundle.ccg_lags_ms, float)
    n_show = len(pick)
    ncol = min(4, n_show)
    nrow = int(np.ceil(n_show / ncol))
    titles = []
    for _, r in pick.iterrows():
        lag = r.get("ccg_peak_lag_ms")
        tag = f" · {lag:+.0f} ms" if lag is not None and np.isfinite(lag) else ""
        titles.append(f"{int(r['u'])}→{int(r['v'])}{tag}")
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=titles,
                        horizontal_spacing=0.05, vertical_spacing=0.12)

    # Shade the window that actually decided ccg_sig for *this* well.
    meta = bundle.ccg_meta or {}
    syn_lo = float(meta.get("syn_lo_ms") or 0.8)
    syn_hi = float(meta.get("syn_hi_ms") or 8.0)
    for k, (_, r) in enumerate(pick.iterrows()):
        i = int(r["row"])
        row, col = k // ncol + 1, k % ncol + 1
        fig.add_trace(go.Bar(
            x=lags, y=counts[i], marker={"color": "rgba(42,120,214,0.55)"},
            hovertemplate="lag %{x:.0f} ms<br>%{y} spikes<extra></extra>",
            showlegend=False,
        ), row=row, col=col)
        if base is not None and np.shape(base) == np.shape(counts):
            fig.add_trace(go.Scatter(
                x=lags, y=np.asarray(base, float)[i], mode="lines",
                line={"color": _INK, "width": 1.5}, hoverinfo="skip",
                showlegend=False,
            ), row=row, col=col)
        for lo, hi in ((syn_lo, syn_hi), (-syn_hi, -syn_lo)):
            fig.add_vrect(x0=lo, x1=hi, line_width=0, fillcolor=_ACCENT,
                          opacity=0.08, row=row, col=col)

    fig.update_annotations(font={"size": 10})
    fig.update_xaxes(title=None, showgrid=False, tickfont={"size": 9})
    fig.update_yaxes(title=None, showgrid=False, tickfont={"size": 9})
    fig.update_layout(
        height=max(height, 150 * nrow), margin={"l": 44, "r": 16, "t": 46, "b": 40},
        bargap=0.0,
        title={"text": f"Top {n_show} correlograms — significant first "
                       f"(counts vs baseline)", "font": {"size": 12}},
    )
    return fig


# --------------------------------------------------------------------------- #
# Node figures
# --------------------------------------------------------------------------- #
_NO_NODES = ("no node metrics for this well<br>"
             "<span style='font-size:11px'>node_metrics.parquet absent — it has "
             "no committed producer</span>")


def fig_cartography(bundle: NetworkBundle, height: int = 380) -> go.Figure:
    """Guimera-Amaral cartography: participation P vs within-module degree z."""
    nm = bundle.node_metrics
    if nm is None or not len(nm) or not {"participation", "z_within"} <= set(nm.columns):
        return _empty_figure(_NO_NODES, height)

    fig = go.Figure()
    for role in ROLE_ORDER:
        d = nm[nm["role"] == role] if "role" in nm.columns else nm.iloc[0:0]
        if not len(d):
            continue
        fig.add_trace(go.Scattergl(
            x=d["participation"], y=d["z_within"], mode="markers",
            name=role.replace("_", " "),
            marker={"size": 9, "color": ROLE_COLORS[role],
                    "line": {"color": _SURFACE, "width": 2}},
            text=[f"unit {int(u)}" for u in d.index],
            customdata=(d["degree"].to_numpy() if "degree" in d.columns
                        else np.zeros(len(d))),
            hovertemplate=("%{text}<br>P %{x:.3f}<br>z %{y:.2f}"
                           "<br>degree %{customdata:.0f}"
                           "<extra>" + role.replace("_", " ") + "</extra>"),
        ))
    # Role is defined by these two cuts, so the guides are the encoding, not decor.
    fig.add_hline(y=Z_HUB, line={"color": _INK3, "width": 1, "dash": "dash"})
    fig.add_vline(x=P_CONNECTOR, line={"color": _INK3, "width": 1, "dash": "dot"})
    fig.update_layout(
        height=height, margin={"l": 56, "r": 16, "t": 34, "b": 44},
        title={"text": "Node cartography (role = position in this plane)",
               "font": {"size": 12}},
        xaxis={"title": {"text": "participation coefficient P", "font": {"size": 11}}},
        yaxis={"title": {"text": "within-module degree z", "font": {"size": 11}}},
        legend={"x": 0.01, "y": 0.99, "font": {"size": 10}},
    )
    return fig


def fig_node_degree(bundle: NetworkBundle, height: int = 320) -> go.Figure:
    """Degree distribution — how far from a random graph the well sits."""
    nm = bundle.node_metrics
    if nm is None or not len(nm) or "degree" not in nm.columns:
        return _empty_figure(_NO_NODES, height)
    deg = nm["degree"].to_numpy(float)
    fig = go.Figure(go.Histogram(
        x=deg, marker={"color": _ACCENT, "line": {"color": _SURFACE, "width": 2}},
        nbinsx=min(30, max(6, int(np.sqrt(len(deg))) * 2)),
        hovertemplate="degree %{x}<br>%{y} units<extra></extra>",
    ))
    fig.add_vline(x=float(np.mean(deg)),
                  line={"color": _INK3, "width": 1, "dash": "dash"},
                  annotation={"text": f"mean {np.mean(deg):.1f}",
                              "font": {"size": 10, "color": _INK3}})
    fig.update_layout(
        height=height, margin={"l": 56, "r": 16, "t": 34, "b": 44},
        bargap=0.06,
        title={"text": "Degree distribution", "font": {"size": 12}},
        xaxis={"title": {"text": "degree", "font": {"size": 11}}},
        yaxis={"title": {"text": "units", "font": {"size": 11}}},
    )
    return fig


def node_table(bundle: NetworkBundle) -> pd.DataFrame:
    """Per-unit node metrics as a tidy frame (the Nodes tab's table view).

    This is also the relief the palette's contrast WARN requires: every role is
    readable as text here, not only as a colour in the cartography scatter.
    """
    nm = bundle.node_metrics
    if nm is None or not len(nm):
        return pd.DataFrame(columns=["unit_id", "role", "degree", "strength",
                                     "betweenness", "participation", "z_within",
                                     "module"])
    out = nm.reset_index().rename(columns={"index": "unit_id"})
    cols = [c for c in ("unit_id", "role", "degree", "strength", "betweenness",
                        "clustering", "local_efficiency", "participation",
                        "z_within", "module") if c in out.columns]
    return out[cols].sort_values("degree", ascending=False)


# --------------------------------------------------------------------------- #
# Criticality figures
# --------------------------------------------------------------------------- #
def _loglog_pdf(x: np.ndarray, n_bins: int = 25):
    """Log-spaced histogram of ``x`` (the F8 estimator, ported)."""
    x = np.asarray(x, float)
    x = x[x > 0]
    if len(x) < 10:
        return None, None
    lo, hi = x.min(), x.max()
    if not (hi > lo):
        return None, None
    bins = np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), n_bins)).astype(int))
    bins = bins[bins > 0]
    if len(bins) < 3:
        return None, None
    h, edges = np.histogram(x, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    m = h > 0
    return centers[m], h[m]


def fig_avalanche_dist(bundle: NetworkBundle, height: int = 340) -> go.Figure:
    """Avalanche size and duration distributions with their MLE power-law fits."""
    av = bundle.avalanches
    if av is None or not len(av):
        return _empty_figure("(no criticality data)", height)

    metrics = bundle.criticality_metrics or {}
    fig = go.Figure()
    specs = [("size", "size", "aval_tau", _ACCENT, "tau"),
             ("duration", "duration", "aval_alpha", "#eb6834", "alpha")]
    for col, label, exp_key, color, sym in specs:
        if col not in av.columns:
            continue
        cx, cy = _loglog_pdf(av[col].to_numpy())
        if cx is None:
            continue
        exp = metrics.get(exp_key)
        name = f"{label}" + (f" ({sym}={exp:.2f})" if isinstance(exp, (int, float))
                             and np.isfinite(exp) else "")
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="markers", name=name,
            marker={"size": 8, "color": color,
                    "line": {"color": _SURFACE, "width": 2}},
            hovertemplate=f"{label} %{{x:.3g}}<br>PDF %{{y:.3g}}<extra></extra>",
        ))
        if isinstance(exp, (int, float)) and np.isfinite(exp) and exp > 0:
            ys = cx.astype(float) ** (-float(exp))
            ys = ys / ys[0] * cy[0]
            fig.add_trace(go.Scatter(
                x=cx, y=ys, mode="lines", showlegend=False, hoverinfo="skip",
                line={"color": color, "width": 2, "dash": "dash"},
            ))
    if not fig.data:
        return _empty_figure("(too few avalanches to bin)", height)
    fig.update_layout(
        height=height, margin={"l": 56, "r": 16, "t": 34, "b": 44},
        title={"text": "Avalanche distributions (MLE power-law overlay)",
               "font": {"size": 12}},
        xaxis={"type": "log", "title": {"text": "avalanche size / duration (bins)",
                                        "font": {"size": 11}}},
        yaxis={"type": "log", "title": {"text": "PDF", "font": {"size": 11}}},
        legend={"x": 0.01, "y": 0.01, "font": {"size": 10}},
    )
    return fig


def fig_crackling(bundle: NetworkBundle, height: int = 340) -> go.Figure:
    """Crackling-noise scaling: mean size vs duration, fitted vs predicted gamma.

    The gap between the two exponents *is* the DCC (Ma et al. 2019), so both
    lines are drawn on the same axes.
    """
    av = bundle.avalanches
    if av is None or not len(av) or not {"size", "duration"} <= set(av.columns):
        return _empty_figure("(no criticality data)", height)
    g = av.groupby("duration")["size"].mean()
    g = g[g.index > 0]
    if not len(g):
        return _empty_figure("(no criticality data)", height)

    m = bundle.criticality_metrics or {}
    x = g.index.to_numpy(float); y = g.to_numpy(float)
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers", name="observed",
        marker={"size": 8, "color": _ACCENT, "line": {"color": _SURFACE, "width": 2}},
        hovertemplate="duration %{x:.0f}<br>mean size %{y:.3g}<extra></extra>",
    ))
    for key, color, label in (("gamma_fit", "#eb6834", "gamma fit"),
                              ("gamma_pred", "#1baf7a", "gamma predicted")):
        gam = m.get(key)
        if not isinstance(gam, (int, float)) or not np.isfinite(gam):
            continue
        ys = x ** float(gam)
        ys = ys / ys[0] * y[0]
        fig.add_trace(go.Scatter(
            x=x, y=ys, mode="lines", name=f"{label} = {gam:.2f}",
            line={"color": color, "width": 2, "dash": "dash"}, hoverinfo="skip",
        ))
    dcc = m.get("dcc")
    title = "Crackling-noise scaling"
    if isinstance(dcc, (int, float)) and np.isfinite(dcc):
        title += f"  ·  DCC = {dcc:.3f}"
    fig.update_layout(
        height=height, margin={"l": 56, "r": 16, "t": 34, "b": 44},
        title={"text": title, "font": {"size": 12}},
        xaxis={"type": "log", "title": {"text": "duration (bins)", "font": {"size": 11}}},
        yaxis={"type": "log", "title": {"text": "mean size", "font": {"size": 11}}},
        legend={"x": 0.01, "y": 0.99, "font": {"size": 10}},
    )
    return fig


# --------------------------------------------------------------------------- #
# Directed (transfer-entropy) figures
# --------------------------------------------------------------------------- #
def fig_te_matrix(bundle: NetworkBundle, height: int = 380) -> go.Figure:
    """Delayed-transfer-entropy matrix (row = source, column = target)."""
    TE = bundle.te_matrix
    if TE is None or TE.size == 0:
        return _empty_figure("(no directed data)", height)
    TE = np.asarray(TE, float)
    uids = bundle.te_unit_ids
    labels = ([str(int(u)) for u in uids] if uids is not None and len(uids) == TE.shape[0]
              else [str(i) for i in range(TE.shape[0])])
    fig = go.Figure(go.Heatmap(
        z=TE, x=labels, y=labels, colorscale=SEQ_SCALE, zmin=0.0,
        colorbar={"title": {"text": "TE", "font": {"size": 11}}, "thickness": 12},
        hovertemplate="source %{y} -> target %{x}<br>TE %{z:.5f}<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin={"l": 56, "r": 16, "t": 34, "b": 48},
        title={"text": "Transfer entropy (row = source, column = target)",
               "font": {"size": 12}},
        xaxis={"title": {"text": "target unit", "font": {"size": 11}},
               "showgrid": False, "constrain": "domain"},
        yaxis={"title": {"text": "source unit", "font": {"size": 11}},
               "showgrid": False, "autorange": "reversed",
               "scaleanchor": "x", "constrain": "domain"},
    )
    return fig


def fig_te_graph(
    bundle: NetworkBundle, top_n: int = 150, height: int = 380
) -> go.Figure:
    """Strongest directed TE edges as arrows on the electrode plane.

    Capped at ``top_n`` arrows: binary spike-train TE is inflated by network-burst
    co-activation, so the significant-edge set is dense and drawing all of it is
    both unreadable and slow. The cap is stated in the title rather than applied
    silently.
    """
    TE = bundle.te_matrix
    uids = bundle.te_unit_ids
    if TE is None or TE.size == 0 or uids is None:
        return _empty_figure("(no directed data)", height)
    if not bundle.positions:
        return _empty_figure(_NO_POSITIONS, height)

    uids = np.asarray(uids)
    xy = _xy_for(uids, bundle.positions)
    ok = np.isfinite(xy[:, 0])
    if not ok.any():
        return _empty_figure(_NO_POSITIONS, height)

    TE = np.asarray(TE, float)
    np.fill_diagonal(TE, 0.0)
    n = TE.shape[0]
    flat = np.argsort(TE, axis=None)[::-1]
    arrows, n_pos = [], 0
    for idx in flat:
        i, j = divmod(int(idx), n)
        if TE[i, j] <= 0:
            break
        n_pos += 1
        if len(arrows) >= top_n or not (ok[i] and ok[j]):
            continue
        arrows.append({
            "x": xy[j, 0], "y": xy[j, 1], "ax": xy[i, 0], "ay": xy[i, 1],
            "xref": "x", "yref": "y", "axref": "x", "ayref": "y",
            "showarrow": True, "arrowhead": 2, "arrowsize": 0.9,
            "arrowwidth": 1, "arrowcolor": "rgba(235,104,52,0.42)",
        })

    out_strength = TE.sum(axis=1)
    fig = go.Figure(go.Scattergl(
        x=xy[ok, 0], y=xy[ok, 1], mode="markers",
        marker={"size": 9, "color": out_strength[ok], "colorscale": SEQ_SCALE,
                "line": {"color": _SURFACE, "width": 2},
                "colorbar": {"title": {"text": "out TE", "font": {"size": 11}},
                             "thickness": 12}},
        text=[f"unit {int(u)}" for u in uids[ok]],
        hovertemplate="%{text}<br>outgoing TE %{marker.color:.4f}<extra></extra>",
        showlegend=False,
    ))
    shown = min(top_n, len(arrows))
    fig.update_layout(
        annotations=arrows, height=height,
        margin={"l": 56, "r": 16, "t": 34, "b": 48},
        title={"text": f"Directed TE graph ({shown} strongest of {n_pos} edges)",
               "font": {"size": 12}},
        xaxis={"title": {"text": "x (um)", "font": {"size": 11}}},
        yaxis={"title": {"text": "y (um)", "font": {"size": 11}},
               "scaleanchor": "x", "scaleratio": 1},
    )
    return fig


# --------------------------------------------------------------------------- #
# Spatial figures
# --------------------------------------------------------------------------- #
def fig_activity_field(bundle: NetworkBundle, height: int = 380) -> go.Figure:
    """Smoothed firing-rate field over the electrode plane."""
    F = bundle.activity_field
    if F is None or F.size == 0:
        return _empty_figure("(no spatial map data)", height)
    F = np.asarray(F, float)
    diag = bundle.spatial_diagnostics or {}
    extent = diag.get("field_extent")
    # `activity_field` is field[row=y, col=x] with row 0 at ymin, so a positive
    # dy from y0=ymin maps rows to microns without a flip.
    kwargs: dict[str, Any] = {}
    dx = dy = 1.0
    x0 = y0 = 0.0
    scaled = False
    if isinstance(extent, (list, tuple)) and len(extent) == 4:
        x0, x1, y0, y1 = (float(v) for v in extent)
        dx = (x1 - x0) / max(F.shape[1], 1)
        dy = (y1 - y0) / max(F.shape[0], 1)
        kwargs = {"x0": x0, "dx": dx, "y0": y0, "dy": dy}
        scaled = True

    unit = "um" if scaled else "bin"
    fig = go.Figure(go.Heatmap(
        z=F, colorscale=SEQ_SCALE,
        colorbar={"title": {"text": "activity", "font": {"size": 11}},
                  "thickness": 12, "x": 1.02},
        hovertemplate=f"x %{{x:.0f}} {unit} · y %{{y:.0f}} {unit}"
                      "<br>%{z:.4f}<extra></extra>",
        **kwargs,
    ))
    # `activity_centroid` is stored in BIN coordinates (spatial_map.spatial_summary),
    # so it has to be mapped through the same transform as the heatmap — plotting
    # it raw against a micron axis would park the marker in the corner.
    centroid = (bundle.spatial_metrics or {}).get("activity_centroid")
    has_centroid = (isinstance(centroid, (list, tuple)) and len(centroid) == 2
                    and all(np.isfinite(float(c)) for c in centroid))
    if has_centroid:
        cx = x0 + (float(centroid[0]) + 0.5) * dx
        cy = y0 + (float(centroid[1]) + 0.5) * dy
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker={"size": 13, "symbol": "x-thin", "color": "#eb6834",
                    "line": {"color": "#eb6834", "width": 3}},
            name="centroid",
            hovertemplate=f"activity centroid<br>x %{{x:.0f}} {unit} · "
                          f"y %{{y:.0f}} {unit}<extra></extra>",
        ))
    fig.update_layout(
        height=height, margin={"l": 56, "r": 70, "t": 34, "b": 48},
        title={"text": "Activity field", "font": {"size": 12}},
        xaxis={"title": {"text": f"x ({unit})", "font": {"size": 11}},
               "showgrid": False},
        yaxis={"title": {"text": f"y ({unit})", "font": {"size": 11}},
               "showgrid": False, "scaleanchor": "x", "scaleratio": 1},
        showlegend=has_centroid,
        legend={"x": 0.01, "y": 0.99, "font": {"size": 10}},
    )
    return fig


def fig_burst_propagation(bundle: NetworkBundle, height: int = 340) -> go.Figure:
    """Per-burst propagation speed against planar-fit quality.

    Speed only means "travelling wave" where the planar fit actually holds, so
    the two are plotted against each other rather than reporting a mean speed
    that a poor fit would make meaningless.
    """
    bp = bundle.burst_propagation
    if bp is None or not len(bp):
        return _empty_figure("(no burst propagation data)", height)
    if not {"speed_um_per_ms", "r2_planar_fit"} <= set(bp.columns):
        return _empty_figure("(no burst propagation data)", height)

    d = bp[np.isfinite(bp["speed_um_per_ms"]) & np.isfinite(bp["r2_planar_fit"])]
    if not len(d):
        return _empty_figure("(no burst propagation data)", height)

    size = d["n_participating"].to_numpy(float) if "n_participating" in d.columns \
        else np.full(len(d), 12.0)
    smax = float(np.nanmax(size)) or 1.0
    fig = go.Figure(go.Scatter(
        x=d["r2_planar_fit"], y=d["speed_um_per_ms"], mode="markers",
        marker={"size": 8 + 14 * size / smax, "color": _ACCENT, "opacity": 0.65,
                "line": {"color": _SURFACE, "width": 2}},
        text=[f"burst {int(i)}" for i in
              (d["burst_index"] if "burst_index" in d.columns else range(len(d)))],
        customdata=size,
        hovertemplate=("%{text}<br>planar r2 %{x:.2f}<br>speed %{y:.0f} um/ms"
                       "<br>%{customdata:.0f} units<extra></extra>"),
        showlegend=False,
    ))
    fig.update_layout(
        height=height, margin={"l": 56, "r": 16, "t": 34, "b": 44},
        title={"text": "Burst propagation — speed vs planar-fit quality "
                       "(marker size = units)", "font": {"size": 12}},
        xaxis={"title": {"text": "planar fit r^2", "font": {"size": 11}}},
        yaxis={"title": {"text": "speed (um/ms)", "font": {"size": 11}}},
    )
    return fig
