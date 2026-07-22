"""Panel registry — the report's figures, decomposed into placeable units.

The batch figures (``fig4_activity_dev.build_f4`` and friends) fuse *content*
with *layout*: each hard-codes a ``figsize``, a ``GridSpec``, which panels exist
and which treatment arms appear. That is fine for a fixed report and useless for
composing one interactively.

This module splits the two. A **panel** is a self-contained unit of content that
knows how to draw itself into a rectangle it is handed; **layout** becomes data
(see :mod:`scripts.report.composer.spec`). ``make_report`` keeps working
unchanged — the ``build_fX`` functions are untouched and simply express one
particular arrangement of the same panels.

Panel signature
---------------
``fn(fig, subspec, ctx) -> None`` — draw into ``subspec``, a
:class:`matplotlib.gridspec.SubplotSpec` pinned to the panel's grid cell. The
full form exists because some panels are composites: F6's posterior panel is a
heatmap plus a trace plus a colourbar sharing an x-axis, and it needs
``subspec.subgridspec(...)``. Panels that only want one axes use the
:func:`simple_panel` decorator and take ``(ax, ctx)``.

The context
-----------
:class:`RenderContext` carries the data roots, the selected treatment groups and
— importantly — a memo table. Ten forest panels in one composition must not each
recompute a 5000-sample bootstrap, and F6c's four panels share one UMAP fit that
takes minutes. Derived tables are computed once per context and shared.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from . import load as L
from . import stats as S
from .report_style import METRIC_LABELS, group_color, ordered_groups

# --------------------------------------------------------------------------- #
# Parameter descriptors — JSON-serialisable so the UI can build its own widgets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Param:
    kind: str
    default: Any
    label: str = ""
    choices: tuple = ()
    min: float | None = None
    max: float | None = None

    def to_json(self) -> dict:
        d = {"kind": self.kind, "default": self.default, "label": self.label}
        if self.choices:
            d["choices"] = list(self.choices)
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        return d


def Choice(default, choices, label=""):
    return Param("choice", default, label, tuple(choices))


def MultiChoice(default, choices, label=""):
    return Param("multichoice", list(default), label, tuple(choices))


def Int(default, label="", min=None, max=None):
    return Param("int", default, label, (), min, max)


def Float(default, label="", min=None, max=None):
    return Param("float", default, label, (), min, max)


def Bool(default, label=""):
    return Param("bool", default, label)


def Text(default, label=""):
    return Param("text", default, label)


# --------------------------------------------------------------------------- #
# Metric families — drives the UI's per-group data-coverage annotation
# --------------------------------------------------------------------------- #
METRIC_FAMILIES: dict[str, list[str]] = {
    "burst": ["nb_rate", "nb_count", "nb_duration_mean",
              "nb_spikes_per_burst_mean", "nb_ibi_mean"],
    "activity": ["median_firing_rate", "n_curated"],
    "ml": ["burst_modulation_index", "burst_type_k", "cluster_n_clusters"],
    "spatial": ["activity_gini", "mean_prop_speed_um_ms"],
    "connectivity": ["mean_sttc", "edge_density", "modularity", "small_worldness",
                     "clustering_coeff", "global_efficiency"],
    "node": ["hub_fraction", "leaf_fraction", "mean_betweenness",
             "participation_mean", "degree_cv", "rich_club", "assortativity"],
    "criticality": ["branching_ratio_mr", "branching_ratio_naive", "dcc",
                    "aval_tau", "aval_alpha", "gamma_fit"],
    "directed": ["mean_te", "te_edge_density", "degree_asymmetry",
                 "reciprocity", "flow_hierarchy"],
}

_FAMILY_OF = {m: fam for fam, ms in METRIC_FAMILIES.items() for m in ms}


def family_of(metric: str) -> str:
    return _FAMILY_OF.get(metric, "other")


def coverage(tidy: pd.DataFrame) -> dict[str, dict[str, int]]:
    """``{canonical_group: {family: n_wells_with_any_value}}``.

    The node / criticality / directed metrics were only ever computed for the
    focus arms, so enabling ``NPH`` would silently produce empty forest rows.
    The UI uses this to say so up front instead.
    """
    out: dict[str, dict[str, int]] = {}
    for g, sub in tidy.groupby("canonical_group"):
        per_fam = {}
        for fam, metrics in METRIC_FAMILIES.items():
            cols = [m for m in metrics if m in sub.columns]
            per_fam[fam] = (
                int(sub[cols].notna().any(axis=1).groupby(sub.well_uid).any().sum())
                if cols else 0
            )
        per_fam["_wells"] = int(sub.well_uid.nunique())
        out[str(g)] = per_fam
    return out


# --------------------------------------------------------------------------- #
# Panel spec + registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PanelSpec:
    id: str
    title: str
    section: str                       # UI palette grouping ("Connectivity", …)
    fn: Callable
    default_w: int = 6                 # grid columns (of 12)
    default_h: int = 4                 # grid row-units
    params: dict[str, Param] = field(default_factory=dict)
    families: tuple[str, ...] = ()     # metric families this panel needs
    description: str = ""

    def to_json(self) -> dict:
        return {
            "id": self.id, "title": self.title, "section": self.section,
            "default_w": self.default_w, "default_h": self.default_h,
            "params": {k: v.to_json() for k, v in self.params.items()},
            "families": list(self.families),
            "description": self.description,
        }

    def resolve_params(self, overrides: dict | None = None) -> dict:
        p = {k: v.default for k, v in self.params.items()}
        for k, v in (overrides or {}).items():
            if k in p:
                p[k] = v
        return p


PANELS: dict[str, PanelSpec] = {}


def register(spec: PanelSpec) -> PanelSpec:
    if spec.id in PANELS:
        raise ValueError(f"duplicate panel id {spec.id!r}")
    PANELS[spec.id] = spec
    return spec


def simple_panel(fn: Callable) -> Callable:
    """Adapt ``fn(ax, ctx, **params)`` to the ``fn(fig, subspec, ctx, **params)`` form."""
    def wrapper(fig, subspec, ctx, **params):
        ax = fig.add_subplot(subspec)
        return fn(ax, ctx, **params)
    wrapper.__name__ = getattr(fn, "__name__", "panel")
    wrapper.__doc__ = fn.__doc__
    return wrapper


class PanelDataMissing(RuntimeError):
    """Raised by a panel when its inputs are absent — drawn as a note, not a crash."""


# --------------------------------------------------------------------------- #
# Render context
# --------------------------------------------------------------------------- #
class RenderContext:
    """Data roots, selected groups, and a memo table shared across panels."""

    def __init__(self, tidy: pd.DataFrame, analysis_root: Path, figure_root: Path,
                 groups: Iterable[str] | None = None, seed: int = 0,
                 extras: dict | None = None):
        self.analysis_root = Path(analysis_root)
        self.figure_root = Path(figure_root)
        self.seed = int(seed)
        self.extras = dict(extras or {})
        all_groups = ordered_groups(tidy.canonical_group.dropna().unique())
        self.all_groups = all_groups
        self.groups = ([g for g in ordered_groups(groups) if g in all_groups]
                       if groups is not None else list(all_groups))
        if "Control" not in self.groups:
            # every response is a difference-in-differences against Control, so
            # it is always kept in the data even when hidden from a plot
            self.groups = ["Control"] + self.groups
        self._tidy_all = tidy
        self.tidy = tidy[tidy.canonical_group.isin(self.groups)].copy()
        self._memo: dict[tuple, Any] = {}
        self._coverage = None

    # -- identity ---------------------------------------------------------- #
    @property
    def treated_groups(self) -> list[str]:
        return [g for g in self.groups if g != "Control"]

    def fingerprint(self) -> str:
        """Stable hash of everything a panel render depends on, minus its params."""
        h = hashlib.sha256()
        h.update(str(self.analysis_root).encode())
        h.update(json.dumps(sorted(self.groups)).encode())
        h.update(str(self.seed).encode())
        h.update(str(len(self._tidy_all)).encode())
        cols = ",".join(sorted(self._tidy_all.columns))
        h.update(cols.encode())
        return h.hexdigest()[:16]

    def coverage(self) -> dict:
        if self._coverage is None:
            self._coverage = coverage(self._tidy_all)
        return self._coverage

    def has_data(self, families: Iterable[str]) -> list[str]:
        """Selected groups with **no** wells in any of ``families``."""
        cov = self.coverage()
        missing = []
        for g in self.groups:
            per = cov.get(g, {})
            if families and all(per.get(f, 0) == 0 for f in families):
                missing.append(g)
        return missing

    # -- memoised derived tables ------------------------------------------- #
    def _memoize(self, key: tuple, build: Callable):
        if key not in self._memo:
            self._memo[key] = build()
        return self._memo[key]

    def response(self, metrics: list[str]) -> pd.DataFrame:
        key = ("response", tuple(sorted(metrics)))

        def build():
            r = S.well_response(self.tidy, metrics=list(metrics))
            return r[r.arm.isin(self.groups)] if len(r) else r
        return self._memoize(key, build)

    def summary(self, metrics: list[str]) -> pd.DataFrame:
        key = ("summary", tuple(sorted(metrics)))
        return self._memoize(
            key, lambda: (S.arm_response_summary(self.response(metrics), seed=self.seed)
                          if len(self.response(metrics)) else pd.DataFrame()))

    def vs_control(self, metrics: list[str]) -> pd.DataFrame:
        key = ("vs_control", tuple(sorted(metrics)))
        return self._memoize(
            key, lambda: (S.arm_response_vs_control(self.response(metrics))
                          if len(self.response(metrics)) else pd.DataFrame()))

    def matched(self, lo: int, hi: int, metrics: list[str] | None = None) -> pd.DataFrame:
        key = ("matched", lo, hi, tuple(sorted(metrics)) if metrics else None)
        return self._memoize(
            key, lambda: S.matched_div_wells(self.tidy, lo, hi, metrics=metrics))

    def units(self, limit: int = 40) -> pd.DataFrame:
        key = ("units", limit)

        def build():
            sample = (self.tidy.sort_values("n_curated", ascending=False)
                      .drop_duplicates("well_uid").head(limit))
            return L.load_units(self.analysis_root, sample)
        return self._memoize(key, build)

    def ml_trace(self, well_uid: str):
        key = ("ml_trace", well_uid)

        def build():
            row = self.tidy[self.tidy.well_uid == well_uid].iloc[0]
            return L.load_ml_trace(self.analysis_root, row)
        return self._memoize(key, build)

    def dataset(self, name: str, build: Callable, **params):
        """Memoise an expensive shared dataset (F6c's UMAP fit, F6b's pooling)."""
        key = ("dataset", name, tuple(sorted(params.items())))
        return self._memoize(key, lambda: build(self, **params))

    # -- representative wells ---------------------------------------------- #
    def example(self, metric: str = "mean_sttc", group: str | None = None,
                div: tuple[int, int] = (14, 24)) -> pd.Series | None:
        """The well-recording closest to the median of ``metric`` in a DIV window."""
        key = ("example", metric, group, div)

        def build():
            sub = self.tidy
            if metric not in sub.columns:
                return None
            sub = sub[(sub.DIV >= div[0]) & (sub.DIV <= div[1]) & sub[metric].notna()]
            if group:
                sub = sub[sub.canonical_group == group]
            if sub.empty:
                return None
            med = sub[metric].median()
            order = (sub[metric] - med).abs().to_numpy().argsort()
            return sub.iloc[order[0]]
        return self._memoize(key, build)


# --------------------------------------------------------------------------- #
# Rendering a panel into a cell
# --------------------------------------------------------------------------- #
def _note(fig, subspec, text: str):
    ax = fig.add_subplot(subspec)
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            fontsize=7, color="0.45", wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def draw_panel(fig, subspec, panel_id: str, ctx: RenderContext,
               params: dict | None = None) -> dict:
    """Draw one registered panel. Never raises — failures become an in-cell note.

    Returns a small record (``{"panel", "status", "warning"}``) that the export
    manifest collects, so a silently-empty panel is visible after the fact.
    """
    spec = PANELS.get(panel_id)
    if spec is None:
        _note(fig, subspec, f"unknown panel\n{panel_id}")
        return {"panel": panel_id, "status": "unknown"}

    missing = ctx.has_data(spec.families)
    resolved = spec.resolve_params(params)
    try:
        spec.fn(fig, subspec, ctx, **resolved)
    except PanelDataMissing as exc:
        _note(fig, subspec, str(exc))
        return {"panel": panel_id, "status": "no-data", "warning": str(exc)}
    except Exception as exc:  # a broken panel must not kill the whole figure
        _note(fig, subspec, f"{spec.title}\nrender failed: {exc}")
        return {"panel": panel_id, "status": "error", "warning": repr(exc)}

    rec: dict[str, Any] = {"panel": panel_id, "status": "ok"}
    if missing:
        rec["warning"] = f"no {'/'.join(spec.families)} data for {', '.join(missing)}"
    return rec


def load_registry() -> dict[str, PanelSpec]:
    """Import the panel modules (which self-register) and return the registry.

    Import is deferred and idempotent so ``panels`` itself stays cheap to import
    and a missing optional dependency in one panel module cannot take the whole
    registry down.
    """
    if not PANELS:
        from . import panels_artifacts  # noqa: F401
        from . import panels_stats  # noqa: F401
    return PANELS


def sections() -> dict[str, list[PanelSpec]]:
    """Registry grouped by UI palette section, in a stable order."""
    out: dict[str, list[PanelSpec]] = {}
    for spec in load_registry().values():
        out.setdefault(spec.section, []).append(spec)
    for specs in out.values():
        specs.sort(key=lambda s: s.title)
    return dict(sorted(out.items()))


__all__ = [
    "PANELS", "PanelSpec", "RenderContext", "PanelDataMissing",
    "register", "simple_panel", "draw_panel", "coverage", "family_of",
    "load_registry", "sections",
    "METRIC_FAMILIES", "METRIC_LABELS", "group_color", "ordered_groups", "np", "pd",
]
