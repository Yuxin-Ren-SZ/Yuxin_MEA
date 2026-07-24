"""Shared, disk-cached data layer for the standalone panel scripts.

Every subplot is its own script, so each one is a fresh process. Without a cache
that would mean re-fitting the pooled-burst UMAP (minutes) once per panel, and
re-walking 1843 per-recording artifacts every time a trajectory is drawn. This
module gives the panels one context object whose expensive derived tables are

* memoised in-process (a batch run through ``render_panels`` pays each once), and
* persisted to ``<figure_root>/report/_cache/`` keyed by a fingerprint of the
  inputs, so a single-panel re-run is instant.

``--refresh-cache`` on any panel drops the persisted entries and recomputes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from yuxin_mea.dataset.metadata import parse_hours_since_media

from . import load as L
from . import stats as S
from .report_style import DEFAULT_CONFIG, report_dir, resolve_roots

logger = logging.getLogger("report.data")

#: The three arms every main figure shows. H2O2 / NPH / AraC are supplement-only.
MAIN_ARMS = ["Control", "IVH_Early", "IVH_Late"]
#: Treated arms (Control is the reference, never listed as a focus arm).
FOCUS_ARMS = ["IVH_Early", "IVH_Late"]

# --------------------------------------------------------------------------- #
# Time since the last media change
# --------------------------------------------------------------------------- #
# A well recorded 30 minutes after its media was changed is not comparable with
# the same well recorded a day later: the acute scans capture the medium swap,
# not the treatment. ``hours_since_media`` (parsed from each recording's assay
# tag) makes that explicit, and these windows keep the report on one timepoint.
#: ~24 h after the media change — the routine post-treatment scan.
MEDIA_WINDOW_POST = (18.0, 30.0)
#: A full day to two days — the pre-treatment baseline.
MEDIA_WINDOW_PRE = (24.0, 48.0)
#: Some early tags record no interval at all ("Test", "Old_config"). The protocol
#: default is a scan the next day, which is what ``_treatment_from_flip`` has
#: always assumed; those rows are admitted on that basis rather than discarded,
#: and the count is logged so the assumption stays visible.
ASSUME_HOURS_WHEN_UNKNOWN = 24.0

# --------------------------------------------------------------------------- #
# Metric families — one per figure.
# --------------------------------------------------------------------------- #
# These are the BH-FDR families. A DiD panel draws one metric but must correct
# over the whole family its figure reports, so every panel script passes the
# figure's list here and then selects its own row. Splitting the family per panel
# would silently make q == p.
F5_METRICS = ["nb_rate", "nb_duration_mean", "nb_spikes_per_burst_mean",
              "nb_ibi_mean", "median_firing_rate"]
F7_METRICS = ["mean_sttc", "edge_density", "clustering_coeff", "modularity",
              "global_efficiency", "small_worldness"]
F7_NODE_METRICS = ["hub_fraction", "leaf_fraction", "participation_mean",
                   "mean_betweenness", "rich_club", "assortativity"]
F7_DIR_METRICS = ["mean_te", "degree_asymmetry", "reciprocity", "flow_hierarchy"]
F8_METRICS = ["branching_ratio_mr", "dcc", "aval_tau", "aval_alpha"]

#: Per-recording burst metrics that live in the detector's ``metrics.json`` but
#: may predate the current roll-up; backfilled onto ``tidy`` when missing.
_BURST_BACKFILL = {
    "nb_ibi_cv": ("network_bursts", "inter_event_interval", "cv"),
    "nb_duration_cv": ("network_bursts", "duration", "cv"),
}


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class _Cache:
    """Fingerprinted pickle cache under ``report/_cache/``."""

    def __init__(self, figure_root, fingerprint: str, refresh: bool = False):
        self.dir = report_dir(figure_root, "_cache")
        self.fingerprint = fingerprint
        self.refresh = bool(refresh)
        self._mem: dict[str, Any] = {}

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(f"{self.fingerprint}|{key}".encode()).hexdigest()[:16]
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:60]
        return self.dir / f"{safe}.{h}.pkl"

    def get(self, key: str, build: Callable[[], Any]) -> Any:
        if key in self._mem:
            return self._mem[key]
        p = self._path(key)
        if p.exists() and not self.refresh:
            try:
                with p.open("rb") as fh:
                    val = pickle.load(fh)
                self._mem[key] = val
                return val
            except Exception as exc:  # noqa: BLE001 — a stale/corrupt entry just recomputes
                logger.warning("cache read failed for %s (%s); recomputing", key, exc)
        logger.info("computing %s", key)
        val = build()
        try:
            with p.open("wb") as fh:
                pickle.dump(val, fh)
        except Exception as exc:  # noqa: BLE001 — caching is an optimisation, never fatal
            logger.warning("cache write failed for %s: %s", key, exc)
        self._mem[key] = val
        return val


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
class PanelContext:
    """Roots, the tidy table, and the memoised derived tables panels ask for."""

    def __init__(self, config_path=None, refresh: bool = False, qpcr_dir=None):
        self.config_path = config_path
        self.analysis_root, self.figure_root = resolve_roots(config_path)
        tidy = L.load_tidy(self.figure_root)
        # The window is part of the cache key, not just of the result: it decides
        # which recordings every derived table is built from, so widening it — or
        # turning it off with ``media_window: null`` — has to invalidate the
        # pickles. Hashing only the on-disk table would let a config change come
        # back served from a cache built under the old cohort, with nothing on
        # screen to say so.
        self.media_window = _media_window_config(config_path)
        fp = hashlib.sha256(
            f"{self.analysis_root}|{len(tidy)}|{','.join(sorted(tidy.columns))}"
            f"|media={self.media_window}".encode()
        ).hexdigest()[:16]
        self.cache = _Cache(self.figure_root, fp, refresh=refresh)
        # Applied here rather than per panel, so every derived table inherits one
        # cohort and no figure can quietly disagree with its neighbours about
        # which recordings the report is about.
        self.tidy = self.cache.get(
            "tidy_backfilled",
            lambda: _restrict_media_window(
                _backfill_hours(_backfill(tidy, self.analysis_root),
                                self.analysis_root),
                self.media_window))
        # qPCR plate exports live beside the MEA analysis tree; the conventional
        # location is used unless a panel is given an explicit one.
        default_qpcr = self.analysis_root / "qPCR"
        self.qpcr_dir = Path(qpcr_dir) if qpcr_dir else (
            default_qpcr if default_qpcr.is_dir() else None)
        _warn_baseline_labels(self.tidy)

    # -- convenience views ------------------------------------------------- #
    @property
    def main(self) -> pd.DataFrame:
        """``tidy`` restricted to the three main-figure arms (by well identity)."""
        return self.cache.get("tidy_main", lambda: _restrict_arms(self.tidy, MAIN_ARMS))

    def well_index(self) -> pd.DataFrame:
        return self.cache.get("well_index", lambda: S.well_index(self.tidy))

    # -- statistics -------------------------------------------------------- #
    def response_by_tau(self, metrics: list[str], arms: list[str] | None = None
                        ) -> pd.DataFrame:
        arms = arms or MAIN_ARMS
        key = f"resp_tau|{','.join(sorted(metrics))}|{','.join(sorted(arms))}"

        def build():
            r = S.well_response_by_tau(self.tidy, metrics=list(metrics))
            return r[r.arm.isin(arms)].reset_index(drop=True)
        return self.cache.get(key, build)

    def did(self, metrics: list[str], arms: list[str] | None = None) -> pd.DataFrame:
        """Chip-level DiD at the primary endpoint, with p and BH-FDR q.

        One row per (arm, metric) — the table every DiD panel labels its glyphs
        from and writes out as ``<panel>_stats.csv``.
        """
        arms = arms or MAIN_ARMS
        key = f"did|{','.join(sorted(metrics))}|{','.join(sorted(arms))}"

        def build():
            resp = self.response_by_tau(metrics, arms)
            return S.did_at_endpoint(resp, focus_arms=[a for a in arms if a != "Control"])
        return self.cache.get(key, build)

    # -- per-unit / artifact tables ---------------------------------------- #
    def units(self, rows: pd.DataFrame, tag: str) -> pd.DataFrame:
        """Concatenated per-unit quality metrics for ``rows`` (cached under ``tag``)."""
        return self.cache.get(f"units|{tag}", lambda: L.load_units(self.analysis_root, rows))

    def dataset(self, name: str, build: Callable[["PanelContext"], Any], **params):
        """Memoise an expensive shared dataset (pooled bursts, UMAP fits, CCG picks)."""
        key = f"ds|{name}|{json.dumps(params, sort_keys=True, default=str)}"
        return self.cache.get(key, lambda: build(self, **params))


    # -- representative recordings ----------------------------------------- #
    def example_pair(self, arm: str = "Control", pre_tau: float = 0.0,
                     post_tau: float = 14.0, metric: str = "n_curated"):
        """``(pre_row, post_row, well_uid)`` for one well of ``arm``.

        ``pre`` is the recording **immediately before** treatment (the latest
        ``tau <= pre_tau``) and ``post`` the one nearest ``post_tau`` days after —
        the immature-vs-mature comparison F4 shows. The well is the one with the
        most curated units among those covering both time points, so the raster
        is legible rather than a handful of sparse rows.
        """
        key = f"pair|{arm}|{pre_tau}|{post_tau}|{metric}"

        def build():
            wi = self.well_index()
            wells = set(wi.loc[wi.arm == arm, "well_uid"])
            best = None
            for wuid, sub in self.tidy.groupby("well_uid"):
                if wuid not in wells:
                    continue
                pre = sub[sub.tau <= pre_tau]
                post = sub[sub.tau > pre_tau]
                if pre.empty or post.empty:
                    continue
                p_row = pre.sort_values("tau").iloc[-1]
                q_row = post.iloc[(post.tau - post_tau).abs().to_numpy().argsort()[0]]
                if abs(float(q_row.tau) - post_tau) > 4:
                    continue
                score = float(min(p_row.get(metric, 0) or 0, q_row.get(metric, 0) or 0))
                if best is None or score > best[0]:
                    best = (score, p_row, q_row, wuid)
            if best is None:
                return (None, None, None)
            return (best[1], best[2], best[3])
        return self.cache.get(key, build)


    def ml_trace_row(self, arm: str = "IVH_Late", min_bursts: int = 8,
                     min_units: int = 25):
        """A recording of ``arm`` whose ML debug trace is on disk.

        The HMM-posterior panel needs ``debug_trace.pkl``, which only exists for
        runs with ``debug=True``; candidates are ranked by burst rate so the
        chosen example actually shows burst structure rather than a flat trace.

        Selection is on the well's arm and restricted to post-treatment
        recordings. Matching ``canonical_group`` instead would silently mean
        "post-treatment" for a treated arm and "anything, including treated wells'
        baselines" for Control — two different things under one parameter.
        """
        key = f"ml_row|{arm}|{min_bursts}|{min_units}"

        def build():
            wi = S.well_index(self.tidy).set_index("well_uid").arm
            t = self.tidy.assign(_arm=self.tidy.well_uid.map(wi))
            cand = t[(t._arm == arm) & (t.tau > 0)
                     & (t.nb_count >= min_bursts)
                     & (t.n_curated >= min_units)]
            for _, r in cand.sort_values("nb_rate", ascending=False).iterrows():
                if L.ml_trace_path(self.analysis_root, r).exists():
                    return r
            return None
        return self.cache.get(key, build)

    def ml_trace(self, row):
        return L.load_ml_trace(self.analysis_root, row)


_CTX: PanelContext | None = None


def context(config_path=None, refresh: bool = False, qpcr_dir=None) -> PanelContext:
    """Process-wide context; reused across panels in a batch render."""
    global _CTX
    if _CTX is None or _CTX.config_path != config_path or refresh:
        _CTX = PanelContext(config_path, refresh=refresh, qpcr_dir=qpcr_dir)
    return _CTX


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _warn_baseline_labels(tidy: pd.DataFrame) -> None:
    """Log, per sample, how many baselines are labelled ``Control`` on a treated well.

    ``canonical_group`` is the groupname the MaxWell chip carried at scan time, and
    the convention differs between chips: CX118 and CX169 name a treated well from
    its first pre-treatment scan, CX138 scanned the whole plate as ``Control`` until
    treatment day. Any selection written against ``canonical_group`` therefore sees
    a different cohort on different chips — treated baselines missing from their own
    arm and pooled into Control. Surfacing the count once makes that visible instead
    of leaving it to reshape selection pools silently.
    """
    if not {"canonical_group", "tau", "well_uid", "sample_id"}.issubset(tidy.columns):
        return
    arm = S.well_index(tidy).set_index("well_uid").arm
    d = tidy.assign(arm=tidy.well_uid.map(arm))
    bad = d[(d.tau < 0) & (d.canonical_group == "Control") & (d.arm != "Control")]
    if bad.empty:
        return
    for sample, sub in bad.groupby("sample_id"):
        logger.warning(
            "%s: %d pre-treatment recordings on %d treated wells are labelled "
            "'Control' (arms: %s) — select on the well's arm, not canonical_group",
            sample, len(sub), sub.well_uid.nunique(),
            ", ".join(sorted(sub.arm.dropna().unique())))


def _media_window_config(config_path) -> dict | None:
    """Read the ``media_window`` block, or the defaults, or ``None`` to disable.

    Absent from the config → the module defaults. Explicitly ``null`` → the filter
    is off and the whole cohort is analysed, which is the escape hatch for
    checking what the window costs without editing code.
    """
    cfg: dict = {}
    try:
        with (Path(config_path) if config_path else DEFAULT_CONFIG).open() as fh:
            cfg = json.load(fh)
    except Exception as exc:  # noqa: BLE001 — a missing config just means defaults
        logger.debug("media_window: config unreadable (%s); using defaults", exc)
    block = cfg.get("media_window", cfg.get("global", {}).get("media_window", {}))
    if block is None:
        return None
    return {
        "pre": tuple(block.get("pre", MEDIA_WINDOW_PRE)),
        "post": tuple(block.get("post", MEDIA_WINDOW_POST)),
        "assume_unknown_hours": block.get("assume_unknown_hours",
                                          ASSUME_HOURS_WHEN_UNKNOWN),
    }


def _restrict_media_window(tidy: pd.DataFrame, window: dict | None) -> pd.DataFrame:
    """Keep recordings taken a comparable time after their last media change.

    Pre-treatment (``tau < 0``) and post-treatment rows get different windows: a
    baseline may sit anywhere from a day to two days after the change, but a
    treatment-period scan has to be the routine ~24 h one. ``tau == 0`` is gated
    with the post window, which rejects it — every tau-0 recording carries a 0 h
    tag by construction, since the treatment date is back-dated from that very
    scan's tag.

    Bounds are inclusive at both ends **deliberately**: the unknown-tag default
    sits exactly on the pre window's lower edge, and an exclusive comparison would
    silently drop the untimed baselines this is meant to keep.
    """
    if window is None or "hours_since_media" not in tidy.columns:
        if window is not None:
            logger.warning("media window skipped: tidy has no hours_since_media "
                           "column — re-run compare_treatment_groups to add it")
        return tidy
    h = pd.to_numeric(tidy.hours_since_media, errors="coerce")
    unknown = h.isna()
    h = h.fillna(window["assume_unknown_hours"])
    pre = tidy.tau < 0
    lo_pre, hi_pre = window["pre"]
    lo_post, hi_post = window["post"]
    keep = np.where(pre,
                    h.between(lo_pre, hi_pre, inclusive="both"),
                    h.between(lo_post, hi_post, inclusive="both"))
    out = tidy[keep].copy()

    before = _pre_post_coverage(tidy)
    after = _pre_post_coverage(out)
    logger.info("media window pre %s post %s: kept %d/%d recordings "
                "(%d admitted on the unknown-tag default of %g h)",
                window["pre"], window["post"], len(out), len(tidy),
                int((unknown & keep).sum()), window["assume_unknown_hours"])
    for sample, n_drop in (tidy[~keep].groupby("sample_id").size().items()):
        logger.info("  %s: %d recordings outside the window", sample, n_drop)
    # A well that loses its last baseline stops being paired and vanishes from
    # every difference-in-differences — the one failure mode of this filter that
    # would not look like a filter.
    lost = sorted(before - after)
    if lost:
        logger.warning("media window cost %d well(s) their pre or post coverage: %s",
                       len(lost), ", ".join(f"{w} ({side})" for w, side in lost[:10]))
    return out


def _pre_post_coverage(tidy: pd.DataFrame) -> set:
    """``{(well_uid, "pre"|"post")}`` for every well that has that side."""
    if tidy.empty or "well_uid" not in tidy.columns:
        return set()
    pre = tidy[tidy.tau < 0].well_uid.unique()
    post = tidy[tidy.tau > 0].well_uid.unique()
    return {(w, "pre") for w in pre} | {(w, "post") for w in post}


def _restrict_arms(tidy: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    """Keep wells whose *treatment identity* is in ``arms``.

    Filtering on ``canonical_group`` would silently drop every treated well's
    pre-treatment recordings (they are labelled ``Control``), which is exactly the
    baseline the difference-in-differences needs. Filter on the well's arm.
    """
    wi = S.well_index(tidy)
    keep = set(wi.loc[wi.arm.isin(arms), "well_uid"])
    return tidy[tidy.well_uid.isin(keep)].copy()


def _backfill(tidy: pd.DataFrame, analysis_root: Path) -> pd.DataFrame:
    """Add burst-variability columns from the detector's per-recording metrics.json.

    ``nb_ibi_cv`` / ``nb_duration_cv`` are already computed by the burst detector
    (``network_bursts.inter_event_interval.cv``) but predate the current roll-up,
    so they are absent from older ``tidy_long.csv``. Read them straight from the
    same source the other ``nb_*`` columns come from rather than re-deriving them
    from spike times — same numbers, no ambiguity.
    """
    missing = [c for c in _BURST_BACKFILL if c not in tidy.columns]
    if not missing:
        return tidy
    out = tidy.copy()
    vals: dict[str, list[float]] = {c: [] for c in missing}
    for _, r in out.iterrows():
        p = L.artifact_path(analysis_root, "burst_detection_data", r,
                            "burst_detection", "metrics.json")
        m: dict = {}
        if p.exists():
            try:
                m = json.loads(p.read_text())
            except Exception:  # noqa: BLE001 — a bad file is a NaN, not a crash
                m = {}
        for c in missing:
            cur: Any = m
            for k in _BURST_BACKFILL[c]:
                cur = cur.get(k) if isinstance(cur, dict) else None
            vals[c].append(float(cur) if isinstance(cur, (int, float)) else np.nan)
    for c in missing:
        out[c] = vals[c]
    n_ok = {c: int(pd.notna(out[c]).sum()) for c in missing}
    logger.info("backfilled burst-variability columns: %s of %d rows", n_ok, len(out))
    return out


def _backfill_hours(tidy: pd.DataFrame, analysis_root: Path) -> pd.DataFrame:
    """Derive ``hours_since_media`` from the assay tags when the roll-up predates it.

    ``compare_treatment_groups`` writes this column now, but the live
    ``tidy_long.csv`` also carries criticality and directed-connectivity columns
    that script does not produce — so re-running it to pick up one new column
    would drop twenty-two others. Deriving here instead keeps both: the column is
    read straight from ``experiment_cache.json``, which is the same source the
    roll-up uses, so the two paths cannot disagree.
    """
    if "hours_since_media" in tidy.columns:
        return tidy
    path = analysis_root / "experiment_cache.json"
    if not path.exists():
        logger.warning("no experiment_cache.json at %s — hours_since_media unavailable",
                       path)
        return tidy
    with path.open() as fh:
        cache = json.load(fh)
    hours = {k: parse_hours_since_media((r.get("metadata") or {}).get("tag"))
             for k, r in cache.items()}
    out = tidy.copy()
    out["hours_since_media"] = out.recording_key.map(hours)
    known = int(pd.notna(out.hours_since_media).sum())
    logger.info("derived hours_since_media for %d of %d rows (%d tags record no "
                "interval)", known, len(out), len(out) - known)
    return out
