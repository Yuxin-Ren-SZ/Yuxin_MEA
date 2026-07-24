#!/usr/bin/env python3
"""Longitudinal treatment-group comparison for MEA burst/firing metrics.

The pipeline computes per-well burst/firing metrics and tags each well with a
free-text ``groupname`` (Control, H2O2 doses, IVH Early/Late, NPH, AraC...), but
nothing pools results by group and compares them. This script does that, while
respecting the awkward parts of the data:

  * it is longitudinal — each physical well = ``(sample_id, plate_id, well_id)``
    recorded on many dates;
  * cultures have different lengths and are treated at different DIV, so absolute
    time is not comparable across wells;
  * treatment is crossed *within* plate (every real plate carries contemporaneous
    Control wells), so the well is the unit of analysis and the plate is a block;
  * neither plating date nor treatment date lives anywhere in the pipeline — the
    user supplies them via a template CSV this script can emit.

Method (see ``doc``/plan for rationale):
  1. Align each recording to tau = (recording date - treatment date) in days.
  2. Normalise each well to its own pre-treatment baseline (tau < 0) — this is the
     single strongest move: it neutralises both culture-length and treatment-DIV
     differences. Response = log2 fold-change (ratio metrics) or post-minus-pre
     (difference metrics) over a post-treatment window.
  3. Collapse to one value per well (unit = well, no pseudoreplication).
  4. Nonparametric stats (scipy only): Mann-Whitney U / Kruskal-Wallis, Cliff's
     delta effect size, permutation p, bootstrap CI, Benjamini-Hochberg FDR — run
     both pooled-by-canonical-arm and within-plate.

Run — step 1, emit the template to fill in::

    python scripts/compare_treatment_groups.py --config pipeline_config.json \
        --emit-template treatment_map.csv

Run — step 2, after filling treatment_map.csv::

    python scripts/compare_treatment_groups.py --config pipeline_config.json \
        --treatment-map treatment_map.csv

Outputs land under ``<figure_root>/treatment_comparison/``: tidy_long.csv,
per_well_summary.csv, stats.csv, report.html, plus PNG figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("compare_treatment_groups")

# --------------------------------------------------------------------------- #
# Metric specification
# --------------------------------------------------------------------------- #
# Each metric: display name -> (source, extractor, kind).
#   source   : "burst" (traditional metrics.json), "ml" (ML diagnostics.json),
#              "curation" (aggregated quality_metrics.pkl).
#   kind     : "ratio" -> response = log2((post+eps)/(base+eps));
#              "diff"  -> response = post - base.
# Burst-typing per-type *proportions* are intentionally NOT used: typing is a
# per-well k-means with per-well cluster ids, so "fraction of type 0" is not
# comparable across wells. We use the cross-well-comparable scalars instead
# (burst_modulation_index, number of burst types).


def _nested(d: dict, *keys: str) -> float | None:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


METRIC_SPECS: dict[str, dict[str, Any]] = {
    # --- Network bursting (traditional burst_detection metrics.json) ---------
    "nb_rate": {"source": "burst", "kind": "ratio",
                "get": lambda m: _nested(m, "network_bursts", "rate")},
    "nb_count": {"source": "burst", "kind": "ratio",
                 "get": lambda m: _nested(m, "network_bursts", "count")},
    "nb_duration_mean": {"source": "burst", "kind": "ratio",
                         "get": lambda m: _nested(m, "network_bursts", "duration", "mean")},
    "nb_spikes_per_burst_mean": {"source": "burst", "kind": "ratio",
                                 "get": lambda m: _nested(m, "network_bursts", "spikes_per_burst", "mean")},
    "nb_ibi_mean": {"source": "burst", "kind": "ratio",
                    "get": lambda m: _nested(m, "network_bursts", "inter_event_interval", "mean")},
    # Burst regularity: CV of the inter-burst intervals (and of burst duration).
    # Both are already computed by the detector; they are the "burst variability"
    # readouts F4 tracks across maturation.
    "nb_ibi_cv": {"source": "burst", "kind": "ratio",
                  "get": lambda m: _nested(m, "network_bursts", "inter_event_interval", "cv")},
    "nb_duration_cv": {"source": "burst", "kind": "ratio",
                       "get": lambda m: _nested(m, "network_bursts", "duration", "cv")},
    # --- Firing / activity (curation quality_metrics) ------------------------
    "median_firing_rate": {"source": "curation", "kind": "ratio",
                           "get": lambda m: m.get("median_firing_rate")},
    "n_curated": {"source": "curation", "kind": "ratio",
                  "get": lambda m: m.get("n_curated")},
    # --- Burst-type composition (ML diagnostics, cross-well-comparable) ------
    "burst_modulation_index": {"source": "ml", "kind": "diff",
                               "get": lambda m: _nested(m, "burst_modulation_index")},
    "burst_type_k": {"source": "ml", "kind": "diff",
                     "get": lambda m: _nested(m, "burst_typing", "k")},
    "cluster_n_clusters": {"source": "ml", "kind": "diff",
                           "get": lambda m: _nested(m, "cluster_n_clusters")},
    # --- Spatial activity map (spatial_map_data spatial_metrics.json) ---------
    # activity_gini / mean_prop_speed are non-negative -> ratio (log2) is valid.
    "activity_gini": {"source": "spatial", "kind": "ratio",
                      "get": lambda m: _nested(m, "activity_gini")},
    "mean_prop_speed_um_ms": {"source": "spatial", "kind": "ratio",
                              "get": lambda m: _nested(m, "mean_prop_speed_um_ms")},
    # --- Functional connectivity (connectivity_data graph_metrics.json) -------
    # mean_sttc in [-1,1] and modularity can be <=0 -> must be diff, not ratio.
    "mean_sttc": {"source": "connectivity", "kind": "diff",
                  "get": lambda m: _nested(m, "mean_sttc")},
    "edge_density": {"source": "connectivity", "kind": "ratio",
                     "get": lambda m: _nested(m, "edge_density")},
    "modularity": {"source": "connectivity", "kind": "diff",
                   "get": lambda m: _nested(m, "modularity")},
    "small_worldness": {"source": "connectivity", "kind": "ratio",
                        "get": lambda m: _nested(m, "small_worldness")},
}

RATIO_EPS = 1e-9  # guards log2 of zero for counts/rates


# --------------------------------------------------------------------------- #
# Config / path helpers
# --------------------------------------------------------------------------- #
def _load_config(config_path: Path) -> tuple[Path, Path]:
    with config_path.open() as fh:
        cfg = json.load(fh)
    g = cfg.get("global", {})
    analysis_root = Path(g["analysis_root"])
    figure_root = Path(g.get("figure_root") or analysis_root)
    return analysis_root, figure_root


def _parse_yymmdd(s: str | None) -> datetime | None:
    """Parse a 6-digit ``YYMMDD`` (2000-2099) into a datetime, else None."""
    if s is None:
        return None
    s = str(s).strip()
    if len(s) != 6 or not s.isdigit():
        return None
    try:
        return datetime.strptime("20" + s, "%Y%m%d")
    except ValueError:
        return None


def _load_experiment_cache(analysis_root: Path) -> dict[str, Any]:
    path = analysis_root / "experiment_cache.json"
    if not path.exists():
        raise SystemExit(f"experiment_cache.json not found at {path}")
    with path.open() as fh:
        return json.load(fh)


def _group_lookup(experiment_cache: dict) -> dict[tuple[str, str], dict[str, str]]:
    """Map (recording_key, well_id) -> {well_name, groupname}."""
    out: dict[tuple[str, str], dict[str, str]] = {}
    for rec_key, rec in experiment_cache.items():
        for wid, w in (rec.get("wells") or {}).items():
            md = w.get("metadata") or {}
            out[(rec_key, wid)] = {
                "well_name": str(md.get("well_name", "?")),
                "groupname": str(md.get("groupname", "?")),
            }
    return out


# --------------------------------------------------------------------------- #
# Template emission
# --------------------------------------------------------------------------- #
TEMPLATE_COLUMNS = [
    "sample_id", "plate_id", "raw_groupname",
    "canonical_group", "role", "plating_date", "treatment_date",
    "n_wells", "date_min", "date_max", "treated_media",
]


def _norm_plating_date(s: str) -> str:
    """Normalise a MaxWell 'Plating Date' (e.g. '30.1.2026' / '30.01.2026') to YYMMDD."""
    s = str(s).strip()
    for fmt in ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%y%m%d")
        except ValueError:
            continue
    return ""


def _treatment_from_flip(flip_date: str, tag: str) -> str:
    """Back-date the media-change (treatment) day from the first treated recording.

    Recordings are taken some hours AFTER the media change; the ``tag`` encodes the
    delay: 'Imm xH' → recorded at treatment (same day); '24H' → 1 day after;
    '48H' → 2 days after. Default (no marker) assumes the usual 24 h delay.
    So treatment_date = flip recording date − delay.
    """
    from datetime import timedelta
    dt = _parse_yymmdd(flip_date)
    if dt is None:
        return ""
    t = str(tag).lower()
    if "imm" in t:
        delta = 0
    elif "48h" in t:
        delta = 2
    elif "24h" in t:
        delta = 1
    else:
        delta = 1  # protocol default: recording done 24 h after media change
    return (dt - timedelta(days=delta)).strftime("%y%m%d")


def emit_template(experiment_cache: dict, out_path: Path) -> None:
    """Enumerate every (sample_id, plate_id, raw_groupname) and write a template.

    Auto-fills what the cached MaxWell annotations already contain, so the user
    only reviews rather than hand-enters dates:

    - ``plating_date`` — from each well's ``Plating Date`` annotation (YYMMDD).
    - ``treatment_date`` — derived from the ``Media`` annotation: the earliest
      recording date on which a group's Media contains ``'+'`` (i.e. flips from
      plain "Maintain"/"Phase" to "Maintain + <drug>"). Control rows (and any
      never-dosed group) inherit the plate's earliest treated date, so controls
      get the same pre/post split (difference-in-differences baseline).
    - ``role`` — 'control' for the exact ``Control`` group, else 'treatment' for
      any group that ever received a ``'+'`` Media, else 'control'.
    - ``treated_media`` — the detected drug Media string(s), for the user to sanity
      check the derivation.

    The user still (a) edits ``canonical_group`` to merge spelling variants across
    plates, (b) deletes unwanted rows (blank/Default groups), and (c) overrides any
    date the Media heuristic got wrong (e.g. a late acute-pharmacology probe).
    """
    from collections import defaultdict

    agg: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"wells": set(), "dates": set(), "plating": set(),
                 "treated_media": set(), "flip": None})
    plate_treatment: dict[tuple[str, str], list] = defaultdict(list)

    for rec_key, rec in experiment_cache.items():
        sid = str(rec.get("sample_id", ""))
        pid = str(rec.get("plate_id", ""))
        date = str(rec.get("date", ""))
        run_id = str(rec.get("run_id", ""))
        tag = str((rec.get("metadata") or {}).get("tag", ""))
        for _wid, w in (rec.get("wells") or {}).items():
            md = w.get("metadata") or {}
            g = str(md.get("groupname", "?"))
            slot = agg[(sid, pid, g)]
            slot["wells"].add((sid, pid, _wid))
            if date:
                slot["dates"].add(date)
            pdate = md.get("Plating Date")
            if pdate:
                slot["plating"].add(str(pdate))
            media = str(md.get("Media") or "")
            if "+" in media:  # "Maintain + <drug>" marks a treated recording
                slot["treated_media"].add(media)
                # remember the EARLIEST treated recording (by date, then run) + its tag
                cand = (date, run_id, tag)
                if slot["flip"] is None or (date, run_id) < slot["flip"][:2]:
                    slot["flip"] = cand

    # per-group treatment date, then plate's earliest for controls
    group_treatment: dict[tuple[str, str, str], str] = {}
    for key, slot in agg.items():
        if slot["flip"]:
            fdate, _run, ftag = slot["flip"]
            td = _treatment_from_flip(fdate, ftag)
            group_treatment[key] = td
            if td:
                plate_treatment[key[:2]].append(td)

    rows = []
    for (sid, pid, g), slot in sorted(agg.items()):
        gl = g.strip().lower()
        treated = slot["flip"] is not None
        role = "control" if gl == "control" else ("treatment" if treated else "control")
        plating = _norm_plating_date(sorted(slot["plating"])[0]) if slot["plating"] else ""
        if role == "treatment":
            treatment_date = group_treatment.get((sid, pid, g), "")
        else:  # control / never-dosed → plate's earliest treated date
            pts = plate_treatment.get((sid, pid))
            treatment_date = min(pts) if pts else ""
        dates = sorted(slot["dates"])
        rows.append({
            "sample_id": sid, "plate_id": pid, "raw_groupname": g,
            "canonical_group": g, "role": role,
            "plating_date": plating, "treatment_date": treatment_date,
            "n_wells": len(slot["wells"]),
            "date_min": dates[0] if dates else "",
            "date_max": dates[-1] if dates else "",
            "treated_media": "; ".join(sorted(slot["treated_media"])[:3]),
        })

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote template with %d (sample,plate,group) rows -> %s", len(rows), out_path)


def load_template(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Read a filled template -> {(sample_id, plate_id, raw_groupname): {...}}."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["sample_id"].strip(), row["plate_id"].strip(),
                   row["raw_groupname"].strip())
            out[key] = {
                "canonical_group": (row.get("canonical_group") or row["raw_groupname"]).strip(),
                "role": (row.get("role") or "").strip().lower(),
                "plating_date": (row.get("plating_date") or "").strip(),
                "treatment_date": (row.get("treatment_date") or "").strip(),
            }
    return out


# --------------------------------------------------------------------------- #
# Loader: walk per-well outputs -> tidy long DataFrame
# --------------------------------------------------------------------------- #
def _parse_metrics_path(metrics_path: Path, burst_dirname: str) -> dict[str, str] | None:
    """Extract ids from a .../<burst_dir>/<key.../<rec>/<well>/<terminal>/metrics.json path."""
    parts = metrics_path.parts
    try:
        i = parts.index(burst_dirname)
    except ValueError:
        return None
    # parts[i+1 : i+6] = sample_id, date, plate_id, scan_type, run_id
    tail = parts[i + 1:]
    if len(tail) < 8:
        return None
    sample_id, date, plate_id, scan_type, run_id, rec_name, well_id = tail[0:7]
    return {
        "sample_id": sample_id, "date": date, "plate_id": plate_id,
        "scan_type": scan_type, "run_id": run_id,
        "rec_name": rec_name, "well_id": well_id,
        "recording_key": "/".join([sample_id, date, plate_id, scan_type, run_id]),
    }


def _read_json(path: Path) -> dict[str, Any]:
    """Load a per-well metrics JSON; empty dict on missing/unreadable file."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.debug("json read fail %s: %s", path, exc)
        return {}


def _curation_median_firing(qm_path: Path) -> dict[str, Any]:
    """One-well curation summary from quality_metrics.pkl (median_firing_rate, n_curated)."""
    if not qm_path.exists():
        return {}
    try:
        qm = pd.read_pickle(qm_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("curation read fail %s: %s", qm_path, exc)
        return {}
    if "curated" not in qm.columns:
        return {}
    n_curated = int(qm["curated"].sum())
    curated = qm.loc[qm["curated"]] if n_curated else qm.iloc[:0]
    mfr = None
    if "firing_rate" in curated.columns and not curated.empty:
        v = curated["firing_rate"].median()
        mfr = None if pd.isna(v) else float(v)
    return {"n_curated": n_curated, "median_firing_rate": mfr}


def build_long_table(
    analysis_root: Path,
    group_lut: dict[tuple[str, str], dict[str, str]],
    burst_dirname: str = "burst_detection_data",
    ml_dirname: str = "ml_burst_data_umap",
    curation_dirname: str = "curation_data",
    spatial_dirname: str = "spatial_map_data",
    conn_dirname: str = "connectivity_data",
    scan_type: str = "Network",
    limit: int | None = None,
) -> pd.DataFrame:
    """Walk per-well outputs and assemble one row per (well x recording)."""
    burst_root = analysis_root / burst_dirname
    if not burst_root.exists():
        raise SystemExit(f"burst root not found: {burst_root}")

    metric_paths = sorted(burst_root.glob(f"*/*/*/{scan_type}/*/*/*/*/metrics.json"))
    if limit:
        metric_paths = metric_paths[:limit]
    logger.info("Found %d %s metrics.json under %s", len(metric_paths), scan_type, burst_root)

    rows: list[dict[str, Any]] = []
    n_no_ml = n_no_cur = 0
    for mp in metric_paths:
        ids = _parse_metrics_path(mp, burst_dirname)
        if ids is None or ids["scan_type"] != scan_type:
            continue
        try:
            burst_metrics = json.loads(mp.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.debug("burst read fail %s: %s", mp, exc)
            continue

        rel = Path(ids["recording_key"]) / ids["rec_name"] / ids["well_id"]
        ml_path = analysis_root / ml_dirname / rel / "ml_burst_detection" / "diagnostics.json"
        cur_path = analysis_root / curation_dirname / rel / "auto_curation" / "quality_metrics.pkl"
        spatial_path = analysis_root / spatial_dirname / rel / "spatial_map" / "spatial_metrics.json"
        conn_path = analysis_root / conn_dirname / rel / "connectivity" / "graph_metrics.json"

        ml_diag: dict[str, Any] = {}
        if ml_path.exists():
            try:
                ml_diag = json.loads(ml_path.read_text())
            except Exception:  # noqa: BLE001
                ml_diag = {}
        else:
            n_no_ml += 1
        cur = _curation_median_firing(cur_path)
        if not cur:
            n_no_cur += 1

        spatial_metrics = _read_json(spatial_path)
        conn_metrics = _read_json(conn_path)

        meta = group_lut.get((ids["recording_key"], ids["well_id"]), {})
        row: dict[str, Any] = {
            **ids,
            "well_name": meta.get("well_name", "?"),
            "groupname": meta.get("groupname", "?"),
        }
        for name, spec in METRIC_SPECS.items():
            if spec["source"] == "burst":
                row[name] = spec["get"](burst_metrics)
            elif spec["source"] == "ml":
                row[name] = spec["get"](ml_diag)
            elif spec["source"] == "curation":
                row[name] = spec["get"](cur)
            elif spec["source"] == "spatial":
                row[name] = spec["get"](spatial_metrics)
            elif spec["source"] == "connectivity":
                row[name] = spec["get"](conn_metrics)
        rows.append(row)

    logger.info("Assembled %d well-recordings (%d missing ML, %d missing curation)",
                len(rows), n_no_ml, n_no_cur)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date_dt"] = df["date"].map(_parse_yymmdd)
    return df


# --------------------------------------------------------------------------- #
# Apply template: canonicalise + tau/DIV + within-well normalisation
# --------------------------------------------------------------------------- #
def apply_template(df: pd.DataFrame, template: dict[tuple[str, str, str], dict[str, Any]]) -> pd.DataFrame:
    """Add canonical_group, role, tau, DIV columns; drop unmapped rows."""
    canon, role, tdate, pdate = [], [], [], []
    for _, r in df.iterrows():
        key = (r["sample_id"], r["plate_id"], r["groupname"])
        t = template.get(key)
        if t is None:
            canon.append(None); role.append(None); tdate.append(None); pdate.append(None)
            continue
        canon.append(t["canonical_group"])
        role.append(t["role"])
        tdate.append(_parse_yymmdd(t["treatment_date"]))
        pdate.append(_parse_yymmdd(t["plating_date"]))
    df = df.copy()
    df["canonical_group"] = canon
    df["role"] = role
    df["_treatment_dt"] = tdate
    df["_plating_dt"] = pdate

    unmapped = df["canonical_group"].isna().sum()
    if unmapped:
        logger.warning("%d rows had no template entry (groupname not in template) — dropped", unmapped)
    df = df[df["canonical_group"].notna()].copy()

    def _days(a, b):
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            return np.nan
        return (a - b).days

    df["tau"] = [_days(dt, tt) for dt, tt in zip(df["date_dt"], df["_treatment_dt"])]
    df["DIV"] = [_days(dt, pt) for dt, pt in zip(df["date_dt"], df["_plating_dt"])]
    df["well_uid"] = df["sample_id"] + "|" + df["plate_id"] + "|" + df["well_id"]
    return df


def per_well_summary(
    df: pd.DataFrame,
    metrics: list[str],
    baseline_window: float | None,
    post_window: float | None,
    min_pre: int = 1,
    min_post: int = 1,
    mode: str = "paired",
) -> pd.DataFrame:
    """One value per (well x metric); unit of analysis = well.

    ``mode="paired"`` (default, preferred): within-well baseline normalisation.
        baseline = median over pre-treatment rows (tau < 0, within baseline_window);
        post     = median over post rows (0 <= tau <= post_window);
        response = log2((post+eps)/(base+eps)) [ratio] or post - base [diff].
        Requires each well to have >= min_pre PRE-treatment recordings — use this
        only if the protocol recorded a baseline before treatment.

    ``mode="cross_sectional"`` (fallback when no baseline exists): response = the
        absolute post-window median (no normalisation); requires only min_post
        post recordings. Correct when wells have no pre-treatment recordings.
    """
    out_rows: list[dict[str, Any]] = []
    excluded = {"no_tau": 0, "no_pre": 0, "no_post": 0}
    for uid, g in df.groupby("well_uid"):
        g = g[g["tau"].notna()]
        if g.empty:
            excluded["no_tau"] += 1
            continue
        pre = g[g["tau"] < 0]
        if baseline_window is not None:
            pre = pre[pre["tau"] >= -baseline_window]
        post = g[g["tau"] >= 0]
        if post_window is not None:
            post = post[post["tau"] <= post_window]

        base_info = g.iloc[0]
        for metric in metrics:
            pre_vals = pd.to_numeric(pre[metric], errors="coerce").dropna()
            post_vals = pd.to_numeric(post[metric], errors="coerce").dropna()
            if len(post_vals) < min_post:
                excluded["no_post"] += 1
                continue
            postm = float(post_vals.median())
            kind = METRIC_SPECS[metric]["kind"]
            if mode == "cross_sectional":
                base = float("nan")
                response = postm  # absolute post-window value, no normalisation
            else:
                if len(pre_vals) < min_pre:
                    excluded["no_pre"] += 1
                    continue
                base = float(pre_vals.median())
                if kind == "ratio":
                    response = float(np.log2((postm + RATIO_EPS) / (base + RATIO_EPS)))
                else:
                    response = float(postm - base)
            out_rows.append({
                "well_uid": uid,
                "sample_id": base_info["sample_id"],
                "plate_id": base_info["plate_id"],
                "well_id": base_info["well_id"],
                "well_name": base_info["well_name"],
                "canonical_group": base_info["canonical_group"],
                "role": base_info["role"],
                "metric": metric,
                "kind": kind,
                "baseline": base,
                "post_median": postm,
                "response": response,
                "n_pre": len(pre_vals),
                "n_post": len(post_vals),
            })
    # excluded counts are per (well,metric); log once
    logger.info("per-well summary: %d (well,metric) rows; excluded no_tau=%d no_pre=%d no_post=%d",
                len(out_rows), excluded["no_tau"], excluded["no_pre"], excluded["no_post"])
    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------- #
# Nonparametric statistics
# --------------------------------------------------------------------------- #
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size in [-1, 1]; >0 means a tends to exceed b."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return np.nan
    # rank-based O((n+m)log(n+m)) computation
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return float((gt - lt) / (a.size * b.size))


def _cliffs_interp(d: float) -> str:
    ad = abs(d)
    if np.isnan(d):
        return "na"
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def bootstrap_cliffs_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                        n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return (np.nan, np.nan)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        ra = rng.choice(a, size=a.size, replace=True)
        rb = rng.choice(b, size=b.size, replace=True)
        deltas[i] = cliffs_delta(ra, rb)
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def compare_arm_vs_control(
    arm_vals: np.ndarray, ctrl_vals: np.ndarray, rng: np.random.Generator,
) -> dict[str, Any]:
    arm_vals = np.asarray(arm_vals, float); ctrl_vals = np.asarray(ctrl_vals, float)
    res: dict[str, Any] = {
        "n_arm": int(arm_vals.size), "n_ctrl": int(ctrl_vals.size),
        "median_arm": float(np.median(arm_vals)) if arm_vals.size else np.nan,
        "median_ctrl": float(np.median(ctrl_vals)) if ctrl_vals.size else np.nan,
        "cliffs_delta": np.nan, "cliffs_ci_low": np.nan, "cliffs_ci_high": np.nan,
        "effect": "na", "U": np.nan, "p_mwu": np.nan, "p_perm": np.nan,
    }
    if arm_vals.size < 2 or ctrl_vals.size < 2:
        return res
    d = cliffs_delta(arm_vals, ctrl_vals)
    res["cliffs_delta"] = d
    res["effect"] = _cliffs_interp(d)
    res["cliffs_ci_low"], res["cliffs_ci_high"] = bootstrap_cliffs_ci(arm_vals, ctrl_vals, rng)
    try:
        u, p = stats.mannwhitneyu(arm_vals, ctrl_vals, alternative="two-sided")
        res["U"], res["p_mwu"] = float(u), float(p)
    except ValueError:
        pass
    try:
        def _stat(x, y):
            return np.median(x) - np.median(y)
        pt = stats.permutation_test(
            (arm_vals, ctrl_vals), _stat, permutation_type="independent",
            n_resamples=5000, alternative="two-sided", rng=rng,
        )
        res["p_perm"] = float(pt.pvalue)
    except Exception as exc:  # noqa: BLE001
        logger.debug("permutation_test failed: %s", exc)
    return res


def run_stats(summary: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """For each metric: each treatment arm vs Control, pooled and within-batch.

    The blocking unit ("batch") is ``(sample_id, plate_id)`` — NOT plate_id
    alone, because the physical MaxWell chip serial (plate_id, e.g. T003346) is
    reused across different cultures/experiments (CX118, CX138, ...). The
    biologically meaningful contemporaneous-control block is the sample_id.

    Also a Kruskal-Wallis omnibus across all arms (incl. Control) per metric,
    pooled. BH-FDR is applied across all pairwise arm-vs-control MWU p-values.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    summary = summary.copy()
    summary["batch"] = summary["sample_id"] + "/" + summary["plate_id"]

    for metric, gm in summary.groupby("metric"):
        controls = gm[gm["role"] == "control"]
        arms = sorted(gm[gm["role"] != "control"]["canonical_group"].unique())

        # Kruskal-Wallis omnibus (pooled) across all canonical groups
        groups = [grp["response"].dropna().to_numpy() for _, grp in gm.groupby("canonical_group")
                  if grp["response"].notna().sum() >= 2]
        p_kw = np.nan
        if len(groups) >= 2:
            try:
                p_kw = float(stats.kruskal(*groups).pvalue)
            except ValueError:
                pass

        for arm in arms:
            arm_all = gm[gm["canonical_group"] == arm]["response"].dropna().to_numpy()
            ctrl_all = controls["response"].dropna().to_numpy()
            r = compare_arm_vs_control(arm_all, ctrl_all, rng)
            rows.append({"metric": metric, "arm": arm, "scope": "pooled",
                         "batch": "ALL", "p_kw_omnibus": p_kw, **r})

            # within-batch: only (sample_id, plate_id) blocks where both present
            for batch in sorted(gm[gm["canonical_group"] == arm]["batch"].unique()):
                a = gm[(gm["canonical_group"] == arm) & (gm["batch"] == batch)]["response"].dropna().to_numpy()
                c = controls[controls["batch"] == batch]["response"].dropna().to_numpy()
                if a.size == 0 or c.size == 0:
                    continue
                rp = compare_arm_vs_control(a, c, rng)
                rows.append({"metric": metric, "arm": arm, "scope": "batch",
                             "batch": batch, "p_kw_omnibus": np.nan, **rp})

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df
    # BH-FDR across pooled MWU p-values (the primary family)
    stats_df["q_bh"] = np.nan
    pooled_mask = (stats_df["scope"] == "pooled") & stats_df["p_mwu"].notna()
    if pooled_mask.any():
        q = stats.false_discovery_control(stats_df.loc[pooled_mask, "p_mwu"].to_numpy(), method="bh")
        stats_df.loc[pooled_mask, "q_bh"] = q
    return stats_df


# --------------------------------------------------------------------------- #
# Figures / report
# --------------------------------------------------------------------------- #
def build_report(long_df: pd.DataFrame, summary: pd.DataFrame, stats_df: pd.DataFrame,
                 metrics: list[str], out_html: Path, out_png_dir: Path) -> None:
    import plotly.graph_objects as go
    import plotly.express as px

    figs_html: list[str] = []

    # ---- (1) tau trajectories: group-median vs tau, control vs arms ----------
    for metric in metrics:
        sub = long_df[long_df["tau"].notna()].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub[sub[metric].notna()]
        if sub.empty:
            continue
        fig = go.Figure()
        for grp, gdf in sub.groupby("canonical_group"):
            agg = (gdf.groupby("tau")[metric]
                   .agg(["median", "count", "sem"]).reset_index().sort_values("tau"))
            fig.add_trace(go.Scatter(
                x=agg["tau"], y=agg["median"], mode="lines+markers", name=str(grp),
                error_y=dict(type="data", array=agg["sem"].fillna(0), visible=True),
                hovertemplate=f"{grp}<br>tau=%{{x}}d<br>{metric}=%{{y:.3g}}<extra></extra>",
            ))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(title=f"Trajectory — {metric} (group median ± SEM vs days since treatment)",
                          xaxis_title="tau = days since treatment", yaxis_title=metric,
                          template="plotly_white", height=380)
        figs_html.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # ---- (2) per-well response boxplots by arm -------------------------------
    if not summary.empty:
        for metric in metrics:
            sub = summary[summary["metric"] == metric]
            if sub.empty:
                continue
            fig = px.box(sub, x="canonical_group", y="response", color="role",
                         points="all", hover_data=["well_name", "plate_id", "baseline", "post_median"],
                         title=f"Per-well normalised response — {metric}")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(template="plotly_white", height=380,
                              yaxis_title="response (log2 FC or post−pre)")
            figs_html.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # ---- (3) effect-size forest (pooled Cliff's delta ± CI) ------------------
    if not stats_df.empty:
        pooled = stats_df[stats_df["scope"] == "pooled"].dropna(subset=["cliffs_delta"]).copy()
        if not pooled.empty:
            pooled["label"] = pooled["metric"] + " · " + pooled["arm"]
            pooled = pooled.sort_values(["metric", "arm"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pooled["cliffs_delta"], y=pooled["label"], mode="markers",
                error_x=dict(type="data",
                             array=(pooled["cliffs_ci_high"] - pooled["cliffs_delta"]).abs(),
                             arrayminus=(pooled["cliffs_delta"] - pooled["cliffs_ci_low"]).abs()),
                marker=dict(size=9),
                text=[f"q={q:.3g}" if pd.notna(q) else "" for q in pooled["q_bh"]],
                hovertemplate="%{y}<br>δ=%{x:.3f}<br>%{text}<extra></extra>",
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(title="Effect size (Cliff's delta, arm vs Control, pooled) — 95% bootstrap CI",
                              xaxis_title="Cliff's delta", template="plotly_white",
                              height=max(360, 26 * len(pooled)))
            figs_html.append(fig.to_html(full_html=False, include_plotlyjs=False))
            # PNG forest
            _forest_png(pooled, out_png_dir / "effect_size_forest.png")

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
        "<title>Treatment-group comparison</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}"
        "h1{font-size:20px}.fig{margin-bottom:28px}</style></head><body>"
        "<h1>Treatment-group comparison (longitudinal, baseline-normalised)</h1>"
        "<p>Response = log2 fold-change (ratio metrics) or post−pre (difference metrics), "
        "each well normalised to its own pre-treatment baseline (paired mode), or the "
        "absolute post-window median (cross-sectional mode). Unit of analysis = well. "
        "BH-FDR (q) is applied to the pooled arm-vs-Control family; within-batch results are "
        "exploratory and unadjusted.</p>"
        + "".join(f"<div class='fig'>{h}</div>" for h in figs_html)
        + "</body></html>"
    )
    out_html.write_text(doc)
    logger.info("Wrote report -> %s", out_html)


def _forest_png(pooled: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(pooled)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * n)))
    y = np.arange(n)
    xerr = np.vstack([
        (pooled["cliffs_delta"] - pooled["cliffs_ci_low"]).abs().to_numpy(),
        (pooled["cliffs_ci_high"] - pooled["cliffs_delta"]).abs().to_numpy(),
    ])
    ax.errorbar(pooled["cliffs_delta"], y, xerr=xerr, fmt="o", color="#3366cc", capsize=3)
    ax.axvline(0, ls="--", color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels(pooled["label"], fontsize=8)
    ax.set_xlabel("Cliff's delta (arm vs Control, pooled)")
    ax.set_title("Effect size — 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote forest PNG -> %s", out_path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True,
                   help="pipeline_config.json (read for analysis_root + figure_root).")
    p.add_argument("--emit-template", type=Path, default=None,
                   help="Write the (sample,plate,group) template CSV to this path and exit.")
    p.add_argument("--treatment-map", type=Path, default=None,
                   help="Filled template CSV (required unless --emit-template).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output dir (default <figure_root>/treatment_comparison).")
    p.add_argument("--metrics", nargs="*", default=list(METRIC_SPECS.keys()),
                   help="Subset of metrics to analyse.")
    p.add_argument("--mode", choices=["paired", "cross_sectional"], default="paired",
                   help="paired = within-well baseline normalisation (needs pre-treatment "
                        "recordings); cross_sectional = absolute post-window value (fallback "
                        "when no baseline exists).")
    p.add_argument("--scan-type", default="Network")
    p.add_argument("--burst-dirname", default="burst_detection_data",
                   help="Source of network-bursting metrics. 'burst_detection_data' = "
                        "traditional detector; 'ml_burst_data_umap' = ML detector.")
    p.add_argument("--ml-dirname", default="ml_burst_data_umap",
                   help="Source of ML burst-typing diagnostics.")
    p.add_argument("--curation-dirname", default="curation_data")
    p.add_argument("--baseline-window", type=float, default=None,
                   help="Max days before treatment to use as baseline (default: all pre).")
    p.add_argument("--post-window", type=float, default=None,
                   help="Max days after treatment to include as post (default: all post).")
    p.add_argument("--min-pre", type=int, default=1)
    p.add_argument("--min-post", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="Cap #metrics.json read (dry run).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s — %(message)s")
    analysis_root, figure_root = _load_config(args.config)
    experiment_cache = _load_experiment_cache(analysis_root)

    if args.emit_template is not None:
        emit_template(experiment_cache, args.emit_template)
        return 0

    if args.treatment_map is None:
        p.error("--treatment-map is required (or use --emit-template to create one).")

    unknown = [m for m in args.metrics if m not in METRIC_SPECS]
    if unknown:
        p.error(f"unknown metrics: {unknown}. Choose from {list(METRIC_SPECS)}")

    out_dir = args.output_dir or (figure_root / "treatment_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "png"
    png_dir.mkdir(exist_ok=True)

    group_lut = _group_lookup(experiment_cache)
    long_df = build_long_table(analysis_root, group_lut, burst_dirname=args.burst_dirname,
                               ml_dirname=args.ml_dirname, curation_dirname=args.curation_dirname,
                               scan_type=args.scan_type, limit=args.limit)
    if long_df.empty:
        raise SystemExit("no well-recordings assembled; check analysis_root / scan_type")

    template = load_template(args.treatment_map)
    long_df = apply_template(long_df, template)
    if long_df.empty:
        raise SystemExit("no rows mapped by template; check sample_id/plate_id/raw_groupname match")

    long_df.to_csv(out_dir / "tidy_long.csv", index=False)

    summary = per_well_summary(long_df, args.metrics, args.baseline_window,
                               args.post_window, args.min_pre, args.min_post, mode=args.mode)
    summary.to_csv(out_dir / "per_well_summary.csv", index=False)
    logger.info("mode=%s — response is %s", args.mode,
                "within-well log2 FC / post−pre (baseline-normalised)" if args.mode == "paired"
                else "absolute post-window median (no baseline)")
    if summary.empty:
        msg = ("per-well summary is empty. In paired mode this usually means wells have no "
               "PRE-treatment recordings — re-run with --mode cross_sectional (no baseline "
               "needed), or check treatment_date entries/windows."
               if args.mode == "paired"
               else "per-well summary is empty — check treatment_date entries and windows.")
        logger.warning("%s Stats/figures skipped.", msg)
        return 0

    stats_df = run_stats(summary, seed=args.seed)
    stats_df.to_csv(out_dir / "stats.csv", index=False)

    build_report(long_df, summary, stats_df, args.metrics,
                 out_dir / "report.html", png_dir)

    # console headline: significant pooled arms
    if not stats_df.empty:
        sig = stats_df[(stats_df["scope"] == "pooled") & (stats_df["q_bh"] < 0.05)]
        logger.info("=== %d pooled arm×metric comparisons at q<0.05 ===", len(sig))
        for _, r in sig.sort_values("q_bh").iterrows():
            logger.info("  %-24s %-16s δ=%+.2f (%s) q=%.3g  n=%d vs %d",
                        r["metric"], r["arm"], r["cliffs_delta"], r["effect"],
                        r["q_bh"], r["n_arm"], r["n_ctrl"])
    logger.info("Done. Outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
