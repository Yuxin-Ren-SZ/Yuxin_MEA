"""Longitudinal QC of Network-scan electrode selection for one (sample, chip, well).

A MaxTwo well carries 26,400 electrodes but only ~1,000 can be routed to an
amplifier at once, so **every Network run re-picks its own subset**. Nothing
else in the pipeline checks whether that subset persists across a culture's
life, yet every longitudinal analysis downstream implicitly assumes some
continuity of the recording sites. This module measures it.

Extraction is deliberately cheap: only ``settings/mapping`` and
``groups/routed/channels`` are read, never ``groups/routed/raw``, so a 20 GB
Network file costs ~0.4 s for all 24 wells.

Three measurement decisions, each forced by what the real data does (verified on
``CX138/T003346/well000``, 22 Network scans):

* **Identity overlap needs a spatially-matched null to be readable.** The mean
  all-pair Jaccard is 0.088 — which looks like noise until you compare it with a
  null that holds each scan's own spatial clustering fixed, against which it is
  hugely enriched. Selection is reproducibly biased toward the same *sites*
  while almost never re-picking the same *electrode*. Reporting either number
  alone gives the wrong verdict, so both are always emitted.
* **Proximity is not the rescue it looks like.** At 17.5 um pitch it is tempting
  to call two nearby-but-different electrodes "the same site", and dilating each
  selection does raise raw overlap (0.080 -> 0.164 at k=1 -> 0.255 at k=2). But
  the matched chance level rises at least as fast, so enrichment *falls*:
  6.2x at k=0, 3.1x at k=1, 2.1x at k=2. Identity, not proximity, is where the
  signal is; dilated overlap is a secondary line, never the headline.
* **Operator config reuse contaminates the decay fit.** Runs tagged
  ``"Using 015 as base"`` reproduce an earlier selection exactly. On the well
  above the two identical pairs land at lag 0 and lag 1 — precisely the bins an
  exponential fit leans on — inflating the half-life by ~1.6x. Reuse is
  detected two independent ways (identical sets, and the run tag itself) and
  excluded from every primary statistic.

The null is computed **exactly, not by sampling**: the 2-D circular
cross-correlation of two boolean masks yields the intersection count for all
26,400 toroidal shifts in a single FFT, so the full null distribution is free
and no ``--n-shuffles`` knob is needed.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from yuxin_mea.analysis.activity_scan import (
    GRID_COLS,
    GRID_ROWS,
    N_ELECTRODES,
    PITCH_UM,
    _gini,
)

logger = logging.getLogger(__name__)

SCAN_TYPE = "Network"
DEFAULT_OUTPUT_SUBDIR = "electrode_selection_qc"

#: The stated experiment window in treatment days (mirrors
#: ``scripts/report/trajectory.EXPERIMENT_RANGE``).
EXPERIMENT_RANGE = (0, 21)

#: Fractions of scans defining the "core" electrode sets reported per well.
CORE_LEVELS = (0.5, 0.8, 1.0)

#: Dilation radii (in electrodes) for the secondary spatial-tolerance overlap.
DILATE_LEVELS = (1, 2)

#: MaxWell run tags name the run a configuration was copied from, e.g.
#: ``"Run #000019 Using 015 as base 15H"``. This is a mechanism-level reuse
#: signal, unlike the threshold-dependent identical-set test.
_BASE_RUN_RE = re.compile(r"using\s+0*(\d+)\s+as\s+base", re.IGNORECASE)

STABILITY_FILE = "stability.json"
COUNTS_FILE = "counts.npz"


class ElectrodeSelectionError(ValueError):
    """Raised when a file or well carries no usable routed-electrode set."""


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def make_well_uid(sample_id: str, plate_id: str, well_id: str) -> str:
    """``sample|plate|well`` — the longitudinal key.

    ``plate_id`` alone is *not* unique: ``T003346`` is reused across CX118,
    CX138, CX157 and CX169 in this dataset, so ``sample_id`` must never be
    dropped from a key or an output path.
    """
    return f"{sample_id}|{plate_id}|{well_id}"


def split_well_uid(uid: str) -> tuple[str, str, str]:
    parts = str(uid).split("|")
    if len(parts) != 3:
        raise ValueError(f"malformed well_uid: {uid!r}")
    return parts[0], parts[1], parts[2]


def parse_yymmdd(value: str | None) -> datetime | None:
    """Parse a 6-digit ``YYMMDD`` (2000-2099), else None.

    Same convention as ``scripts/compare_treatment_groups._parse_yymmdd`` so
    dates agree with the treatment map and the report layer.
    """
    if value is None:
        return None
    s = str(value).strip()
    if len(s) != 6 or not s.isdigit():
        return None
    try:
        return datetime.strptime("20" + s, "%Y%m%d")
    except ValueError:
        return None


def parse_base_run(tag: str | None) -> str | None:
    """Run id a configuration was copied from, from the recording tag."""
    if not tag:
        return None
    m = _BASE_RUN_RE.search(str(tag))
    if not m:
        return None
    return f"{int(m.group(1)):06d}"


def day_delta(a: datetime | None, b: datetime | None) -> int | None:
    if a is None or b is None:
        return None
    return int((a - b).days)


# --------------------------------------------------------------------------- #
# Stage 1 — extraction
# --------------------------------------------------------------------------- #
def node_electrodes(node: Any) -> np.ndarray:
    """Routed electrode ids for one ``data_store`` node, sorted and unique.

    The channel list and the mapping table agree on every healthy file checked,
    but an aborted run need not be healthy, so the *intersection* is kept rather
    than trusting ``mapping`` wholesale. A large drop is logged rather than
    silently swallowed.
    """
    try:
        mapping = node["settings"]["mapping"][:]
    except (KeyError, OSError, TypeError):
        return np.empty(0, dtype=np.int64)
    try:
        channel = np.asarray(mapping["channel"], dtype=np.int64)
        electrode = np.asarray(mapping["electrode"], dtype=np.int64)
    except (KeyError, IndexError, ValueError):
        return np.empty(0, dtype=np.int64)

    try:
        routed = np.asarray(node["groups"]["routed"]["channels"][:], dtype=np.int64)
    except (KeyError, OSError, TypeError, ValueError):
        routed = None

    if routed is not None and routed.size:
        keep = np.isin(channel, routed)
        if keep.sum() < 0.5 * channel.size:
            logger.warning(
                "routed/mapping mismatch: %d of %d mapped channels are routed",
                int(keep.sum()), int(channel.size))
        electrode = electrode[keep]

    electrode = electrode[(electrode >= 0) & (electrode < N_ELECTRODES)]
    return np.unique(electrode)


def read_selection(h5_path: str | Path,
                   well_indices: Iterable[int] | None = None) -> dict[int, np.ndarray]:
    """``{well_index: sorted electrode ids}`` for one Network file.

    Reads metadata only — the multi-GB ``groups/routed/raw`` store is never
    touched. A Network well maps to exactly one ``data_store`` node in every
    file checked, but nodes are unioned anyway so a multi-config file degrades
    to "everything this well ever routed" instead of silently keeping one node.
    """
    import h5py

    wanted = None if well_indices is None else {int(w) for w in well_indices}
    out: dict[int, np.ndarray] = {}
    with h5py.File(str(h5_path), "r") as h5:
        store = h5.get("data_store")
        if store is None:
            raise ElectrodeSelectionError(f"no /data_store in {h5_path}")
        for name in store:
            node = store[name]
            try:
                well_index = int(node["well_id"][0])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if wanted is not None and well_index not in wanted:
                continue
            electrodes = node_electrodes(node)
            if electrodes.size == 0:
                continue
            prev = out.get(well_index)
            out[well_index] = (electrodes if prev is None
                               else np.union1d(prev, electrodes))
    if not out:
        raise ElectrodeSelectionError(f"no routed electrodes found in {h5_path}")
    return out


def iter_network_entries(entries: dict, *,
                         samples: set[str] | None = None,
                         plates: set[str] | None = None,
                         recordings: set[str] | None = None) -> Iterator[tuple[str, dict]]:
    """Network entries of ``experiment_cache.json`` matching the filters."""
    for key, entry in sorted(entries.items()):
        if not isinstance(entry, dict) or entry.get("scan_type") != SCAN_TYPE:
            continue
        if recordings and key not in recordings:
            continue
        if samples and entry.get("sample_id") not in samples:
            continue
        if plates and entry.get("plate_id") not in plates:
            continue
        yield key, entry


def _entry_well_meta(entry: dict, well_id: str) -> dict:
    wells = entry.get("wells") or {}
    rec = wells.get(well_id) or {}
    meta = rec.get("metadata") if isinstance(rec, dict) else None
    return meta if isinstance(meta, dict) else {}


def _plating_date_from_meta(meta: dict) -> str | None:
    """``'13.3.2026'`` -> ``'260313'``; None when absent or unparseable."""
    raw = str(meta.get("Plating Date") or "").strip()
    if not raw:
        return None
    for sep in (".", "/", "-"):
        if sep in raw:
            parts = [p for p in raw.split(sep) if p]
            if len(parts) == 3:
                try:
                    d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    return None
                return f"{y % 100:02d}{m:02d}{d:02d}"
    return None


def extract_selection_index(entries: dict, data_root: str | Path, *,
                            samples: set[str] | None = None,
                            plates: set[str] | None = None,
                            recordings: set[str] | None = None,
                            wells: set[int] | None = None) -> tuple[Any, Any]:
    """Read every in-scope Network file into ``(index_df, scans_df)``.

    ``index_df`` is one row per (scan, electrode); ``scans_df`` one row per
    (scan, well) carrying the dates, group and reuse tag.
    """
    import pandas as pd

    data_root = Path(data_root)
    index_rows: list[dict] = []
    scan_rows: list[dict] = []
    n_files = n_skipped = 0

    for key, entry in iter_network_entries(entries, samples=samples, plates=plates,
                                           recordings=recordings):
        path = data_root / str(entry.get("data_path", ""))
        try:
            size = path.stat().st_size
        except OSError as exc:
            logger.warning("skipping %s: %s", key, exc)
            n_skipped += 1
            continue
        # One recording in this dataset is a zero-byte stub; opening it would
        # kill the sweep, so it is skipped with a warning (same guard as
        # scripts/run_activity_scan.discover_jobs).
        if size == 0:
            logger.warning("skipping %s: zero-size %s", key, path)
            n_skipped += 1
            continue

        try:
            selection = read_selection(path, wells)
        except Exception as exc:  # noqa: BLE001 — one bad file must not stop the sweep
            logger.warning("skipping %s: %s", key, exc)
            n_skipped += 1
            continue
        n_files += 1

        meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        tag = str(meta.get("tag") or "")
        base_run = parse_base_run(tag)
        sample_id = str(entry.get("sample_id") or "")
        plate_id = str(entry.get("plate_id") or "")
        date = str(entry.get("date") or "")
        run_id = str(entry.get("run_id") or "")

        rec_of_well: dict[str, str] = {}
        for rec_name, well_list in (entry.get("h5_recordings") or {}).items():
            for w in well_list or []:
                rec_of_well[str(w)] = str(rec_name)

        for well_index in sorted(selection):
            well_id = f"well{well_index:03d}"
            electrodes = selection[well_index]
            uid = make_well_uid(sample_id, plate_id, well_id)
            wmeta = _entry_well_meta(entry, well_id)
            scan_rows.append({
                "recording_key": key,
                "sample_id": sample_id,
                "plate_id": plate_id,
                "date": date,
                "run_id": run_id,
                "rec_name": rec_of_well.get(well_id, ""),
                "well_id": well_id,
                "well_uid": uid,
                "n_routed": int(electrodes.size),
                "raw_groupname": str(wmeta.get("groupname") or ""),
                "well_name": str(wmeta.get("well_name") or ""),
                "plating_date_meta": _plating_date_from_meta(wmeta) or "",
                "tag": tag,
                "base_run": base_run or "",
            })
            index_rows.append({
                "well_uid": uid,
                "recording_key": key,
                "electrode": electrodes,
            })

    logger.info("read %d Network file(s); %d skipped", n_files, n_skipped)

    if index_rows:
        index_df = pd.DataFrame({
            "well_uid": np.repeat([r["well_uid"] for r in index_rows],
                                  [r["electrode"].size for r in index_rows]),
            "recording_key": np.repeat([r["recording_key"] for r in index_rows],
                                       [r["electrode"].size for r in index_rows]),
            "electrode": np.concatenate([r["electrode"] for r in index_rows]),
        })
        index_df["electrode"] = index_df["electrode"].astype(np.int32)
        index_df["row"] = (index_df["electrode"] // GRID_COLS).astype(np.int16)
        index_df["col"] = (index_df["electrode"] % GRID_COLS).astype(np.int16)
    else:
        index_df = pd.DataFrame(columns=["well_uid", "recording_key", "electrode",
                                         "row", "col"])
    scans_df = pd.DataFrame(scan_rows)
    return index_df, scans_df


# --------------------------------------------------------------------------- #
# Stage 2 — window resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WindowSpec:
    """What the user asked for."""
    kind: str = "tau"                 # tau | date | div | all
    lo: float | None = EXPERIMENT_RANGE[0]
    hi: float | None = EXPERIMENT_RANGE[1]

    def label(self) -> str:
        if self.kind == "all":
            return "all scans"
        unit = {"tau": "treatment day", "div": "DIV", "date": "date"}[self.kind]
        return f"{unit} {self.lo:g}–{self.hi:g}" if self.kind != "date" else \
            f"date {int(self.lo)}–{int(self.hi)}"


@dataclass(frozen=True)
class Window:
    """What could actually be resolved for one well."""
    kind: str
    lo: float | None
    hi: float | None
    anchor: str                       # exact | plate | unanchored | n/a
    treatment_date: str | None = None
    plating_date: str | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind, "lo": self.lo, "hi": self.hi,
                "anchor": self.anchor, "treatment_date": self.treatment_date,
                "plating_date": self.plating_date, "note": self.note}


def load_treatment_map(path: str | Path) -> dict[tuple[str, str, str], dict]:
    """``treatment_map.csv`` -> ``{(sample, plate, raw_groupname): row}``.

    Same key and columns as ``scripts/compare_treatment_groups.load_template``.
    """
    out: dict[tuple[str, str, str], dict] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                key = (row["sample_id"].strip(), row["plate_id"].strip(),
                       row["raw_groupname"].strip())
            except KeyError:
                continue
            out[key] = {
                "canonical_group": (row.get("canonical_group")
                                    or row.get("raw_groupname") or "").strip(),
                "role": (row.get("role") or "").strip().lower(),
                "plating_date": (row.get("plating_date") or "").strip(),
                "treatment_date": (row.get("treatment_date") or "").strip(),
            }
    return out


def _plate_fallback(tmap: dict, sample_id: str, plate_id: str,
                    field_name: str) -> str | None:
    """Earliest ``field_name`` across every group of one (sample, plate)."""
    dates = []
    for (s, p, _g), row in tmap.items():
        if s != sample_id or p != plate_id:
            continue
        d = parse_yymmdd(row.get(field_name))
        if d is not None:
            dates.append((d, row[field_name]))
    if not dates:
        return None
    return min(dates, key=lambda t: t[0])[1]


def resolve_window(spec: WindowSpec, tmap: dict, sample_id: str, plate_id: str,
                   raw_groupname: str, *,
                   plating_fallback: str | None = None) -> Window:
    """Turn a requested window into one a given well can actually be judged by.

    The anchor is always recorded, so a number is never silently produced off a
    guessed treatment date:

    1. exact ``(sample, plate, raw_groupname)`` hit in the treatment map;
    2. else the earliest anchor on that ``(sample, plate)``;
    3. else no anchor at all — fall back to all scans, marked ``unanchored``.
    """
    row = tmap.get((sample_id, plate_id, raw_groupname))
    treatment = (row or {}).get("treatment_date") or None
    plating = (row or {}).get("plating_date") or None
    anchor = "exact" if row else "missing"

    if spec.kind in ("all", "date"):
        return Window(spec.kind, spec.lo, spec.hi, "n/a",
                      treatment or None, plating or plating_fallback,
                      note=("every scan requested" if spec.kind == "all"
                            else "explicit date range requested"))

    field_name = "treatment_date" if spec.kind == "tau" else "plating_date"
    value = treatment if spec.kind == "tau" else plating
    if not value or parse_yymmdd(value) is None:
        value = _plate_fallback(tmap, sample_id, plate_id, field_name)
        anchor = "plate" if value else "unanchored"
    if (not value or parse_yymmdd(value) is None) and spec.kind == "div":
        # The plating date is also stamped on each well's own metadata, which
        # covers plates the treatment map never listed.
        value = plating_fallback
        anchor = "plate" if value else "unanchored"

    if not value or parse_yymmdd(value) is None:
        return Window("all", None, None, "unanchored",
                      treatment or None, plating or plating_fallback,
                      note=f"no {field_name} for this well; using all scans")

    if spec.kind == "tau":
        treatment = value
    else:
        plating = value
    return Window(spec.kind, spec.lo, spec.hi, anchor,
                  treatment or None, plating or plating_fallback)


def well_groupname(scans: Sequence[dict]) -> tuple[str, list[str]]:
    """The well's group, plus every distinct name it carried.

    A well's ``groupname`` is per-*recording* metadata and really does change
    mid-experiment: on ``CX138/well000`` the first five scans read ``Control``
    and everything from the treatment date on reads ``NPH``. Wells appear to be
    relabelled from a placeholder when they are treated, so the **most recent**
    non-empty name is the well's true group — taking the first would look the
    treatment map up under the placeholder and silently lose the anchor.
    """
    seen: list[str] = []
    for s in scans:
        name = str(s.get("raw_groupname") or "").strip()
        if name and name not in seen:
            seen.append(name)
    latest = ""
    for s in reversed(list(scans)):
        name = str(s.get("raw_groupname") or "").strip()
        if name:
            latest = name
            break
    return latest, seen


def annotate_scans(scans: Sequence[dict], window: Window) -> list[dict]:
    """Attach ``tau`` / ``DIV`` to each scan, wherever the anchors allow it."""
    t0 = parse_yymmdd(window.treatment_date)
    p0 = parse_yymmdd(window.plating_date)
    out = []
    for s in scans:
        d = parse_yymmdd(s.get("date"))
        row = dict(s)
        row["tau"] = day_delta(d, t0)
        row["DIV"] = day_delta(d, p0)
        out.append(row)
    return out


def select_window(scans: Sequence[dict], window: Window) -> list[dict]:
    """Scans falling inside ``window``, in chronological order."""
    rows = annotate_scans(scans, window)
    if window.kind == "all":
        keep = rows
    elif window.kind == "date":
        lo, hi = int(window.lo), int(window.hi)
        keep = [r for r in rows
                if r.get("date") and lo <= int(r["date"]) <= hi]
    else:
        key = "tau" if window.kind == "tau" else "DIV"
        keep = [r for r in rows
                if r.get(key) is not None and window.lo <= r[key] <= window.hi]
    return sorted(keep, key=lambda r: (str(r.get("date")), str(r.get("run_id"))))


def collapse_same_day(scans: Sequence[dict], mode: str,
                      sets: dict[str, np.ndarray]) -> tuple[list[dict], dict]:
    """Optionally fold repeat runs on one date into a single scan.

    Same-date repeats are real and informative (mean J 0.305 at lag 0 vs 0.088
    overall on the reference well), so the default is to keep them.
    """
    if mode == "no":
        return list(scans), sets
    by_date: dict[str, list[dict]] = {}
    for s in scans:
        by_date.setdefault(str(s.get("date")), []).append(s)
    out_scans, out_sets = [], {}
    for date in sorted(by_date):
        group = by_date[date]
        head = dict(group[0])
        if mode == "union":
            merged = np.unique(np.concatenate(
                [sets[g["recording_key"]] for g in group]))
            head["run_id"] = "+".join(g["run_id"] for g in group)
            head["n_routed"] = int(merged.size)
            head["collapsed_from"] = [g["recording_key"] for g in group]
            out_sets[head["recording_key"]] = merged
        else:  # "first"
            head["collapsed_from"] = [g["recording_key"] for g in group]
            out_sets[head["recording_key"]] = sets[head["recording_key"]]
        out_scans.append(head)
    return out_scans, out_sets


# --------------------------------------------------------------------------- #
# Stage 3 — metrics
# --------------------------------------------------------------------------- #
def electrode_mask(electrodes: np.ndarray) -> np.ndarray:
    """Boolean ``(120, 220)`` mask of a selection."""
    mask = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    e = np.asarray(electrodes, dtype=np.int64)
    e = e[(e >= 0) & (e < N_ELECTRODES)]
    mask[e // GRID_COLS, e % GRID_COLS] = True
    return mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Identity Jaccard between two electrode-id arrays."""
    sa, sb = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    union = len(sa | sb)
    if union == 0:
        return float("nan")
    return len(sa & sb) / union


def dilate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    """8-connected dilation by ``k`` electrodes (identity when ``k <= 0``)."""
    if k <= 0:
        return mask
    from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure

    structure = iterate_structure(generate_binary_structure(2, 2), k)
    return binary_dilation(mask, structure=structure)


def shift_null(mask_a: np.ndarray, mask_b: np.ndarray,
               fft_a: np.ndarray | None = None,
               fft_b: np.ndarray | None = None) -> dict:
    """Exact toroidal-shift null for the overlap of two masks.

    Shifting one mask over the torus preserves its count *and* its internal
    spatial clustering, which a uniform-random null does not — and clustering is
    exactly what would otherwise masquerade as reproducible site selection.

    The 2-D circular cross-correlation gives the intersection count at every one
    of the 26,400 shifts in a single FFT, so the whole null distribution is
    available rather than a sample of it. The observed value is the zero shift
    and is excluded from the null.

    An exact percentile is reported rather than a z-score: shifting a clustered
    mask by a little still overlaps it heavily, so the null carries a heavy right
    tail and is nowhere near Gaussian.
    """
    a = np.asarray(mask_a, dtype=float)
    b = np.asarray(mask_b, dtype=float)
    na, nb = float(a.sum()), float(b.sum())
    if na == 0 or nb == 0:
        return {"observed": float("nan"), "p": float("nan"),
                "enrichment": float("nan"), "null_median": float("nan")}

    fa = np.fft.rfft2(a) if fft_a is None else fft_a
    fb = np.fft.rfft2(b) if fft_b is None else fft_b
    # corr[dr, dc] == |A ∩ roll(B, (dr, dc))|
    corr = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
    inter = np.rint(corr)                     # counts, not floats
    np.clip(inter, 0.0, min(na, nb), out=inter)
    union = na + nb - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, np.nan)

    observed = float(jac[0, 0])
    null = np.delete(jac.ravel(), 0)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {"observed": observed, "p": float("nan"),
                "enrichment": float("nan"), "null_median": float("nan")}
    median = float(np.median(null))
    p = float((1 + int(np.sum(null >= observed))) / (null.size + 1))
    enrichment = float(observed / median) if median > 0 else float("inf")
    return {"observed": observed, "p": p, "enrichment": enrichment,
            "null_median": median}


def _exp_half_life(lags: np.ndarray, values: np.ndarray) -> float:
    """Half-life (days) of an exponential fitted to a lag-decay curve."""
    lags = np.asarray(lags, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(lags) & np.isfinite(values) & (values > 0)
    if ok.sum() < 3:
        return float("nan")
    slope, _intercept = np.polyfit(lags[ok], np.log(values[ok]), 1)
    if slope >= 0:
        return float("inf")
    return float(np.log(2.0) / -slope)


def _lag_curve(pairs: Sequence[dict], *, dedup: bool) -> dict:
    """Mean Jaccard binned by whole-day lag, plus the fitted half-life."""
    bins: dict[int, list[float]] = {}
    for p in pairs:
        if dedup and p.get("reuse"):
            continue
        lag = p.get("lag_days")
        j = p.get("jaccard")
        if lag is None or j is None or not np.isfinite(j):
            continue
        bins.setdefault(int(lag), []).append(float(j))
    lags = sorted(bins)
    means = [float(np.mean(bins[k])) for k in lags]
    counts = [len(bins[k]) for k in lags]
    return {
        "lag_days": lags,
        "mean_jaccard": means,
        "n_pairs": counts,
        "half_life_days": _exp_half_life(np.array(lags), np.array(means)),
    }


def _verdict(stability: dict, min_scans: int) -> tuple[str, str]:
    """``(band, sentence)`` — withheld below ``min_scans`` in-window scans."""
    n = stability["n_scans"]
    if n < min_scans:
        return "insufficient", (
            f"Insufficient scans (n={n} < {min_scans}) in this window — "
            "no stability verdict.")

    core80 = stability["core_fraction"].get("0.8", 0.0)
    half_life = stability["lag_decay_dedup"]["half_life_days"]
    hl = half_life if np.isfinite(half_life) else 0.0
    mean_j = stability["mean_jaccard_dedup"]

    if core80 >= 0.5 and hl >= 21:
        band = "stable"
        text = ("Selection is stable: a majority of routed electrodes recur in "
                "at least 80% of scans.")
    elif core80 >= 0.2 or hl >= 14:
        band = "partial"
        text = ("Selection is partially stable: a persistent core exists but "
                "most electrodes turn over across the window.")
    elif core80 >= 0.05 or hl >= 7:
        band = "weak"
        text = ("Selection is weakly stable: only a small core recurs, and "
                "overlap decays within about a week.")
    else:
        band = "unstable"
        text = ("Selection is effectively re-routed each scan: no persistent "
                "electrode core, and overlap decays within days.")

    enrich = stability.get("shift_null", {}).get("median_enrichment")
    if enrich is not None and np.isfinite(enrich) and enrich > 1.5:
        text += (f" Against a spatially-matched null the overlap is still "
                 f"{enrich:.1f}x enriched, so scans revisit the same *regions* "
                 f"even while individual electrodes change (mean identity "
                 f"Jaccard {mean_j:.3f}).")
    return band, text


@dataclass
class WellSelectionQC:
    """Everything the renderers and the report need for one well."""
    well_uid: str
    sample_id: str
    plate_id: str
    well_id: str
    window: Window
    window_spec: WindowSpec
    scans: list[dict]
    sets: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    counts: np.ndarray = field(repr=False,
                               default_factory=lambda: np.zeros((GRID_ROWS, GRID_COLS),
                                                                dtype=np.int32))
    pairs: list[dict] = field(default_factory=list, repr=False)
    stability: dict = field(default_factory=dict)
    n_scans_total: int = 0

    @property
    def n_scans(self) -> int:
        return len(self.scans)


def compute_stability(well_uid: str, scans: Sequence[dict],
                      sets: dict[str, np.ndarray], *,
                      window: Window,
                      window_spec: WindowSpec | None = None,
                      min_scans: int = 3,
                      dilate_levels: Sequence[int] = DILATE_LEVELS,
                      n_scans_total: int | None = None) -> WellSelectionQC:
    """All stability metrics for one well over its in-window scans."""
    sample_id, plate_id, well_id = split_well_uid(well_uid)
    scans = list(scans)
    keys = [s["recording_key"] for s in scans]
    n = len(scans)

    counts = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
    masks: list[np.ndarray] = []
    for k in keys:
        m = electrode_mask(sets[k])
        masks.append(m)
        counts += m

    res = WellSelectionQC(
        well_uid=well_uid, sample_id=sample_id, plate_id=plate_id, well_id=well_id,
        window=window, window_spec=window_spec or WindowSpec(),
        scans=scans, sets={k: sets[k] for k in keys}, counts=counts,
        n_scans_total=n_scans_total if n_scans_total is not None else n,
    )

    n_routed = np.array([int(sets[k].size) for k in keys], dtype=float)
    per_electrode = counts[counts > 0]
    union = int(per_electrode.size)

    core_counts = {f"{lvl:g}": int(np.sum(counts >= max(1, int(np.ceil(lvl * n)))))
                   for lvl in CORE_LEVELS} if n else {f"{lvl:g}": 0 for lvl in CORE_LEVELS}
    median_routed = float(np.median(n_routed)) if n_routed.size else 0.0
    core_fraction = {k: (v / median_routed if median_routed > 0 else float("nan"))
                     for k, v in core_counts.items()}

    # Pairwise: identity Jaccard, exact shift null, dilated overlap. FFTs are
    # cached per scan so the cost is one inverse transform per pair.
    ffts = [np.fft.rfft2(m.astype(float)) for m in masks]
    dil_masks = {k: [dilate_mask(m, k) for m in masks] for k in dilate_levels}
    dil_ffts = {k: [np.fft.rfft2(m.astype(float)) for m in v]
                for k, v in dil_masks.items()}

    dates = [parse_yymmdd(s.get("date")) for s in scans]
    base_runs = [str(s.get("base_run") or "") for s in scans]
    run_ids = [str(s.get("run_id") or "") for s in scans]

    pairs: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            j_id = jaccard(sets[keys[i]], sets[keys[j]])
            identical = bool(j_id == 1.0)
            # A base config with a small edit gives J ~ 0.9 and slips past the
            # identical-set test, so the run tag is checked independently.
            declared = bool((base_runs[j] and base_runs[j] == run_ids[i])
                            or (base_runs[i] and base_runs[i] == run_ids[j])
                            or (base_runs[i] and base_runs[i] == base_runs[j]))
            null = shift_null(masks[i], masks[j], ffts[i], ffts[j])
            row = {
                "i": i, "j": j,
                "recording_i": keys[i], "recording_j": keys[j],
                "date_i": scans[i].get("date"), "date_j": scans[j].get("date"),
                "lag_days": day_delta(dates[j], dates[i]),
                "jaccard": j_id,
                "identical": identical,
                "declared_base": declared,
                "reuse": bool(identical or declared),
                "null_p": null["p"],
                "null_enrichment": null["enrichment"],
                "null_median": null["null_median"],
            }
            for k in dilate_levels:
                dn = shift_null(dil_masks[k][i], dil_masks[k][j],
                                dil_ffts[k][i], dil_ffts[k][j])
                row[f"jaccard_dil{k}"] = dn["observed"]
                row[f"enrichment_dil{k}"] = dn["enrichment"]
            pairs.append(row)
    res.pairs = pairs

    def _mean(field_name: str, dedup: bool) -> float:
        vals = [p[field_name] for p in pairs
                if (not dedup or not p["reuse"]) and np.isfinite(p[field_name])]
        return float(np.mean(vals)) if vals else float("nan")

    adjacent = [p for p in pairs if p["j"] == p["i"] + 1]
    adj_dedup = [p["jaccard"] for p in adjacent if not p["reuse"]]

    centroids = []
    for m in masks:
        rows, cols = np.nonzero(m)
        if rows.size:
            centroids.append((float(cols.mean() * PITCH_UM),
                              float(rows.mean() * PITCH_UM)))
    centroid_drift = float("nan")
    if len(centroids) >= 2:
        c = np.array(centroids)
        centroid_drift = float(np.hypot(*(c[-1] - c[0])))

    enrichments = [p["null_enrichment"] for p in pairs
                   if not p["reuse"] and np.isfinite(p["null_enrichment"])]
    p_values = [p["null_p"] for p in pairs if not p["reuse"] and np.isfinite(p["null_p"])]

    stability = {
        "well_uid": well_uid,
        "sample_id": sample_id,
        "plate_id": plate_id,
        "well_id": well_id,
        "n_scans": n,
        "n_scans_total": res.n_scans_total,
        "n_routed_min": float(n_routed.min()) if n_routed.size else float("nan"),
        "n_routed_median": median_routed,
        "n_routed_max": float(n_routed.max()) if n_routed.size else float("nan"),
        "union_electrodes": union,
        "union_fraction": union / N_ELECTRODES,
        "n_never_selected": N_ELECTRODES - union,
        "core_counts": core_counts,
        "core_fraction": core_fraction,
        "gini": _gini(per_electrode.astype(float)) if union else float("nan"),
        "mean_jaccard_dedup": _mean("jaccard", True),
        "mean_jaccard_raw": _mean("jaccard", False),
        "median_jaccard_dedup": float(np.median(
            [p["jaccard"] for p in pairs if not p["reuse"]])) if any(
            not p["reuse"] for p in pairs) else float("nan"),
        "adjacent_jaccard_dedup": float(np.mean(adj_dedup)) if adj_dedup else float("nan"),
        "n_pairs": len(pairs),
        "n_identical_pairs": int(sum(p["identical"] for p in pairs)),
        "n_declared_base_pairs": int(sum(p["declared_base"] for p in pairs)),
        "n_reuse_pairs": int(sum(p["reuse"] for p in pairs)),
        "shift_null": {
            "median_enrichment": float(np.median(enrichments)) if enrichments else float("nan"),
            "median_p": float(np.median(p_values)) if p_values else float("nan"),
            "frac_p_below_0.01": (float(np.mean(np.array(p_values) < 0.01))
                                  if p_values else float("nan")),
        },
        "dilated": {
            f"k{k}": {
                "mean_jaccard": _mean(f"jaccard_dil{k}", True),
                "median_enrichment": float(np.median(
                    [p[f"enrichment_dil{k}"] for p in pairs
                     if not p["reuse"] and np.isfinite(p[f"enrichment_dil{k}"])]))
                if any(not p["reuse"] and np.isfinite(p[f"enrichment_dil{k}"])
                       for p in pairs) else float("nan"),
            } for k in dilate_levels
        },
        "centroid_drift_um": centroid_drift,
        "lag_decay_dedup": _lag_curve(pairs, dedup=True),
        "lag_decay_raw": _lag_curve(pairs, dedup=False),
        # Consecutive-scan overlaps only — small enough to keep in the JSON so
        # the scan table and the report can show "J vs previous" without
        # re-reading pairs.csv.
        "adjacent_pairs": [
            {"i": p["i"], "j": p["j"], "jaccard": p["jaccard"],
             "lag_days": p["lag_days"], "reuse": p["reuse"]}
            for p in adjacent
        ],
    }
    band, sentence = _verdict(stability, min_scans)
    stability["verdict"] = band
    stability["verdict_text"] = sentence
    stability["min_scans"] = min_scans
    res.stability = stability
    return res


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def output_root(analysis_root: str | Path, override: str | None = None) -> Path:
    if override:
        return Path(override)
    return Path(analysis_root) / DEFAULT_OUTPUT_SUBDIR


def well_dir(root: str | Path, sample_id: str, plate_id: str, well_id: str) -> Path:
    return Path(root) / sample_id / plate_id / well_id


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_well_qc(res: WellSelectionQC, out_dir: str | Path) -> Path:
    """Persist one well's QC: ``stability.json`` + ``counts.npz`` + ``pairs.csv``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "well_uid": res.well_uid,
        "sample_id": res.sample_id,
        "plate_id": res.plate_id,
        "well_id": res.well_id,
        "window": res.window.as_dict(),
        "window_request": {"kind": res.window_spec.kind,
                           "lo": res.window_spec.lo, "hi": res.window_spec.hi},
        "scans": res.scans,
        "stability": res.stability,
    }
    (out / STABILITY_FILE).write_text(json.dumps(_jsonable(payload), indent=1))

    electrodes = np.nonzero(res.counts.ravel())[0].astype(np.int32)
    np.savez_compressed(
        out / COUNTS_FILE,
        counts=res.counts.astype(np.int32),
        electrode=electrodes,
        count=res.counts.ravel()[electrodes].astype(np.int32),
        n_scans=np.array([res.n_scans], dtype=np.int32),
    )

    if res.pairs:
        import pandas as pd
        pd.DataFrame(res.pairs).to_csv(out / "pairs.csv", index=False)
    return out


def load_well_qc(out_dir: str | Path) -> dict | None:
    """Read back ``stability.json`` (+ the count grid), or None if absent."""
    d = Path(out_dir)
    path = d / STABILITY_FILE
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None
    npz = d / COUNTS_FILE
    if npz.exists():
        try:
            with np.load(npz) as z:
                payload["counts"] = z["counts"]
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("could not read %s: %s", npz, exc)
    return payload


# --------------------------------------------------------------------------- #
# Visualization 2 — selection-history animation
# --------------------------------------------------------------------------- #
def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    v = value.lstrip("#")
    return tuple(int(v[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _valid_video(path: Path) -> bool:
    """True if ``path`` is a readable video (catches a missing moov atom)."""
    import subprocess

    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001 — ffprobe missing/timeout => treat as invalid
        return False
    return r.returncode == 0 and r.stdout.strip() not in ("", "0", "N/A")


def history_frames(res: WellSelectionQC, *, decay_days: float = 7.0,
                   alpha_min: float = 0.06,
                   past_color: str = "#4a4a4a",
                   current_color: str = "#b2182b") -> np.ndarray:
    """``(n_scans, 120, 220, 3)`` RGB frames: current scan over faded history.

    The fade is a property of the **scan**, not of the electrode: every
    electrode of a past scan is drawn at one alpha set by how many *days* that
    scan sits before the current one, which is what "greyed out by temporal
    distance" means. Frames composite oldest-first so a recent scan overwrites
    an older one where they share an electrode.
    """
    n = res.n_scans
    frames = np.ones((max(n, 1), GRID_ROWS, GRID_COLS, 3), dtype=np.float32)
    if n == 0:
        return frames

    masks = [electrode_mask(res.sets[s["recording_key"]]) for s in res.scans]
    dates = [parse_yymmdd(s.get("date")) for s in res.scans]
    past_rgb = np.array(_hex_to_rgb(past_color), dtype=np.float32)
    cur_rgb = np.array(_hex_to_rgb(current_color), dtype=np.float32)

    for i in range(n):
        canvas = frames[i]
        for j in range(i):                       # oldest first
            gap = day_delta(dates[i], dates[j])
            # Fall back to index distance when a date is unparseable, so a bad
            # date degrades the fade rather than dropping the scan.
            gap = abs(gap) if gap is not None else (i - j)
            alpha = alpha_min + (1.0 - alpha_min) * float(
                np.exp(-gap / max(decay_days, 1e-6)))
            m = masks[j]
            canvas[m] = canvas[m] * (1.0 - alpha) + past_rgb * alpha
        canvas[masks[i]] = cur_rgb               # current scan, fully opaque
    return frames


def render_history_animation(res: WellSelectionQC, out_path: str | Path, *,
                             decay_days: float = 7.0, fps: int = 12,
                             hold_frames: int = 12, fmt: str = "mp4",
                             dpi: int = 110) -> Path:
    """Vis 2 — one held frame per scan, past selections fading with day distance.

    Each scan is held for ``hold_frames`` frames instead of writing at ~1 fps:
    a 1 fps MP4 is technically correct but scrubs badly in most players.

    Written to a sibling temp file and validated before being moved into place,
    so an interrupted ffmpeg can never leave an unplayable file that looks
    finished; a GIF is written instead when ffmpeg is missing or fails.
    """
    import os

    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if res.n_scans == 0:
        raise ElectrodeSelectionError(f"{res.well_uid}: no scans to animate")

    frames = history_frames(res, decay_days=decay_days)
    extent = [0, GRID_COLS * PITCH_UM, 0, GRID_ROWS * PITCH_UM]

    fig = Figure(figsize=(7.6, 4.8), dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, facecolor="white")
    im = ax.imshow(frames[0], origin="lower", aspect="equal",
                   interpolation="nearest", extent=extent)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    title = ax.set_title("", fontsize=10)
    caption = fig.text(0.01, 0.01, "", fontsize=8, color="#84807a")

    adjacent = {p["j"]: p["jaccard"]
                for p in (res.stability.get("adjacent_pairs") or [])}

    def _label(i: int) -> tuple[str, str]:
        s = res.scans[i]
        bits = [f"{s.get('date')}", f"run {s.get('run_id')}"]
        if s.get("tau") is not None:
            bits.append(f"D{s['tau']:+d}")
        if s.get("DIV") is not None:
            bits.append(f"DIV {s['DIV']}")
        head = f"{res.well_uid}  ·  " + "  ·  ".join(bits)
        j = adjacent.get(i)
        foot = (f"scan {i + 1}/{res.n_scans} · {s.get('n_routed')} routed"
                + (f" · J vs previous {j:.3f}" if j is not None and np.isfinite(j)
                   else "")
                + f" · grey = past scans, fading over {decay_days:g} d")
        return head, foot

    def update(frame: int):
        i = min(frame // max(hold_frames, 1), res.n_scans - 1)
        im.set_data(frames[i])
        head, foot = _label(i)
        title.set_text(head)
        caption.set_text(foot)
        return im, title, caption

    n_frames = res.n_scans * max(hold_frames, 1)
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000.0 / max(fps, 1), blit=False)

    if fmt == "gif":
        gif = out_path.with_suffix(".gif")
        anim.save(str(gif), writer=PillowWriter(fps=fps), dpi=dpi)
        return gif

    tmp = out_path.with_name(out_path.stem + ".tmp" + out_path.suffix)
    try:
        anim.save(str(tmp), writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
        if not _valid_video(tmp):
            raise RuntimeError("ffmpeg produced an unreadable file (no moov atom?)")
        os.replace(tmp, out_path)
        return out_path
    except Exception as exc:  # noqa: BLE001 — ffmpeg missing/failed/incomplete
        tmp.unlink(missing_ok=True)
        gif = out_path.with_suffix(".gif")
        logger.warning("ffmpeg failed (%s); falling back to GIF %s", exc, gif.name)
        anim.save(str(gif), writer=PillowWriter(fps=fps), dpi=dpi)
        return gif
