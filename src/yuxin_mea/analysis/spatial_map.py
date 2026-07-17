"""Spatial activity maps and network-burst propagation on the electrode plane.

New computation (see ``doc/spatial_connectivity_plan.md`` §A). Two products per
physical well:

1. **Activity field** — a Gaussian-smoothed 2-D map of per-unit firing rate over
   the electrode plane, plus a spatial-concentration scalar (Gini of the field).
2. **Burst propagation** — for each detected network burst, each participating
   unit's first-spike latency from burst onset is fit to a plane
   ``latency ≈ a·x + b·y + c``; the planar gradient gives a propagation speed,
   direction, and origin, summarised per well (median speed / planarity).

House style mirrors :mod:`yuxin_mea.analysis.burst_detector` /
:mod:`ml_burst_detector`: a frozen ``SpatialMapConfig`` with ``from_task_params``,
a ``SpatialMapResults`` container, a ``compute_spatial_map`` entry taking the
``dict[unit_id -> np.ndarray(seconds)]`` spike-times contract, and a
``SpatialMapError`` for insufficient input.

Unit locations come from the curation ``quality_metrics.pkl`` (``loc_x``/``loc_y``
columns) — the only source existing code populates (``unit_locations.npy`` has no
reader in the repo; treat it as a fallback the task may pass in). Firing rate is
the ``firing_rate`` column of the same frame.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, fields as _dc_fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


class SpatialMapError(ValueError):
    """Raised when unit positions / spikes are insufficient for a spatial map."""


@dataclass(frozen=True)
class SpatialMapConfig:
    """Tunable parameters for the spatial activity map + burst propagation."""

    bins: int = 64
    sigma_um: float = 50.0
    # Fraction of the field's max used to define the "active" support (area frac).
    active_thresh_frac: float = 0.05
    # A network burst is used for propagation only if at least this many units
    # both have a position and fire inside the burst window (plane fit needs ≥3).
    min_participation: int = 5

    @classmethod
    def from_task_params(cls, params: dict) -> "SpatialMapConfig":
        """Build a config from a task-params dict; cast by default type, ignore extras."""
        kwargs: dict = {}
        for f in _dc_fields(cls):
            if f.name not in params:
                continue
            v = params[f.name]
            if isinstance(f.default, bool):
                kwargs[f.name] = bool(v)
            elif isinstance(f.default, int):
                kwargs[f.name] = int(v)
            elif isinstance(f.default, float):
                kwargs[f.name] = float(v)
            elif isinstance(f.default, str):
                kwargs[f.name] = str(v)
            else:
                kwargs[f.name] = v
        return cls(**kwargs)


@dataclass
class SpatialMapResults:
    """Structured output of :func:`compute_spatial_map`.

    ``activity_field`` is a ``(bins, bins)`` normalised (sum=1) firing-rate map.
    ``burst_propagation`` has one row per network burst used for the plane fit
    (columns: ``burst_index, origin_x, origin_y, speed_um_per_ms, direction_rad,
    r2_planar_fit, n_participating, delay_ms``); empty if no usable burst.
    ``metrics`` are the well-level scalars (schema = :func:`spatial_summary`).
    ``diagnostics`` records ``n_units, bins, sigma_um, n_bursts_used`` and the
    field extent needed to place the map on the electrode plane.
    """

    activity_field: np.ndarray
    burst_propagation: pd.DataFrame
    metrics: dict
    diagnostics: dict


# ---------------------------------------------------------------------------
# Positions / field
# ---------------------------------------------------------------------------
_PROP_COLUMNS = [
    "burst_index", "origin_x", "origin_y", "speed_um_per_ms",
    "direction_rad", "r2_planar_fit", "n_participating", "delay_ms",
]


def unit_positions(quality_metrics: pd.DataFrame) -> np.ndarray:
    """Return ``(n, 2)`` µm positions (``loc_x``, ``loc_y``) for the frame's rows.

    Row order follows ``quality_metrics.index`` (unit ids). Raises if the columns
    are absent so the caller fails loudly rather than mapping bursts onto NaNs.
    """
    if "loc_x" not in quality_metrics.columns or "loc_y" not in quality_metrics.columns:
        raise SpatialMapError(
            "quality_metrics is missing loc_x/loc_y columns; cannot place units "
            "on the electrode plane."
        )
    return quality_metrics[["loc_x", "loc_y"]].to_numpy(dtype=float)


def activity_field(
    pos: np.ndarray,
    firing_rate: np.ndarray,
    *,
    bins: int = 64,
    sigma_um: float = 50.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Gaussian-smoothed 2-D histogram of firing rate over the electrode plane.

    Returns ``(field, extent)`` where ``field`` is ``(bins, bins)`` normalised to
    sum 1 and ``extent = (xmin, xmax, ymin, ymax)`` in µm (imshow convention).
    ``sigma_um`` is converted to bins via the per-axis bin width.
    """
    pos = np.asarray(pos, dtype=float)
    firing_rate = np.asarray(firing_rate, dtype=float)
    if pos.shape[0] == 0:
        raise SpatialMapError("no unit positions for activity field")
    xmin, ymin = pos.min(axis=0)
    xmax, ymax = pos.max(axis=0)
    # Guard degenerate extent (all units at one point / colinear) with a pad.
    if xmax - xmin < 1e-6:
        xmin, xmax = xmin - 1.0, xmax + 1.0
    if ymax - ymin < 1e-6:
        ymin, ymax = ymin - 1.0, ymax + 1.0
    hist, xedges, yedges = np.histogram2d(
        pos[:, 0], pos[:, 1], bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
        weights=np.clip(firing_rate, 0.0, None),
    )
    bin_w_x = (xmax - xmin) / bins
    bin_w_y = (ymax - ymin) / bins
    sig_x = sigma_um / bin_w_x if bin_w_x > 0 else 0.0
    sig_y = sigma_um / bin_w_y if bin_w_y > 0 else 0.0
    field = gaussian_filter(hist, sigma=(sig_x, sig_y), mode="constant")
    total = field.sum()
    if total > 0:
        field = field / total
    # histogram2d indexes [x, y]; transpose so field[row=y, col=x] for imshow.
    return field.T, (float(xmin), float(xmax), float(ymin), float(ymax))


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative 1-D array (0 = uniform, →1 = concentrated)."""
    x = np.sort(np.asarray(values, dtype=float).ravel())
    x = x[x >= 0]
    if x.size == 0 or x.sum() == 0:
        return 0.0
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2.0 * cum.sum() / cum[-1]) / n)


# ---------------------------------------------------------------------------
# Burst propagation
# ---------------------------------------------------------------------------
def _first_latencies(
    spikes: dict,
    units: list,
    pos_by_unit: dict,
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First-spike latency (s) after ``start`` and positions for participating units."""
    lat, xy = [], []
    for u in units:
        st = spikes.get(u)
        if st is None or u not in pos_by_unit:
            continue
        in_win = st[(st >= start) & (st < end)]
        if in_win.size == 0:
            continue
        lat.append(float(in_win[0]) - start)
        xy.append(pos_by_unit[u])
    return np.asarray(lat, dtype=float), np.asarray(xy, dtype=float)


def _plane_fit(lat_s: np.ndarray, xy: np.ndarray) -> tuple[float, float, float, float, float]:
    """Fit latency(ms) ≈ a·x + b·y + c. Returns (speed_um_per_ms, dir_rad, r2, ox, oy)."""
    lat_ms = lat_s * 1e3
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(xy.shape[0])])
    coef, *_ = np.linalg.lstsq(A, lat_ms, rcond=None)
    a, b, _c = coef
    pred = A @ coef
    ss_res = float(np.sum((lat_ms - pred) ** 2))
    ss_tot = float(np.sum((lat_ms - lat_ms.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    grad = float(np.hypot(a, b))  # ms per µm
    speed = 1.0 / grad if grad > 1e-12 else float("nan")  # µm per ms
    direction = float(np.arctan2(b, a))
    origin_idx = int(np.argmin(lat_s))
    return speed, direction, float(r2), float(xy[origin_idx, 0]), float(xy[origin_idx, 1])


def burst_propagation(
    spikes: dict,
    pos_by_unit: dict,
    bursts: pd.DataFrame,
    *,
    min_participation: int = 5,
) -> pd.DataFrame:
    """Per-burst planar propagation fit. One row per burst with ≥ min_participation
    positioned, firing units (and ≥3, required for a plane)."""
    if bursts is None or bursts.empty or "start" not in bursts.columns:
        return pd.DataFrame(columns=_PROP_COLUMNS)
    units = list(spikes.keys())
    rows = []
    for bi, (_, b) in enumerate(bursts.iterrows()):
        start, end = float(b["start"]), float(b["end"])
        lat, xy = _first_latencies(spikes, units, pos_by_unit, start, end)
        if lat.size < max(3, min_participation):
            continue
        speed, direction, r2, ox, oy = _plane_fit(lat, xy)
        rows.append({
            "burst_index": bi,
            "origin_x": ox,
            "origin_y": oy,
            "speed_um_per_ms": speed,
            "direction_rad": direction,
            "r2_planar_fit": r2,
            "n_participating": int(lat.size),
            "delay_ms": float((lat.max() - lat.min()) * 1e3),
        })
    if not rows:
        return pd.DataFrame(columns=_PROP_COLUMNS)
    return pd.DataFrame(rows, columns=_PROP_COLUMNS)


def spatial_summary(
    field: np.ndarray,
    prop: pd.DataFrame,
    *,
    active_thresh_frac: float = 0.05,
) -> dict:
    """Well-level scalars from the field + propagation table."""
    field = np.asarray(field, dtype=float)
    fmax = float(field.max()) if field.size else 0.0
    active_frac = (
        float(np.mean(field >= active_thresh_frac * fmax)) if fmax > 0 else 0.0
    )
    # Field-weighted centroid in bin coordinates (row=y, col=x).
    if field.sum() > 0:
        ys, xs = np.indices(field.shape)
        cx = float((xs * field).sum() / field.sum())
        cy = float((ys * field).sum() / field.sum())
    else:
        cx = cy = float("nan")
    speeds = prop["speed_um_per_ms"].to_numpy(float) if len(prop) else np.array([])
    speeds = speeds[np.isfinite(speeds)]
    r2s = prop["r2_planar_fit"].to_numpy(float) if len(prop) else np.array([])
    delays = prop["delay_ms"].to_numpy(float) if len(prop) else np.array([])
    return {
        "active_area_frac": active_frac,
        "activity_centroid": [cx, cy],
        "activity_gini": _gini(field),
        "mean_prop_speed_um_ms": float(np.median(speeds)) if speeds.size else float("nan"),
        "prop_planarity": float(np.median(r2s)) if r2s.size else float("nan"),
        "mean_prop_delay_ms": float(np.median(delays)) if delays.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Entry point + writer
# ---------------------------------------------------------------------------
def compute_spatial_map(
    spike_times: dict,
    quality_metrics: pd.DataFrame,
    bursts: pd.DataFrame | None = None,
    config: SpatialMapConfig | None = None,
) -> SpatialMapResults:
    """Compute the activity field + burst propagation for one well.

    ``spike_times`` : ``{unit_id -> np.ndarray(spike times, seconds)}`` (curated).
    ``quality_metrics`` : DataFrame indexed by unit_id with ``loc_x, loc_y,
    firing_rate`` columns (curation output).
    ``bursts`` : network-burst table (``start``/``end`` s). Missing/empty → the
    field is still produced and propagation is skipped (``n_bursts_used=0``).
    """
    cfg = config or SpatialMapConfig()
    if not spike_times:
        raise SpatialMapError("no spike trains provided")

    # Restrict to curated units present in both the spikes and the metrics frame.
    units = [u for u in spike_times if u in quality_metrics.index]
    if len(units) == 0:
        raise SpatialMapError(
            "no overlap between spike-train unit ids and quality_metrics index"
        )
    qm = quality_metrics.loc[units]
    pos = unit_positions(qm)
    if "firing_rate" in qm.columns:
        fr = qm["firing_rate"].to_numpy(dtype=float)
    else:
        # Fall back to empirical rate = n_spikes / span.
        spans = []
        for u in units:
            st = np.asarray(spike_times[u], float)
            spans.append(st.size)
        fr = np.asarray(spans, dtype=float)
    fr = np.nan_to_num(fr, nan=0.0)

    field, extent = activity_field(
        pos, fr, bins=cfg.bins, sigma_um=cfg.sigma_um
    )

    pos_by_unit = {u: pos[i] for i, u in enumerate(units)}
    spikes_arr = {u: np.asarray(spike_times[u], dtype=float) for u in units}
    prop = burst_propagation(
        spikes_arr, pos_by_unit, bursts, min_participation=cfg.min_participation
    )

    metrics = spatial_summary(field, prop, active_thresh_frac=cfg.active_thresh_frac)
    diagnostics = {
        "n_units": int(len(units)),
        "bins": int(cfg.bins),
        "sigma_um": float(cfg.sigma_um),
        "n_bursts_total": int(0 if bursts is None else len(bursts)),
        "n_bursts_used": int(len(prop)),
        "field_extent": [float(v) for v in extent],
    }
    return SpatialMapResults(
        activity_field=field,
        burst_propagation=prop,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def _atomic_json_write(data: dict, dest: Path) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, dest)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_spatial_map(results: SpatialMapResults, output_dir: Path) -> None:
    """Persist a ``SpatialMapResults`` bundle (mirrors PickleBurstOutputWriter).

    Layout::

        activity_field.npy       (bins, bins) normalised field
        burst_propagation.parquet  one row per burst
        spatial_metrics.json     well-level scalars (spatial_summary keys)
        diagnostics.json         n_units, bins, sigma_um, n_bursts_used, extent
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "activity_field.npy", results.activity_field)
    results.burst_propagation.to_parquet(output_dir / "burst_propagation.parquet")
    _atomic_json_write(results.metrics, output_dir / "spatial_metrics.json")
    _atomic_json_write(results.diagnostics, output_dir / "diagnostics.json")
