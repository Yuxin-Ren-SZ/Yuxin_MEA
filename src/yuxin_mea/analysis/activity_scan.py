"""Per-electrode activity maps for a MaxTwo ActivityScan recording.

An ActivityScan is the checkerboard sweep that precedes a Network run. A well
carries 26,400 electrodes but only ~1,000 can be routed to an amplifier at once,
so the assay cycles through a set of configurations (14 on most recordings, 29
on the denser ones) and each well must be *assembled* across them:

    data.raw.h5
      └── data_store/dataNNNN         ← one (well, configuration) pair
            ├── well_id               ← physical well 0…23 (global, not per-rec)
            ├── settings/mapping      ← (channel, electrode, x_um, y_um) per routed channel
            ├── settings/lsb          ← µV per LSB
            └── groups/routed/raw     ← (n_channels, n_samples) uint16, 30 s @ 10 kHz

There is no spike sorting here — activity is measured directly from the traces
by MAD-threshold peak detection, which is what an ActivityScan is for (the
instrument uses the same idea to pick the 1,020 electrodes a Network run keeps).

Two deliberate choices in the signal path:

* **Detection is causal, amplitude is zero-phase.** The store is streamed
  block-wise through a causal ``sosfilt`` (carrying filter state across blocks)
  so memory stays at one block rather than the 1.2 GB a whole filtered store
  would take. But a causal highpass shifts and flattens the trough — measured
  −33 µV where the zero-phase trough of the same spikes was −44 µV — so every
  detected event's amplitude is re-measured by ``sosfiltfilt`` on a short raw
  snippet around it. Detection cost stays streaming; amplitudes stay in
  calibrated µV comparable with the (zero-phase) Network pipeline.
* **Noise is estimated on the first block** (10 s of the 30 s by default) and
  used both as the detection threshold and as the reported ``noise_uv``, so the
  threshold a well was detected with is always the number the noise map shows.
  A MAD over 100k samples per channel is stable; a whole-trace estimate would
  need either a second filtering pass or the memory this design avoids.

Electrode geometry is exact and verified on the raw files: pitch 17.5 µm,
grid 220 × 120, and ``electrode_id == row * 220 + col``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Array geometry (MaxTwo well). Verified against settings/mapping on the raw h5:
# x ∈ {0, 17.5, …, 3832.5} (220 columns), y ∈ {0, …, 2082.5} (120 rows), and
# every electrode id equals row * 220 + col.
GRID_COLS = 220
GRID_ROWS = 120
PITCH_UM = 17.5
N_ELECTRODES = GRID_ROWS * GRID_COLS  # 26,400

MAP_NAMES = ("firing_rate", "amp_median", "noise", "coverage")


class ActivityScanError(ValueError):
    """Raised when a well carries no usable configuration in the file."""


@dataclass
class ActivityScanConfig:
    """Detection + assembly parameters. All of these land in the params hash."""

    highpass_hz: float = 300.0
    filter_order: int = 3
    thresh_mad: float = 5.0          # threshold = -thresh_mad * (median|x| / 0.6745)
    refractory_ms: float = 1.0       # dead time between accepted crossings
    align_ms: float = 2.0            # trough searched within ±align_ms of the crossing
    snippet_pad_ms: float = 5.0      # raw context each side for the zero-phase re-measure
    block_samples: int = 100_000     # streaming block (raw chunk time dim is 200)
    noise_block_stride: int = 4      # subsample stride for the first-block MAD
    active_min_rate_hz: float = 0.1  # default "active electrode" cut (viewer can re-threshold)

    @classmethod
    def from_task_params(cls, p: dict) -> "ActivityScanConfig":
        f = {k: p[k] for k in cls.__dataclass_fields__ if k in p}
        return cls(**{k: type(getattr(cls(), k))(v) for k, v in f.items()})

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class WellActivityResults:
    electrodes: "object"                  # DataFrame, one row per unique electrode
    maps: dict[str, np.ndarray]           # (GRID_ROWS, GRID_COLS) arrays, NaN off-coverage
    stats: dict
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# File structure
# --------------------------------------------------------------------------- #
def well_store_nodes(h5: Any, well_index: int) -> list[str]:
    """Return ``data_store`` keys belonging to one physical well, in file order.

    Read from ``/data_store`` rather than the ``/wells/wellXXX`` convenience view:
    both index the same nodes on a healthy file, but ``data_store`` is the one
    structure every recording in this dataset was verified to carry, including
    the short/aborted ones (1 and 18 configurations).
    """
    store = h5["data_store"]
    out: list[str] = []
    for key in store:
        try:
            if int(store[key]["well_id"][0]) == int(well_index):
                out.append(key)
        except (KeyError, IndexError, TypeError):
            continue
    return out


def wells_in_file(h5: Any) -> list[int]:
    """Sorted physical well indices present in the file."""
    store = h5["data_store"]
    found: set[int] = set()
    for key in store:
        try:
            found.add(int(store[key]["well_id"][0]))
        except (KeyError, IndexError, TypeError):
            continue
    return sorted(found)


def well_label(well_index: int) -> str:
    """``0 -> 'well000'`` — the id form used by the rest of the pipeline."""
    return f"well{int(well_index):03d}"


def well_index_from_label(well_id: str) -> int:
    return int(str(well_id).replace("well", ""))


# --------------------------------------------------------------------------- #
# Per-configuration detection
# --------------------------------------------------------------------------- #
def _sos(cfg: ActivityScanConfig, fs: float):
    from scipy.signal import butter

    return butter(cfg.filter_order, cfg.highpass_hz, btype="highpass",
                  fs=fs, output="sos")


def _accept_with_refractory(idx: np.ndarray, refrac: int,
                            last: int = -(10 ** 12)) -> tuple[np.ndarray, int]:
    """Keep crossings ``refrac`` samples apart, continuing across block edges.

    ``last`` is the absolute index of the previously accepted crossing on this
    channel (from an earlier block), so an event that straddles a block boundary
    is not counted twice. Returns the accepted indices and the new ``last``.
    """
    if idx.size == 0:
        return idx, last
    keep = []
    for i in idx:
        i = int(i)
        if i - last >= refrac:
            keep.append(i)
            last = i
    return np.asarray(keep, dtype=np.int64), last


def _zero_phase_troughs(
    raw_uv: np.ndarray,      # (n_events, snippet_len) raw µV snippets, same channel
    sos,
    center: int,
    half: int,
) -> np.ndarray:
    """Zero-phase filter the snippets and return the trough near their centre."""
    from scipy.signal import sosfiltfilt

    if raw_uv.size == 0:
        return np.empty(0, dtype=np.float32)
    n = raw_uv.shape[1]
    # filtfilt needs padlen < n; snippets are built long enough, but stay safe
    padlen = min(3 * (2 * len(sos) + 1), max(0, n - 1))
    y = sosfiltfilt(sos, raw_uv, axis=1, padlen=padlen)
    lo = max(0, center - half)
    hi = min(n, center + half + 1)
    return y[:, lo:hi].min(axis=1).astype(np.float32)


def store_metrics(node: Any, cfg: ActivityScanConfig) -> dict:
    """Per-routed-channel activity for one (well, configuration) node.

    Returns arrays aligned to ``groups/routed/channels`` order:
    ``electrode, x_um, y_um, n_spikes, noise_uv, rms_uv, amp_median_uv,
    amp_p10_uv, amp_max_uv`` plus scalars ``duration_s``, ``fs``, ``n_samples``.
    """
    from scipy.signal import sosfilt, sosfilt_zi

    settings = node["settings"]
    lsb_uv = float(settings["lsb"][0]) * 1e6
    fs = float(settings["sampling"][0])
    mapping = settings["mapping"][:]
    routed = node["groups"]["routed"]
    channels = routed["channels"][:]
    raw = routed["raw"]
    n_ch, n_samples = raw.shape

    # channel -> (electrode, x, y); raw rows follow `channels`, not `mapping` order
    by_channel = {int(c): (int(e), float(x), float(y))
                  for c, e, x, y in zip(mapping["channel"], mapping["electrode"],
                                        mapping["x"], mapping["y"])}
    electrode = np.full(n_ch, -1, dtype=np.int64)
    x_um = np.full(n_ch, np.nan, dtype=np.float64)
    y_um = np.full(n_ch, np.nan, dtype=np.float64)
    for row, ch in enumerate(channels):
        hit = by_channel.get(int(ch))
        if hit is not None:
            electrode[row], x_um[row], y_um[row] = hit

    sos = _sos(cfg, fs)
    refrac = max(1, int(round(cfg.refractory_ms * fs / 1000.0)))
    half = max(1, int(round(cfg.align_ms * fs / 1000.0)))
    pad = max(half + 2, int(round(cfg.snippet_pad_ms * fs / 1000.0)))

    zi = None
    noise = None
    thr = None
    sumsq = np.zeros(n_ch, dtype=np.float64)
    counts = np.zeros(n_ch, dtype=np.int64)
    amps: list[list[np.ndarray]] = [[] for _ in range(n_ch)]
    # carried across blocks so a crossing at a block edge is neither split nor
    # double-counted: whether the channel was already below threshold, and the
    # absolute index of its last accepted event
    prev_below = np.zeros(n_ch, dtype=bool)
    last_accept = np.full(n_ch, -(10 ** 12), dtype=np.int64)
    tail_raw = np.zeros((n_ch, 0), dtype=np.float32)   # raw context across blocks
    tail_len = 0
    window = np.arange(-pad, pad + 1)

    for start in range(0, n_samples, cfg.block_samples):
        stop = min(n_samples, start + cfg.block_samples)
        x = raw[:, start:stop].astype(np.float32) * lsb_uv
        if zi is None:
            # scale the steady-state initial condition by the first sample so the
            # filter does not ring in from zero at t=0
            zi = sosfilt_zi(sos)[:, None, :] * x[:, 0][None, :, None]
        y, zi = sosfilt(sos, x, axis=1, zi=zi)
        y = y.astype(np.float32, copy=False)
        sumsq += np.einsum("ij,ij->i", y, y, dtype=np.float64)

        if noise is None:
            sample = np.abs(y[:, ::cfg.noise_block_stride])
            noise = (np.median(sample, axis=1) / 0.6745).astype(np.float64)
            noise[~np.isfinite(noise) | (noise <= 0)] = np.nan
            thr = np.where(np.isfinite(noise), -cfg.thresh_mad * noise,
                           -np.inf).astype(np.float32)

        below = y < thr[:, None]
        onset = np.zeros_like(below)
        onset[:, 1:] = below[:, 1:] & ~below[:, :-1]
        onset[:, 0] = below[:, 0] & ~prev_below
        prev_below = below[:, -1].copy()

        if onset.any():
            for ch in np.flatnonzero(onset.any(axis=1)):
                idx = np.flatnonzero(onset[ch]) + start
                idx, last_accept[ch] = _accept_with_refractory(
                    idx, refrac, int(last_accept[ch]))
                if idx.size == 0:
                    continue
                counts[ch] += idx.size
                # raw snippets for the zero-phase amplitude re-measure; the
                # previous block's tail supplies context for events near the edge
                ctx = np.concatenate([tail_raw[ch], x[ch]]) if tail_len else x[ch]
                base = start - tail_len
                rel = idx - base
                good = (rel >= pad) & (rel + pad < ctx.size)
                if not good.any():
                    continue
                snips = ctx[rel[good][:, None] + window[None, :]]
                amps[ch].append(_zero_phase_troughs(snips, sos, pad, half))

        tail_len = min(pad, x.shape[1])
        tail_raw = x[:, -tail_len:].copy() if tail_len else np.zeros((n_ch, 0), np.float32)

    duration_s = n_samples / fs
    rms = np.sqrt(sumsq / max(1, n_samples))
    amp_median = np.full(n_ch, np.nan)
    amp_p10 = np.full(n_ch, np.nan)
    amp_max = np.full(n_ch, np.nan)
    for ch, parts in enumerate(amps):
        if not parts:
            continue
        a = np.concatenate(parts)
        a = a[np.isfinite(a)]
        if a.size:
            amp_median[ch] = float(np.median(a))
            amp_p10[ch] = float(np.percentile(a, 10))
            amp_max[ch] = float(a.min())

    return {
        "electrode": electrode,
        "x_um": x_um,
        "y_um": y_um,
        "n_spikes": counts,
        "noise_uv": noise if noise is not None else np.full(n_ch, np.nan),
        "rms_uv": rms,
        "amp_median_uv": amp_median,
        "amp_p10_uv": amp_p10,
        "amp_max_uv": amp_max,
        "duration_s": duration_s,
        "fs": fs,
        "n_samples": int(n_samples),
        "n_channels": int(n_ch),
        "lsb_uv": lsb_uv,
    }


def online_spike_counts(node: Any) -> dict[int, int]:
    """Vendor's own threshold detector, counted per **electrode**.

    Carried as a cross-check column only: the instrument's ``amplitude`` field is
    not in µV (measured ratio ≈ 7.8 against the zero-phase trough), but the event
    times align with the raw, so the counts are a legitimate independent detector
    to compare ranks against.
    """
    try:
        spikes = node["spikes"][:]
        mapping = node["settings"]["mapping"][:]
    except (KeyError, OSError):
        return {}
    if len(spikes) == 0:
        return {}
    el_of_channel = {int(c): int(e)
                     for c, e in zip(mapping["channel"], mapping["electrode"])}
    out: dict[int, int] = {}
    for ch in spikes["channel"]:
        el = el_of_channel.get(int(ch))
        if el is not None:
            out[el] = out.get(el, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# Assembly across configurations
# --------------------------------------------------------------------------- #
def _aggregate(frames: "list", pd) -> "object":
    """Union the per-configuration tables into one row per electrode.

    ~153 of a well's ~13k electrodes are routed in two configurations. Every
    configuration is the same length, so any "keep the better one" rule would
    decay into an arbitrary first-wins; instead the duplicates are pooled:
    spikes and durations add, rate is the pooled rate, noise/rms average, and
    the amplitude quantiles are averaged weighted by each configuration's spike
    count (a pooled median of the underlying events is not recoverable without
    keeping every event).
    """
    df = pd.concat(frames, ignore_index=True)
    df = df[df["electrode"] >= 0]
    if df.empty:
        return df

    def _wmean(values, weights):
        v = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        m = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not m.any():
            return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        return float(np.sum(v[m] * w[m]) / np.sum(w[m]))

    rows = []
    for electrode, grp in df.groupby("electrode", sort=True):
        n_spikes = float(grp["n_spikes"].sum())
        duration = float(grp["duration_s"].sum())
        rows.append({
            "electrode": int(electrode),
            "x_um": float(grp["x_um"].iloc[0]),
            "y_um": float(grp["y_um"].iloc[0]),
            "row": int(round(float(grp["y_um"].iloc[0]) / PITCH_UM)),
            "col": int(round(float(grp["x_um"].iloc[0]) / PITCH_UM)),
            "n_configs": int(len(grp)),
            "n_spikes": int(n_spikes),
            "duration_s": duration,
            "firing_rate_hz": n_spikes / duration if duration > 0 else np.nan,
            "noise_uv": float(np.nanmean(grp["noise_uv"])),
            "rms_uv": float(np.nanmean(grp["rms_uv"])),
            "amp_median_uv": _wmean(grp["amp_median_uv"], grp["n_spikes"]),
            "amp_p10_uv": _wmean(grp["amp_p10_uv"], grp["n_spikes"]),
            "amp_max_uv": float(np.nanmin(grp["amp_max_uv"]))
            if np.isfinite(grp["amp_max_uv"]).any() else np.nan,
            "online_n_spikes": int(grp["online_n_spikes"].sum()),
        })
    return pd.DataFrame(rows).set_index("electrode").sort_index()


def grid_from_electrodes(electrodes: "object", column: str,
                         fill: float = np.nan) -> np.ndarray:
    """Scatter one electrode column onto the (120, 220) array grid."""
    grid = np.full((GRID_ROWS, GRID_COLS), fill, dtype=np.float64)
    if electrodes is None or len(electrodes) == 0:
        return grid
    rows = electrodes["row"].to_numpy()
    cols = electrodes["col"].to_numpy()
    vals = electrodes[column].to_numpy(dtype=float)
    ok = ((rows >= 0) & (rows < GRID_ROWS) & (cols >= 0) & (cols < GRID_COLS))
    grid[rows[ok], cols[ok]] = vals[ok]
    return grid


def _gini(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size == 0 or v.sum() <= 0:
        return float("nan")
    v = np.sort(v)
    n = v.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1.0) / n)


def _largest_active_component(active_grid: np.ndarray) -> int:
    """Size of the biggest 8-connected blob of active electrodes."""
    try:
        from scipy.ndimage import label
    except ImportError:  # pragma: no cover - scipy is a hard dep of the detector
        return -1
    if not active_grid.any():
        return 0
    lab, n = label(active_grid, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return 0
    return int(np.bincount(lab.ravel())[1:].max())


def well_stats(electrodes: "object", cfg: ActivityScanConfig,
               diagnostics: dict) -> dict:
    """Well-level summary. Both the observed and the coverage-corrected area."""
    n_scanned = int(len(electrodes))
    coverage_frac = n_scanned / N_ELECTRODES if n_scanned else 0.0
    if n_scanned == 0:
        return {"n_scanned": 0, "coverage_frac": 0.0, "n_active": 0,
                "active_frac": float("nan"), "active_area_mm2": float("nan"),
                "active_min_rate_hz": cfg.active_min_rate_hz}

    fr = electrodes["firing_rate_hz"].to_numpy(dtype=float)
    active = np.isfinite(fr) & (fr >= cfg.active_min_rate_hz)
    n_active = int(active.sum())
    amp = electrodes["amp_median_uv"].to_numpy(dtype=float)
    noise = electrodes["noise_uv"].to_numpy(dtype=float)

    x = electrodes["x_um"].to_numpy(dtype=float)
    y = electrodes["y_um"].to_numpy(dtype=float)
    w = np.where(np.isfinite(fr), fr, 0.0)
    if w.sum() > 0:
        cx, cy = float(np.sum(w * x) / w.sum()), float(np.sum(w * y) / w.sum())
    else:
        cx = cy = float("nan")

    active_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    rows = electrodes["row"].to_numpy()
    cols = electrodes["col"].to_numpy()
    ok = ((rows >= 0) & (rows < GRID_ROWS) & (cols >= 0) & (cols < GRID_COLS))
    active_grid[rows[ok & active], cols[ok & active]] = True

    # Observed count vs the array-extrapolated area: only ~49 % of the electrodes
    # are scanned, so the two answer different questions and both are reported.
    area_per_electrode_mm2 = (PITCH_UM ** 2) / 1e6
    return {
        "n_scanned": n_scanned,
        "coverage_frac": coverage_frac,
        "n_active": n_active,
        "active_frac": n_active / n_scanned,
        "active_area_mm2": (n_active / coverage_frac * area_per_electrode_mm2
                            if coverage_frac > 0 else float("nan")),
        "active_min_rate_hz": cfg.active_min_rate_hz,
        "largest_active_component": _largest_active_component(active_grid),
        "fr_mean_active_hz": float(np.nanmean(fr[active])) if n_active else float("nan"),
        "fr_median_active_hz": float(np.nanmedian(fr[active])) if n_active else float("nan"),
        "fr_mean_all_hz": float(np.nanmean(fr)),
        "fr_max_hz": float(np.nanmax(fr)) if np.isfinite(fr).any() else float("nan"),
        "total_rate_hz": float(np.nansum(fr)),
        "fr_gini": _gini(fr),
        "amp_median_uv": float(np.nanmedian(amp[active])) if n_active else float("nan"),
        "amp_p10_uv": (float(np.nanpercentile(electrodes["amp_p10_uv"].to_numpy(float)[active], 50))
                       if n_active else float("nan")),
        "noise_median_uv": float(np.nanmedian(noise)),
        "activity_centroid_um": [cx, cy],
        "n_configs": int(diagnostics.get("n_configs", 0)),
        "duration_total_s": float(diagnostics.get("duration_total_s", float("nan"))),
        "online_total_spikes": int(np.nansum(
            electrodes["online_n_spikes"].to_numpy(dtype=float))),
    }


def compute_well_activity(h5_path: str | Path, well_index: int,
                          config: ActivityScanConfig | None = None,
                          h5: Any = None) -> WellActivityResults:
    """Assemble one well's activity across every configuration in the file."""
    import h5py
    import pandas as pd

    cfg = config or ActivityScanConfig()
    opened = None
    if h5 is None:
        opened = h5py.File(str(h5_path), "r")
        h5 = opened
    t0 = time.time()
    try:
        keys = well_store_nodes(h5, well_index)
        if not keys:
            raise ActivityScanError(
                f"well {well_index} has no data_store node in {h5_path}")

        frames = []
        per_config = []
        duration_total = 0.0
        for key in keys:
            node = h5["data_store"][key]
            t_cfg = time.time()
            try:
                m = store_metrics(node, cfg)
            except Exception as exc:  # noqa: BLE001 — one bad config must not sink the well
                per_config.append({"store": key, "error": f"{type(exc).__name__}: {exc}"})
                continue
            online = online_spike_counts(node)
            frame = pd.DataFrame({
                "electrode": m["electrode"],
                "x_um": m["x_um"],
                "y_um": m["y_um"],
                "n_spikes": m["n_spikes"],
                "duration_s": m["duration_s"],
                "noise_uv": m["noise_uv"],
                "rms_uv": m["rms_uv"],
                "amp_median_uv": m["amp_median_uv"],
                "amp_p10_uv": m["amp_p10_uv"],
                "amp_max_uv": m["amp_max_uv"],
            })
            frame["online_n_spikes"] = frame["electrode"].map(online).fillna(0).astype(int)
            frames.append(frame)
            duration_total += m["duration_s"]
            per_config.append({
                "store": key,
                "recording_id": int(node["recording_id"][0]),
                "n_channels": m["n_channels"],
                "n_samples": m["n_samples"],
                "fs": m["fs"],
                "duration_s": m["duration_s"],
                "lsb_uv": m["lsb_uv"],
                "noise_median_uv": float(np.nanmedian(m["noise_uv"])),
                "n_spikes": int(np.nansum(m["n_spikes"])),
                "online_n_spikes": int(sum(online.values())),
                "elapsed_s": round(time.time() - t_cfg, 2),
            })

        if not frames:
            raise ActivityScanError(
                f"well {well_index}: every configuration failed to read")

        electrodes = _aggregate(frames, pd)
        diagnostics = {
            "well_index": int(well_index),
            "well_id": well_label(well_index),
            "h5_path": str(h5_path),
            "n_configs": len(frames),
            "n_configs_found": len(keys),
            "duration_total_s": duration_total,
            "per_config": per_config,
            "params": cfg.as_dict(),
            "elapsed_s": round(time.time() - t0, 2),
        }
        stats = well_stats(electrodes, cfg, diagnostics)
        maps = {
            "firing_rate": grid_from_electrodes(electrodes, "firing_rate_hz"),
            "amp_median": grid_from_electrodes(electrodes, "amp_median_uv"),
            "noise": grid_from_electrodes(electrodes, "noise_uv"),
            "coverage": grid_from_electrodes(electrodes, "n_configs", fill=0.0),
        }
        return WellActivityResults(electrodes, maps, stats, diagnostics)
    finally:
        if opened is not None:
            opened.close()


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def write_activity_scan(results: WellActivityResults, output_dir: str | Path) -> Path:
    """Write ``electrodes.parquet`` + ``maps.npz`` + ``stats.json`` + diagnostics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.electrodes.to_parquet(out / "electrodes.parquet")
    np.savez_compressed(
        out / "maps.npz",
        **{k: np.asarray(v, dtype=np.float32) for k, v in results.maps.items()},
    )
    (out / "stats.json").write_text(json.dumps(results.stats, indent=1, default=float))
    (out / "diagnostics.json").write_text(
        json.dumps(results.diagnostics, indent=1, default=float))
    return out


def load_activity_scan(output_dir: str | Path) -> WellActivityResults:
    """Read back a written well (viewer entry point)."""
    import pandas as pd

    d = Path(output_dir)
    electrodes = pd.read_parquet(d / "electrodes.parquet")
    with np.load(d / "maps.npz") as z:
        maps = {k: z[k] for k in z.files}
    stats = json.loads((d / "stats.json").read_text())
    diag_path = d / "diagnostics.json"
    diagnostics = json.loads(diag_path.read_text()) if diag_path.exists() else {}
    return WellActivityResults(electrodes, maps, stats, diagnostics)


def active_area_curve(electrodes: "object",
                      thresholds: np.ndarray | None = None) -> tuple:
    """(thresholds, active fraction) — lets the viewer move the 'active' cut.

    The default 0.1 Hz cut is 3 events in 30 s, so what counts as active is a
    judgement rather than a fact; the curve makes the sensitivity visible.
    """
    if thresholds is None:
        thresholds = np.logspace(-2, 1, 40)
    fr = electrodes["firing_rate_hz"].to_numpy(dtype=float)
    fr = fr[np.isfinite(fr)]
    if fr.size == 0:
        return thresholds, np.full(thresholds.size, np.nan)
    frac = np.array([float((fr >= t).mean()) for t in thresholds])
    return thresholds, frac
