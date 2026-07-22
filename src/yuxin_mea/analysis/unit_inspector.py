"""Per-unit (single-neuron) inspector library.

Pure-library loaders + Plotly figure builders for the dashboard
``Single-unit`` section (``yuxin_mea.dashboard.pages.unit_inspector``). No Dash
imports and — deliberately — **no SpikeInterface / torch import**: everything is
read straight off the ``SortingAnalyzer`` binary-folder that ``AnalyzerTask``
writes, with plain ``np.load`` / ``pd.read_csv`` / ``json``. Loading a full
``SortingAnalyzer`` would drag in torch (seconds of import) for what is a handful
of array reads, so we reconstruct the same views by hand.

On-disk layout (per well), rooted at the AnalyzerTask output dir ``…/analyzer``::

    analyzer/
      sorting/spikes.npy                 # structured: sample_index, unit_index, …
      sorting/numpysorting_info.json     # sampling_frequency
      sorting/properties/KSLabel.npy     # good/mua per unit (positional)
      sparsity_mask.npy                  # (n_units, n_chan) bool
      extensions/templates/average.npy   # (n_units, n_samp, n_chan) dense
      extensions/templates/std.npy
      extensions/templates/params.json   # ms_before / ms_after
      extensions/waveforms/waveforms.npy # (n_sel, n_samp, max_sparse) random subset
      extensions/random_spikes/random_spikes_indices.npy  # indices into spikes.npy
      extensions/spike_amplitudes/amplitudes.npy          # (n_all_spikes,)
      extensions/quality_metrics/metrics.csv
      extensions/template_metrics/metrics.csv
      extensions/unit_locations/unit_locations.npy        # (n_units, >=2)

The waveform overlay maps snippets to a unit exactly the way SI's
``get_waveforms_one_unit`` does: ``waveforms[i]`` corresponds to global spike
``random_spikes_indices[i]``, whose unit is ``spikes['unit_index'][…]``; the
sparse channel columns are left-aligned to ``nonzero(sparsity_mask[unit])``.
Verified on-disk (peak-column mean vs dense template peak channel: corr = 1.0).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Kilosort label -> our good/MUA/noise vocabulary (noise never comes from KS).
KS_LABEL_MAP = {"good": "good", "mua": "MUA"}
_WAVE_COLOR = "rgba(120,120,120,0.22)"
_TEMPLATE_COLOR = "#c0623a"
_STD_FILL = "rgba(192,98,58,0.15)"
_HILITE = "#c0623a"
_REFRACTORY_MS = 2.0


# --------------------------------------------------------------------------- #
# Discovery (reads pipeline_cache.json as plain data — no pipeline import)
# --------------------------------------------------------------------------- #
def list_analyzer_recordings(analysis_root: Path | str) -> list[str]:
    """Recording keys with a completed ``analyzer`` task, from the pipeline cache."""
    from yuxin_mea.analysis.burst_inspector import _load_pipeline_cache

    cache = _load_pipeline_cache(analysis_root)
    recs: set[str] = set()
    for entry in cache.values():
        if not isinstance(entry, dict):
            continue
        task = (entry.get("tasks") or {}).get("analyzer")
        rk = entry.get("recording_key")
        if task and task.get("status") == "complete" and rk:
            recs.add(rk)
    return sorted(recs)


def analyzer_dirs_for_recording(
    analysis_root: Path | str, recording_key: str
) -> dict[str, Path]:
    """Return ``{well_id: <…>/analyzer}`` for one recording (pipeline-cache manifest).

    Falls back to walking ``analysis_root/analyzer_data`` when the cache has no
    manifest entry for the recording.
    """
    from yuxin_mea.analysis.burst_inspector import well_output_dirs_from_cache

    dirs = well_output_dirs_from_cache(analysis_root, recording_key, "analyzer")
    if dirs:
        return dirs
    # Fallback: convention walk.
    base = Path(analysis_root) / "analyzer_data" / recording_key
    out: dict[str, Path] = {}
    if base.exists():
        for adir in base.rglob("analyzer"):
            if (adir / "extensions" / "templates" / "average.npy").exists():
                # Key by well_id (last path part) to match the cache-manifest form
                # from well_output_dirs_from_cache.
                out[adir.parent.name] = adir
    return out


# --------------------------------------------------------------------------- #
# Bundle + loader
# --------------------------------------------------------------------------- #
@dataclass
class UnitBundle:
    """Everything the unit-inspector panels need for one well."""

    recording_key: str
    rec_name: str
    well_id: str
    analyzer_dir: Path
    fs: float
    ms_before: float
    ms_after: float
    unit_ids: list[int]
    templates: np.ndarray          # (n_units, n_samp, n_chan) dense
    templates_std: np.ndarray | None
    sparsity_mask: np.ndarray | None  # (n_units, n_chan) bool
    metrics: pd.DataFrame          # index = unit_id
    locations: np.ndarray | None   # (n_units, >=2)
    spikes: np.ndarray             # structured (sample_index, unit_index, …)
    rand_idx: np.ndarray | None    # (n_sel,) into `spikes`
    waveforms: np.ndarray | None   # (n_sel, n_samp, max_sparse)
    amplitudes: np.ndarray | None  # (n_all_spikes,)

    # -- indexing helpers --
    def n_units(self) -> int:
        return len(self.unit_ids)

    def unit_pos(self, unit_id: int) -> int:
        """Positional index (into templates / spikes.unit_index) for a unit_id."""
        return self.unit_ids.index(int(unit_id))

    def time_ms(self) -> np.ndarray:
        n_samp = self.templates.shape[1]
        return np.linspace(-self.ms_before, self.ms_after, n_samp)

    def peak_channel(self, pos: int) -> int:
        """Dense channel index of the widest peak-to-peak template for a unit."""
        wf = self.templates[pos]
        return int(np.argmax(wf.max(0) - wf.min(0)))

    def unit_spike_samples(self, unit_id: int) -> np.ndarray:
        pos = self.unit_pos(unit_id)
        return self.spikes["sample_index"][self.spikes["unit_index"] == pos]

    def unit_spike_times(self, unit_id: int) -> np.ndarray:
        return self.unit_spike_samples(unit_id).astype(float) / self.fs

    def unit_amplitudes(self, unit_id: int) -> np.ndarray | None:
        if self.amplitudes is None:
            return None
        pos = self.unit_pos(unit_id)
        return self.amplitudes[self.spikes["unit_index"] == pos]

    def unit_peak_waveforms(
        self, unit_id: int, max_waves: int = 200
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(waves[n, n_samp], template_peak[n_samp])`` on the peak channel.

        ``waves`` are the stored random-subset snippets for this unit (the
        standard Phy/SI QC overlay); ``None`` when snippets are unavailable.
        """
        if self.waveforms is None or self.rand_idx is None or self.sparsity_mask is None:
            return None
        pos = self.unit_pos(unit_id)
        sel_unit = self.spikes["unit_index"][self.rand_idx]
        rows = np.nonzero(sel_unit == pos)[0]
        if rows.size == 0:
            return None
        wfs = self.waveforms[rows]  # (k, n_samp, max_sparse)
        chans_u = np.nonzero(self.sparsity_mask[pos])[0]  # dense idxs, ascending
        pk = self.peak_channel(pos)
        hit = np.nonzero(chans_u == pk)[0]
        if hit.size:
            col = int(hit[0])
        else:  # peak not in sparse set (rare) — widest sparse column
            avg = wfs.mean(0)
            col = int(np.argmax(avg.max(0) - avg.min(0)))
        peak_waves = wfs[:, :, col]
        if peak_waves.shape[0] > max_waves:
            idx = np.linspace(0, peak_waves.shape[0] - 1, max_waves).astype(int)
            peak_waves = peak_waves[idx]
        return peak_waves, self.templates[pos][:, pk]


def _read_json(path: Path) -> dict:
    try:
        with path.open() as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _load_fs(analyzer_dir: Path) -> float:
    for rel in (
        "sorting/numpysorting_info.json",
        "recording_info/recording_attributes.json",
    ):
        sf = _read_json(analyzer_dir / rel).get("sampling_frequency")
        if sf:
            return float(sf)
    logger.warning("sampling_frequency not found under %s; defaulting to 10000 Hz",
                   analyzer_dir)
    return 10000.0


def _load_npy(path: Path, *, allow_pickle: bool = False) -> np.ndarray | None:
    try:
        return np.load(path, allow_pickle=allow_pickle)
    except (OSError, ValueError):
        return None


def _build_metrics(
    analyzer_dir: Path,
    unit_ids: list[int],
    spikes: np.ndarray,
    locations: np.ndarray | None,
    auto_curated: pd.Series | None,
) -> pd.DataFrame:
    ext = analyzer_dir / "extensions"
    qm = pd.read_csv(ext / "quality_metrics" / "metrics.csv", index_col=0)
    tm_path = ext / "template_metrics" / "metrics.csv"
    if tm_path.exists():
        tm = pd.read_csv(tm_path, index_col=0)
        metrics = qm.join(tm, how="left", rsuffix="_tm")
    else:
        metrics = qm.copy()

    if locations is not None and locations.shape[0] == len(metrics):
        metrics["loc_x"] = locations[:, 0]
        metrics["loc_y"] = locations[:, 1]

    ks = _load_npy(analyzer_dir / "sorting" / "properties" / "KSLabel.npy",
                   allow_pickle=True)
    if ks is not None and len(ks) == len(metrics):
        metrics["ks_label"] = [str(x) for x in ks]
        metrics["quality"] = [KS_LABEL_MAP.get(str(x), "unsorted") for x in ks]

    counts = np.bincount(spikes["unit_index"].astype(int),
                         minlength=len(unit_ids))
    metrics["n_spikes"] = [int(counts[pos]) for pos in range(len(unit_ids))]

    if auto_curated is not None:
        metrics["auto_curated"] = auto_curated.reindex(metrics.index).astype("boolean")
    return metrics


def load_unit_bundle(
    analyzer_dir: Path | str,
    *,
    recording_key: str = "",
    rec_name: str = "",
    well_id: str = "",
    curation_dir: Path | str | None = None,
) -> UnitBundle:
    """Load one well's SortingAnalyzer views into a :class:`UnitBundle`.

    ``curation_dir`` (the ``auto_curation`` output dir) is optional; when present
    its ``quality_metrics.pkl`` supplies the auto ``curated`` flag as a column.
    """
    analyzer_dir = Path(analyzer_dir)
    ext = analyzer_dir / "extensions"

    templates = np.load(ext / "templates" / "average.npy")
    templates_std = _load_npy(ext / "templates" / "std.npy")
    params = _read_json(ext / "templates" / "params.json")
    ms_before = float(params.get("ms_before", 1.0))
    ms_after = float(params.get("ms_after", 2.0))

    spikes = np.load(analyzer_dir / "sorting" / "spikes.npy")
    sparsity_mask = _load_npy(analyzer_dir / "sparsity_mask.npy")
    locations = _load_npy(ext / "unit_locations" / "unit_locations.npy")
    rand_idx = _load_npy(ext / "random_spikes" / "random_spikes_indices.npy")
    waveforms = _load_npy(ext / "waveforms" / "waveforms.npy")
    amplitudes = _load_npy(ext / "spike_amplitudes" / "amplitudes.npy")

    qm_index = pd.read_csv(ext / "quality_metrics" / "metrics.csv",
                           index_col=0).index
    unit_ids = [int(u) for u in qm_index]

    auto_curated: pd.Series | None = None
    if curation_dir is not None:
        pkl = Path(curation_dir) / "quality_metrics.pkl"
        if pkl.exists():
            try:
                cur = pd.read_pickle(pkl)
                if "curated" in cur:
                    auto_curated = cur["curated"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not read auto-curation pkl %s: %s", pkl, exc)

    metrics = _build_metrics(analyzer_dir, unit_ids, spikes, locations, auto_curated)

    return UnitBundle(
        recording_key=recording_key,
        rec_name=rec_name,
        well_id=well_id,
        analyzer_dir=analyzer_dir,
        fs=_load_fs(analyzer_dir),
        ms_before=ms_before,
        ms_after=ms_after,
        unit_ids=unit_ids,
        templates=templates,
        templates_std=templates_std,
        sparsity_mask=sparsity_mask,
        metrics=metrics,
        locations=locations,
        spikes=spikes,
        rand_idx=rand_idx,
        waveforms=waveforms,
        amplitudes=amplitudes,
    )


# --------------------------------------------------------------------------- #
# Signature-keyed memoize (single-user dashboard; bust by mtime+size)
# --------------------------------------------------------------------------- #
_BUNDLE_MEMO: dict[tuple, UnitBundle] = {}
_MEMO_MAX = 8


def _sig(analyzer_dir: Path, curation_dir: Path | None) -> tuple:
    files = [
        analyzer_dir / "extensions" / "templates" / "average.npy",
        analyzer_dir / "extensions" / "waveforms" / "waveforms.npy",
        analyzer_dir / "sorting" / "spikes.npy",
    ]
    if curation_dir is not None:
        files.append(Path(curation_dir) / "quality_metrics.pkl")
    out = []
    for p in files:
        try:
            st = p.stat()
            out.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(p), -1, -1))
    return tuple(out)


def load_unit_bundle_cached(
    analyzer_dir: Path | str,
    *,
    recording_key: str = "",
    rec_name: str = "",
    well_id: str = "",
    curation_dir: Path | str | None = None,
) -> UnitBundle:
    """Memoized :func:`load_unit_bundle` keyed by source-file staleness signature."""
    adir = Path(analyzer_dir)
    cdir = Path(curation_dir) if curation_dir is not None else None
    key = (str(adir), _sig(adir, cdir))
    hit = _BUNDLE_MEMO.get(key)
    if hit is not None:
        return hit
    bundle = load_unit_bundle(
        adir, recording_key=recording_key, rec_name=rec_name,
        well_id=well_id, curation_dir=cdir,
    )
    _BUNDLE_MEMO[key] = bundle
    while len(_BUNDLE_MEMO) > _MEMO_MAX:
        _BUNDLE_MEMO.pop(next(iter(_BUNDLE_MEMO)))
    return bundle


def clear_cache() -> None:
    """Drop memoized bundles (tests / rescan)."""
    _BUNDLE_MEMO.clear()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _empty_figure(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color="#84807a"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=height)
    return fig


def fig_unit_template(bundle: UnitBundle, unit_id: int, height: int = 320) -> go.Figure:
    """Template on the unit's sparse channels — peak channel bold with ±std band."""
    pos = bundle.unit_pos(unit_id)
    t = bundle.time_ms()
    pk = bundle.peak_channel(pos)
    fig = go.Figure()

    chans = (np.nonzero(bundle.sparsity_mask[pos])[0]
             if bundle.sparsity_mask is not None else np.array([pk]))
    # ±std band on the peak channel.
    if bundle.templates_std is not None:
        std = bundle.templates_std[pos][:, pk]
        mean = bundle.templates[pos][:, pk]
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([mean + std, (mean - std)[::-1]]),
            fill="toself", fillcolor=_STD_FILL, line=dict(width=0),
            hoverinfo="skip", showlegend=False, name="±std"))
    # Faint traces for the non-peak sparse channels.
    for ch in chans:
        if ch == pk:
            continue
        fig.add_trace(go.Scatter(
            x=t, y=bundle.templates[pos][:, ch], mode="lines",
            line=dict(color="rgba(120,120,120,0.5)", width=1),
            hoverinfo="skip", showlegend=False))
    # Peak channel, bold.
    fig.add_trace(go.Scatter(
        x=t, y=bundle.templates[pos][:, pk], mode="lines",
        line=dict(color=_TEMPLATE_COLOR, width=2.4),
        name=f"peak ch {pk}"))
    fig.update_layout(
        margin=dict(l=48, r=12, t=28, b=40), height=height,
        title=dict(text=f"Template — unit {unit_id}", x=0.02, font=dict(size=12)),
        xaxis_title="time (ms)", yaxis_title="µV",
        showlegend=True, legend=dict(font=dict(size=10)))
    return fig


def fig_unit_overlay(
    bundle: UnitBundle, unit_id: int, max_waves: int = 500, height: int = 320
) -> go.Figure:
    """All stored raw snippets (peak channel) overlaid behind the template.

    ``waveforms.npy`` holds up to ~500 spikes/unit (SI ``random_spikes``); the
    default cap shows the full stored set — the standard Phy/SI QC overlay.
    """
    got = bundle.unit_peak_waveforms(unit_id, max_waves=max_waves)
    if got is None:
        return _empty_figure("No waveform snippets stored for this unit.", height)
    waves, template_peak = got
    t = bundle.time_ms()
    # Many polylines as ONE trace: separate each snippet with a NaN break.
    n, n_samp = waves.shape
    xs = np.tile(np.append(t, np.nan), n)
    ys = np.concatenate([np.append(waves[i], np.nan) for i in range(n)])
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="lines",
        line=dict(color=_WAVE_COLOR, width=0.6),
        hoverinfo="skip", name=f"{n} snippets"))
    fig.add_trace(go.Scatter(
        x=t, y=template_peak, mode="lines",
        line=dict(color=_TEMPLATE_COLOR, width=2.6), name="template"))
    fig.update_layout(
        margin=dict(l=48, r=12, t=28, b=40), height=height,
        title=dict(text=f"Waveform overlay — unit {unit_id}", x=0.02,
                   font=dict(size=12)),
        xaxis_title="time (ms)", yaxis_title="µV",
        showlegend=True, legend=dict(font=dict(size=10)))
    return fig


def fig_isi_hist(
    bundle: UnitBundle, unit_id: int, bin_ms: float = 1.0, max_ms: float = 100.0,
    height: int = 260,
) -> go.Figure:
    """Inter-spike-interval histogram with the 2 ms refractory marker."""
    st = np.sort(bundle.unit_spike_times(unit_id))
    if st.size < 2:
        return _empty_figure("Too few spikes for ISI.", height)
    isi = np.diff(st) * 1000.0  # ms
    isi = isi[isi <= max_ms]
    bins = np.arange(0, max_ms + bin_ms, bin_ms)
    counts, edges = np.histogram(isi, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=counts, marker_color="#5b7fac",
                           width=bin_ms, name="ISI"))
    fig.add_vline(x=_REFRACTORY_MS, line=dict(color=_HILITE, width=1, dash="dash"),
                  annotation_text="2 ms", annotation_font_size=9)
    fig.update_layout(
        margin=dict(l=48, r=12, t=28, b=40), height=height,
        title=dict(text="ISI histogram", x=0.02, font=dict(size=12)),
        xaxis_title="inter-spike interval (ms)", yaxis_title="count",
        showlegend=False)
    return fig


def fig_autocorrelogram(
    bundle: UnitBundle, unit_id: int, bin_ms: float = 1.0, window_ms: float = 50.0,
    height: int = 260,
) -> go.Figure:
    """Spike-train autocorrelogram (symmetric, self-lag excluded)."""
    t = np.sort(bundle.unit_spike_times(unit_id)) * 1000.0  # ms
    if t.size < 2:
        return _empty_figure("Too few spikes for autocorrelogram.", height)
    bins = np.arange(-window_ms, window_ms + bin_ms, bin_ms)
    acc = np.zeros(len(bins) - 1, dtype=np.int64)
    lo = np.searchsorted(t, t - window_ms, side="left")
    hi = np.searchsorted(t, t + window_ms, side="right")
    for i in range(t.size):
        neigh = t[lo[i]:hi[i]]
        d = neigh - t[i]
        d = d[d != 0.0]
        if d.size:
            acc += np.histogram(d, bins=bins)[0]
    centers = (bins[:-1] + bins[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=acc, marker_color="#5d9e7e",
                           width=bin_ms, name="ACG"))
    fig.update_layout(
        margin=dict(l=48, r=12, t=28, b=40), height=height,
        title=dict(text="Autocorrelogram", x=0.02, font=dict(size=12)),
        xaxis_title="lag (ms)", yaxis_title="count", showlegend=False)
    return fig


def fig_amplitude_time(
    bundle: UnitBundle, unit_id: int, height: int = 260, max_points: int = 8000
) -> go.Figure:
    """Per-spike amplitude over the recording (drift / stability view)."""
    amps = bundle.unit_amplitudes(unit_id)
    if amps is None or amps.size == 0:
        return _empty_figure("No spike amplitudes stored.", height)
    times = bundle.unit_spike_times(unit_id)
    if times.size > max_points:
        idx = np.linspace(0, times.size - 1, max_points).astype(int)
        times, amps = times[idx], amps[idx]
    fig = go.Figure(go.Scattergl(
        x=times, y=amps, mode="markers",
        marker=dict(size=2.5, color="#5b7fac", opacity=0.4), name="spikes"))
    fig.update_layout(
        margin=dict(l=48, r=12, t=28, b=40), height=height,
        title=dict(text="Amplitude over time", x=0.02, font=dict(size=12)),
        xaxis_title="time (s)", yaxis_title="amplitude (µV)", showlegend=False)
    return fig


def fig_probe_location(
    bundle: UnitBundle, unit_id: int, height: int = 320
) -> go.Figure:
    """All units on the probe, selected unit highlighted."""
    if bundle.locations is None or bundle.locations.shape[1] < 2:
        return _empty_figure("No unit locations available.", height)
    xy = bundle.locations[:, :2]
    pos = bundle.unit_pos(unit_id)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xy[:, 0], y=xy[:, 1], mode="markers",
        marker=dict(size=6, color="rgba(130,128,122,0.55)"),
        text=[f"unit {u}" for u in bundle.unit_ids], hoverinfo="text",
        name="units"))
    fig.add_trace(go.Scatter(
        x=[xy[pos, 0]], y=[xy[pos, 1]], mode="markers",
        marker=dict(size=13, color=_HILITE, line=dict(width=1.5, color="#fff")),
        text=[f"unit {unit_id}"], hoverinfo="text", name="selected"))
    fig.update_layout(
        margin=dict(l=48, r=12, t=28, b=40), height=height,
        title=dict(text="Unit location on probe", x=0.02, font=dict(size=12)),
        xaxis_title="x (µm)", yaxis_title="y (µm)",
        showlegend=False, yaxis_scaleanchor="x", yaxis_scaleratio=1)
    return fig


# --------------------------------------------------------------------------- #
# Metric summaries for the page
# --------------------------------------------------------------------------- #
_KV_METRICS = [
    ("firing_rate", "firing rate (Hz)", "{:.3g}"),
    ("amplitude_median", "amplitude (µV)", "{:.1f}"),
    ("presence_ratio", "presence ratio", "{:.3f}"),
    ("rp_contamination", "rp contamination", "{:.3f}"),
    ("peak_to_trough_duration", "peak→trough (ms)", "{:.3g}"),
    ("n_spikes", "n spikes", "{:.0f}"),
]


def unit_metrics_kv(bundle: UnitBundle, unit_id: int) -> dict[str, str]:
    """Compact KV summary for one unit (rendered as a card by the page)."""
    row = bundle.metrics.loc[unit_id]
    out: dict[str, str] = {}
    for col, label, fmt in _KV_METRICS:
        if col not in row or pd.isna(row[col]):
            continue
        val = float(row[col])
        if col == "peak_to_trough_duration":  # stored in seconds
            val *= 1000.0
        out[label] = fmt.format(val)
    if "ks_label" in row and not pd.isna(row["ks_label"]):
        out["Kilosort label"] = str(row["ks_label"])
    if "auto_curated" in row and not pd.isna(row["auto_curated"]):
        out["auto curated"] = "pass" if bool(row["auto_curated"]) else "reject"
    return out


def unit_table(bundle: UnitBundle) -> pd.DataFrame:
    """Per-unit summary table for the master list (one row per unit)."""
    cols = [c for c in ("firing_rate", "amplitude_median", "presence_ratio",
                        "rp_contamination", "n_spikes", "quality", "ks_label",
                        "auto_curated") if c in bundle.metrics.columns]
    tbl = bundle.metrics[cols].copy()
    tbl.insert(0, "unit_id", tbl.index)
    return tbl.reset_index(drop=True)
