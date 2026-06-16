"""Burst detector diagnostic — pure library (no Dash imports).

Public API:
- ``BatchResults`` dataclass holding all detector outputs.
- ``run_batch_generic`` to execute a burst detector (``traditional`` or
  ``ml``) on every Kilosort source under a root directory.
- ``load_or_run_batch`` — caching wrapper: pickle to
  ``<analysis_root>/burst_diagnostic_cache/<key>.pkl``.
- ``fig_generic_summary`` returning a ``plotly.graph_objects.Figure``.
- ``save_html`` to write a figure to disk.

The Dash-app construction lives in
``yuxin_mea.dashboard.pages.burst_diagnostic`` and reuses the figure
functions from this module. Keep this module free of ``dash`` imports so
notebooks and unit tests can use the figures without a web-stack dependency.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BatchResults:
    """All detector outputs for every recording in a batch.

    ``recording_names`` returns a sorted list for stable iteration order.
    ``method`` indicates which detector produced the results.
    """

    spike_times: dict[str, dict] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    results_no_gate: dict[str, Any] = field(default_factory=dict)
    method: str = "ml"

    @property
    def recording_names(self) -> list[str]:
        return sorted(self.results.keys())

    def result(self, name: str, kind: str = "default") -> Any:
        return self.results[name] if kind == "default" else self.results_no_gate[name]


# ---------------------------------------------------------------------------
# Kilosort loaders
# ---------------------------------------------------------------------------


def is_kilosort_dir(path: Path) -> bool:
    """A Kilosort output directory has spike_times.npy, spike_clusters.npy, and params.py."""
    return path.is_dir() and all(
        (path / fn).exists()
        for fn in ("spike_times.npy", "spike_clusters.npy", "params.py")
    )


def discover_real_spike_sources(root: Path | None) -> list[Path]:
    """Discover all Kilosort sources (or curated spike-time files) under ``root``."""
    if root is None or not root.exists():
        return []
    if root.is_file() or is_kilosort_dir(root):
        return [root]
    curated = sorted(root.rglob("curated_spike_times.npy"))
    ks_dirs = sorted(
        {p.parent for p in root.rglob("spike_times.npy") if is_kilosort_dir(p.parent)}
    )
    curated_parents = {p.parent for p in curated}
    sources = curated + [p for p in ks_dirs if p not in curated_parents]
    return sorted(set(sources), key=lambda p: str(p))


def _read_kilosort_sample_rate(params_path: Path) -> float:
    values: dict[str, object] = {}
    exec(params_path.read_text(), {}, values)
    for key in ("sample_rate", "fs", "sampling_rate"):
        if key in values:
            return float(values[key])
    raise ValueError(f"Could not find sample_rate/fs/sampling_rate in {params_path}")


def _read_kilosort_keep_clusters(
    ks_dir: Path, labels: set[str] | None
) -> set[int] | None:
    if labels is None:
        return None
    for filename in ("cluster_group.tsv", "cluster_KSLabel.tsv"):
        label_path = ks_dir / filename
        if not label_path.exists():
            continue
        table = pd.read_csv(label_path, sep="\t")
        if "cluster_id" not in table.columns:
            continue
        label_col = next(
            (c for c in ("group", "KSLabel", "label") if c in table.columns), None
        )
        if label_col is None:
            continue
        keep = table[
            table[label_col].astype(str).str.lower().isin({l.lower() for l in labels})
        ]
        return set(keep["cluster_id"].astype(int).tolist())
    return None


def load_kilosort_spike_times(
    ks_dir: Path,
    labels: set[str] | None = frozenset({"good"}),
) -> dict[str, np.ndarray]:
    """Load spike times from a Kilosort directory, optionally filtered by labels."""
    ks_dir = Path(ks_dir)
    sample_rate = _read_kilosort_sample_rate(ks_dir / "params.py")
    spike_samples = np.load(ks_dir / "spike_times.npy").reshape(-1).astype(float)
    spike_clusters = np.load(ks_dir / "spike_clusters.npy").reshape(-1).astype(int)
    keep_clusters = _read_kilosort_keep_clusters(ks_dir, labels)
    out: dict[str, np.ndarray] = {}
    for cid in np.unique(spike_clusters):
        if keep_clusters is not None and int(cid) not in keep_clusters:
            continue
        times = spike_samples[spike_clusters == cid] / sample_rate
        times = times[np.isfinite(times) & (times >= 0.0)]
        if times.size:
            out[f"cluster_{int(cid):03d}"] = np.sort(times.astype(float))
    if not out:
        raise ValueError(f"No clusters found in {ks_dir}")
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_html(fig: go.Figure, path: Path, offline: bool = False) -> Path:
    """Write ``fig`` to ``path`` as an HTML file (Plotly JS from CDN by default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotlyjs = True if offline else "cdn"
    fig.write_html(str(path), include_plotlyjs=plotlyjs)
    return path


def fig_generic_summary(batch: BatchResults) -> go.Figure:
    """Bar chart of burstlet / network burst counts per recording."""
    names = batch.recording_names
    if not names:
        fig = go.Figure()
        fig.add_annotation(
            text="(no recordings)", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#888"),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    n_burstlets = []
    n_network = []
    for name in names:
        res = batch.results.get(name)
        if res is not None:
            n_burstlets.append(len(res.burstlets))
            n_network.append(len(res.network_bursts))
        else:
            n_burstlets.append(0)
            n_network.append(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=n_burstlets, name="burstlets", marker_color="#1f77b4",
    ))
    fig.add_trace(go.Bar(
        x=names, y=n_network, name="network bursts", marker_color="#ff7f0e",
    ))
    fig.update_layout(
        title=f"{batch.method} burst detection — event counts per recording",
        barmode="group",
        height=400,
        margin=dict(l=60, r=20, t=60, b=100),
        xaxis_title="recording",
        yaxis_title="count",
        xaxis_tickangle=-45,
    )
    return fig


# ---------------------------------------------------------------------------
# Disk cache for BatchResults (so dashboard reloads don't recompute)
# ---------------------------------------------------------------------------

_CACHE_VERSION = 1  # Bump if BatchResults shape changes.


def cache_key(root: Path) -> str:
    """Deterministic cache key for a Kilosort root.

    Hashes the absolute path string only — does not stat the input files.
    Use the dashboard's Recompute button (or delete the cache file) when
    the underlying spike-time data changes.
    """
    h = hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:16]
    return f"v{_CACHE_VERSION}_{h}"


def cache_path(analysis_root: Path, key: str) -> Path:
    """Per-analysis cache location: <analysis_root>/burst_diagnostic_cache/<key>.pkl."""
    return Path(analysis_root) / "burst_diagnostic_cache" / f"{key}.pkl"


def run_batch_generic(
    sources: list[Path],
    method: str,
    labels: set[str] | None = frozenset({"good"}),
    verbose: bool = True,
) -> BatchResults:
    """Run a burst detector on every source.

    Supports ``"traditional"`` and ``"ml"`` methods. Returns a BatchResults
    with ``results`` populated.
    """
    batch = BatchResults(method=method)
    for source in sources:
        name = source.name if source.is_dir() else source.parent.name
        st = load_kilosort_spike_times(source, labels=labels)
        batch.spike_times[name] = st

        if method == "traditional":
            from yuxin_mea.analysis.burst_detector import compute_network_bursts
            res = compute_network_bursts(st)
        elif method == "ml":
            from yuxin_mea.analysis.ml_burst_detector import compute_ml_bursts
            res = compute_ml_bursts(st)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        batch.results[name] = res
        if verbose:
            print(f"{name}  {method}: {len(res.burstlets)} burstlets")
    return batch


def load_or_run_batch(
    root: Path,
    analysis_root: Path | None,
    *,
    force_recompute: bool = False,
    method: str = "ml",
) -> tuple[BatchResults, bool]:
    """Return ``(batch, came_from_cache)``. Writes the cache on a fresh run.

    ``analysis_root=None`` disables on-disk caching and always runs fresh —
    safe to call from a dashboard whose config has no ``analysis_root`` yet.

    ``labels`` is intentionally not a parameter. The cache key hashes only
    the path, so a label change would silently return stale data. Callers
    needing non-default labels must invoke :func:`run_batch_generic` directly
    and own their own cache decisions.
    """
    sources = discover_real_spike_sources(root)
    if not sources:
        raise FileNotFoundError(f"No Kilosort sources under {root}")

    ck = f"{cache_key(root)}_{method}"
    cache_file: Path | None = None
    if analysis_root is not None:
        cache_file = cache_path(analysis_root, ck)
        if not force_recompute and cache_file.exists():
            with cache_file.open("rb") as fh:
                return pickle.load(fh), True

    batch = run_batch_generic(
        sources, method=method, labels=frozenset({"good"}), verbose=False,
    )

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as fh:
            pickle.dump(batch, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return batch, False
