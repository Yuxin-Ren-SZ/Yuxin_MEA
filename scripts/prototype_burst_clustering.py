"""Throwaway prototype: compare clustering algorithms for burst-trajectory detection.

The ML burst detector clusters bin-level features with HDBSCAN in the raw ~26-D
space, where the burst is a low-density *filament* that HDBSCAN dumps into noise.
This script does NOT touch the detector or config — it loads existing debug traces
and renders, per well, a UMAP(2) view colored three ways:

    (a) current HDBSCAN labels (noise = gray)
    (b) Method A  — HDBSCAN on a UMAP embedding   → burst vs non-burst
    (c) Method B  — Spectral clustering (kNN)      → burst vs non-burst

In every method, "burst" = union of clusters whose mean burst-posterior
(post_frac_gt_0_5) exceeds an adaptive threshold (background median + k·MAD), so
the small outer non-burst clusters are excluded by construction.

A per-well table reports, for each method: n burst bins, recall of post_frac>0.5
bins, false-positive rate on resting bins (post_frac<0.1), number of contiguous
burst runs (event proxy), and cluster count. The visual overlay is the primary
judge; the table numbers are a secondary, partly-circular sanity check.

Usage:
    python scripts/prototype_burst_clustering.py --config pipeline_config.json
    python scripts/prototype_burst_clustering.py --config pipeline_config.json --wells well000 well023
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Reuse the inspector's trace loader and the detector's own z-norm logic so the
# clustered space matches exactly what the detector sees.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import inspect_ml_bursts as insp  # noqa: E402
from yuxin_mea.analysis.ml_burst_cluster import (  # noqa: E402
    _background_mask_from_feature,
    _znorm_with_stats,
)

RANKING = "post_frac_gt_0_5"
BURST_COLOR = "#d62728"
NONBURST_COLOR = "rgba(150,150,150,0.40)"
NOISE_COLOR = "rgba(180,180,180,0.55)"
CLUSTER_PALETTE = [
    "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
    "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2",
    "#dbdb8d", "#9edae5", "#393b79", "#637939", "#8c6d31",
]


# ---------------------------------------------------------------------------
# Clustering candidates
# ---------------------------------------------------------------------------


def _umap_embed(Xn: np.ndarray, *, n_components: int = 5) -> np.ndarray:
    """UMAP(min_dist=0) embedding tuned for clustering (not viz)."""
    import umap

    n_comp = int(min(n_components, Xn.shape[1], max(2, Xn.shape[0] - 1)))
    return umap.UMAP(
        n_components=n_comp, n_neighbors=30, min_dist=0.0,
        metric="euclidean", random_state=42,
    ).fit_transform(Xn)


def _pca_embed(Xn: np.ndarray, *, n_components: int = 10) -> np.ndarray:
    from sklearn.decomposition import PCA

    n_comp = int(min(n_components, Xn.shape[1], max(2, Xn.shape[0] - 1)))
    return PCA(n_components=n_comp, random_state=42).fit_transform(Xn)


def _hdbscan_on(emb: np.ndarray) -> np.ndarray:
    """HDBSCAN on a precomputed embedding."""
    import hdbscan

    return hdbscan.HDBSCAN(
        min_cluster_size=30, min_samples=5, cluster_selection_method="eom",
    ).fit_predict(emb).astype(int)


def _spectral(Xn: np.ndarray, *, k_range=(2, 3, 4, 5, 6, 7, 8)) -> np.ndarray:
    """Spectral clustering on a kNN affinity graph.

    k chosen by the largest normalized-Laplacian eigengap over ``k_range``
    (first-pass heuristic for visual comparison, not final-tuned).
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh

    n = Xn.shape[0]
    knn = max(10, min(30, n - 1))
    # Symmetric kNN connectivity → affinity for the eigengap probe.
    A = kneighbors_graph(Xn, n_neighbors=knn, mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T)
    L = laplacian(A, normed=True)
    kmax = min(max(k_range) + 1, n - 1)
    try:
        vals = np.sort(eigsh(L, k=kmax, which="SM", return_eigenvectors=False))
        gaps = np.diff(vals)
        # eigengap among candidate ks (gap after the k-th eigenvalue)
        cand = [k for k in k_range if 1 <= k < len(gaps)]
        best_k = max(cand, key=lambda k: gaps[k - 1]) if cand else 4
    except Exception:
        best_k = 4
    sc = SpectralClustering(
        n_clusters=int(best_k), affinity="nearest_neighbors",
        n_neighbors=knn, assign_labels="kmeans", random_state=42,
    )
    return sc.fit_predict(Xn).astype(int), int(best_k)


# ---------------------------------------------------------------------------
# Shared posterior-based burst selection
# ---------------------------------------------------------------------------


def _burst_mask_from_clusters(
    labels: np.ndarray, rank_raw: np.ndarray, *, mad_scale: float = 3.0
) -> tuple[np.ndarray, set[int], float]:
    """burst = union of clusters whose mean ranking exceeds bg-median + k·MAD.

    Background = bins not in the highest-ranking cluster pool; we use the global
    lower-half as the baseline reference so the threshold is stable.
    """
    base = rank_raw[rank_raw <= np.median(rank_raw)]
    med = float(np.median(base)) if base.size else float(np.median(rank_raw))
    mad = float(np.median(np.abs(base - med))) if base.size else 0.0
    thr = med + mad_scale * max(mad, 1e-6)
    burst_clusters: set[int] = set()
    for c in sorted(set(int(v) for v in labels.tolist())):
        if c < 0:
            continue  # noise is never burst
        m = labels == c
        if m.any() and float(rank_raw[m].mean()) > thr:
            burst_clusters.add(c)
    mask = np.isin(labels, list(burst_clusters)) if burst_clusters else np.zeros_like(labels, bool)
    return mask, burst_clusters, thr


# ---------------------------------------------------------------------------
# Metrics + plotting
# ---------------------------------------------------------------------------


def _contiguous_runs(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    return int(np.count_nonzero(np.diff(mask.astype(int)) == 1) + (1 if mask[0] else 0))


def _method_stats(burst_mask: np.ndarray, rank_raw: np.ndarray, n_clusters: int) -> dict:
    hi = rank_raw > 0.5      # majority of units in burst state
    lo = rank_raw < 0.1      # clearly resting
    recall = float(burst_mask[hi].mean()) if hi.any() else float("nan")
    fp = float(burst_mask[lo].mean()) if lo.any() else float("nan")
    return {
        "n_burst": int(burst_mask.sum()),
        "recall_hi": recall,
        "fp_resting": fp,
        "runs": _contiguous_runs(burst_mask),
        "n_clusters": n_clusters,
    }


def _scatter_by_cluster(coords, labels, name_prefix):
    """Color points by raw cluster id (noise gray)."""
    traces = []
    palette = iter(CLUSTER_PALETTE)
    cmap = {}
    for c in sorted(set(int(v) for v in labels.tolist())):
        cmap[c] = NOISE_COLOR if c == -1 else next(palette, "#666666")
    for c in sorted(cmap):
        m = labels == c
        if not m.any():
            continue
        traces.append(go.Scattergl(
            x=coords[m, 0], y=coords[m, 1], mode="markers",
            marker=dict(size=3 if c == -1 else 4, color=cmap[c],
                        opacity=0.35 if c == -1 else 0.8, line=dict(width=0)),
            name=f"{name_prefix} {c}", showlegend=False,
        ))
    return traces


def _scatter_by_burst(coords, burst_mask):
    traces = []
    for is_b, col, nm in ((False, NONBURST_COLOR, "non-burst"), (True, BURST_COLOR, "burst")):
        m = burst_mask if is_b else ~burst_mask
        if not m.any():
            continue
        traces.append(go.Scattergl(
            x=coords[m, 0], y=coords[m, 1], mode="markers",
            marker=dict(size=4 if is_b else 3, color=col,
                        opacity=0.85 if is_b else 0.35, line=dict(width=0)),
            name=f"{nm} (n={int(m.sum())})", showlegend=True,
        ))
    return traces


def _table(stats_by_method: dict) -> go.Table:
    methods = list(stats_by_method)
    def col(key, fmt):
        return [fmt(stats_by_method[m][key]) for m in methods]
    f3 = lambda v: "—" if v != v else f"{v:.3f}"
    return go.Table(
        header=dict(values=["<b>method</b>", "<b>n_burst</b>", "<b>recall hi</b>",
                            "<b>FP resting</b>", "<b>runs</b>", "<b>n_clusters</b>"],
                    fill_color="#eee", align="left", font=dict(size=10)),
        cells=dict(values=[methods, col("n_burst", str), col("recall_hi", f3),
                           col("fp_resting", f3), col("runs", str),
                           col("n_clusters", str)],
                   align="left", font=dict(size=10), height=20),
    )


def build_figure(wid, coords, methods, stats):
    """methods: list of (name, labels, burst_mask). Row1 cluster, row2 burst."""
    ncol = len(methods)
    specs = [
        [{"type": "xy"} for _ in range(ncol)],
        [{"type": "xy"} for _ in range(ncol)],
        [{"type": "table", "colspan": ncol}] + [None] * (ncol - 1),
    ]
    titles = ([f"{nm} — clusters" for nm, _, _ in methods]
              + [f"{nm} — burst" for nm, _, _ in methods]
              + ["metrics"])
    fig = make_subplots(rows=3, cols=ncol, specs=specs, subplot_titles=titles,
                        row_heights=[0.42, 0.42, 0.16],
                        vertical_spacing=0.08, horizontal_spacing=0.03)
    for j, (nm, labels, burst) in enumerate(methods, start=1):
        for tr in _scatter_by_cluster(coords, labels, nm):
            fig.add_trace(tr, row=1, col=j)
        for tr in _scatter_by_burst(coords, burst):
            fig.add_trace(tr, row=2, col=j)
    fig.add_trace(_table(stats), row=3, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(
        title=f"<b>{wid}</b> — clustering prototype (UMAP-2 view; burst = high-posterior clusters)",
        height=900, width=1600, plot_bgcolor="white", showlegend=True,
        legend=dict(orientation="h", y=-0.04, font=dict(size=9)),
        margin=dict(l=30, r=20, t=70, b=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--wells", nargs="*", default=["well000", "well015", "well023"],
                   help="well ids to prototype (default: 3 representative wells)")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)

    cfg = json.load(args.config.open())
    analysis_root = Path(cfg["global"]["analysis_root"])
    out_dir = Path(args.output_dir or (Path(cfg["global"].get("figure_root") or analysis_root)
                                       / "_prototype_clustering"))
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = sorted(glob.glob(os.path.join(str(analysis_root), "**",
                                           "ml_burst_detection", "debug_trace.pkl"),
                              recursive=True))
    picked = [t for t in traces if any(w in t for w in args.wells)]
    if not picked:
        print(f"No traces matched wells {args.wells} under {analysis_root}")
        return 1

    import umap

    print(f"{'well':<10} {'method':<18} {'n_burst':>7} {'recall_hi':>9} "
          f"{'FP_rest':>7} {'runs':>5} {'ncl':>4}")
    print("-" * 66)

    for tp in picked:
        tr = insp._load_trace(Path(tp))
        if tr is None or getattr(tr, "feature_matrix", None) is None:
            continue
        X = np.asarray(tr.feature_matrix, float)
        names = list(tr.feature_names or [])
        if RANKING not in names:
            continue
        rank_raw = X[:, names.index(RANKING)]
        wid = next((s for s in tp.split("/") if s.startswith("well")), "well?")

        bg = _background_mask_from_feature(rank_raw, 0.5)
        Xn, _, _ = _znorm_with_stats(X, bg)

        # Shared 2-D UMAP view (what the user inspects).
        coords = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.05,
                           metric="euclidean", random_state=42).fit_transform(Xn)

        cur_labels = np.asarray(tr.hdbscan_labels) if getattr(tr, "hdbscan_labels", None) is not None \
            else np.full(X.shape[0], -1, int)

        # Clustering embeddings (shared where possible).
        umap5 = _umap_embed(Xn, n_components=5)
        pca10 = _pca_embed(Xn, n_components=10)

        a_labels = _hdbscan_on(umap5)                 # A: HDBSCAN on UMAP-5
        bp_labels, bp_k = _spectral(pca10)            # B-pca: spectral on PCA-10
        bu_labels, bu_k = _spectral(umap5)            # B-umap: spectral on UMAP-5
        braw_labels, braw_k = _spectral(Xn)           # B-raw: spectral on raw 26-D (baseline)

        cur_burst = cur_labels == _current_burst_label(tr)
        a_burst, _, _ = _burst_mask_from_clusters(a_labels, rank_raw)
        bp_burst, _, _ = _burst_mask_from_clusters(bp_labels, rank_raw)
        bu_burst, _, _ = _burst_mask_from_clusters(bu_labels, rank_raw)
        braw_burst, _, _ = _burst_mask_from_clusters(braw_labels, rank_raw)

        nc = lambda lab: len({int(c) for c in lab if c >= 0})
        methods = [
            ("current", cur_labels, cur_burst),
            ("A:hdbscan-umap", a_labels, a_burst),
            ("B:spectral-pca10", bp_labels, bp_burst),
            ("B:spectral-umap5", bu_labels, bu_burst),
            ("B:spectral-raw", braw_labels, braw_burst),
        ]
        kmap = {"current": nc(cur_labels), "A:hdbscan-umap": nc(a_labels),
                "B:spectral-pca10": bp_k, "B:spectral-umap5": bu_k,
                "B:spectral-raw": braw_k}
        stats = {nm: _method_stats(burst, rank_raw, kmap[nm]) for nm, _, burst in methods}
        for m, s in stats.items():
            print(f"{wid:<10} {m:<18} {s['n_burst']:>7} {s['recall_hi']:>9.3f} "
                  f"{s['fp_resting']:>7.3f} {s['runs']:>5} {s['n_clusters']:>4}")
        print("-" * 66)

        fig = build_figure(wid, coords, methods, stats)
        out = out_dir / f"{wid}_clustering_prototype.html"
        fig.write_html(out, include_plotlyjs="cdn", full_html=True)
        print(f"wrote {out}")

    print(f"\nDone. HTMLs under {out_dir}")
    return 0


def _current_burst_label(trace) -> int:
    """Burst cluster id the detector picked (top-ranked), or a sentinel."""
    cr = getattr(trace, "cluster_ranking", None)
    if cr:
        try:
            return int(next(iter(cr)))
        except Exception:
            pass
    bl = getattr(trace, "burst_label", None)
    try:
        return int(bl)
    except Exception:
        return -999


if __name__ == "__main__":
    sys.exit(main())
