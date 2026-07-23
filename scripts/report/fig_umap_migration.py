"""F6b — Developmental migration of resting- and bursting-state manifolds.

For one physical well, **all its recordings across DIV are pooled into a single
UMAP embedding** (one StandardScaler + one UMAP fit over the concatenated per-bin
feature matrices). This is the crucial design point: because the embedding is fit
once on every timepoint together, the axes/loadings are identical for DIV 7 and
DIV 30, so where a bin lands is comparable across development — separate per-day
UMAPs would have incomparable axes.

Rows = a Control well and IVH wells (same chip, CX118, to avoid a chip confound).
Columns are **treatment-relative timepoints** — Day 0, Day 14, Day 21 of
treatment (``tau`` = days post-treatment; nearest available recording is used and
its actual day annotated; if Day 21 was not recorded the latest day is shown and
labelled). On each timepoint's fixed axes the resting-state bins (blue) and
bursting-state bins (red) reveal how the network's dynamics move after treatment.

Reuses the pooled-embedding idea from ``scripts/well_recording_umap_overlay.py``.
Requires ``debug=True`` ML-burst runs (``debug_trace.pkl`` present).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load as L
from . import stats as S
from .report_style import caption, save_fig

# UMAP params mirror the pipeline; n_components=2 for display.
_UMAP_KW = dict(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)
# The manifold is *fit* on a subsample for speed, then every bin is *transformed*
# onto those fixed axes — so displayed coverage is 100% while the (slow) fit stays
# bounded. Raise _FIT_SAMPLE for a finer manifold at the cost of fit time.
_FIT_SAMPLE = 20000          # bins used to FIT the UMAP per well
_GREY_SAMPLE = 12000         # bins drawn for the light context layer (display only)
ROW_ARMS = ["Control", "IVH_Early", "IVH_Late"]
# Hand-picked, chip-matched trio for the published F6b (from the candidate
# gallery). CX138 tops out ~Day 15 post-treatment, so timepoints are 0/7/14.
SELECTED_WELLS = {
    "Control":   "CX138|T003346|well015",
    "IVH_Early": "CX138|T003346|well001",
    "IVH_Late":  "CX138|T003346|well014",
}
# 4 columns auto-spaced across each well's own post-treatment window (evenly
# from Day 0 to its latest recorded day), so all four are distinct even when a
# well doesn't reach a fixed Day 21. Column headers are stage labels; each panel
# annotates its actual treatment day.
N_TIMEPOINTS = 4
COL_STAGES = ["Day 0\n(treatment)", "early\n(~⅓ window)",
              "mid\n(~⅔ window)", "latest\nrecorded"]
_REST_C = "#4477aa"
_BURST_C = "#d62728"


def _stride_keep(n: int, k: int, seed: int = 0) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def _pick_well(tidy, analysis_root, arm: str):
    """Well of this arm with the most DIV-spanning debug traces (chip CX118)."""
    wi = S.well_index(tidy)
    wells = wi[(wi.arm == arm) & (wi.chip == "CX118")].well_uid
    best = None
    for w in wells:
        sub = tidy[tidy.well_uid == w]
        divs = [int(r.DIV) for _, r in sub.iterrows()
                if L.ml_trace_path(analysis_root, r).exists()]
        if divs and min(divs) <= 10 and max(divs) >= 28:
            if best is None or len(divs) > best[1]:
                best = (w, len(divs))
    return None if best is None else best[0]


def _thin_recordings(sub, max_recordings):
    """Stride-select up to ``max_recordings`` rows spanning the DIV range."""
    if max_recordings is None or len(sub) <= max_recordings:
        return sub
    idx = np.linspace(0, len(sub) - 1, max_recordings).round().astype(int)
    return sub.iloc[np.unique(idx)]


def _pool_well(tidy, analysis_root, well_uid: str, fit_sample: int = _FIT_SAMPLE,
               max_recordings: int | None = None):
    """Pool bins of a well's recordings → shared 2-D UMAP.

    UMAP is *fit* on a random ``fit_sample`` subsample (speed), then every loaded
    bin is *transformed* onto the fitted axes. ``max_recordings`` thins the
    recording set (for fast gallery thumbnails); None = use all recordings.
    Returns dict with emb (N,2), div/tau/is_burst (N,) or None if no data.
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    sub = _thin_recordings(tidy[tidy.well_uid == well_uid].sort_values("DIV"),
                           max_recordings)
    Xs, divs, taus, burst = [], [], [], []
    for _, r in sub.iterrows():
        p = L.ml_trace_path(analysis_root, r)
        if not p.exists():
            continue
        tr = L.load_ml_trace(analysis_root, r)
        X = np.asarray(getattr(tr, "feature_matrix", None), dtype=float)
        labels = np.asarray(getattr(tr, "hdbscan_labels", []))
        if X is None or X.ndim != 2 or len(labels) != X.shape[0]:
            continue
        bl = set(getattr(tr, "burst_labels", []) or [])
        Xs.append(X)                                    # keep every bin
        divs.append(np.full(X.shape[0], int(r.DIV)))
        taus.append(np.full(X.shape[0], int(r.tau)))
        burst.append(np.array([lab in bl for lab in labels]))
    if not Xs:
        return None
    Xall = np.nan_to_num(np.vstack(Xs))
    Z = StandardScaler().fit_transform(Xall)
    fit_idx = _stride_keep(Z.shape[0], fit_sample)      # subsample only for the fit
    reducer = umap.UMAP(**_UMAP_KW).fit(Z[fit_idx])
    emb = np.asarray(reducer.transform(Z))              # transform 100% of bins
    return dict(emb=emb, div=np.concatenate(divs),
                tau=np.concatenate(taus), is_burst=np.concatenate(burst))


def _auto_days(tau: np.ndarray, n: int = N_TIMEPOINTS) -> list[int]:
    """`n` treatment days evenly spaced across the well's available post-Tx
    recordings (Day 0 → latest). Uses actual recorded days; distinct when the
    well has ≥n post recordings."""
    post = np.unique(tau[tau >= 0])
    if len(post) == 0:
        return []
    idx = np.unique(np.linspace(0, len(post) - 1, n).round().astype(int))
    return [int(d) for d in post[idx]]


def build_umap_migration(tidy, analysis_root):
    import matplotlib.pyplot as plt

    pooled = {}
    for arm in ROW_ARMS:
        w = SELECTED_WELLS.get(arm) or _pick_well(tidy, analysis_root, arm)
        pooled[arm] = (_pool_well(tidy, analysis_root, w), w) if w else (None, None)

    n_rows, n_cols = len(ROW_ARMS), N_TIMEPOINTS
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows),
                             squeeze=False)
    for ri, arm in enumerate(ROW_ARMS):
        d, _ = pooled[arm]
        grey_idx = (_stride_keep(len(d["emb"]), _GREY_SAMPLE)
                    if d is not None else None)
        days = _auto_days(d["tau"]) if d is not None else []
        for ci in range(n_cols):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(COL_STAGES[ci], fontsize=8, fontweight="bold")
            if d is None:
                ax.text(0.5, 0.5, "no debug traces", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                continue
            # faint context: subsample of the full embedding (display only)
            g = d["emb"][grey_idx]
            ax.scatter(g[:, 0], g[:, 1], s=1.0, c="0.88", lw=0, zorder=1,
                       rasterized=True)
            day = days[ci] if ci < len(days) else None
            if day is not None:
                # colored day = 100% of that day's bins
                sel = d["tau"] == day
                rest = sel & ~d["is_burst"]
                brst = sel & d["is_burst"]
                ax.scatter(d["emb"][rest, 0], d["emb"][rest, 1], s=2,
                           c=_REST_C, lw=0, alpha=0.5, zorder=2, rasterized=True)
                ax.scatter(d["emb"][brst, 0], d["emb"][brst, 1], s=4,
                           c=_BURST_C, lw=0, alpha=0.75, zorder=3, rasterized=True)
            note = f"Day {day}" if day is not None else "n/a"
            ax.text(0.03, 0.96, note, transform=ax.transAxes, fontsize=6.5,
                    va="top", ha="left", color="0.25")
            if ci == 0:
                ax.set_ylabel(arm, fontsize=8, fontweight="bold")
    # single resting/burst legend
    axes[0, -1].scatter([], [], s=8, c=_REST_C, label="resting")
    axes[0, -1].scatter([], [], s=8, c=_BURST_C, label="bursting")
    axes[0, -1].legend(loc="lower right", fontsize=6, markerscale=1.5)
    fig.suptitle("Figure 6b — Post-treatment network-state dynamics "
                 "(per-well pooled UMAP; fixed axes across treatment days)",
                 fontsize=8.5, y=1.0)
    caption(fig,
        "Post-treatment network-state migration. One row per arm (a chip-matched "
        "CX138 trio: Control well015, IVH_Early well001, IVH_Late well014); each "
        "column is a developmental stage, auto-spaced from the treatment day "
        "(tau 0) to the well's latest recording. Every point is a time bin "
        "embedded in that well's pooled UMAP (all days fit together, so the axes "
        "are fixed across columns and the drift is comparable); colour = network "
        "state (burst vs rest). Shows how the resting-state cluster migrates and "
        "the trajectory reshapes after treatment. Illustrative single-well "
        "panels, not a group statistic.")
    return fig, {arm: (None if d is None else
                       {int(t): int((d["tau"] == t).sum())
                        for t in _auto_days(d["tau"])})
                 for arm, (d, _) in pooled.items()}


def render(tidy, analysis_root, figure_root):
    fig, meta = build_umap_migration(tidy, analysis_root)
    return save_fig(fig, "F6b_umap_migration", figure_root, subdir="main"), meta


# --------------------------------------------------------------------------- #
# Candidate gallery — one pooled-UMAP-by-DIV thumbnail per candidate well, so a
# representative chip-matched (Control, IVH_Early, IVH_Late) trio can be picked.
# --------------------------------------------------------------------------- #
def _pool_worker(payload):
    """Picklable worker: pool one well (own tidy subset) → (well_uid, dict)."""
    import pandas as pd
    analysis_root, records, well_uid, fit_sample, max_recordings = payload
    sub = pd.DataFrame.from_records(records)
    try:
        d = _pool_well(sub, analysis_root, well_uid, fit_sample, max_recordings)
    except Exception:
        d = None
    return well_uid, d


def render_candidate_gallery(tidy, analysis_root, figure_root, chip: str,
                             arms=("Control", "IVH_Early", "IVH_Late"),
                             fit_sample: int = 3000, max_recordings: int = 8,
                             workers: int = 8):
    import matplotlib.pyplot as plt
    from concurrent.futures import ProcessPoolExecutor

    wi = S.well_index(tidy)
    per_arm = {a: sorted(wi[(wi.arm == a) & (wi.chip == chip)].well_uid) for a in arms}
    all_wells = [w for a in arms for w in per_arm[a]]
    cols = ["well_uid", "DIV", "tau", "recording_key", "rec_name",
            "well_id", "sample_id", "canonical_group"]
    payloads = [(analysis_root,
                 tidy.loc[tidy.well_uid == w, cols].to_dict("records"),
                 w, fit_sample, max_recordings) for w in all_wells]
    pooled: dict = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for wuid, d in ex.map(_pool_worker, payloads):
            pooled[wuid] = d

    ncols = max((len(v) for v in per_arm.values()), default=1)
    nrows = len(arms)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows),
                             squeeze=False)
    divs_all = tidy[tidy.sample_id == chip].DIV
    vmin, vmax = int(divs_all.min()), int(divs_all.max())
    sc = None
    for ri, arm in enumerate(arms):
        wells = per_arm[arm]
        for ci in range(ncols):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            if ci >= len(wells):
                ax.set_visible(False)
                continue
            w = wells[ci]
            d = pooled.get(w)
            wid = w.split("|")[-1]
            if d is None:
                ax.text(0.5, 0.5, f"{wid}\nno traces", ha="center", va="center",
                        transform=ax.transAxes, fontsize=6)
                continue
            sc = ax.scatter(d["emb"][:, 0], d["emb"][:, 1], s=1.5, c=d["div"],
                            cmap="viridis", vmin=vmin, vmax=vmax, lw=0, alpha=0.7,
                            rasterized=True)
            ax.scatter(d["emb"][d["is_burst"], 0], d["emb"][d["is_burst"], 1],
                       s=2, c=_BURST_C, lw=0, alpha=0.7, rasterized=True)
            post = np.unique(d["tau"][d["tau"] >= 0])
            ax.set_title(f"{wid}  DIV{int(d['div'].min())}-{int(d['div'].max())}\n"
                         f"maxDay{int(post.max()) if len(post) else 'NA'}",
                         fontsize=6)
            if ci == 0:
                ax.set_ylabel(arm, fontsize=8, fontweight="bold")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, shrink=0.5, pad=0.02)
        cb.set_label("DIV", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"F6b candidate gallery — {chip} (red = burst bins; "
                 f"DIV colour; pick one chip-matched trio)", fontsize=9, y=1.0)
    return save_fig(fig, f"F6b_gallery_{chip}", figure_root, subdir="gallery",
                    formats=("png",))
