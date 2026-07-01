"""Throwaway prototype: trajectory-coherence burst selection (read-only).

Goal: replace the static per-cluster magnitude threshold in
`_select_burst_clusters` with an *intrinsic*, self-calibrating rule based on the
temporal trajectory structure that UMAP exposes — so it adapts across cultures
with no magnitude knob.

Hypothesis (from CX138/000012 diagnosis):
  - A real network burst is a COHERENT MULTI-CLUSTER EXCURSION off the dwelling
    manifold: long contiguous runs, clusters that co-travel in temporal sequence
    (onset->peak->decay), recurring with similar composition.
  - Junk type 1 (A2 cl2,3): isolated single-bin flickers — high amplitude but no
    coherence / co-travel.
  - Junk type 2 (D5 cl22): a diffuse pervasive elevated-baseline state — high
    occupancy, low activity, part of the dwelling manifold, not a transient.

This script computes per-cluster INTRINSIC structural metrics (all relative /
scale-free) and prints them next to the "known" answer so we can find the rule:
  A2 real = {0,4,5}, junk = {2,3}
  D5 real = {1,5,6,7,8,13,15} (tight high-PF sweeps), junk = {22} (+ marginal)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import inspect_ml_bursts as insp  # noqa: E402

np.set_printoptions(suppress=True, precision=3)


def _runs(mask: np.ndarray):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return list(zip(s, e))


def analyze(d: str, tag: str, known_real: set[int], known_junk: set[int]):
    t = pickle.load(open(d + "/debug_trace.pkl", "rb"))
    lab = np.asarray(t.hdbscan_labels)
    tc = np.asarray(t.t_centers)
    X = np.asarray(t.feature_matrix)
    fn = list(t.feature_names)
    N = len(lab)
    pf = fn.index("post_frac_gt_0_5")
    part_i = fn.index("participation")
    clusters = sorted(c for c in set(lab) if c >= 0)

    # ---- UMAP 2D for geometry (relative amplitude) ----
    res = insp._compute_umap_axes(t, max_bins=20000, n_neighbors=30, min_dist=0.05)
    coords, _, _, kept = res
    full = np.full((N, 2), np.nan)
    full[kept] = coords
    klab = lab[kept]

    # ---- dwelling manifold (data-driven): the trajectory's quiescent core ----
    # global rest = cluster with the LOWEST mean participation (most silent),
    # tie-broken by highest occupancy. No magnitude threshold — pure ranking.
    part_by_c = {c: float(X[lab == c, part_i].mean()) for c in clusters}
    occ_by_c = {c: int((lab == c).sum()) for c in clusters}
    rest = min(clusters, key=lambda c: (part_by_c[c], -occ_by_c[c]))
    rest_centroid = np.nanmean(full[klab == rest], axis=0)
    # trajectory spread (scale): median distance of all bins from rest centroid
    dist_all = np.hypot(full[:, 0] - rest_centroid[0], full[:, 1] - rest_centroid[1])
    spread = np.nanmedian(dist_all)

    # ---- transition graph ----
    nxt = lab[1:]
    cur = lab[:-1]

    # ---- excursions off rest (for co-travel) ----
    notrest = lab != rest
    excur = _runs(notrest)
    # map each bin to its excursion's distinct-cluster count
    multi_bin = np.zeros(N, bool)
    for s, e in excur:
        comp = {int(c) for c in lab[s:e] if c >= 0}
        if len(comp) >= 2:
            multi_bin[s:e] = True

    # ---- feature-space amplitude (z-normed dist from rest centroid) ----
    # avoids any UMAP dependency in the detector — works in 'none' embedding too.
    mu = np.asarray(t.scaler_mean, float)
    sd = np.asarray(t.scaler_std, float)
    sd = np.where(sd < 1e-9, 1e-9, sd)
    Z = (X - mu) / sd
    zrest = Z[lab == rest].mean(axis=0)
    zdist = np.linalg.norm(Z - zrest, axis=1)
    zspread = float(np.median(zdist))

    print(f"\n===== {tag} =====  N={N}  rest=cl{rest} (part={part_by_c[rest]:.3f})  "
          f"spread_umap={spread:.2f} spread_feat={zspread:.2f}")
    hdr = ("cl  occ%  mRun  toRest  cotrav  amp_um  amp_ft  part   PF    | known  PICK")
    print(hdr)
    rows = {}
    TO_REST_MAX = 0.5
    AMP_MIN = 1.3        # relative to trajectory spread (scale-free)
    for c in clusters:
        m = lab == c
        n = int(m.sum())
        runs = _runs(m)
        mrun = n / max(len(runs), 1)
        to_rest = float(((cur == c) & (nxt == rest)).sum() / max((cur == c).sum(), 1))
        cotr = float(multi_bin[m].mean())
        amp = float(np.nanmean(dist_all[m]) / spread) if spread > 0 else 0.0
        amp_ft = float(zdist[m].mean() / zspread) if zspread > 0 else 0.0
        is_rest = c == rest
        pick = (not is_rest) and (to_rest < TO_REST_MAX) and (amp >= AMP_MIN)
        lab_known = "REAL" if c in known_real else ("junk" if c in known_junk else "rest" if is_rest else "-")
        rows[c] = dict(to_rest=to_rest, amp=amp, amp_ft=amp_ft, pick=pick, known=lab_known)
        flag = ""
        if lab_known == "REAL" and not pick: flag = " <<MISS"
        if lab_known == "junk" and pick: flag = " <<FALSE+"
        print(f"{c:3d} {100*n/N:5.1f} {mrun:5.1f} {to_rest:6.2f} {cotr:6.2f} "
              f"{amp:6.2f} {amp_ft:6.2f} {part_by_c[c]:5.3f} {float(X[m,pf].mean()):5.3f}  "
              f"| {lab_known:4s} {'BURST' if pick else '.'}{flag}")
    picked = {c for c, r in rows.items() if r["pick"]}
    miss = known_real - picked
    falsepos = known_junk & picked
    print(f"  RULE to_rest<{TO_REST_MAX} & amp_um>={AMP_MIN}: picked={sorted(picked)}")
    print(f"  recall(known real): {len(known_real & picked)}/{len(known_real)} miss={sorted(miss)} | falsepos(known junk)={sorted(falsepos)}")
    return rows


def sweep_all(B: str, to_rest_max=0.5, amp_min=1.3):
    """Fast feature-space-amplitude rule across all 24 wells. No UMAP recompute.
    Reports per well: rest, n_clusters, current burst_labels count vs new pick."""
    import glob, json
    print(f"\n##### 24-well sweep (feature-space amp, to_rest<{to_rest_max} & amp_ft>={amp_min}) #####")
    print(f"{'well':>7} {'nclu':>4} {'cur':>4} {'new':>4} {'rest':>4} {'dropped(cur-new)':>18}")
    for d in sorted(glob.glob(B + "/rec*/well*/ml_burst_detection")):
        tp = Path(d) / "debug_trace.pkl"
        if not tp.exists():
            continue
        t = pickle.load(open(tp, "rb"))
        lab = np.asarray(t.hdbscan_labels)
        X = np.asarray(t.feature_matrix, float)
        fn = list(t.feature_names)
        part_i = fn.index("participation")
        clusters = sorted(c for c in set(lab) if c >= 0)
        if not clusters:
            continue
        occ = {c: int((lab == c).sum()) for c in clusters}
        partc = {c: float(X[lab == c, part_i].mean()) for c in clusters}
        rest = min(clusters, key=lambda c: (partc[c], -occ[c]))
        mu = np.asarray(t.scaler_mean, float); sd = np.asarray(t.scaler_std, float)
        sd = np.where(sd < 1e-9, 1e-9, sd)
        Z = (X - mu) / sd
        zrest = Z[lab == rest].mean(0)
        zd = np.linalg.norm(Z - zrest, axis=1)
        zspread = float(np.median(zd)) or 1.0
        cur = lab[:-1]; nxt = lab[1:]
        pick = []
        for c in clusters:
            if c == rest:
                continue
            tr = float(((cur == c) & (nxt == rest)).sum() / max((cur == c).sum(), 1))
            amp = float(zd[lab == c].mean() / zspread)
            if tr < to_rest_max and amp >= amp_min:
                pick.append(c)
        cur_bl = set(int(x) for x in (t.burst_labels or []))
        well = d.split("/well")[1].split("/")[0]
        dropped = sorted(cur_bl - set(pick))
        print(f"{well:>7} {len(clusters):4d} {len(cur_bl):4d} {len(pick):4d} {rest:4d}  {str(dropped):>18}")


if __name__ == "__main__":
    B = ("/mnt/benshalom-nas/analysis/Sadegh/new/CX138/ml_burst_data_umap/"
         "CX138/260325/T003346/Network/000012")
    if "--sweep" in sys.argv:
        sweep_all(B)
    else:
        analyze(f"{B}/rec0000/well001/ml_burst_detection", "A2 well001",
                known_real={0, 4, 5}, known_junk={2, 3})
        analyze(f"{B}/rec0003/well022/ml_burst_detection", "D5 well022",
                known_real={1, 5, 6, 7, 8, 13, 15}, known_junk={22})
