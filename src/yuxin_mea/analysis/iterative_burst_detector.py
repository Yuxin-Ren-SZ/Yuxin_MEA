"""Iterative contrast-maximizing network burst detector.

Algorithm overview
------------------
The core problem: a single participation-signal threshold works poorly when
recordings differ in baseline firing rate, unit count, or noise level, because
it has no way to learn what "different from background" means for a given well.

This detector instead treats burst detection as an iterative two-class
separation problem:

  1. PERMISSIVE SEED  — over-detect using a low threshold on the participation
     signal. This produces many false positives but ensures no true burst is
     missed at the start.

  2. FEATURE MATRIX  — compute 8 biologically motivated signals per time bin:
       F0  PFR              population firing rate (Hz)
       F1  P                participation fraction (fraction of active units)
       F2–F5  FF×4          spatial Fano Factor at 4 temporal scales
       F6  LLR              per-unit Poisson log-likelihood ratio vs background
       F7  burstiness       mean instantaneous ISI reciprocal

  3. ITERATE until convergence:
       a. Estimate per-unit background rates from non-candidate bins
       b. Update the LLR feature (depends on background estimate)
       c. Z-score all features relative to background
       d. Fit Fisher's linear discriminant on the current burst/non-burst labels
          to find the projection direction w that maximises between-class
          variance relative to within-class variance
       e. Compute composite(t) = X_norm[t] @ w  (single discriminant score)
       f. Re-threshold composite to get new burst/non-burst labels
       g. Trim candidate boundaries where composite falls below extent threshold
       h. Merge nearby candidates whose separating valley is still "bursty"
       i. Check convergence: stop when < 0.5 % of bins change label

  4. HIERARCHY  — apply the same two-tier merge used by the original detector:
       burstlets  →(merge_strict)→  network_bursts
       network_bursts  →(merge_clustered)→  superbursts

  5. BURSTLET GATE — use burstlet-level llr_aggregate as a soft quality gate
     to remove bridge-like false positives without deleting an entire mixed
     recording that contains both silent and bursty sections.

  6. OUTPUT  — BurstResults with the standard schema plus four quality columns
     per event: llr_aggregate, composite_peak, composite_mean, ff_peak.

Key design choices
------------------
- Fisher LDA is re-fit every iteration so the composite signal adapts to
  whichever features actually discriminate in each recording. Iteration 0 uses
  a fixed participation-heavy prior to bootstrap the label set.
- Background rates are per-unit (not pooled) so heterogeneous networks where
  some units fire much faster than others are handled correctly.
- Fano Factor is computed at four temporal scales. Short scales capture fast
  synchrony inside a burstlet; coarser scales capture slower network-wide
  co-activation. Fisher LDA learns which scales are informative.
- LLR is signed (positive = elevated above background, negative = suppressed),
  so it contributes positively to the composite during bursts and near-zero
  during quiet periods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


class IterativeBurstError(ValueError):
    """Raised when spike data is insufficient for iterative burst detection."""


@dataclass(frozen=True)
class IterativeBurstConfig:
    """All tunable parameters for the iterative burst detector.

    Phase 1 — permissive seeding
    ----------------------------
    permissive_mad_scale : float
        MAD multiplier for the initial participation threshold (0.30 = very
        permissive, roughly half the normal 0.75 setting). Lower → more seed
        candidates (more false positives that iterations must eliminate).
    permissive_percentile : float
        Fallback initial threshold when spread_mad is near zero (sparse or
        uniform recordings where MAD-based thresholding is meaningless).
        70.0 means "top 30 % of bins by participation fraction".
    mad_fallback_threshold : float
        If spread_mad < this value the percentile fallback is used instead of
        the MAD method.

    Phase 3 — iterative refinement
    --------------------------------
    composite_mad_scale : float
        MAD multiplier applied to the composite signal background distribution
        to set the burst/non-burst threshold each iteration. Matches the
        original detector's 0.75 default once the composite is well-calibrated.
    extent_frac : float
        After re-thresholding, candidate edges are trimmed inward until the
        composite value exceeds max(threshold, extent_frac × peak_composite).
        Controls how tightly boundaries follow the burst envelope.
    merge_floor_frac : float
        During iteration, two adjacent candidates are merged if their separating
        valley is above merge_floor_frac × threshold. 0.70 is relaxed (allows
        merges even when the valley dips somewhat below the detection threshold).
        The final hierarchy merge uses a stricter criterion.
    network_merge_gap_min_s : float
        Minimum gap enforced for the network-burst merge stage in the hierarchy,
        regardless of the biological ISI estimate.
    max_iterations : int
        Hard cap on refinement iterations (safety valve; typically converges
        in 5–10 iterations).
    convergence_eps : float
        Stop iterating when fewer than this fraction of bins change their
        burst/non-burst label relative to the previous iteration.

    Fisher regularization
    ---------------------
    fisher_alpha_frac : float
        Ridge regularization for the within-class scatter matrix S_W:
            alpha = fisher_alpha_frac × trace(S_W) / n_features
        Prevents numerical issues when features are nearly collinear or one
        class has very few samples. 1e-3 is mild regularization.

    Fano Factor scales
    ------------------
    ff_scale_multipliers : tuple of float
        Each multiplier is applied to the adaptive bin size to produce one
        Fano Factor feature. E.g. (0.5, 1.0, 2.0, 5.0) with bin_size=20 ms
        gives FF at 10 ms, 20 ms, 40 ms, 100 ms. Clamped to [5, 100] ms.
        Fisher LDA learns which scale(s) best discriminate for each recording.
    """

    # Phase 1 — initial detection
    permissive_mad_scale: float = 0.30
    permissive_percentile: float = 70.0
    mad_fallback_threshold: float = 0.01

    # Phase 3 — iteration
    composite_mad_scale: float = 0.75
    extent_frac: float = 0.30
    merge_floor_frac: float = 0.70
    network_merge_gap_min_s: float = 0.75
    max_iterations: int = 20
    convergence_eps: float = 0.005

    # Fisher regularization
    fisher_alpha_frac: float = 1e-3

    # FF multi-scale bin-size multipliers
    ff_scale_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)

    # Event-level modulation gate
    min_burst_modulation: float = 0.1
    """Minimum burstlet-level llr_aggregate required for an event to survive.
    LLR is used because it is an absolute measure (deviation from each unit's
    own baseline) that does not collapse to near-zero for dense uniform-noise
    recordings the way the composite signal does.  A value ≤ 0 disables the
    gate and keeps the pre-filtered burstlets."""

    # Post-convergence event clustering
    cluster_events: bool = True
    """After convergence, fit a multi-component GMM on per-event quality
    features and merge similar components before discarding the noise-like
    clusters. Automatically skipped when fewer than cluster_min_events are
    detected."""

    cluster_initial_components: int = 6
    """Initial number of GMM components used before similarity-based merging.
    Larger values let the detector over-segment first and then merge similar
    components when the iteration itself cannot separate them cleanly."""

    cluster_min_events: int = 5
    """Minimum detected events required to attempt GMM clustering."""

    cluster_min_separation: float = 1.5
    """Maximum normalised Euclidean distance between component means for them
    to be merged in standardized feature space."""

    # Inner partitioner — bin-level burst/background separation each iteration
    inner_partitioner: str = "fisher_lda"
    """Which method discriminates burst vs background bins each iteration.

    ``"fisher_lda"`` (default) fits Fisher's linear discriminant on the
    z-normed feature matrix.  Augmented with two safeguards added after the
    diagnostic on ``cx138_44_02`` revealed that the LDA could flip sign on
    heterogeneous (silence + tonic + burst) recordings:

      - Silence excision: bins with zero active units are removed from both
        the burst and background classes, and from the ``_znorm`` background
        statistics, so the "background" class is not dragged toward zero by
        long silent stretches.
      - Sign pinning: a new ``w`` is rejected (the previous iteration's
        weights are kept) when ``w_PFR``, ``w_P``, or ``w_LLR`` is negative.
        These three features are biologically constrained to be elevated
        during bursts; a negative weight is a signal the LDA found the
        wrong contrast and would otherwise drag the next iteration in the
        wrong direction.

    ``"gmm_em"`` fits a BIC-adaptive Gaussian Mixture Model on the same
    feature matrix and uses the burst-component posterior as the per-bin
    score.  Best suited to recordings where the latent regime structure is
    not well captured by a single discriminant direction.  Currently less
    stable across iterations than the LDA path on simple 2-regime data
    because GMM relabels its components each fit.
    """

    gmm_k_range: tuple[int, int] = (2, 3)
    """Inclusive ``(k_min, k_max)`` component count range swept by the
    BIC-based GMM-EM partitioner each iteration.  Capped at 3 by default
    because the natural latent structure of MEA recordings is
    silence / tonic firing / true burst — allowing more components causes
    the burst regime itself to split into ramp/peak/tail subclusters and
    destabilises the burst-component selection across iterations."""

    gmm_bic_margin: float = 5.0
    """A new ``k*`` must beat the previous iteration's ``k*`` BIC by this
    margin before the chosen component count is allowed to change.  Suppresses
    k-flapping when several values of k fit the data nearly equally well."""

    gmm_em_n_init: int = 5
    """``GaussianMixture(n_init=...)`` — number of random restarts per fit."""

    gmm_em_reg_covar: float = 1e-4
    """``GaussianMixture(reg_covar=...)``.  Slightly above the sklearn default
    because the z-normed feature matrix contains near-collinear FF columns
    that can produce singular covariance estimates."""

    gmm_burst_score_weights: tuple[float, ...] = (
        0.20, 0.25, 0.05, 0.10, 0.10, 0.05, 0.20, 0.05,
    )
    """Burst-component scoring prior aligned to the bin feature order
    ``[PFR, P, FF0, FF1, FF2, FF3, LLR, burst]``.  After the GMM is fit and
    near-duplicate components are merged, each merged group's centroid is
    scored by ``weights @ centroid`` (centroid is in standardized space).
    The highest-scoring group is designated as the burst cluster, so the
    sign pattern of this prior pins burst semantics (high P, high LLR, high
    PFR, high burstiness) regardless of how the GMM labels its components."""

    gmm_posterior_threshold: float | None = 0.5
    """Candidate threshold for the GMM-EM posterior.  Because the posterior is
    naturally bounded in ``[0, 1]`` and tends to be near-binary at convergence,
    a fixed cut at ``0.5`` is the natural choice (every bin where the burst
    component is more likely than the rest combined).  Set to ``None`` to fall
    back to the MAD-based threshold used by the Fisher path
    (``median(composite[bg]) + composite_mad_scale * MAD(composite[bg])``)."""

    # Fisher LDA stability safeguards (only used when inner_partitioner == "fisher_lda")
    lda_exclude_silence: bool = True
    """Exclude truly silent bins (``active_unit_count == 0``) from both the
    burst and background classes inside ``_fit_fisher``, and from the
    background statistics used by ``_znorm``.  Prevents heterogeneous
    recordings (silence + tonic + burst) from collapsing the LDA onto the
    silence-vs-everything contrast — the failure mode observed on
    ``cx138_44_02`` where ``w_PFR`` converged to ``-0.81``."""

    lda_sign_pinned_feature_names: tuple[str, ...] = ("PFR", "P", "LLR")
    """Feature names whose Fisher weight must be non-negative for the new
    direction to be accepted.  When any pinned weight is negative the LDA
    step is rejected and the previous iteration's ``w`` is kept (as for
    other degenerate solves).  These features are biologically constrained
    to be elevated during bursts; a negative weight is a sign-flip
    indicating the LDA found the wrong contrast."""

    gmm_component_merge_distance: float = 0.5
    """Standardized-Euclidean distance below which sibling GMM components are
    collapsed into a single group before centroid scoring.  Prevents
    degenerate EM solutions (two near-identical components for the same
    regime) from splitting the burst posterior across two columns."""

    gmm_burst_top_fraction: float = 1.0
    """Selects which merged groups contribute to the burst posterior.  Keeps
    every group whose centroid score is at least
    ``gmm_burst_top_fraction × top_score``.

    ``1.0`` (default) → top group only.  Use a smaller value (e.g. ``0.5``)
    when you want a multi-component burst regime (ramp + peak + plateau) to
    be unioned together rather than collapsing onto a single component.  In
    practice the default of 1.0 combined with ``gmm_k_range = (2, 3)`` is
    the most stable across iterations — the GMM's three clusters already
    correspond to silence / tonic / burst, and the top group is
    unambiguously the burst one."""


@dataclass
class IterativeBurstTrace:
    """Optional diagnostic bundle for inspecting why candidates get killed.

    When passed to ``compute_iterative_bursts`` the detector populates this in
    place with intermediate state from every kill stage (iteration trimming,
    participation floor, BMI gate, GMM clustering). Pass ``None`` (default) and
    no extra state is captured. Used by ``notebooks/07_*.ipynb`` to visualise
    the kill pipeline and PCA-project the GMM feature space.
    """

    iterations: list[dict] = field(default_factory=list)
    burstlets_pre_gates: list[dict] = field(default_factory=list)
    participation_gate: dict | None = None
    bmi_gate: dict | None = None
    gmm: dict | None = None
    t_centers: np.ndarray | None = None
    bin_size: float | None = None
    feature_names: list[str] | None = None
    unit_ids: list[str] | None = None


# ---------------------------------------------------------------------------
# Helpers — summary statistics (used for aggregate metrics output)
# ---------------------------------------------------------------------------

def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0}
    mean_val = float(x.mean())
    std_val = float(x.std())
    cv = std_val / mean_val if abs(mean_val) > 1e-12 else float("nan")
    return {"mean": mean_val, "std": std_val, "cv": float(cv)}


def _level_metrics(events: list[dict], total_dur: float) -> dict:
    if not events:
        return {}
    starts = [ev["start"] for ev in events]
    return {
        "count": len(events),
        "rate": len(events) / total_dur,
        "duration": _stats([ev["duration_s"] for ev in events]),
        "inter_event_interval": _stats(np.diff(starts)) if len(starts) > 1 else _stats([]),
        "intensity": _stats([ev["synchrony_energy"] for ev in events]),
        "participation": _stats([ev["participation"] for ev in events]),
        "spikes_per_burst": _stats([ev["total_spikes"] for ev in events]),
        "burst_peak": _stats([ev["burst_peak"] for ev in events]),
        "peak_synchrony": _stats([ev["peak_synchrony"] for ev in events]),
    }


# ---------------------------------------------------------------------------
# Helpers — feature computation (F0–F7)
# ---------------------------------------------------------------------------

def _compute_spike_matrix(
    spike_times: dict,
    units: list,
    bins: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Build the (n_units × n_bins) spike count matrix.

    Each row is a unit; each column is an adaptive time bin. This matrix is
    the shared substrate for all per-bin features: summing columns gives PFR
    and participation; computing Var/Mean across rows at each column gives the
    spatial Fano Factor; comparing against per-unit background expectations
    gives the LLR signal.
    """
    matrix = np.zeros((len(units), n_bins), dtype=np.float32)
    for i, u in enumerate(units):
        spk = np.asarray(spike_times[u])
        if spk.size > 0:
            counts, _ = np.histogram(spk, bins=bins)
            matrix[i] = counts
    return matrix


def _compute_multiscale_ff(
    spike_matrix: np.ndarray,
    bins: np.ndarray,
    t_centers: np.ndarray,
    bin_size: float,
    ff_scale_multipliers: tuple,
) -> np.ndarray:
    """Spatial Fano Factor at multiple temporal scales. Returns (n_bins, n_scales).

    Fano Factor (FF) at a given temporal scale Δt:
        FF(t) = Var( spike_counts_per_unit in [t, t+Δt) ) /
                Mean( spike_counts_per_unit in [t, t+Δt) )

    For independent Poisson firing:  FF ≈ 1  (variance equals mean).
    During synchronised bursting:    FF >> 1  (units fire together, so the
        variance across units is much larger than the mean count).
    During silence:                  FF → 0  (all units have 0 counts).

    Why multiple scales?
    A short burstlet (< 100 ms) shows elevated FF at fine scales (5–20 ms)
    but is too brief to leave a strong signal at coarse scales. A long network
    burst shows elevated FF at all scales. By providing 4 scales, Fisher's
    discriminant can learn which temporal resolution is most informative for
    each recording, rather than committing to a single scale up front.

    Implementation:
    Rather than re-histogramming from raw spike times at each scale, we reuse
    the fine-resolution spike_matrix by summing columns that fall within each
    coarse bin. This is equivalent to re-binning but avoids re-reading spike
    times. The coarse FF value is then broadcast back to all fine bins within
    that coarse window, and lightly smoothed (sigma=1.5 bins) to reduce
    single-bin artefacts.
    """
    n_bins = len(t_centers)
    n_scales = len(ff_scale_multipliers)
    ff_signals = np.zeros((n_bins, n_scales))
    rec_start = float(bins[0])
    rec_end = float(bins[-1])

    # TODO is there a faster way to do this without a Python loop over scales? The loop is only over 4 scales so it's not a bottleneck, but it would be cleaner if we could vectorize it.
    for k, mult in enumerate(ff_scale_multipliers):
        # Clamp scale to [5 ms, 100 ms] regardless of the adaptive bin size
        # TODO parameterize the min and max.
        scale_dt = float(np.clip(mult * bin_size, 0.005, 0.1))

        coarse_bins = np.arange(rec_start, rec_end + scale_dt, scale_dt)
        n_coarse = len(coarse_bins) - 1
        if n_coarse < 1:
            continue

        # Map each fine-resolution bin center to its coarse bin index
        coarse_idx = np.clip(
            np.searchsorted(coarse_bins, t_centers, side="right") - 1,
            0, n_coarse - 1,
        )

        # Aggregate per-unit spike counts into coarse bins via bincount (vectorized)
        n_units_ff = spike_matrix.shape[0]
        unit_coarse = np.zeros((n_units_ff, n_coarse), dtype=np.float64)
        for i in range(n_units_ff):
            unit_coarse[i] = np.bincount(coarse_idx, weights=spike_matrix[i], minlength=n_coarse)
        mean_c = unit_coarse.mean(axis=0)   # (n_coarse,)
        var_c = unit_coarse.var(axis=0)     # (n_coarse,)
        valid = mean_c > 1e-9
        ff_coarse = np.where(valid, var_c / np.where(valid, mean_c, 1.0), 0.0)

        # Broadcast coarse FF back to fine resolution, then lightly smooth
        # TODO is the smoothing necessary? Lookslike yes.
        ff_fine = ff_coarse[coarse_idx]
        ff_signals[:, k] = gaussian_filter1d(ff_fine, sigma=1.5)

    return ff_signals


def _compute_llr_signal(
    spike_matrix: np.ndarray,
    lambda_bg_per_unit: np.ndarray,
    bin_size: float,
    sigma: float,
) -> np.ndarray:
    """Signed per-unit Poisson log-likelihood ratio, averaged and smoothed.

    For each bin t and each unit u, the Poisson LLR tests whether the observed
    spike count n_u(t) is consistent with the unit's background firing rate:

        H0: spikes ~ Poisson( λ_u_bg × bin_size )
        H1: spikes ~ Poisson( λ_u_obs )

    The score-test form of the LLR is:
        LLR_u(t) = 2 × [ n_u(t) × ln( n_u(t) / expected_u ) − (n_u(t) − expected_u) ]
    where expected_u = λ_u_bg × bin_size.

    This is always non-negative (it's a KL divergence × 2). We apply a sign:
        +  when n_u(t) > expected_u  (burst: firing above background)
        −  when n_u(t) < expected_u  (suppression: firing below background)

    Then average across units. The resulting signal is:
        ≫ 0  during bursts  (most units fire far above their individual baselines)
        ≈ 0  during background  (observed ≈ expected by construction of λ_bg)
        < 0  rare, only when a unit is transiently suppressed below its baseline

    The signal is smoothed with sigma_slow (≈ 5 × biological ISI in bins) to
    integrate evidence over a biologically meaningful window rather than reacting
    to single-bin coincidences in sparse units.

    Why per-unit λ_bg?
    In heterogeneous MEA networks, some units fire at 0.1 Hz and others at 5 Hz.
    A shared global background rate would make the fast unit look "always bursting"
    and the slow unit look "never bursting". Per-unit rates give each unit equal
    sensitivity to detect its own rate elevation during a burst.
    """
    # expected_u shape: (n_units, 1) — broadcasts across bins
    expected = (lambda_bg_per_unit * bin_size)[:, np.newaxis]

    # Numerically safe substitutes for log(0) and 0/0 cases
    safe_expected = np.where(expected > 1e-12, expected, 1.0)
    safe_n = np.where(spike_matrix > 0, spike_matrix, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Full two-sided LLR (always ≥ 0)
        llr_u = np.where(
            (spike_matrix > 0) & (expected > 1e-12),
            2.0 * (spike_matrix * np.log(safe_n / safe_expected) - (spike_matrix - expected)),
            # When n=0: LLR = 2×(0 − (0 − expected)) = 2×expected > 0, but
            # sign will be −1 since 0 < expected, so contribution is −2×expected ≈ 0
            np.where(expected > 1e-12, 2.0 * expected, 0.0),
        )
        # Apply sign so the signal is directional: + = elevated, − = suppressed
        signed = np.sign(spike_matrix - expected) * np.abs(llr_u)

    # Average across units, then smooth to integrate evidence over ~5 ISI window
    llr = np.mean(signed, axis=0)
    return gaussian_filter1d(llr, sigma=sigma)


def _compute_burstiness(
    spike_times: dict,
    units: list,
    bins: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Per-bin mean instantaneous firing rate (reciprocal of ISI), smoothed.

    For each consecutive spike pair (t_i, t_{i+1}) of a unit, the instantaneous
    rate 1/ISI is assigned to the bin containing the ISI midpoint. Bins that
    fall in a burst have many short ISIs → high instantaneous rate. Background
    bins have only occasional long ISIs → low instantaneous rate.

    This differs from the population firing rate (PFR) in that it measures
    *within-unit* temporal tightness rather than total spike count. A single
    very fast unit in a bin elevates burstiness without elevating PFR much,
    while a burst where all units fire once each elevates PFR without affecting
    per-unit ISIs. Together, PFR and burstiness cover complementary aspects
    of burst structure.

    Light Gaussian smoothing (sigma=2 bins) fills gaps where a unit had no ISI
    midpoint in a given bin despite active firing in adjacent bins.
    """
    raw = np.zeros(n_bins)
    cnt = np.zeros(n_bins)

    for u in units:
        spk = np.sort(np.asarray(spike_times[u]))
        if len(spk) < 2:
            continue
        isi = np.diff(spk)
        midpoints = (spk[:-1] + spk[1:]) / 2
        # Place each ISI's reciprocal at its midpoint bin
        idx = np.clip(np.searchsorted(bins[1:], midpoints), 0, n_bins - 1)
        inst = 1.0 / np.maximum(isi, 1e-6)  # cap at 1 MHz to avoid inf
        np.add.at(raw, idx, inst)
        np.add.at(cnt, idx, 1)

    # Divide only where at least one ISI midpoint landed
    result = np.divide(raw, cnt, out=np.zeros_like(raw), where=cnt > 0)
    return gaussian_filter1d(result, sigma=2.0)


# ---------------------------------------------------------------------------
# Helpers — Fisher's linear discriminant
# ---------------------------------------------------------------------------

def _znorm(
    X: np.ndarray,
    bg_mask: np.ndarray,
) -> np.ndarray:
    """Z-score feature matrix using statistics from background bins only.

    Features have very different units and scales (Hz for PFR, dimensionless
    [0,1] for participation, dimensionless ≥ 0 for FF, LLR in nats²). After
    Z-scoring relative to the background distribution, all features are on the
    same scale (standard deviations above/below background mean) so Fisher's
    discriminant can meaningfully compare their contributions.

    Using background-only statistics (not global statistics) ensures the
    Z-score reflects "how unusual is this bin compared to quiet periods" rather
    than "how unusual is this bin compared to the mix of burst and non-burst".
    """
    bg = X[bg_mask]
    mu = bg.mean(axis=0)
    std = bg.std(axis=0)
    # Replace near-zero std with 1.0 to avoid division by zero on constant features
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mu) / std


def _fit_fisher(
    X_norm: np.ndarray,
    labels: np.ndarray,
    alpha_frac: float,
) -> np.ndarray | None:
    """Compute Fisher's linear discriminant direction.

    Fisher's LDA finds the unit-norm weight vector w that maximises the ratio
    of between-class variance to within-class variance when projecting X onto w:

        w* = S_W^{-1} (μ_burst − μ_background)

    where S_W = S_burst + S_background is the total within-class scatter matrix.

    Geometric intuition: w points in the direction that is simultaneously
    (a) aligned with the vector from the background centroid to the burst
        centroid (separating the class means), and
    (b) perpendicular to the directions of high within-class spread (ignoring
        variance shared by both classes).

    Iteration dynamics: each iteration re-labels bins based on the previous
    iteration's composite signal, re-estimates background rates, and re-fits w.
    As the burst/background partition improves, the class centroids become more
    extreme (bursts look more bursty, background looks flatter) and S_W shrinks,
    so the discriminant becomes sharper. This is analogous to EM for a Gaussian
    mixture model.

    Ridge regularisation (alpha × I) is added to S_W before solving to handle:
    - near-collinear features (e.g., FF at adjacent scales are correlated)
    - small sample sizes in one class (very sparse recordings)
    alpha = alpha_frac × trace(S_W) / n_features scales with the data magnitude.

    Returns None when either class has < 3 samples or the solve fails, in which
    case the caller keeps the previous iteration's weight vector.
    """
    n_features = X_norm.shape[1]
    idx1 = labels == 1   # burst bins
    idx0 = labels == 0   # background bins

    if idx1.sum() < 3 or idx0.sum() < 3:
        return None

    X1, X0 = X_norm[idx1], X_norm[idx0]
    mu1, mu0 = X1.mean(axis=0), X0.mean(axis=0)

    # Within-class scatter: sum of deviations from each class mean
    S_W = (X1 - mu1).T @ (X1 - mu1) + (X0 - mu0).T @ (X0 - mu0)

    # Ridge regularisation scaled to the matrix's own magnitude
    alpha = alpha_frac * float(np.trace(S_W)) / n_features
    S_W_reg = S_W + alpha * np.eye(n_features)

    try:
        # Solve S_W_reg @ w = (μ1 − μ0) rather than explicitly inverting S_W
        w = np.linalg.solve(S_W_reg, mu1 - mu0)
    except np.linalg.LinAlgError:
        return None

    norm = float(np.linalg.norm(w))
    return w / norm if norm > 1e-12 else None


def _fit_gmm_em(
    X_norm: np.ndarray,
    prev_k: int | None,
    config: "IterativeBurstConfig",
    prev_burst_centroid: np.ndarray | None = None,
) -> dict | None:
    """Fit a multi-component GMM and return the burst-component posterior.

    Replacement for ``_fit_fisher`` when ``config.inner_partitioner == "gmm_em"``.
    Sweeps ``k`` over ``config.gmm_k_range`` (inclusive), picks ``k*`` by lowest
    BIC (with ``gmm_bic_margin`` hysteresis against ``prev_k``), merges
    near-duplicate components, and identifies the burst cluster as the merged
    group whose centroid scores highest under ``gmm_burst_score_weights``.

    Returns a dict with keys:
        burst_posterior      : (n_bins,) float — Σ resp. over burst group members
        burst_centroid       : (n_features,) — burst group's standardized centroid
        k_chosen             : int — number of components actually fit
        bic_by_k             : dict[int, float]
        component_means      : (k, n_features) — raw GMM component means
        merged_groups        : list[dict] — output of _merge_component_groups
        group_scores         : list[float] — score per merged group
        burst_group_members  : list[int] — GMM component indices in the burst group

    Returns ``None`` on degenerate fits (too few samples, sklearn missing, or
    all merged groups score ≤ 0).  The caller keeps the previous iteration's
    posterior in that case (parallel to ``_fit_fisher`` returning ``None``).
    """
    n_bins, n_features = X_norm.shape
    k_min, k_max = config.gmm_k_range
    k_max = min(int(k_max), max(2, n_bins // 10))  # need ≥ ~10 bins per component
    k_min = max(2, int(k_min))
    if k_max < k_min:
        return None

    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None

    weights_arr = np.asarray(config.gmm_burst_score_weights, dtype=float)
    if weights_arr.size < n_features:
        weights_arr = np.concatenate([weights_arr, np.zeros(n_features - weights_arr.size)])
    else:
        weights_arr = weights_arr[:n_features]

    bic_by_k: dict[int, float] = {}
    fits: dict[int, GaussianMixture] = {}
    for k in range(k_min, k_max + 1):
        try:
            gm = GaussianMixture(
                n_components=k,
                n_init=int(config.gmm_em_n_init),
                reg_covar=float(config.gmm_em_reg_covar),
                random_state=42,
            )
            gm.fit(X_norm)
        except Exception:
            continue
        bic_by_k[k] = float(gm.bic(X_norm))
        fits[k] = gm

    if not fits:
        return None

    best_k = min(bic_by_k, key=bic_by_k.get)
    # Hysteresis: keep prev_k unless new candidate beats it by margin
    if prev_k is not None and prev_k in bic_by_k and prev_k != best_k:
        if bic_by_k[prev_k] - bic_by_k[best_k] < float(config.gmm_bic_margin):
            best_k = prev_k

    gm = fits[best_k]
    means = np.asarray(gm.means_)            # (k, n_features)
    weights = np.asarray(gm.weights_)        # (k,)

    merged_groups = _merge_component_groups(
        means, weights, float(config.gmm_component_merge_distance)
    )

    group_scores = [
        float(weights_arr @ np.asarray(g["centroid"])) for g in merged_groups
    ]
    if not group_scores:
        return None

    top_score = max(group_scores)
    fraction = float(config.gmm_burst_top_fraction)

    # Identity persistence: when a previous burst centroid is available, anchor
    # the burst cluster to the group whose centroid is closest to it. Otherwise
    # (first GMM iteration) anchor by argmax burst-prior score. Without this,
    # the GMM relabels components randomly each iteration and the iteration
    # loop oscillates between regimes (the "burst" identity hops between the
    # tight peak component and a broader mid-mass component as the candidate
    # mask shrinks and grows).
    if prev_burst_centroid is not None:
        prev_c = np.asarray(prev_burst_centroid, dtype=float)
        if prev_c.shape[0] == n_features:
            dists = [
                float(np.linalg.norm(np.asarray(g["centroid"]) - prev_c))
                for g in merged_groups
            ]
            anchor_idx = int(np.argmin(dists))
        else:
            anchor_idx = int(np.argmax(group_scores))
    else:
        anchor_idx = int(np.argmax(group_scores))

    anchor_score = group_scores[anchor_idx]

    if anchor_score <= 0.0 or fraction >= 1.0:
        burst_group_indices = [anchor_idx]
    else:
        cutoff = fraction * anchor_score
        burst_group_indices = [
            i for i, s in enumerate(group_scores) if s >= cutoff
        ]
        if anchor_idx not in burst_group_indices:
            burst_group_indices.append(anchor_idx)

    burst_group_members: list[int] = []
    for gi in burst_group_indices:
        burst_group_members.extend(int(m) for m in merged_groups[gi]["members"])

    # Burst centroid for diagnostics: weight-weighted average across selected
    # groups (so a multi-component burst is summarised by one centroid).
    sel_centroids = np.stack([
        np.asarray(merged_groups[gi]["centroid"], dtype=float)
        for gi in burst_group_indices
    ])
    sel_weights = np.array([
        float(merged_groups[gi].get("weight", 0.0)) for gi in burst_group_indices
    ])
    if sel_weights.sum() > 1e-12:
        burst_centroid = (sel_centroids * sel_weights[:, None]).sum(axis=0) / sel_weights.sum()
    else:
        burst_centroid = sel_centroids.mean(axis=0)

    resp = gm.predict_proba(X_norm)          # (n_bins, k)
    burst_posterior = resp[:, burst_group_members].sum(axis=1)

    return {
        "burst_posterior": burst_posterior,
        "burst_centroid": burst_centroid,
        "k_chosen": int(best_k),
        "bic_by_k": bic_by_k,
        "component_means": means,
        "component_weights": weights,
        "merged_groups": merged_groups,
        "group_scores": group_scores,
        "burst_group_members": burst_group_members,
    }


# ---------------------------------------------------------------------------
# Helpers — candidate management
# ---------------------------------------------------------------------------

def _mask_to_candidates(mask: np.ndarray, bins: np.ndarray) -> list[dict]:
    """Convert a boolean bin mask to a list of contiguous above-threshold regions.

    Each candidate dict stores both the continuous time boundaries (start/end
    in seconds, using bin edges) and the array indices (start_idx/end_idx) so
    downstream steps can slice the feature matrix without recomputing indices.
    """
    candidates = []
    n = len(mask)
    in_b = False
    s_idx = 0
    for t in range(n):
        if mask[t] and not in_b:
            s_idx = t
            in_b = True
        elif not mask[t] and in_b:
            candidates.append({"start": float(bins[s_idx]), "end": float(bins[t]),
                                "start_idx": s_idx, "end_idx": t - 1})
            in_b = False
    if in_b:
        candidates.append({"start": float(bins[s_idx]), "end": float(bins[n]),
                            "start_idx": s_idx, "end_idx": n - 1})
    return candidates


def _candidates_to_mask(candidates: list[dict], t_centers: np.ndarray) -> np.ndarray:
    """Reconstruct a boolean bin mask from a candidate list.

    Prefers stored index fields (O(1) per candidate) and falls back to a
    time-based search when indices are absent (e.g. after external construction).
    """
    mask = np.zeros(len(t_centers), dtype=bool)
    for c in candidates:
        s, e = c.get("start_idx"), c.get("end_idx")
        if s is not None and e is not None:
            mask[s:e + 1] = True
        else:
            mask |= (t_centers >= c["start"]) & (t_centers < c["end"])
    return mask


def _trim_candidate(
    c: dict,
    composite: np.ndarray,
    threshold: float,
    extent_frac: float,
    bins: np.ndarray,
    n_bins: int,
) -> dict | None:
    """Trim a candidate's edges to the region where composite exceeds the extent threshold.

    The extent threshold is the tighter of:
      - the global detection threshold (ensures the boundary is in burst regime)
      - extent_frac × peak_composite (proportional to the burst's own peak,
        so weak bursts don't get artificially wide boundaries)

    The walk proceeds from each edge inward toward the peak, stopping as soon
    as composite crosses the extent threshold. The peak itself always satisfies
    composite >= threshold (it was detected above threshold), so the walk
    always terminates before reaching the peak. If the walk exhausts the
    candidate without finding a valid boundary, the candidate is discarded.

    This step is crucial for accurate duration estimation: the permissive
    initial detection produces overly wide candidates that include low-composite
    flanking bins. Trimming removes those flanks each iteration, so by
    convergence the boundaries tightly enclose the burst.
    """
    s, e = c["start_idx"], c["end_idx"]
    if s > e:
        return None

    peak_rel = int(np.argmax(composite[s:e + 1]))
    peak_idx = s + peak_rel
    peak_val = composite[peak_idx]
    # Tighter of global threshold and fraction of peak
    ext_thr = max(threshold, extent_frac * peak_val)

    # Walk left edge rightward until composite is strong enough
    while s < peak_idx and composite[s] < ext_thr:
        s += 1
    # Walk right edge leftward until composite is strong enough
    while e > peak_idx and composite[e] < ext_thr:
        e -= 1

    if s > e or float(bins[e + 1]) - float(bins[s]) <= 0:
        return None
    return {"start": float(bins[s]), "end": float(bins[e + 1]),
            "start_idx": s, "end_idx": e}


def _valley_min(
    prev: dict,
    nxt: dict,
    composite: np.ndarray,
    t_centers: np.ndarray,
) -> float | None:
    """Minimum composite value in the valley between two adjacent candidates.

    Returns None when there are no bins between the two candidates (they are
    immediately adjacent), in which case callers use the gap duration as a
    proxy.
    """
    mask = (t_centers >= prev["end"]) & (t_centers <= nxt["start"])
    if not mask.any():
        return None
    v = composite[mask]
    return float(v.min()) if v.size > 0 else None


def _iter_merge(
    candidates: list[dict],
    composite: np.ndarray,
    t_centers: np.ndarray,
    bin_size: float,
    gap_s: float,
    threshold: float,
    floor_frac: float,
) -> list[dict]:
    """Merge adjacent candidates during iteration using a relaxed valley condition.

    Two candidates are merged when both conditions hold:
      1. Their temporal gap is ≤ gap_s  (burstlet_merge_gap_s = 3 × ISI)
      2. The valley between them stays above floor_frac × threshold
         (default 0.70 × threshold = "still 70 % as intense as burst regime")

    The relaxed floor (70 % rather than 100 %) allows the algorithm to merge
    candidates that are clearly part of the same burst even if the composite
    signal dips briefly in the valley, which is common for fast oscillations
    within a burst. The final hierarchy merge (Phase 4) applies a stricter
    criterion on the converged candidates.
    """
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x["start"])
    merged = []
    cur = candidates[0].copy()
    for nxt in candidates[1:]:
        gap = nxt["start"] - cur["end"]
        vm = _valley_min(cur, nxt, composite, t_centers)
        # If no bins exist in the valley, use gap width as a proxy
        valley_ok = (gap <= bin_size) if vm is None else (vm >= floor_frac * threshold)
        if gap <= gap_s and valley_ok:
            # Extend current candidate to absorb nxt
            cur = {"start": cur["start"], "end": nxt["end"],
                   "start_idx": cur["start_idx"], "end_idx": nxt["end_idx"]}
        else:
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)
    return merged


# ---------------------------------------------------------------------------
# Helpers — hierarchy finalization (Phase 4)
# ---------------------------------------------------------------------------

def _finalize_event(
    evs: list[dict],
    s: float,
    e: float,
    units: list,
    spike_times: dict,
    n_units: int,
    composite: np.ndarray,
    t_centers: np.ndarray,
    spike_counts_total: np.ndarray,
    pfr: np.ndarray,
    ws_sharp: np.ndarray,
    ws_smooth: np.ndarray,
    ff1: np.ndarray,
    llr_signal: np.ndarray,
    bin_size: float,
) -> dict:
    """Aggregate sub-events (evs) into a single merged event spanning [s, e).

    The participation and rate metrics (peak_synchrony, synchrony_energy,
    burst_peak) are computed over the full merged window [s, e) rather than
    summarised from sub-events, so they reflect the true extent of the merged
    event. The peak_time is inherited from the sub-event with the highest
    participation peak, so it always points at the moment of maximum synchrony.

    The four quality columns (llr_aggregate, composite_peak, composite_mean,
    ff_peak) are also measured over [s, e) so they reflect the merged event's
    statistical footprint in the converged feature space.
    """
    in_ev = (t_centers >= s) & (t_centers < e)
    participating = sum(
        1 for u in units if np.any((spike_times[u] >= s) & (spike_times[u] < e))
    )
    total_spikes = int(spike_counts_total[in_ev].sum())
    # Peak time: from the sub-event with the highest participation peak
    best = max(evs, key=lambda x: x["peak_synchrony"])

    comp_vals = composite[in_ev]
    return {
        "start": float(s),
        "end": float(e),
        "duration_s": float(e - s),
        "peak_synchrony": float(ws_sharp[in_ev].max()) if in_ev.any() else 0.0,
        "peak_time": float(best["peak_time"]),
        "synchrony_energy": float(ws_smooth[in_ev].sum() * bin_size) if in_ev.any() else 0.0,
        "participation": participating / n_units,
        "total_spikes": total_spikes,
        "burst_peak": float(pfr[in_ev].max()) if in_ev.any() else 0.0,
        "fragment_count": sum(ev.get("fragment_count", 1) for ev in evs),
        "n_sub_events": len(evs),
        # Quality columns unique to this detector
        "llr_aggregate": float(llr_signal[in_ev].mean()) if in_ev.any() else 0.0,
        "composite_peak": float(comp_vals.max()) if comp_vals.size > 0 else 0.0,
        "composite_mean": float(comp_vals.mean()) if comp_vals.size > 0 else 0.0,
        "ff_peak": float(ff1[in_ev].max()) if in_ev.any() else 0.0,
    }


def _merge_strict_hier(
    events: list[dict],
    gap: float,
    threshold: float,
    **ctx,
) -> list[dict]:
    """Merge burstlets → network bursts using the strict valley condition.

    Two burstlets are merged into one network burst only if the composite
    signal in the valley between them stays at or above the detection threshold
    (i.e. the valley itself is still "in burst regime"). This is the most
    conservative merge criterion and is appropriate for the first hierarchy
    level where we want to join only tightly coupled burstlets.
    """
    if not events:
        return []
    composite, t_centers, bin_size = ctx["composite"], ctx["t_centers"], ctx["bin_size"]
    events = sorted(events, key=lambda x: x["start"])
    merged, curr_evs = [], [events[0]]
    s, e = events[0]["start"], events[0]["end"]
    for nxt in events[1:]:
        vm = _valley_min(curr_evs[-1], nxt, composite, t_centers)
        # Strict: valley must stay above the full detection threshold
        valley_ok = (nxt["start"] - e <= bin_size) if vm is None else (vm >= threshold)
        if (nxt["start"] - e) <= gap and valley_ok:
            curr_evs.append(nxt)
            e = max(e, nxt["end"])
        else:
            merged.append(_finalize_event(curr_evs, s, e, **ctx))
            curr_evs, s, e = [nxt], nxt["start"], nxt["end"]
    merged.append(_finalize_event(curr_evs, s, e, **ctx))
    return [m for m in merged if m["duration_s"] > 0]


def _merge_clustered_hier(
    events: list[dict],
    gap: float,
    baseline: float,
    threshold: float,
    **ctx,
) -> list[dict]:
    """Merge network bursts → superbursts using the relaxed valley condition.

    Two network bursts are merged into a superburst when the valley between
    them dips below the detection threshold but stays above the background
    baseline — i.e., the network is not fully silent between them. This captures
    the "burst cluster" phenomenon where a sequence of network bursts are driven
    by a slow oscillation, separated by partial quieting rather than true silence.

    Requires ≥ 2 sub-events (network bursts) per superburst, so isolated
    network bursts are never promoted to superbursts.
    """
    if not events:
        return []
    composite, t_centers, bin_size = ctx["composite"], ctx["t_centers"], ctx["bin_size"]
    events = sorted(events, key=lambda x: x["start"])
    merged, curr_evs = [], [events[0]]
    s, e = events[0]["start"], events[0]["end"]
    for nxt in events[1:]:
        vm = _valley_min(curr_evs[-1], nxt, composite, t_centers)
        # Relaxed: valley must be above baseline but is allowed below threshold
        valley_ok = (nxt["start"] - e <= bin_size) if vm is None else (baseline < vm < threshold)
        if (nxt["start"] - e) <= gap and valley_ok:
            curr_evs.append(nxt)
            e = max(e, nxt["end"])
        else:
            merged.append(_finalize_event(curr_evs, s, e, **ctx))
            curr_evs, s, e = [nxt], nxt["start"], nxt["end"]
    merged.append(_finalize_event(curr_evs, s, e, **ctx))
    return [m for m in merged if m["duration_s"] > 0 and m["n_sub_events"] >= 2]


# ---------------------------------------------------------------------------
# Helpers — event modulation and component merging
# ---------------------------------------------------------------------------

def _merge_component_groups(
    centroids: np.ndarray,
    weights: np.ndarray,
    max_distance: float,
) -> list[dict]:
    """Agglomeratively merge component centroids that are too similar."""
    groups = [
        {
            "members": [i],
            "centroid": np.asarray(centroids[i], dtype=float).copy(),
            "weight": float(weights[i]),
        }
        for i in range(len(centroids))
    ]

    if len(groups) <= 1:
        return groups

    while True:
        best_i = best_j = -1
        best_dist = float("inf")
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                dist = float(np.linalg.norm(groups[i]["centroid"] - groups[j]["centroid"]))
                if dist < best_dist:
                    best_dist = dist
                    best_i, best_j = i, j

        if best_i < 0 or best_dist > max_distance:
            break

        left = groups[best_i]
        right = groups[best_j]
        total_weight = left["weight"] + right["weight"]
        merged_centroid = (
            left["centroid"] * left["weight"] + right["centroid"] * right["weight"]
        ) / max(total_weight, 1e-12)
        groups[best_i] = {
            "members": left["members"] + right["members"],
            "centroid": merged_centroid,
            "weight": total_weight,
        }
        del groups[best_j]

    return groups


# ---------------------------------------------------------------------------
# Helpers — post-convergence GMM event clustering
# ---------------------------------------------------------------------------

def _cluster_events(
    events: list[dict],
    config: "IterativeBurstConfig",
    debug: bool,
    trace: "IterativeBurstTrace | None" = None,
) -> tuple[list[dict], float | None]:
    """Cluster events with a multi-component GMM and merge similar components.

    The detector intentionally over-segments first.  This helper then fits a
    small GMM with more than two initial components, merges components whose
    centroids are too close in standardised feature space, and keeps only the
    clusters that still look burst-like after the merge.
    """
    n = len(events)
    if n < config.cluster_min_events:
        if trace is not None:
            trace.gmm = {"skipped": "too_few_events", "n_events": n}
        return events, None

    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        if debug:
            print("[cluster] sklearn not available — skipping event clustering")
        if trace is not None:
            trace.gmm = {"skipped": "sklearn_missing", "n_events": n}
        return events, None

    feature_cols = [
        "composite_peak", "composite_mean", "llr_aggregate",
        "ff_peak", "participation", "burst_peak",
    ]
    X = np.array([[ev[c] for c in feature_cols] for ev in events], dtype=float)
    if not np.isfinite(X).all():
        if trace is not None:
            trace.gmm = {"skipped": "nonfinite_features", "n_events": n}
        return events, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components = max(2, min(int(config.cluster_initial_components), n))

    try:
        gm = GaussianMixture(
            n_components=n_components,
            n_init=10,
            random_state=42,
            reg_covar=1e-6,
        )
        labels = gm.fit_predict(X_scaled)
    except Exception:
        if trace is not None:
            trace.gmm = {"skipped": "gmm_fit_failed", "n_events": n}
        return events, None

    merged_groups = _merge_component_groups(
        gm.means_,
        gm.weights_,
        float(config.cluster_min_separation),
    )
    def _score_centroid(centroid: np.ndarray) -> float:
        return float(
            0.35 * centroid[0]
            + 0.15 * centroid[1]
            + 0.20 * centroid[2]
            + 0.10 * centroid[3]
            + 0.15 * centroid[4]
            + 0.05 * centroid[5]
        )

    merged_groups.sort(key=lambda g: _score_centroid(g["centroid"]), reverse=True)
    cluster_scores = [_score_centroid(g["centroid"]) for g in merged_groups]

    if len(merged_groups) > 1:
        sep = float(
            min(
                np.linalg.norm(merged_groups[i]["centroid"] - merged_groups[j]["centroid"])
                for i in range(len(merged_groups))
                for j in range(i + 1, len(merged_groups))
            )
        )
    else:
        sep = None

    def _record_gmm(kept_mask: np.ndarray, decision: str) -> None:
        if trace is None:
            return
        trace.gmm = {
            "decision": decision,
            "feature_names": list(feature_cols),
            "X": X.copy(),
            "X_scaled": X_scaled.copy(),
            "labels": np.asarray(labels).copy(),
            "component_means_scaled": np.asarray(gm.means_).copy(),
            "component_weights": np.asarray(gm.weights_).copy(),
            "merged_groups": [
                {
                    "centroid": np.asarray(g["centroid"]).copy(),
                    "members": list(g["members"]),
                    "weight": float(g.get("weight", 0.0)),
                }
                for g in merged_groups
            ],
            "cluster_scores": list(cluster_scores),
            "kept_event_mask": kept_mask.astype(bool),
            "separation": sep,
            "score_weights": [0.35, 0.15, 0.20, 0.10, 0.15, 0.05],
            "n_initial_components": n_components,
        }

    if len(merged_groups) <= 1:
        if debug:
            print(
                f"[cluster] only one merged component after GMM; keeping all {n} events"
            )
        _record_gmm(np.ones(n, dtype=bool), "single_merged_component")
        return events, None

    best_score = max(cluster_scores)
    if best_score <= 0.0:
        if debug:
            print(
                f"[cluster] all merged components scored <= 0; keeping all {n} events"
            )
        _record_gmm(np.ones(n, dtype=bool), "all_scores_nonpositive")
        return events, sep

    keep_groups = [
        group for group, score in zip(merged_groups, cluster_scores)
        if score >= 0.0
    ]
    if not keep_groups:
        keep_groups = [merged_groups[0]]

    component_to_group: dict[int, int] = {}
    for group_idx, group in enumerate(keep_groups):
        for member in group["members"]:
            component_to_group[member] = group_idx

    kept_mask = np.array([lb in component_to_group for lb in labels], dtype=bool)
    kept: list[dict] = [ev for ev, keep in zip(events, kept_mask) if keep]

    if debug:
        group_sizes = ",".join(str(len(g["members"])) for g in merged_groups)
        print(
            f"[cluster] init={n_components} merged={len(merged_groups)}"
            f" sizes=[{group_sizes}] kept={len(kept)}/{n}"
        )

    _record_gmm(kept_mask, "filtered_by_score")
    return kept, sep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_iterative_bursts(
    spike_times: dict[str, np.ndarray],
    config: IterativeBurstConfig | None = None,
    debug: bool = False,
    trace: "IterativeBurstTrace | None" = None,
) -> "BurstResults":
    """Detect burstlets, network bursts, and superbursts via iterative LDA.

    Args:
        spike_times: Mapping of unit_id → spike time array (seconds).
        config: Detection parameters. Uses IterativeBurstConfig defaults if None.
        debug: When True, print a one-line status at iteration 0, every 5
            iterations, and the final converged iteration showing candidate
            count, convergence delta, composite threshold, and the top-3
            Fisher weights. Also prints the burst modulation gate decision and
            GMM clustering outcome.

    Returns:
        BurstResults with per-level DataFrames (burstlets / network_bursts /
        superbursts), aggregate metrics, diagnostics, and plot_data signals.
        Each event DataFrame carries four additional quality columns not present
        in the standard detector: llr_aggregate, composite_peak, composite_mean,
        ff_peak.
        diagnostics additionally contains: burst_modulation_index,
        burst_activity_detected, burst_modulation_candidates_scored,
        burst_modulation_candidates_kept, cluster_separation, cluster_n_kept.

    Raises:
        IterativeBurstError: When spike_times is empty or contains no spikes.
    """
    from .burst_detector import BurstResults

    if config is None:
        config = IterativeBurstConfig()

    units = list(spike_times.keys())
    if not units:
        raise IterativeBurstError("spike_times contains no units")

    non_empty = [spike_times[u] for u in units if len(spike_times[u]) > 0]
    if not non_empty:
        raise IterativeBurstError("spike_times contains no spikes")

    all_spikes = np.sort(np.concatenate(non_empty))
    rec_start, rec_end = float(all_spikes[0]), float(all_spikes[-1])
    total_dur = rec_end - rec_start
    if total_dur < 1e-6:
        raise IterativeBurstError("spike_times spans insufficient duration (< 1 µs)")

    # -----------------------------------------------------------------------
    # Phase 1a: Adaptive bin size from the population median log-ISI
    #
    # The median ISI (in log space, to be robust to outliers) sets the bin size
    # so that each bin is roughly one "typical spike interval" wide.  Clamped
    # to [10, 30] ms: finer bins produce too many empty bins in sparse data;
    # coarser bins blur the temporal structure of fast bursts.
    #
    # Downstream time constants are also derived from biological_isi_s:
    #   sigma_fast  ≈ 1 bin   — narrow smoothing that preserves burst peaks
    #   sigma_slow  ≈ 5 ISIs  — broad smoothing for rate and LLR integration
    #   burstlet_merge_gap_s = 3 ISIs  — gap within which two burstlets
    #       are still considered a single event
    #   network_merge_gap_s  = 10 ISIs — gap for the superburst hierarchy
    # -----------------------------------------------------------------------
    all_log_isis: list[float] = []
    for u in units:
        t = np.unique(np.sort(spike_times[u]))
        if len(t) >= 2:
            isi = np.diff(t)
            isi = isi[isi > 0]
            if isi.size > 0:
                all_log_isis.extend(np.log10(isi).tolist())

    biological_isi_s = 10 ** float(np.median(all_log_isis)) if all_log_isis else 0.1
    adaptive_bin_ms = float(np.clip(biological_isi_s * 1000, 10, 30))
    bin_size = adaptive_bin_ms / 1000.0

    bins = np.arange(rec_start, rec_end + bin_size, bin_size)
    t_centers = (bins[:-1] + bins[1:]) / 2
    n_bins = len(t_centers)
    n_units = len(units)

    isi_bins = biological_isi_s / bin_size  # ISI length in bin units
    sigma_fast = float(np.clip(isi_bins, 1, 2))
    sigma_slow = float(np.clip(5.0 * isi_bins, 3, 8))

    burstlet_merge_gap_s = 3.0 * biological_isi_s
    network_merge_gap_s = max(10.0 * biological_isi_s, config.network_merge_gap_min_s)

    # -----------------------------------------------------------------------
    # Phase 1b: Participation signal and permissive initial candidates
    #
    # ws_sharp = fast-smoothed fraction of active units per bin. This is the
    # same signal used by the original parameter-free detector. It serves two
    # roles here:
    #   (1) Seeding: an intentionally low threshold (0.30 × MAD instead of
    #       0.75 × MAD) gives many false-positive candidates that the iteration
    #       will eliminate. Starting permissive is safer than starting strict
    #       — a missed burst can never be recovered, but a false positive will
    #       be trimmed away.
    #   (2) Reporting: peak_synchrony and peak_time in the output DataFrames
    #       are always derived from ws_sharp (not composite) for compatibility
    #       with viewers that interpret these as participation fractions.
    #
    # Fallback to percentile thresholding when spread_mad < mad_fallback_threshold:
    # For very sparse recordings the participation signal is nearly all zeros with
    # rare spikes. MAD is also near zero, making any MAD multiplier meaningless.
    # Taking the top 30 % of bins by participation is always well-defined.
    # -----------------------------------------------------------------------
    spike_matrix = _compute_spike_matrix(spike_times, units, bins, n_bins)
    spike_counts_total = spike_matrix.sum(axis=0)
    active_unit_counts = (spike_matrix > 0).sum(axis=0).astype(float)

    participation_raw = active_unit_counts / max(1, n_units)
    pfr = spike_counts_total / bin_size           # population firing rate (Hz)
    rate_per_unit = pfr / max(1, n_units)         # per-unit average rate (Hz)

    ws_sharp = gaussian_filter1d(participation_raw, sigma_fast)
    ws_smooth = gaussian_filter1d(rate_per_unit, sigma_slow)

    participation_floor_count = (
        max(5, 0.15 * n_units) if n_units < 50 else max(10, 0.05 * n_units)
    )
    participation_floor = participation_floor_count / max(1, n_units)

    baseline_init = float(np.median(ws_sharp))
    spread_mad_init = float(np.median(np.abs(ws_sharp - baseline_init)))

    if spread_mad_init > config.mad_fallback_threshold:
        init_threshold = baseline_init + config.permissive_mad_scale * spread_mad_init
        init_method = "mad"
    else:
        # Sparse/uniform recording: fall back to top-percentile floor
        init_threshold = float(np.percentile(ws_sharp, config.permissive_percentile))
        init_method = "percentile"

    init_threshold = max(participation_floor, init_threshold)

    # Seed candidates: every contiguous run of bins above init_threshold
    candidates = _mask_to_candidates(ws_sharp >= init_threshold, bins)

    # -----------------------------------------------------------------------
    # Phase 2: Static feature signals
    #
    # FF signals and burstiness are computed once outside the loop because they
    # do not depend on the background rate estimate (λ_bg). Only the LLR column
    # is recomputed inside the loop as λ_bg updates.
    #
    # Feature matrix layout (columns):
    #   [0]      PFR              — raw population firing rate (Hz)
    #   [1]      P                — participation fraction
    #   [2..5]   FF_scale[0..3]   — spatial Fano Factor at 4 scales
    #   [llr_idx] LLR             — signed per-unit Poisson log-LR (placeholder, updated per iter)
    #   [-1]     burstiness       — mean instantaneous ISI reciprocal
    # -----------------------------------------------------------------------
    ff_signals = _compute_multiscale_ff(
        spike_matrix, bins, t_centers, bin_size, config.ff_scale_multipliers
    )
    ff_scales_ms = [
        float(np.clip(m * adaptive_bin_ms, 5.0, 100.0))
        for m in config.ff_scale_multipliers
    ]

    burstiness = _compute_burstiness(spike_times, units, bins, n_bins)

    n_ff = len(config.ff_scale_multipliers)
    n_features = 2 + n_ff + 2   # PFR + P + FF×n_ff + LLR + burstiness
    llr_idx = 2 + n_ff           # column index of LLR in X

    # Build feature matrix with placeholder zeros for the LLR column
    X = np.column_stack([pfr, participation_raw, ff_signals, np.zeros(n_bins), burstiness])

    # Bootstrap weight vector for iteration 0 (before Fisher can be applied):
    # participation-heavy prior (1.5) reflects domain knowledge that synchronous
    # recruitment of units is the most reliable burst indicator.
    w_prior = np.array([1.0, 1.5] + [0.5] * n_ff + [1.0, 0.5])
    w = w_prior / float(np.linalg.norm(w_prior))

    # Global fallback background rate: used when a unit has zero spikes in the
    # background window (e.g. a unit that only fires during bursts)
    global_lambda_bg = float(pfr.mean()) / max(1, n_units)

    # -----------------------------------------------------------------------
    # Phase 3: Iterative refinement
    #
    # Each iteration follows these steps:
    #
    #   3a. Background estimation
    #       Identify background bins (bins not in any current candidate).
    #       Compute per-unit background firing rate λ_u_bg from those bins.
    #       Fall back to global_lambda_bg for any unit with zero background spikes.
    #
    #   3b. LLR update
    #       Recompute the LLR signal using the updated λ_bg.  Because background
    #       bins change each iteration, the LLR signal "tightens" around true
    #       burst periods as the partition improves.
    #
    #   3c. Z-normalisation
    #       Z-score all features using background bin statistics so that each
    #       feature contributes on a common scale (standard deviations above
    #       background mean) to the Fisher discriminant.
    #
    #   3d. Fisher LDA (skipped on iteration 0, which uses w_prior)
    #       Fit the LDA direction on the current burst/background labels.
    #       The weight vector w that maximises between-class / within-class
    #       variance is solved via w = S_W^{-1}(μ_burst − μ_background).
    #       If Fisher returns None (degenerate case), keep the previous w.
    #
    #   3e. Composite signal and new threshold
    #       composite(t) = X_norm[t] @ w   (scalar projection per bin)
    #       Threshold derived from the median + MAD of composite in background
    #       bins, mirroring the original detector's robust threshold.
    #
    #   3f. Boundary trimming
    #       Walk each new candidate's edges inward until composite exceeds the
    #       extent threshold.  This progressively tightens boundaries that were
    #       loose in earlier iterations.
    #
    #   3g. Iteration-level merge
    #       Join nearby trimmed candidates whose valley stays above 70 % of
    #       threshold.  This is intentionally relaxed so the algorithm can
    #       experiment with merged states; the final strict hierarchy merge in
    #       Phase 4 will split any over-merged events.
    #
    #   Convergence: < convergence_eps fraction of bins changed label → stop.
    # -----------------------------------------------------------------------
    feature_names = ["PFR", "P", *[f"FF{k}" for k in range(n_ff)], "LLR", "burst"]

    prev_mask = _candidates_to_mask(candidates, t_centers)
    composite = np.zeros(n_bins)
    composite_threshold = 0.0
    composite_baseline = 0.0
    composite_mad = 1e-6
    n_iterations = 0
    convergence_delta = 1.0
    lambda_bg_per_unit = np.full(n_units, global_lambda_bg)
    llr_signal = np.zeros(n_bins)
    use_gmm = (config.inner_partitioner == "gmm_em")
    prev_k: int | None = None
    prev_burst_centroid: np.ndarray | None = None
    gmm_info: dict | None = None
    burst_centroid_final: np.ndarray | None = None

    # Silence mask: bins with truly zero active units. When LDA silence
    # excision is enabled, these bins are dropped from both classes in
    # _fit_fisher and from the z-norm background statistics, so a long
    # silent stretch can't pull the bg-class mean toward zero and flip the
    # discriminant sign on heterogeneous recordings.
    silence_mask = (active_unit_counts == 0)
    lda_pinned_indices = tuple(
        feature_names.index(name)
        for name in config.lda_sign_pinned_feature_names
        if name in feature_names
    )

    if debug:
        print(
            f"[iter init]  seed_candidates={len(candidates):>4d}"
            f"  method={init_method}  threshold={init_threshold:.4f}"
        )

    for iteration in range(config.max_iterations):
        n_iterations = iteration + 1

        # 3a. Background bins = all bins not covered by any candidate.
        # When LDA silence excision is on, drop truly silent bins (zero active
        # units) from the bg mask used for z-norm and per-unit rate estimates;
        # they're not "background firing", they're "no signal", and including
        # them in the bg class pulls μ_bg toward zero and flips the LDA on
        # heterogeneous recordings.
        candidate_mask = _candidates_to_mask(candidates, t_centers)
        bg_mask = ~candidate_mask
        if not use_gmm and config.lda_exclude_silence:
            bg_mask = bg_mask & ~silence_mask
        if bg_mask.sum() < 2:
            # Edge case: candidates cover almost the whole recording.
            # Fall back to treating all bins as background to avoid
            # Z-scoring on an empty set.
            bg_mask = np.ones(n_bins, dtype=bool)

        # Per-unit background rate from background bins only
        n_bg_bins = bg_mask.sum()
        lambda_bg_per_unit = spike_matrix[:, bg_mask].sum(axis=1) / (n_bg_bins * bin_size)
        # Replace units with zero background spikes with the global estimate
        lambda_bg_per_unit = np.where(
            lambda_bg_per_unit > 0, lambda_bg_per_unit, global_lambda_bg
        )

        # 3b. Recompute LLR with updated per-unit background rates
        llr_signal = _compute_llr_signal(spike_matrix, lambda_bg_per_unit, bin_size, sigma_slow)
        X[:, llr_idx] = llr_signal  # update LLR column in-place

        # 3c. Z-score all features relative to background
        X_norm = _znorm(X, bg_mask)

        # 3d. Partition bins into burst vs background.
        # Iteration 0 always uses the w_prior bootstrap composite so the GMM
        # has a meaningful candidate mask to refine on iteration 1.
        # From iteration 1 onward, dispatch to GMM-EM or Fisher LDA per config.
        gmm_info = None
        if iteration == 0 or not use_gmm:
            if iteration > 0:
                # Silence-excluded label vector + matrix for the Fisher fit:
                # bins flagged silent are not handed to the LDA on either
                # side of the contrast.  The composite is still computed on
                # the full X_norm so the candidate mask covers all bins.
                if config.lda_exclude_silence:
                    fit_mask = ~silence_mask
                    fit_labels = candidate_mask[fit_mask].astype(int)
                    fit_X = X_norm[fit_mask]
                else:
                    fit_labels = candidate_mask.astype(int)
                    fit_X = X_norm
                w_new = _fit_fisher(fit_X, fit_labels, config.fisher_alpha_frac)
                # Sign pinning: reject the new direction if a biologically
                # sign-constrained feature (PFR/P/LLR) has a negative weight.
                # That's the signature of the LDA having locked onto silence
                # as the "burst" class (cx138_44_02 with w_PFR=-0.81).
                if w_new is not None and lda_pinned_indices:
                    if any(w_new[i] < 0 for i in lda_pinned_indices):
                        w_new = None
                if w_new is not None:
                    w = w_new
            composite = X_norm @ w
        else:
            gmm_info = _fit_gmm_em(
                X_norm, prev_k, config, prev_burst_centroid=prev_burst_centroid,
            )
            if gmm_info is None:
                # Degenerate fit — fall back to the previous iteration's posterior.
                # composite, composite_threshold from last iter are still in scope.
                pass
            else:
                composite = gmm_info["burst_posterior"]
                prev_k = gmm_info["k_chosen"]
                prev_burst_centroid = gmm_info["burst_centroid"]
                burst_centroid_final = gmm_info["burst_centroid"]

        # 3e. Threshold composite using background distribution (or a fixed
        # posterior cut when configured for the GMM path).
        comp_bg = composite[bg_mask]
        composite_baseline = float(np.median(comp_bg))
        composite_mad = float(np.median(np.abs(comp_bg - composite_baseline)))
        if use_gmm and iteration > 0 and config.gmm_posterior_threshold is not None:
            composite_threshold = float(config.gmm_posterior_threshold)
        else:
            # max(..., 1e-6) prevents a zero-MAD threshold on constant composite
            composite_threshold = composite_baseline + config.composite_mad_scale * max(
                composite_mad, 1e-6
            )

        # All above-threshold connected regions become new candidate seeds
        new_candidates_raw = _mask_to_candidates(composite >= composite_threshold, bins)

        # 3f. Trim each candidate's boundaries to where composite is strong enough
        trimmed: list[dict] = []
        for c in new_candidates_raw:
            tc = _trim_candidate(c, composite, composite_threshold, config.extent_frac, bins, n_bins)
            if tc is not None:
                trimmed.append(tc)

        # 3g. Merge nearby trimmed candidates (relaxed valley during iteration)
        candidates = _iter_merge(
            trimmed, composite, t_centers, bin_size,
            burstlet_merge_gap_s, composite_threshold, config.merge_floor_frac,
        )

        # Convergence: measure fraction of bins that changed burst/background label
        new_mask = _candidates_to_mask(candidates, t_centers)
        convergence_delta = float(np.sum(new_mask != prev_mask)) / n_bins
        prev_mask = new_mask

        converged = convergence_delta < config.convergence_eps

        if trace is not None:
            entry = {
                "iter": int(iteration),
                "n_candidates": len(candidates),
                "candidates": [dict(c) for c in candidates],
                "composite": composite.copy(),
                "composite_threshold": float(composite_threshold),
                "composite_baseline": float(composite_baseline),
                "composite_mad": float(composite_mad),
                "w": w.copy(),
                "lambda_bg_per_unit": lambda_bg_per_unit.copy(),
                "X_norm": X_norm.copy(),
                "candidate_mask_in": candidate_mask.copy(),
                "candidate_mask": new_mask.copy(),
                "convergence_delta": float(convergence_delta),
                "converged": bool(converged),
            }
            if gmm_info is not None:
                entry["k_chosen"] = int(gmm_info["k_chosen"])
                entry["bic_by_k"] = dict(gmm_info["bic_by_k"])
                entry["gmm_centroids_scaled"] = gmm_info["component_means"].copy()
                entry["gmm_component_weights"] = gmm_info["component_weights"].copy()
                entry["gmm_group_scores"] = list(gmm_info["group_scores"])
                entry["burst_group_members"] = list(gmm_info["burst_group_members"])
                entry["burst_centroid"] = gmm_info["burst_centroid"].copy()
            trace.iterations.append(entry)

        if debug and (iteration == 0 or iteration % 5 == 0 or converged):
            tag = "  CONVERGED" if converged else ""
            if gmm_info is not None:
                centroid = gmm_info["burst_centroid"]
                top3 = sorted(zip(feature_names, centroid), key=lambda x: -abs(x[1]))[:3]
                cstr = "  ".join(f"{nm}={v:+.2f}" for nm, v in top3)
                print(
                    f"[iter {iteration:>3d}]  candidates={len(candidates):>4d}"
                    f"  delta={convergence_delta:.4f}"
                    f"  thr={composite_threshold:.3f}"
                    f"  k={gmm_info['k_chosen']}"
                    f"  burst centroid: {cstr}{tag}"
                )
            else:
                top3 = sorted(zip(feature_names, w), key=lambda x: -abs(x[1]))[:3]
                wstr = "  ".join(f"{nm}={v:+.2f}" for nm, v in top3)
                print(
                    f"[iter {iteration:>3d}]  candidates={len(candidates):>4d}"
                    f"  delta={convergence_delta:.4f}"
                    f"  thr={composite_threshold:.3f}"
                    f"  top weights: {wstr}{tag}"
                )

        if converged:
            break

    # -----------------------------------------------------------------------
    # Phase 4a: Materialise burstlet events from converged candidates
    #
    # Each converged candidate becomes one burstlet.  Per-event scalars are
    # computed over the candidate's exact time window.  peak_synchrony and
    # peak_time are derived from ws_sharp (participation signal) for
    # compatibility with existing viewers; the four quality columns are from
    # the composite/LLR/FF signals that drove the iterative detection.
    # -----------------------------------------------------------------------
    # Use the 1× bin-scale FF column as the representative FF for quality output
    ff1 = ff_signals[:, min(1, n_ff - 1)]

    burstlets_raw: list[dict] = []
    pre_floor_events: list[dict] = []  # all candidates that produced a valid window
    dropped_by_floor: list[dict] = []
    for c in candidates:
        s_t, e_t = c["start"], c["end"]
        duration_s = e_t - s_t
        if duration_s <= 0:
            continue

        in_ev = (t_centers >= s_t) & (t_centers < e_t)
        if not in_ev.any():
            continue

        participating = sum(
            1 for u in units if np.any((spike_times[u] >= s_t) & (spike_times[u] < e_t))
        )
        total_spikes = int(spike_counts_total[in_ev].sum())

        # Peak time: bin with highest participation within this burstlet
        peak_abs_idx = int(np.where(in_ev)[0][np.argmax(ws_sharp[in_ev])])
        peak_synchrony = float(ws_sharp[peak_abs_idx])
        peak_time = float(t_centers[peak_abs_idx])
        comp_vals = composite[in_ev]

        event = {
            "start": float(s_t),
            "end": float(e_t),
            "duration_s": float(duration_s),
            "peak_synchrony": peak_synchrony,
            "peak_time": peak_time,
            "synchrony_energy": float(ws_smooth[in_ev].sum() * bin_size),
            "participation": participating / n_units,
            "total_spikes": total_spikes,
            "burst_peak": float(pfr[in_ev].max()),
            "fragment_count": 1,
            # Quality columns: how strongly did the composite signal and LLR flag this event?
            "llr_aggregate": float(llr_signal[in_ev].mean()),
            "composite_peak": float(comp_vals.max()),
            "composite_mean": float(comp_vals.mean()),
            "ff_peak": float(ff1[in_ev].max()),
        }

        if trace is not None:
            pre_floor_events.append(dict(event))

        if peak_synchrony < participation_floor:
            if trace is not None:
                dropped_by_floor.append(dict(event))
            continue

        burstlets_raw.append(event)

    if trace is not None:
        trace.burstlets_pre_gates = pre_floor_events
        trace.participation_gate = {
            "floor": float(participation_floor),
            "floor_count": float(participation_floor_count),
            "n_pre": len(pre_floor_events),
            "n_post": len(burstlets_raw),
            "n_dropped": len(dropped_by_floor),
            "dropped_events": dropped_by_floor,
        }

    # -----------------------------------------------------------------------
    # Burst modulation gate — burstlet-level LLR filter
    #
    # The recording-level gate was too brittle for heterogeneous wells.  We
    # now use the burstlet's own llr_aggregate so long bridge-like events can
    # be dropped without deleting a whole burst-rich section.  A value <= 0
    # disables the gate and keeps the pre-filtered burstlets.
    # -----------------------------------------------------------------------
    burst_modulation_scores = [float(ev["llr_aggregate"]) for ev in burstlets_raw]
    burst_modulation_index = max(burst_modulation_scores) if burst_modulation_scores else 0.0
    if trace is not None:
        trace.bmi_gate = {
            "threshold": float(config.min_burst_modulation),
            "enabled": bool(config.min_burst_modulation > 0),
            "llr_aggregate": list(burst_modulation_scores),
            "n_pre": len(burstlets_raw),
            "pre_events": [dict(ev) for ev in burstlets_raw],
        }
    if config.min_burst_modulation > 0:
        kept_burstlets = [
            ev for ev in burstlets_raw
            if float(ev["llr_aggregate"]) >= config.min_burst_modulation
        ]
        if debug and len(kept_burstlets) < len(burstlets_raw):
            print(
                f"[gate] burstlet llr gate kept {len(kept_burstlets)}/{len(burstlets_raw)} "
                f"events at min={config.min_burst_modulation:.2f}"
            )
        burstlets_raw = kept_burstlets
    if trace is not None:
        trace.bmi_gate["n_post"] = len(burstlets_raw)

    # -----------------------------------------------------------------------
    # GMM event clustering — discard noise burstlets
    #
    # After materialisation, a 2-component GMM on per-event quality features
    # separates high-confidence burst events from marginal noise events that
    # the iteration-level threshold still admitted.  This handles the
    # over-segmentation seen in CX138_27_10 and CX138_31_5 where the iterative
    # detector finds many small inter-burst events.
    # -----------------------------------------------------------------------
    cluster_sep: float | None = None
    if config.cluster_events:
        burstlets_raw, cluster_sep = _cluster_events(burstlets_raw, config, debug, trace=trace)

    # -----------------------------------------------------------------------
    # Phase 4b: Two-tier hierarchy merge
    #
    # burstlets  →[merge_strict]→   network_bursts
    #     Valley must stay ≥ composite_threshold (still "in burst regime").
    #     Gap ≤ burstlet_merge_gap_s = 3 × biological_isi_s.
    #
    # network_bursts  →[merge_clustered]→  superbursts
    #     Valley is allowed to dip below threshold but must stay above
    #     composite_baseline (not fully silent between bursts).
    #     Gap ≤ network_merge_gap_s = max(10 × ISI, 0.75 s).
    #     Requires ≥ 2 network bursts per superburst.
    # -----------------------------------------------------------------------
    hier_ctx = dict(
        units=units, spike_times=spike_times, n_units=n_units,
        composite=composite, t_centers=t_centers, bin_size=bin_size,
        spike_counts_total=spike_counts_total, pfr=pfr,
        ws_sharp=ws_sharp, ws_smooth=ws_smooth, ff1=ff1, llr_signal=llr_signal,
    )

    network_bursts = _merge_strict_hier(
        burstlets_raw, gap=burstlet_merge_gap_s, threshold=composite_threshold, **hier_ctx
    )
    superbursts = _merge_clustered_hier(
        network_bursts, gap=network_merge_gap_s,
        baseline=composite_baseline, threshold=composite_threshold, **hier_ctx
    )

    # -----------------------------------------------------------------------
    # Phase 5: Assemble and return BurstResults
    # -----------------------------------------------------------------------
    def _to_df(evs: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(evs) if evs else pd.DataFrame()

    metrics = {
        "burstlets": _level_metrics(burstlets_raw, total_dur),
        "network_bursts": _level_metrics(network_bursts, total_dur),
        "superbursts": _level_metrics(superbursts, total_dur),
    }

    diagnostics = {
        # Adaptive binning
        "adaptive_bin_ms": adaptive_bin_ms,
        "biological_isi_s": biological_isi_s,
        "sigma_fast_bins": sigma_fast,
        "sigma_slow_bins": sigma_slow,
        # Iteration summary
        "n_iterations": n_iterations,
        "convergence_delta": float(convergence_delta),
        "init_method": init_method,           # "mad" or "percentile"
        # Final composite calibration
        "composite_threshold": float(composite_threshold),
        "composite_baseline": float(composite_baseline),
        "participation_floor": float(participation_floor),
        # Learnt feature weights — inspect to understand which features drove detection.
        # For the GMM-EM partitioner this is the burst component's standardized
        # centroid; for the Fisher LDA path it is the discriminant direction w.
        "feature_weights_final": (
            burst_centroid_final.tolist()
            if (use_gmm and burst_centroid_final is not None)
            else w.tolist()
        ),
        "inner_partitioner": config.inner_partitioner,
        "k_chosen_final": int(prev_k) if (use_gmm and prev_k is not None) else None,
        # Per-unit background rates — useful for diagnosing heterogeneous networks
        "lambda_bg_per_unit": {u: float(lambda_bg_per_unit[i]) for i, u in enumerate(units)},
        # Merge geometry
        "burstlet_merge_gap_s": float(burstlet_merge_gap_s),
        "network_merge_gap_s": float(network_merge_gap_s),
        # FF scales actually used
        "ff_scales_ms": ff_scales_ms,
        "n_units": n_units,
        # Quality gate and clustering
        "burst_modulation_index": float(burst_modulation_index),
        "burst_activity_detected": bool(burstlets_raw),
        "burst_modulation_candidates_scored": len(burst_modulation_scores),
        "burst_modulation_candidates_kept": len(candidates),
        "cluster_separation": cluster_sep,
        "cluster_n_kept": len(burstlets_raw),
    }

    plot_data = {
        "t": t_centers,
        "participation_signal": ws_sharp,        # fast-smoothed participation (original signal)
        "rate_signal": ws_smooth,                # slow-smoothed per-unit rate
        "composite_signal": composite,           # Fisher LDA projection (main detection signal)
        "ff_signal": ff1,                        # Fano Factor at 1× bin scale
        "llr_signal": llr_signal,                # signed per-unit Poisson LLR
        "burst_peak_times": np.array([b["peak_time"] for b in network_bursts]),
        "burst_peak_values": np.array([b["peak_synchrony"] for b in network_bursts]),
        "participation_baseline": float(baseline_init),
        "participation_threshold": float(init_threshold),
    }

    if trace is not None:
        trace.t_centers = t_centers.copy()
        trace.bin_size = float(bin_size)
        trace.feature_names = list(feature_names)
        trace.unit_ids = [str(u) for u in units]

    return BurstResults(
        burstlets=_to_df(burstlets_raw),
        network_bursts=_to_df(network_bursts),
        superbursts=_to_df(superbursts),
        metrics=metrics,
        diagnostics=diagnostics,
        plot_data=plot_data,
    )
