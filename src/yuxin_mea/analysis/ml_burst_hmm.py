"""Per-unit 2-state Poisson HMM for burst-state inference.

Each unit gets its own 2-state HMM with Poisson emissions:
    state 0 = background    (rate λ_bg)
    state 1 = burst         (rate λ_burst, by construction λ_burst > λ_bg)

For an observed count sequence n_u(t) (counts of unit u in bin t), the model
parameters (λ_bg_u, λ_burst_u, transition_u, start_prob_u) are learned by
Baum-Welch (EM with forward-backward). The forward-backward pass then produces
the per-bin posterior p(state=burst | n_u(0:T)) used downstream as a calibrated
"burst-likeness" signal that respects each unit's own background/burst rate.

Design notes
------------
- Hand-rolled implementation (no hmmlearn dependency required). The state space
  is 2, sequences are <50k bins, so a numpy Baum-Welch is fast enough.
- Stable log-space forward-backward to avoid underflow on long sequences.
- Quantile-based initialization: λ_bg ≈ 25th-percentile rate, λ_burst ≈
  95th-percentile rate. K-means on counts is an alternative — selectable via
  ``init_strategy``.
- Identifiability flip: after EM, swap so state 1 = the higher-rate state.
- Units with too few spikes or a degenerate fit (λ_burst/λ_bg below
  min_rate_ratio) are flagged via ``UnitHMMFit.skipped_reason``; their posterior
  rows are filled with NaN so aggregation (np.nanmean / np.nanstd) drops them.
- joblib parallel across units when n_jobs != 1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from scipy.special import gammaln


@dataclass
class UnitHMMFit:
    """Result of fitting a 2-state Poisson HMM to one unit's bin counts.

    Attributes
    ----------
    unit_id
        Unit identifier (same key as in spike_times dict).
    lambda_bg, lambda_burst
        Per-bin Poisson rates for the background and burst states *in counts per
        bin* (i.e. λ_state × bin_size). Stored in per-bin form so likelihood
        computation does not need to re-multiply by bin_size.
    transition
        2x2 row-stochastic transition matrix. transition[i, j] = P(state_{t+1}=j | state_t=i).
    start_prob
        Initial state distribution (length 2).
    converged
        True if Baum-Welch hit the tolerance criterion before max_iter.
    n_iter
        Number of EM iterations performed.
    loglik
        Final marginal log-likelihood of the observed sequence under the fit.
    skipped_reason
        None when the fit was kept; one of {"low_spike_count", "low_rate_ratio",
        "no_convergence", "numerical_failure"} when the unit's posterior should
        be treated as missing (NaN row).
    """

    unit_id: str
    lambda_bg: float = float("nan")
    lambda_burst: float = float("nan")
    transition: np.ndarray = field(
        default_factory=lambda: np.array([[0.95, 0.05], [0.30, 0.70]], dtype=float)
    )
    start_prob: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.1], dtype=float))
    converged: bool = False
    n_iter: int = 0
    loglik: float = float("nan")
    skipped_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------


def _quantile_init(counts: np.ndarray) -> tuple[float, float]:
    """Quantile-based initial rate estimates.

    λ_bg ~ 25th percentile (the low-firing baseline) clipped to a small floor.
    λ_burst ~ 95th percentile clipped to ≥ 2 × λ_bg so the EM starts from a
    well-separated initialization.
    """
    if counts.size == 0:
        return 0.05, 0.5
    q25 = float(np.quantile(counts, 0.25))
    q95 = float(np.quantile(counts, 0.95))
    lam_bg = max(q25, 0.01)
    lam_burst = max(q95, 2.0 * lam_bg, lam_bg + 0.5)
    return lam_bg, lam_burst


def _kmeans_init(counts: np.ndarray, random_state: int) -> tuple[float, float]:
    """1D k-means with k=2 to get initial rates.

    Falls back to quantile init if k-means returns degenerate centers.
    """
    rng = np.random.default_rng(random_state)
    x = counts.astype(float)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return _quantile_init(counts)
    # Initialise two centers at low and high tails
    centers = np.array([lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)], dtype=float)
    for _ in range(20):
        d0 = np.abs(x - centers[0])
        d1 = np.abs(x - centers[1])
        labels = (d1 < d0).astype(int)
        new_centers = centers.copy()
        for k in (0, 1):
            sel = labels == k
            if sel.any():
                new_centers[k] = float(x[sel].mean())
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    lam_bg, lam_burst = float(min(centers)), float(max(centers))
    if lam_burst < 2.0 * max(lam_bg, 1e-3):
        # Degenerate k-means: jitter and re-try quantile init
        _ = rng.random()  # touch rng so different states differ
        return _quantile_init(counts)
    return max(lam_bg, 1e-3), lam_burst


# ---------------------------------------------------------------------------
# Forward-backward (log-space)
# ---------------------------------------------------------------------------


def _poisson_logpmf(counts: np.ndarray, lam_per_bin: float) -> np.ndarray:
    """log P(n | Poisson(lam)) for an array of counts.

    Numerically safe for lam=0 (returns log(δ_{n,0}) treated as a large negative
    number for n>0 so it doesn't poison forward-backward).
    """
    lam = max(float(lam_per_bin), 1e-12)
    # log(lam^n * e^{-lam} / n!) = n*log(lam) - lam - gammaln(n+1)
    return counts * np.log(lam) - lam - gammaln(counts + 1.0)


def _forward_backward(
    log_emit: np.ndarray,  # (T, 2) log P(n_t | state=k)
    log_trans: np.ndarray,  # (2, 2) log P(state_{t+1}=j | state_t=i)
    log_start: np.ndarray,  # (2,) log P(state_0=k)
) -> tuple[np.ndarray, np.ndarray, float]:
    """Stable log-space forward-backward for a 2-state HMM.

    Returns
    -------
    log_alpha : (T, 2)
    log_beta  : (T, 2)
    loglik    : float — log P(observations) = logsumexp(log_alpha[-1])
    """
    T = log_emit.shape[0]
    log_alpha = np.empty((T, 2), dtype=float)
    log_beta = np.empty((T, 2), dtype=float)
    log_alpha[0] = log_start + log_emit[0]
    for t in range(1, T):
        # log_alpha[t, j] = log_emit[t, j] + logsumexp_i (log_alpha[t-1, i] + log_trans[i, j])
        prev = log_alpha[t - 1][:, None] + log_trans  # (2, 2): i x j
        log_alpha[t] = log_emit[t] + _logsumexp_axis(prev, axis=0)
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        # log_beta[t, i] = logsumexp_j (log_trans[i, j] + log_emit[t+1, j] + log_beta[t+1, j])
        nxt = log_trans + (log_emit[t + 1] + log_beta[t + 1])[None, :]
        log_beta[t] = _logsumexp_axis(nxt, axis=1)
    loglik = float(_logsumexp_vec(log_alpha[-1]))
    return log_alpha, log_beta, loglik


def _logsumexp_vec(x: np.ndarray) -> float:
    m = float(x.max())
    if not np.isfinite(m):
        return float(m)
    return m + float(np.log(np.exp(x - m).sum()))


def _logsumexp_axis(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(m), m, 0.0)
    out = safe + np.log(np.sum(np.exp(x - safe), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


# ---------------------------------------------------------------------------
# Baum-Welch (EM)
# ---------------------------------------------------------------------------


def _baum_welch(
    counts: np.ndarray,
    lam_bg_init: float,
    lam_burst_init: float,
    max_iter: int,
    tol: float,
) -> tuple[float, float, np.ndarray, np.ndarray, float, int, bool]:
    """Run Baum-Welch EM for a 2-state Poisson HMM. Returns fitted parameters.

    Returns (lambda_bg, lambda_burst, transition, start_prob, loglik, n_iter, converged).
    """
    T = counts.shape[0]
    lam = np.array([lam_bg_init, lam_burst_init], dtype=float)
    # A reasonable initial transition matrix biased toward staying in the
    # current state (bursts are temporally extended, not random per-bin events).
    transition = np.array([[0.95, 0.05], [0.30, 0.70]], dtype=float)
    start_prob = np.array([0.9, 0.1], dtype=float)

    prev_ll = -np.inf
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        # Emission log-prob matrix
        log_emit = np.column_stack(
            [_poisson_logpmf(counts, lam[0]), _poisson_logpmf(counts, lam[1])]
        )
        log_trans = np.log(np.clip(transition, 1e-12, None))
        log_start = np.log(np.clip(start_prob, 1e-12, None))

        log_alpha, log_beta, loglik = _forward_backward(log_emit, log_trans, log_start)
        # Posterior gamma: γ_t(k) = P(state_t = k | obs)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp_axis(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)  # (T, 2)

        # Pairwise xi: ξ_t(i, j) summed across t for the transition update
        # log ξ_t(i,j) = log_alpha[t,i] + log_trans[i,j] + log_emit[t+1,j] + log_beta[t+1,j] - loglik
        if T > 1:
            log_xi_t = (
                log_alpha[:-1, :, None]
                + log_trans[None, :, :]
                + log_emit[1:, None, :]
                + log_beta[1:, None, :]
                - loglik
            )
            xi_sum = np.exp(log_xi_t).sum(axis=0)  # (2, 2)
        else:
            xi_sum = np.zeros((2, 2), dtype=float)

        # M-step
        start_prob = gamma[0] / max(gamma[0].sum(), 1e-12)
        row_sums = xi_sum.sum(axis=1, keepdims=True)
        transition = np.where(row_sums > 1e-12, xi_sum / np.where(row_sums > 1e-12, row_sums, 1.0), transition)
        # Re-estimate Poisson rates: λ_k = Σ_t γ_t(k) * n_t / Σ_t γ_t(k)
        gamma_sum = gamma.sum(axis=0)  # (2,)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam_new = (gamma * counts[:, None]).sum(axis=0) / np.where(
                gamma_sum > 1e-12, gamma_sum, 1.0
            )
        # Guard against collapsed states
        lam = np.where(gamma_sum > 1e-12, lam_new, lam)
        lam = np.maximum(lam, 1e-6)

        if abs(loglik - prev_ll) < tol:
            converged = True
            break
        prev_ll = loglik

    return float(lam[0]), float(lam[1]), transition, start_prob, float(loglik), int(n_iter), bool(converged)


# ---------------------------------------------------------------------------
# Public API: per-unit fit and posterior
# ---------------------------------------------------------------------------


def fit_unit_hmm(
    counts: np.ndarray,
    bin_size: float,
    *,
    unit_id: str = "",
    max_iter: int = 100,
    tol: float = 1e-3,
    min_spikes: int = 50,
    init_strategy: str = "quantile",
    min_rate_ratio: float = 1.5,
    random_state: int = 42,
) -> UnitHMMFit:
    """Fit a 2-state Poisson HMM to a unit's bin-count sequence.

    Parameters
    ----------
    counts
        Integer (or float) array of spike counts per bin for one unit, shape (T,).
    bin_size
        Bin width in seconds. Used only to convert per-bin λ to Hz for
        diagnostics (the model itself operates in counts-per-bin).
    unit_id
        Identifier carried through to the returned ``UnitHMMFit``.
    max_iter
        Hard cap on Baum-Welch iterations.
    tol
        Stop when |Δ log-likelihood| < tol between iterations.
    min_spikes
        Units with fewer total spikes are skipped (`skipped_reason="low_spike_count"`).
    init_strategy
        ``"quantile"`` (default) or ``"kmeans"``.
    min_rate_ratio
        After fitting, reject the fit if λ_burst / λ_bg < min_rate_ratio.
    random_state
        Seeds k-means init only; Baum-Welch is deterministic given init.
    """
    counts = np.asarray(counts, dtype=float)
    fit = UnitHMMFit(unit_id=str(unit_id))

    total_spikes = float(counts.sum())
    if total_spikes < float(min_spikes):
        fit.skipped_reason = "low_spike_count"
        return fit

    if counts.size < 4:
        fit.skipped_reason = "low_spike_count"
        return fit

    if init_strategy == "kmeans":
        lam_bg_init, lam_burst_init = _kmeans_init(counts, random_state)
    else:
        lam_bg_init, lam_burst_init = _quantile_init(counts)

    if lam_burst_init / max(lam_bg_init, 1e-6) < min_rate_ratio:
        fit.skipped_reason = "low_rate_ratio"
        fit.lambda_bg = float(lam_bg_init) / max(bin_size, 1e-12)
        fit.lambda_burst = float(lam_burst_init) / max(bin_size, 1e-12)
        return fit

    try:
        lam_bg, lam_burst, transition, start_prob, loglik, n_iter, converged = _baum_welch(
            counts, lam_bg_init, lam_burst_init, max_iter, tol,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        fit.skipped_reason = "numerical_failure"
        return fit

    # Identifiability: state 1 must be the higher-rate ("burst") state.
    if lam_bg > lam_burst:
        lam_bg, lam_burst = lam_burst, lam_bg
        transition = transition[::-1, ::-1].copy()
        start_prob = start_prob[::-1].copy()

    # Convert per-bin rates to Hz for the persistent fields (so downstream
    # diagnostics are interpretable). The internal posterior still recomputes
    # the per-bin rates from these via × bin_size.
    fit.lambda_bg = float(lam_bg) / max(bin_size, 1e-12)
    fit.lambda_burst = float(lam_burst) / max(bin_size, 1e-12)
    fit.transition = transition
    fit.start_prob = start_prob
    fit.converged = converged
    fit.n_iter = n_iter
    fit.loglik = loglik

    if fit.lambda_burst / max(fit.lambda_bg, 1e-6) < min_rate_ratio:
        fit.skipped_reason = "low_rate_ratio"

    return fit


def posterior_burst(counts: np.ndarray, fit: UnitHMMFit, bin_size: float) -> np.ndarray:
    """Forward-backward posterior p(state=burst | obs) per bin.

    Returns
    -------
    posterior : (T,) array of P(burst-state) in [0, 1], or NaN array of length T
        when the fit is missing or marked skipped.
    """
    counts = np.asarray(counts, dtype=float)
    T = counts.shape[0]
    if fit.skipped_reason is not None or not np.isfinite(fit.lambda_bg) or not np.isfinite(fit.lambda_burst):
        return np.full(T, np.nan, dtype=float)

    lam_bg_bin = float(fit.lambda_bg) * float(bin_size)
    lam_burst_bin = float(fit.lambda_burst) * float(bin_size)
    log_emit = np.column_stack(
        [_poisson_logpmf(counts, lam_bg_bin), _poisson_logpmf(counts, lam_burst_bin)]
    )
    log_trans = np.log(np.clip(fit.transition, 1e-12, None))
    log_start = np.log(np.clip(fit.start_prob, 1e-12, None))
    log_alpha, log_beta, _ = _forward_backward(log_emit, log_trans, log_start)
    log_gamma = log_alpha + log_beta
    log_gamma -= _logsumexp_axis(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)
    # state 1 is burst by construction (identifiability flip in fit_unit_hmm)
    return gamma[:, 1]


def fit_all_units(
    spike_matrix: np.ndarray,
    unit_ids: list[str],
    bin_size: float,
    *,
    max_iter: int = 100,
    tol: float = 1e-3,
    min_spikes: int = 50,
    init_strategy: str = "quantile",
    min_rate_ratio: float = 1.5,
    random_state: int = 42,
    n_jobs: int = 1,
) -> tuple[list[UnitHMMFit], np.ndarray]:
    """Fit per-unit HMMs and return (fits, posterior_matrix).

    Parameters
    ----------
    spike_matrix
        (n_units, n_bins) float array of per-bin counts.
    unit_ids
        Identifiers aligned to rows of ``spike_matrix``.
    n_jobs
        joblib parallelism. ``1`` = serial. Use ``-1`` for all cores. Falls
        back to serial when joblib is not available.

    Returns
    -------
    fits
        Length-n_units list of ``UnitHMMFit``. Index aligns to rows of
        ``spike_matrix`` / ``unit_ids``.
    posterior_matrix
        (n_units, n_bins) float array. Skipped units have rows filled with NaN.
    """
    spike_matrix = np.asarray(spike_matrix, dtype=float)
    n_units, n_bins = spike_matrix.shape
    if len(unit_ids) != n_units:
        raise ValueError(
            f"unit_ids has {len(unit_ids)} entries but spike_matrix has {n_units} rows"
        )

    def _fit_one(idx: int) -> UnitHMMFit:
        return fit_unit_hmm(
            spike_matrix[idx],
            bin_size,
            unit_id=str(unit_ids[idx]),
            max_iter=max_iter,
            tol=tol,
            min_spikes=min_spikes,
            init_strategy=init_strategy,
            min_rate_ratio=min_rate_ratio,
            random_state=random_state,
        )

    parallel_jobs = int(n_jobs)
    fits: list[UnitHMMFit]
    if parallel_jobs == 1 or n_units <= 1:
        fits = [_fit_one(i) for i in range(n_units)]
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            fits = [_fit_one(i) for i in range(n_units)]
        else:
            fits = list(
                Parallel(n_jobs=parallel_jobs, backend="loky")(
                    delayed(_fit_one)(i) for i in range(n_units)
                )
            )

    posterior_matrix = np.full((n_units, n_bins), np.nan, dtype=float)
    for i, fit in enumerate(fits):
        if fit.skipped_reason is None:
            posterior_matrix[i] = posterior_burst(spike_matrix[i], fit, bin_size)

    return fits, posterior_matrix


def fit_to_dict(fit: UnitHMMFit) -> dict:
    """Serializable view of a fit (numpy arrays converted to lists)."""
    d = asdict(fit)
    d["transition"] = fit.transition.tolist()
    d["start_prob"] = fit.start_prob.tolist()
    return d
