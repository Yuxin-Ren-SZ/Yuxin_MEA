"""Neuronal-avalanche criticality analysis for one well-recording.

Grounded in the standard references:

* **Avalanche detection** — Beggs & Plenz (2003): pool the curated spikes into a
  population raster, bin at Δt = the mean inter-event interval (IEI) across the
  array, and define an avalanche as a run of consecutive non-empty bins bounded
  by empty bins. Avalanche *size* = spikes in the run; *duration* = number of bins.
* **Power-law exponents** — fit the avalanche size (τ) and duration (α)
  distributions by MLE with KS-optimised ``x_min`` and compare against
  lognormal/exponential (``powerlaw``, Alstott et al. 2014 / Clauset et al. 2009).
* **DCC** (Deviation from Criticality Coefficient, Ma et al. 2019 *Neuron*):
  ``DCC = |γ_fit − γ_pred|`` where ``γ_pred = (α−1)/(τ−1)`` (crackling-noise
  relation) and ``γ_fit`` is the slope of mean-size vs duration.
* **Branching ratio** — the subsampling-robust MR estimator (``mrestimator``,
  Wilting & Priesemann 2018 *Nat Commun*) plus a naive avalanche estimator for
  comparison. At criticality m ≈ 1, DCC → 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class CriticalityError(ValueError):
    """Raised when a well has too little activity for avalanche statistics."""


@dataclass
class CriticalityConfig:
    min_spikes: int = 200          # pooled spikes required for any statistics
    min_avalanches: int = 30       # avalanches required for a power-law fit
    bin_mode: str = "mean_iei"     # "mean_iei" (Beggs-Plenz) or "fixed"
    fixed_bin_s: float = 0.004
    mr_max_bin_s: float = 0.05     # cap the (possibly tiny) IEI bin for the MR estimator
    mr_kmax: int = 150             # MR autocorrelation steps
    seed: int = 0

    @classmethod
    def from_task_params(cls, p: dict) -> "CriticalityConfig":
        f = {k: p[k] for k in cls.__dataclass_fields__ if k in p}
        return cls(**f)


@dataclass
class CriticalityResults:
    avalanches: "object"           # DataFrame [size, duration]
    metrics: dict
    powerlaw_fits: dict
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Avalanche detection
# --------------------------------------------------------------------------- #
def _pool(spike_times: dict) -> np.ndarray:
    if not spike_times:
        return np.array([])
    return np.sort(np.concatenate([np.asarray(s, float).ravel()
                                   for s in spike_times.values() if len(s)]))


def _bin_width(pooled: np.ndarray, cfg: CriticalityConfig) -> float:
    if cfg.bin_mode == "fixed":
        return float(cfg.fixed_bin_s)
    iei = np.diff(pooled)
    iei = iei[iei > 0]
    return float(np.mean(iei)) if iei.size else float(cfg.fixed_bin_s)


def detect_avalanches(pooled: np.ndarray, dt: float):
    """Return (sizes, durations, counts) — counts is the binned population raster."""
    t0, t1 = pooled[0], pooled[-1]
    nbins = max(1, int(np.ceil((t1 - t0) / dt)))
    counts, _ = np.histogram(pooled, bins=nbins, range=(t0, t0 + nbins * dt))
    active = counts > 0
    sizes, durs = [], []
    i, n = 0, len(active)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            sizes.append(int(counts[i:j].sum()))
            durs.append(int(j - i))
            i = j
        else:
            i += 1
    return np.array(sizes), np.array(durs), counts


# --------------------------------------------------------------------------- #
# Power-law + crackling-noise (DCC)
# --------------------------------------------------------------------------- #
def _fit_powerlaw(x: np.ndarray) -> dict:
    """MLE power-law exponent with KS x_min + comparison to lognormal/exp."""
    import powerlaw

    x = x[x > 0]
    unfittable = {"exponent": np.nan, "xmin": np.nan, "ks": np.nan,
                  "R_lognormal": np.nan, "p_lognormal": np.nan, "n": int(len(x))}
    if len(x) < 20 or np.unique(x).size < 3:
        return unfittable
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # `powerlaw.Fit` builds each candidate distribution lazily in
        # ``Fit.__getattr__``, so the MLE actually runs on first attribute access
        # — ``fit.alpha``/``fit.xmin``/``fit.power_law`` all fit here, not above.
        # On a degenerate avalanche distribution (e.g. a 2-unit well) the search
        # lands on an x_min that excludes every point and powerlaw raises
        # ValueError("No data points in defined range of the distribution") from
        # inside scipy's optimiser. There is then no exponent to report, so the
        # honest answer is the all-NaN row: the well still gets a criticality
        # output recording n_avalanches, rather than failing the whole task and
        # leaving the well with no output at all.
        try:
            fit = powerlaw.Fit(x, discrete=True, verbose=False)
            exponent = float(fit.alpha)
            xmin = float(fit.xmin)
            # KS distance; .KS() is broken in powerlaw 2.0.0, hence .D
            ks = float(getattr(fit.power_law, "D", np.nan))
        except Exception:  # noqa: BLE001 — degenerate input, not a code fault
            return unfittable
        # The lognormal comparison is a goodness-of-fit diagnostic, not the
        # estimate, so it degrades on its own without discarding the exponent.
        try:
            R, p = fit.distribution_compare("power_law", "lognormal",
                                            normalized_ratio=True)
            R, p = float(R), float(p)
        except Exception:  # noqa: BLE001
            R = p = float("nan")
    return {"exponent": exponent, "xmin": xmin,
            "ks": ks, "R_lognormal": R,
            "p_lognormal": p, "n": int(len(x))}


def _gamma_fit(sizes: np.ndarray, durs: np.ndarray) -> float:
    """Slope of log(mean avalanche size) vs log(duration) (crackling noise)."""
    if len(sizes) < 10:
        return np.nan
    df = {}
    for s, d in zip(sizes, durs):
        df.setdefault(d, []).append(s)
    xs = np.array(sorted(k for k, v in df.items() if len(v) >= 3 and k > 0))
    if len(xs) < 3:
        return np.nan
    ys = np.array([np.mean(df[k]) for k in xs])
    coef = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(coef[0])


# --------------------------------------------------------------------------- #
# Branching ratio
# --------------------------------------------------------------------------- #
def _branching_naive(counts: np.ndarray) -> float:
    """Naive σ = mean_t a_{t+1}/a_t over active-to-next transitions."""
    a = counts.astype(float)
    prev, nxt = a[:-1], a[1:]
    m = prev > 0
    return float(np.mean(nxt[m] / prev[m])) if m.any() else np.nan


def _branching_mr(counts: np.ndarray, dt: float, cfg: CriticalityConfig) -> tuple:
    """MR estimator branching ratio + intrinsic timescale (Wilting-Priesemann)."""
    import contextlib
    import io
    import logging
    import warnings
    try:
        import mrestimator as mre
        logging.getLogger("mrestimator").setLevel(logging.ERROR)
        act = counts.astype(float)
        kmax = int(min(cfg.mr_kmax, max(10, len(act) // 4)))
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            rk = mre.coefficients(act, steps=(1, kmax), dt=1, numboot=0, method="ts")
            fit = mre.fit(rk, fitfunc="exponential")
        return float(fit.mre), float(fit.tau) * dt   # timescale: steps → seconds
    except Exception:  # noqa: BLE001
        return np.nan, np.nan


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def compute_criticality(spike_times: dict,
                        config: CriticalityConfig | None = None) -> CriticalityResults:
    import pandas as pd

    cfg = config or CriticalityConfig()
    pooled = _pool(spike_times)
    if pooled.size < cfg.min_spikes:
        raise CriticalityError(f"only {pooled.size} pooled spikes (< {cfg.min_spikes})")

    dt = _bin_width(pooled, cfg)
    sizes, durs, counts = detect_avalanches(pooled, dt)
    n_av = len(sizes)

    size_fit = _fit_powerlaw(sizes) if n_av >= cfg.min_avalanches else \
        {"exponent": np.nan, "xmin": np.nan, "ks": np.nan}
    dur_fit = _fit_powerlaw(durs) if n_av >= cfg.min_avalanches else \
        {"exponent": np.nan, "xmin": np.nan, "ks": np.nan}
    tau = size_fit["exponent"]
    alpha = dur_fit["exponent"]
    gamma_fit = _gamma_fit(sizes, durs)
    gamma_pred = ((alpha - 1.0) / (tau - 1.0)
                  if np.isfinite(alpha) and np.isfinite(tau) and (tau - 1.0) != 0
                  else np.nan)
    dcc = abs(gamma_fit - gamma_pred) if np.isfinite(gamma_fit) and np.isfinite(gamma_pred) else np.nan

    br_naive = _branching_naive(counts)
    br_mr, tau_c = _branching_mr(counts, dt, cfg)

    metrics = {
        "branching_ratio_mr": br_mr,
        "branching_ratio_naive": br_naive,
        "tau_C_s": tau_c,
        "aval_tau": tau,
        "aval_alpha": alpha,
        "gamma_fit": gamma_fit,
        "gamma_pred": gamma_pred,
        "dcc": dcc,
        "n_avalanches": int(n_av),
        "mean_aval_size": float(np.mean(sizes)) if n_av else np.nan,
        "mean_aval_duration_bins": float(np.mean(durs)) if n_av else np.nan,
    }
    diagnostics = {
        "dt_s": dt, "n_bins": int(len(counts)), "n_pooled_spikes": int(pooled.size),
        "n_units": int(len(spike_times)), "bin_mode": cfg.bin_mode,
        "recording_span_s": float(pooled[-1] - pooled[0]),
    }
    avalanches = pd.DataFrame({"size": sizes, "duration": durs})
    return CriticalityResults(avalanches, metrics,
                              {"size": size_fit, "duration": dur_fit}, diagnostics)


def empty_criticality_results(reason: str = "") -> CriticalityResults:
    import pandas as pd
    nan = float("nan")
    metrics = {k: nan for k in (
        "branching_ratio_mr", "branching_ratio_naive", "tau_C_s", "aval_tau",
        "aval_alpha", "gamma_fit", "gamma_pred", "dcc", "mean_aval_size",
        "mean_aval_duration_bins")}
    metrics["n_avalanches"] = 0
    return CriticalityResults(pd.DataFrame(columns=["size", "duration"]),
                              metrics, {}, {"reason": reason})


def write_criticality(results: CriticalityResults, output_dir) -> None:
    import json
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.avalanches.to_parquet(out / "avalanches.parquet")
    (out / "criticality_metrics.json").write_text(json.dumps(results.metrics))
    (out / "powerlaw_fits.json").write_text(json.dumps(results.powerlaw_fits))
    (out / "diagnostics.json").write_text(json.dumps(results.diagnostics))
