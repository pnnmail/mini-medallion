"""P1/P2 -- the exact decomposition of a sizing decision, and its null.

    mean(w * r) = mean(w) * mean(r) + cov(w, r)

is an identity, not a fit.  Applied to two forecasts it separates "holds more on
average" from "holds more at the right moments", which is precisely the
distinction M1 could not make and therefore could not explain.

Any residual is printed, never absorbed.
"""
from __future__ import annotations

import numpy as np

from config_m2 import TRADING_DAYS


def _clean(w, r, mask):
    w = np.asarray(w, float)[mask]
    r = np.asarray(r, float)[mask]
    ok = np.isfinite(w) & np.isfinite(r)
    return w[ok], r[ok]


def pop_cov(a, b) -> float:
    """Population covariance -- the one that makes the identity close exactly."""
    return float(np.mean(a * b) - np.mean(a) * np.mean(b))


def decompose_pair(w_a, w_b, fwd_ret, mask):
    """Exposure vs timing split of the mean-P&L difference between two models."""
    a, r = _clean(w_a, fwd_ret, mask)
    b, r2 = _clean(w_b, fwd_ret, mask)
    n = min(len(a), len(b), len(r), len(r2))
    a, b, r = a[:n], b[:n], r[:n]
    if n < 60:
        return {}
    mean_r = float(np.mean(r))
    ma, mb = float(np.mean(a)), float(np.mean(b))
    ca, cb = pop_cov(a, r), pop_cov(b, r)
    exposure = (ma - mb) * mean_r
    timing = ca - cb
    delta = float(np.mean(a * r) - np.mean(b * r))
    return {
        "n": int(n), "mean_r": mean_r,
        "mean_w_a": ma, "mean_w_b": mb,
        "cov_a": ca, "cov_b": cb,
        "exposure_term": float(exposure), "timing_term": float(timing),
        "delta_mean_total": delta,
        "residual": float(delta - exposure - timing),
        "exposure_term_ann": float(exposure * TRADING_DAYS),
        "timing_term_ann": float(timing * TRADING_DAYS),
        "delta_mean_ann": float(delta * TRADING_DAYS),
    }


def sharpe_attribution(pnl_a, pnl_b, mask):
    """Exact two-way attribution of a Sharpe difference, interaction shown.

    Sharpe = mean * sqrt(252) / sd, so the difference splits into a mean effect
    at the old volatility, a volatility effect at the old mean, and an explicit
    interaction term. An attribution that hides the interaction is a story, not
    an accounting.
    """
    a = np.asarray(pnl_a, float)[mask]
    b = np.asarray(pnl_b, float)[mask]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 60:
        return {}
    ma, mb = float(np.mean(a)), float(np.mean(b))
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    if sa <= 0 or sb <= 0:
        return {}
    k = np.sqrt(TRADING_DAYS)
    sh_a, sh_b = k * ma / sa, k * mb / sb
    mean_effect = k * (ma - mb) / sb
    vol_effect = k * mb * (1.0 / sa - 1.0 / sb)
    interaction = k * (ma - mb) * (1.0 / sa - 1.0 / sb)
    return {"sharpe_a": sh_a, "sharpe_b": sh_b, "delta_sharpe": sh_a - sh_b,
            "mean_effect": mean_effect, "vol_effect": vol_effect,
            "interaction": interaction,
            "residual": (sh_a - sh_b) - (mean_effect + vol_effect + interaction),
            "sd_a": sa, "sd_b": sb}


# ------------------------------------------------------------------ null
def circular_shift_null(d, r, n_rep, rng, statistic="cov"):
    """Null distribution for cov(d, r) with the cross-link destroyed.

    A random circular shift of d against r preserves the ENTIRE autocovariance
    structure of both series and destroys only their cross-dependence. That is
    exactly the null wanted here: the spurious covariance two persistent series
    generate when nothing connects them. Resampling pairs would preserve the
    link and answer a different question.
    """
    d = np.asarray(d, float)
    r = np.asarray(r, float)
    ok = np.isfinite(d) & np.isfinite(r)
    d, r = d[ok], r[ok]
    n = len(d)
    if n < 200:
        return np.array([]), np.nan
    obs = pop_cov(d, r) if statistic == "cov" else float(np.corrcoef(d, r)[0, 1])
    shifts = rng.integers(1, n, size=n_rep)
    out = np.empty(n_rep)
    for i, s in enumerate(shifts):
        ds = np.roll(d, int(s))
        out[i] = pop_cov(ds, r) if statistic == "cov" \
            else float(np.corrcoef(ds, r)[0, 1])
    return out, obs


def null_summary(null_draws, obs, pct=95):
    if len(null_draws) < 50 or not np.isfinite(obs):
        return {"p95_abs": np.nan, "p_two_sided": np.nan, "beats_null": False}
    a = np.abs(null_draws)
    thr = float(np.percentile(a, pct))
    p = float((a >= abs(obs)).mean())
    return {"p95_abs": thr, "p_two_sided": max(p, 1.0 / len(null_draws)),
            "beats_null": bool(abs(obs) > thr)}
