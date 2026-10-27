"""M1 -- the statistical estimator.

CRITICAL INVARIANT: the null harness (M2) and the real run (M5) must call the
SAME functions here.  Comparing an ordinary null t-stat against a Newey-West
real t-stat is a silent methodological error, so there is exactly one
implementation of each statistic and both paths import it.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from config import POSITION_CLIP, TRADING_DAYS

EULER_GAMMA = 0.5772156649015329


# ------------------------------------------------------------------ helpers
def _finite_pair(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


# ------------------------------------------------------------------ IC
def spearman_ic(x, y) -> float:
    x, y = _finite_pair(x, y)
    if len(x) < 30 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(sps.spearmanr(x, y).statistic)


# ------------------------------------------------------- Newey-West t-stat
def newey_west_slope(y, x, lag: int):
    """OLS y = a + b*x with Bartlett-kernel HAC standard error on b.

    `lag` must be H-1 for an H-day overlapping forward target.
    Returns (beta, t_stat, p_value_two_sided, n_obs).
    """
    y, x = _finite_pair(y, x)          # note: order kept consistent below
    n = len(y)
    if n < 30 or np.std(x) == 0:
        return np.nan, np.nan, np.nan, n
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, n
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    u = X * e[:, None]
    S = u.T @ u
    L = max(int(lag), 0)
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = u[l:].T @ u[:-l]
        S = S + w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    var_b = V[1, 1]
    if not np.isfinite(var_b) or var_b <= 0:
        return float(beta[1]), np.nan, np.nan, n
    t = float(beta[1] / np.sqrt(var_b))
    p = float(2.0 * (1.0 - sps.norm.cdf(abs(t))))
    return float(beta[1]), t, p, n


def newey_west_mean_tstat(series, lag: int):
    """HAC t-stat for the mean of a (possibly autocorrelated) P&L series."""
    v = np.asarray(series, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 30:
        return np.nan, np.nan, n
    mu = v.mean()
    e = v - mu
    S = float(e @ e)
    L = max(int(lag), 0)
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        S += 2.0 * w * float(e[l:] @ e[:-l])
    var_mu = S / (n * n)
    if var_mu <= 0 or not np.isfinite(var_mu):
        return np.nan, np.nan, n
    t = mu / np.sqrt(var_mu)
    return float(t), float(2.0 * (1.0 - sps.norm.cdf(abs(t)))), n


# ------------------------------------------------------------------ P&L
def build_position(x, sign: float, horizon: int, mode: str) -> np.ndarray:
    """Signal -> position, held for `horizon` days as an overlapping portfolio.

    mode 'continuous': w = clip(z, +-CLIP)/CLIP  in [-1, 1]
    mode 'direction' : w = sign(z)               in {-1, +1}
    The overlapping average is the standard Jegadeesh-Titman construction: it
    turns an H-day holding period into an honest daily P&L series instead of
    an overlapping-return artefact.
    """
    z = np.asarray(x, float)
    if mode == "direction":
        w = np.sign(z)
    else:
        w = np.clip(z, -POSITION_CLIP, POSITION_CLIP) / POSITION_CLIP
    w = sign * w
    w = np.where(np.isfinite(w), w, 0.0)
    n = len(w)
    if horizon <= 1 or n <= horizon:
        return w
    c = np.cumsum(np.concatenate(([0.0], w)))
    out = np.zeros(n)
    out[horizon - 1:] = (c[horizon:] - c[: n - horizon + 1]) / horizon
    return out


def strategy_stats(position, fwd1_return, cost_bps: float, n_legs: int):
    """Daily P&L, Sharpe, turnover and drawdown for a unit-notional strategy.

    `cost_bps` is ROUND-TRIP.  A full open-and-close cycle accumulates
    sum|dW| = 2, so the per-unit-turnover charge is cost_bps/2, applied to each
    leg (a pair trade touches two instruments).
    """
    w = np.asarray(position, float)
    r = np.asarray(fwd1_return, float)
    m = np.isfinite(w) & np.isfinite(r)
    w = np.where(m, w, 0.0)
    r = np.where(m, r, 0.0)
    n = int(m.sum())
    if n < 60:
        return dict(n=n, gross_sharpe=np.nan, net_sharpe=np.nan, turnover_annual=np.nan,
                    max_drawdown=np.nan, skew=np.nan, kurtosis=np.nan,
                    pnl=np.zeros_like(r), mean_gross=np.nan, mean_net=np.nan)
    dw = np.abs(np.diff(np.concatenate(([0.0], w))))
    gross = w * r
    cost = dw * (cost_bps / 2.0 / 1e4) * n_legs
    net = gross - cost
    sd = gross.std(ddof=1)
    sd_n = net.std(ddof=1)
    eq = np.cumsum(net)
    dd = eq - np.maximum.accumulate(eq)
    return dict(
        n=n,
        gross_sharpe=float(gross.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
        net_sharpe=float(net.mean() / sd_n * np.sqrt(TRADING_DAYS)) if sd_n > 0 else np.nan,
        turnover_annual=float(dw.sum() * TRADING_DAYS / len(dw)),
        max_drawdown=float(dd.min()),
        skew=float(sps.skew(net[np.isfinite(net)])),
        kurtosis=float(sps.kurtosis(net[np.isfinite(net)], fisher=False)),
        pnl=net, mean_gross=float(gross.mean()), mean_net=float(net.mean()),
    )


# --------------------------------------------------- stationary bootstrap
def stationary_bootstrap_index(n: int, mean_block: int, rng) -> np.ndarray:
    """Politis-Romano stationary block bootstrap indices (vectorised, wrap-around).

    Preserves local dependence, volatility clustering and fat tails.
    A plain IID permutation is FORBIDDEN here: it destroys the volatility
    clustering that is precisely what inflates t-statistics.
    """
    p = 1.0 / max(mean_block, 1)
    restart = rng.random(n) < p
    restart[0] = True
    seg = np.cumsum(restart) - 1
    pos = np.arange(n)
    seg_start = np.zeros(seg[-1] + 1, dtype=np.int64)
    seg_start[seg[restart]] = pos[restart]
    starts = rng.integers(0, n, size=seg[-1] + 1)
    offset = pos - seg_start[seg]
    return (starts[seg] + offset) % n


def bootstrap_mean_pvalue(series, mean_block: int, rng, n_rep: int = 2000):
    """Two-sided bootstrap p-value and CI for the mean of an autocorrelated series.

    Resamples the DEMEANED series, so the reference distribution is centred on
    zero -- this is the null of 'no edge', not a CI around the point estimate.
    """
    v = np.asarray(series, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 60:
        return np.nan, (np.nan, np.nan)
    obs = v.mean()
    centred = v - obs
    boot = np.empty(n_rep)
    for i in range(n_rep):
        idx = stationary_bootstrap_index(n, mean_block, rng)
        boot[i] = centred[idx].mean()
    p = float((np.abs(boot) >= abs(obs)).mean())
    lo, hi = np.percentile(boot + obs, [2.5, 97.5])
    return max(p, 1.0 / n_rep), (float(lo), float(hi))


# ------------------------------------------------------- multiple testing
def benjamini_hochberg(pvals, alpha: float):
    """Returns (rejected_mask, adjusted_pvalues)."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    adj = np.full(len(p), np.nan)
    rej = np.zeros(len(p), bool)
    if ok.sum() == 0:
        return rej, adj
    idx = np.where(ok)[0]
    ps = p[idx]
    order = np.argsort(ps)
    m = len(ps)
    ranked = ps[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    adj_sorted = np.empty(m)
    adj_sorted[order] = q
    adj[idx] = adj_sorted
    rej[idx] = adj_sorted <= alpha
    return rej, adj


def deflated_sharpe_ratio(sr_daily, n_trials, n_obs, skew, kurtosis, sr_variance):
    """Bailey & Lopez de Prado deflated Sharpe.

    sr_daily is per-observation (NOT annualised); sr_variance is the variance of
    the Sharpe ratios across the `n_trials` trials actually run.
    """
    if not np.isfinite(sr_daily) or n_obs < 30 or n_trials < 2:
        return np.nan
    if not np.isfinite(sr_variance) or sr_variance <= 0:
        return np.nan
    N = float(n_trials)
    z1 = sps.norm.ppf(1.0 - 1.0 / N)
    z2 = sps.norm.ppf(1.0 - 1.0 / (N * np.e))
    sr_star = np.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    denom = 1.0 - skew * sr_daily + (kurtosis - 1.0) / 4.0 * sr_daily ** 2
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    stat = (sr_daily - sr_star) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(sps.norm.cdf(stat))


def bootstrap_slope_pvalue(y, x, mean_block: int, rng, n_rep: int = 2000):
    """Block-bootstrap p-value and CI for the slope of y on x.

    Pairs (x, y) are resampled jointly in blocks, so the dependence structure of
    BOTH series and their overlap is preserved.  The null distribution is the
    bootstrap distribution recentred on the point estimate (the standard pivot),
    which is the correct reference for 'the slope is zero'.

    This is the independent check on Newey-West demanded for overlapping
    targets; where the two disagree, the bootstrap is the conservative one.
    """
    y, x = _finite_pair(y, x)
    n = len(y)
    if n < 100 or np.std(x) == 0:
        return np.nan, (np.nan, np.nan)
    sx, sy = x.sum(), y.sum()
    b_hat = (n * (x * y).sum() - sx * sy) / (n * (x * x).sum() - sx * sx)
    boot = np.empty(n_rep)
    for i in range(n_rep):
        idx = stationary_bootstrap_index(n, mean_block, rng)
        xb, yb = x[idx], y[idx]
        sxb, syb = xb.sum(), yb.sum()
        den = n * (xb * xb).sum() - sxb * sxb
        boot[i] = ((n * (xb * yb).sum() - sxb * syb) / den) if den != 0 else np.nan
    boot = boot[np.isfinite(boot)]
    if len(boot) < 100:
        return np.nan, (np.nan, np.nan)
    centred = boot - b_hat
    p = float((np.abs(centred) >= abs(b_hat)).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return max(p, 1.0 / len(boot)), (float(lo), float(hi))
