"""M1 -- feature calculators. Every transform here is strictly causal.

The look-ahead test in scripts/m1_pipeline_checks.py is the enforcement
mechanism: a feature that peeks fails the shift test and stops the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import AL_REG_WINDOW, TRADING_DAYS


# ------------------------------------------------------------------ basics
def log_returns(price: pd.Series) -> pd.Series:
    p = price.astype(float)
    p = p.where(p > 0)
    return np.log(p).diff()


def zscore_causal(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling z-score using only data up to and including t."""
    mp = min_periods or window
    m = series.rolling(window, min_periods=mp).mean()
    s = series.rolling(window, min_periods=mp).std(ddof=1)
    return (series - m) / s.replace(0.0, np.nan)


def log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a = a.astype(float).where(lambda v: v > 0)
    b = b.astype(float).where(lambda v: v > 0)
    return np.log(a) - np.log(b)


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualised trailing realised volatility (causal)."""
    return np.sqrt(TRADING_DAYS * (returns ** 2).rolling(window, min_periods=window).mean())


# ------------------------------------------------------------------ half-life
def half_life(series: pd.Series):
    """AR(1) half-life of a spread. Returns (value, note)."""
    s = series.dropna()
    if len(s) < 50:
        return np.nan, "insufficient_data"
    y = s.values[1:]
    x = s.values[:-1]
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx, sxy = (x * x).sum(), (x * y).sum()
    den = n * sxx - sx * sx
    if den == 0:
        return np.nan, "degenerate"
    b = (n * sxy - sx * sy) / den
    if b <= 0:
        return np.nan, "negative_ar"
    if b >= 1:
        return np.nan, "non_stationary"
    return float(-np.log(2.0) / np.log(b)), "ok"


# ------------------------------------------------- Avellaneda-Lee residual
def _rolling_ols_ab(y: np.ndarray, x: np.ndarray, n: int):
    """Rolling simple-OLS intercept/slope over a trailing window of length n."""
    cy = np.concatenate(([0.0], np.cumsum(y)))
    cx = np.concatenate(([0.0], np.cumsum(x)))
    cxx = np.concatenate(([0.0], np.cumsum(x * x)))
    cxy = np.concatenate(([0.0], np.cumsum(x * y)))
    sy = cy[n:] - cy[:-n]
    sx = cx[n:] - cx[:-n]
    sxx = cxx[n:] - cxx[:-n]
    sxy = cxy[n:] - cxy[:-n]
    den = n * sxx - sx * sx
    den = np.where(np.abs(den) < 1e-300, np.nan, den)
    beta = (n * sxy - sx * sy) / den
    alpha = (sy - beta * sx) / n
    return alpha, beta, cy, cx


def avellaneda_lee_score(asset_returns: pd.Series, macro_diff: pd.Series,
                         window: int = AL_REG_WINDOW) -> pd.DataFrame:
    """Avellaneda & Lee s-score on the CUMULATIVE residual path.

    The naive version -- regress today's return on today's factor change and
    z-score the residual -- measures nothing: an OLS residual is idiosyncratic
    daily noise with essentially no autocorrelation.  The construction that
    carries information is:

        1. rolling-window OLS   r_t = a + b * d(factor)_t + e_t
        2. cumulative residual path inside the window  X_k = sum_{j<=k} e_j
        3. fit OU / AR(1) to X, take the equilibrium level m and sigma_eq
        4. s_t = (X_N - m) / sigma_eq,  and X_N == 0 by OLS construction,
           so s_t = -m / sigma_eq

    Everything uses the trailing window only; no full-sample fit.
    Returns columns: s_score, ar_b, kappa, half_life_days.
    """
    df = pd.concat([asset_returns.rename("r"), macro_diff.rename("dm")], axis=1).dropna()
    if len(df) < window + 5:
        return pd.DataFrame(index=asset_returns.index,
                            columns=["s_score", "ar_b", "kappa", "half_life_days"], dtype=float)
    r = df["r"].to_numpy(float)
    dm = df["dm"].to_numpy(float)
    n = int(window)

    alpha, beta, cy, cx = _rolling_ols_ab(r, dm, n)
    # window ending at position t (inclusive) starts at s0 = t - n + 1;
    # windows are indexed 0..T-n, window i ends at t = i + n - 1
    from numpy.lib.stride_tricks import sliding_window_view
    winY = sliding_window_view(cy, n + 1)          # (T-n+1, n+1)
    winX = sliding_window_view(cx, n + 1)
    A = winY - winY[:, :1]                          # cumulative return inside window
    B = winX - winX[:, :1]                          # cumulative factor change inside window
    k = np.arange(n + 1, dtype=float)[None, :]
    X = A - k * alpha[:, None] - beta[:, None] * B  # cumulative residual path, X[:,0]=0

    y_ = X[:, 1:]
    x_ = X[:, :-1]
    m_ = n
    sx = x_.sum(1); sy = y_.sum(1)
    sxx = (x_ * x_).sum(1); sxy = (x_ * y_).sum(1); syy = (y_ * y_).sum(1)
    den = m_ * sxx - sx * sx
    den = np.where(np.abs(den) < 1e-300, np.nan, den)
    b_ar = (m_ * sxy - sx * sy) / den
    a_ar = (sy - b_ar * sx) / m_
    sse = syy - a_ar * sy - b_ar * sxy
    sse = np.where(sse < 0, np.nan, sse)
    var_xi = sse / (m_ - 2)

    ok = (b_ar > 0) & (b_ar < 1) & np.isfinite(var_xi)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_eq = np.where(ok, a_ar / (1.0 - b_ar), np.nan)
        sigma_eq = np.where(ok, np.sqrt(var_xi / (1.0 - b_ar ** 2)), np.nan)
        s_score = np.where(sigma_eq > 0, -mean_eq / sigma_eq, np.nan)
        kappa = np.where(ok, -np.log(b_ar) * TRADING_DAYS, np.nan)
        hl = np.where(ok, -np.log(2.0) / np.log(b_ar), np.nan)

    idx = df.index[n - 1:]
    out = pd.DataFrame({"s_score": s_score, "ar_b": b_ar, "kappa": kappa,
                        "half_life_days": hl}, index=idx)
    return out.reindex(asset_returns.index)


# ------------------------------------------------------------------ builder
def build_features(prices: pd.DataFrame, macro: pd.Series, spec) -> dict[str, pd.Series]:
    """Build every feature in the frozen search space.

    `prices` is a wide adjusted-close panel; `macro` is the T10Y2Y level.
    Identical code path is used for real data (M5) and synthetic data (M2).
    """
    from config import (PAIRS, REGIME_PAIRS, RVOL_WINDOWS, VOL_SPILLOVER,
                        XCORR_LAGS, Z_WINDOWS, RESIDUAL_SPECS, STD_WINDOW)

    rets = {c: log_returns(prices[c]) for c in prices.columns}
    feats: dict[str, pd.Series] = {}

    for a, b in PAIRS:
        lr = log_ratio(prices[a], prices[b])
        for w in Z_WINDOWS:
            feats[f"z_{a}_{b}_{w}"] = zscore_causal(lr, w)

    for a, b in REGIME_PAIRS:
        lr = log_ratio(prices[a], prices[b])
        for w in Z_WINDOWS:
            feats[f"z_{a}_{b}_{w}"] = zscore_causal(lr, w)

    for asset, series_id in RESIDUAL_SPECS:
        dm = macro.reindex(prices.index).diff()
        al = avellaneda_lee_score(rets[asset], dm, AL_REG_WINDOW)
        feats[f"al_{asset}_{series_id}_{AL_REG_WINDOW}"] = al["s_score"]

    for src, _dst in VOL_SPILLOVER:
        for w in RVOL_WINDOWS:
            rv = realized_vol(rets[src], w)
            base = zscore_causal(rv, STD_WINDOW)
            for lag in XCORR_LAGS:
                feats[f"rvol_{src}_{w}_lag{lag}"] = base.shift(lag)

    return feats
