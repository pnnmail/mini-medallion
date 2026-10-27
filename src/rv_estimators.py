"""M1.3 -- realised variance estimators.

Close-to-close throws away the high and the low; with OHLC available that is
simply inefficient.  But range-based estimators cannot see the overnight gap at
all, and M0 established twice, on different data and different periods, that the
overnight gap carries essentially the whole return of these instruments.  So the
fourth estimator restores it explicitly.

The estimator is chosen by out-of-sample forecast accuracy, not by elegance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config_m1 import RV_FLOOR

_GK_C = 2.0 * np.log(2.0) - 1.0
_PK_C = 1.0 / (4.0 * np.log(2.0))


def _cols(d: pd.DataFrame):
    f = (d["adj_close"].astype(float) / d["close"].astype(float))
    o = d["open"].astype(float) * f
    h = d["high"].astype(float) * f
    l = d["low"].astype(float) * f
    c = d["adj_close"].astype(float)
    return o, h, l, c


def daily_variance(d: pd.DataFrame, estimator: str) -> pd.Series:
    """Daily variance estimate. All four are in the same units (per-day variance)."""
    o, h, l, c = _cols(d)
    cp = c.shift(1)
    if estimator == "cc":
        v = np.log(c / cp) ** 2
    elif estimator == "parkinson":
        v = _PK_C * np.log(h / l) ** 2
    elif estimator == "gk":
        v = 0.5 * np.log(h / l) ** 2 - _GK_C * np.log(c / o) ** 2
    elif estimator == "gk_overnight":
        # intraday Garman-Klass plus the squared overnight jump the range cannot see
        gk = 0.5 * np.log(h / l) ** 2 - _GK_C * np.log(c / o) ** 2
        v = gk + np.log(o / cp) ** 2
    else:
        raise ValueError(f"unknown estimator {estimator}")
    return v.clip(lower=RV_FLOOR).replace([np.inf, -np.inf], np.nan)


def overnight_intraday_variance(d: pd.DataFrame):
    """The two components separately, for candidate family 4."""
    o, h, l, c = _cols(d)
    cp = c.shift(1)
    on = (np.log(o / cp) ** 2).clip(lower=RV_FLOOR)
    idv = (np.log(c / o) ** 2).clip(lower=RV_FLOOR)
    return on.replace([np.inf, -np.inf], np.nan), idv.replace([np.inf, -np.inf], np.nan)


def downside_semivariance(d: pd.DataFrame, window: int) -> pd.Series:
    """Trailing mean of squared NEGATIVE close-to-close returns."""
    c = d["adj_close"].astype(float)
    r = np.log(c).diff()
    neg = (r.where(r < 0, 0.0)) ** 2
    return neg.rolling(window, min_periods=window).mean().clip(lower=RV_FLOOR)


def trailing_mean_variance(v: pd.Series, window: int) -> pd.Series:
    return v.rolling(window, min_periods=window).mean().clip(lower=RV_FLOOR)


def forward_mean_variance(v: pd.Series, horizon: int) -> pd.Series:
    """Mean daily variance over t+1 .. t+H, stamped at t. The forecast TARGET."""
    return v.rolling(horizon, min_periods=horizon).mean().shift(-horizon)


def annualise_vol(var_daily) -> np.ndarray:
    return np.sqrt(252.0 * np.asarray(var_daily, float))


# ------------------------------------------------------------------ losses
def qlike(actual_var, pred_var) -> float:
    """QLIKE loss. Robust to a noisy volatility proxy (Patton 2011); lower is better."""
    a = np.asarray(actual_var, float)
    p = np.asarray(pred_var, float)
    m = np.isfinite(a) & np.isfinite(p) & (a > 0) & (p > 0)
    if m.sum() < 30:
        return np.nan
    return float(np.mean(np.log(p[m]) + a[m] / p[m]))


def mse(actual_var, pred_var) -> float:
    a = np.asarray(actual_var, float)
    p = np.asarray(pred_var, float)
    m = np.isfinite(a) & np.isfinite(p)
    return float(np.mean((a[m] - p[m]) ** 2)) if m.sum() >= 30 else np.nan


def r2_oos(actual, pred, benchmark) -> float:
    """Campbell-Thompson out-of-sample R^2 against a stated benchmark forecast."""
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    b = np.asarray(benchmark, float)
    m = np.isfinite(a) & np.isfinite(p) & np.isfinite(b)
    if m.sum() < 30:
        return np.nan
    sse = np.sum((a[m] - p[m]) ** 2)
    sst = np.sum((a[m] - b[m]) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else np.nan
