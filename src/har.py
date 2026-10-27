"""N2 -- HAR-RV baseline (Corsi 2009) and the lower-bound benchmarks.

The baseline is declared here, before any search, and does not change afterwards.

Every forecast is produced by an EXPANDING-window OLS that respects the
publication lag: to forecast at t on horizon H, only observations whose target
was already realised by t (that is, s <= t - H) may enter the fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config_m1 import EWMA_LAMBDA, HAR_LAGS, HAR_MIN_TRAIN_OBS, MA_WINDOW, RV_FLOOR


def har_regressors(rv_daily: pd.Series) -> pd.DataFrame:
    """log daily / weekly / monthly trailing mean variance, all causal."""
    d, w, m = HAR_LAGS
    v = rv_daily.clip(lower=RV_FLOOR)
    out = pd.DataFrame(index=v.index)
    out["har_d"] = np.log(v.rolling(d, min_periods=d).mean())
    out["har_w"] = np.log(v.rolling(w, min_periods=w).mean())
    out["har_m"] = np.log(v.rolling(m, min_periods=m).mean())
    return out


def har_target(rv_daily: pd.Series, horizon: int) -> pd.Series:
    """log of the mean daily variance over t+1 .. t+H, stamped at t."""
    v = rv_daily.clip(lower=RV_FLOOR)
    fwd = v.rolling(horizon, min_periods=horizon).mean().shift(-horizon)
    return np.log(fwd)


def expanding_ols_forecast(X: np.ndarray, y: np.ndarray, horizon: int,
                           min_obs: int = HAR_MIN_TRAIN_OBS):
    """Expanding-window OLS forecast of y from X, honouring the H-day lag.

    Returns (pred_log, resid_var, n_used) aligned with the input rows.
    Implemented by accumulating X'X / X'y so each step is O(k^2) and the whole
    path is exact -- no refitting shortcuts, no look-ahead.
    """
    n, k = X.shape
    Xa = np.column_stack([np.ones(n), X])
    kk = k + 1
    valid = np.isfinite(y) & np.isfinite(Xa).all(1)
    Xv = np.where(valid[:, None], Xa, 0.0)
    yv = np.where(valid, y, 0.0)

    cum_XX = np.cumsum(np.einsum("ni,nj->nij", Xv, Xv), axis=0)
    cum_Xy = np.cumsum(Xv * yv[:, None], axis=0)
    cum_yy = np.cumsum(yv * yv)
    cum_n = np.cumsum(valid.astype(np.int64))

    pred = np.full(n, np.nan)
    rvar = np.full(n, np.nan)
    nused = np.zeros(n, dtype=np.int64)

    t = np.arange(n)
    j = t - horizon
    ok = (j >= 0)
    ok[ok] &= cum_n[j[ok]] >= min_obs
    ok &= np.isfinite(Xa).all(1)
    if not ok.any():
        return pred, rvar, nused

    jj = j[ok]
    A = cum_XX[jj] + np.eye(kk) * 1e-10          # ridge-free; tiny jitter for safety
    b = cum_Xy[jj]
    try:
        beta = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return pred, rvar, nused
    pred[ok] = np.einsum("ni,ni->n", Xa[ok], beta)
    sse = cum_yy[jj] - np.einsum("ni,ni->n", beta, b)
    dof = np.maximum(cum_n[jj] - kk, 1)
    rvar[ok] = np.maximum(sse, 0.0) / dof
    nused[ok] = cum_n[jj]
    return pred, rvar, nused


def log_forecast_to_variance(pred_log, resid_var):
    """E[V] = exp(mu + s^2/2) for a log-normal forecast."""
    p = np.asarray(pred_log, float)
    s = np.asarray(resid_var, float)
    s = np.where(np.isfinite(s), s, 0.0)
    return np.exp(p + 0.5 * s)


def har_forecast(rv_daily: pd.Series, horizon: int, extra: pd.DataFrame | None = None,
                 min_obs: int = HAR_MIN_TRAIN_OBS, rv_target: pd.Series | None = None):
    """HAR-RV forecast, optionally augmented with extra regressors.

    `rv_daily` supplies the REGRESSORS; `rv_target` supplies the target and
    defaults to the same series.  Keeping them separable is what makes the
    estimator comparison fair: every estimator is judged against one common
    target, so differences in level are absorbed by the intercept and only
    forecasting content is compared.

    Returns a DataFrame with pred_log, pred_var, resid_var, n_used and the target.
    """
    Xdf = har_regressors(rv_daily)
    if extra is not None and len(extra.columns):
        Xdf = Xdf.join(extra, how="left")
    y = har_target(rv_daily if rv_target is None else rv_target, horizon)
    y = y.reindex(Xdf.index)
    pred_log, rvar, nused = expanding_ols_forecast(
        Xdf.to_numpy(float), y.to_numpy(float), horizon, min_obs)
    return pd.DataFrame({
        "pred_log": pred_log,
        "pred_var": log_forecast_to_variance(pred_log, rvar),
        "resid_var": rvar, "n_used": nused,
        "target_log": y.to_numpy(float),
        "target_var": np.exp(y.to_numpy(float)),
    }, index=Xdf.index)


# ------------------------------------------------------------------ benchmarks
def naive_forecast(rv_daily: pd.Series) -> pd.Series:
    """Random walk in variance: tomorrow looks like today."""
    return rv_daily.clip(lower=RV_FLOOR)


def ma_forecast(rv_daily: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    return rv_daily.clip(lower=RV_FLOOR).rolling(window, min_periods=window).mean()


def ewma_forecast(rv_daily: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """RiskMetrics EWMA, causal by construction."""
    v = rv_daily.clip(lower=RV_FLOOR)
    return v.ewm(alpha=1.0 - lam, adjust=False, min_periods=MA_WINDOW).mean()
