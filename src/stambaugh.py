"""N1 -- predictive regression with HAC errors and the Stambaugh bias correction.

    r_{t+1} = a + b * w_t + u_{t+1}

The regressor is a volatility-forecast-driven weight: it is highly persistent,
and its innovations are correlated with return innovations by construction,
because sigma_hat responds to past returns. That is the textbook Stambaugh
setting, so the raw OLS slope is biased and the corrected one is reported
beside it rather than instead of it.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def _ols(y, X):
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    return beta, y - X @ beta, XtX_inv


def hac_se_slope(y, x, lag):
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta, e, XtX_inv = _ols(y, X)
    u = X * e[:, None]
    S = u.T @ u
    L = max(int(lag), 0)
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = u[l:].T @ u[:-l]
        S = S + w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return float(beta[1]), float(np.sqrt(max(V[1, 1], 0.0))), e, float(beta[0])


def predictive_regression(fwd_ret, weight, hac_lag: int):
    """Slope of next-day return on today's weight, with the Stambaugh correction.

    Returns raw slope, HAC t, the AR(1) persistence of the regressor, the
    estimated bias and the corrected slope.
    """
    y = np.asarray(fwd_ret, float)
    x = np.asarray(weight, float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = len(y)
    if n < 200 or np.std(x) == 0:
        return {}

    b, se, u, a = hac_se_slope(y, x, hac_lag)
    t = b / se if se > 0 else np.nan

    # AR(1) of the regressor
    x1, x0 = x[1:], x[:-1]
    X = np.column_stack([np.ones(len(x0)), x0])
    coef, v, _ = _ols(x1, X)
    phi = float(coef[1])

    # Stambaugh: E[b_hat] - b = gamma * E[phi_hat - phi], E[phi_hat - phi] ~ -(1+3phi)/T
    # so b = b_hat + gamma * (1 + 3 phi) / T
    u_al = u[1:]                      # align the two innovation series
    v_al = v[: len(u_al)]
    k = min(len(u_al), len(v_al))
    u_al, v_al = u_al[:k], v_al[:k]
    var_v = float(np.var(v_al, ddof=1))
    cov_uv = float(np.cov(u_al, v_al, ddof=1)[0, 1]) if var_v > 0 else np.nan
    gamma = cov_uv / var_v if var_v > 0 else np.nan
    bias = -gamma * (1.0 + 3.0 * phi) / n if np.isfinite(gamma) else np.nan
    b_adj = b - bias if np.isfinite(bias) else np.nan

    return {
        "n": int(n), "b_raw": b, "se_hac": se, "t_hac": float(t),
        "p_hac": float(2.0 * (1.0 - sps.norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
        "phi_ar1_regressor": phi,
        "gamma_cov_uv_over_var_v": float(gamma) if np.isfinite(gamma) else np.nan,
        "stambaugh_bias": float(bias) if np.isfinite(bias) else np.nan,
        "b_stambaugh_adjusted": float(b_adj) if np.isfinite(b_adj) else np.nan,
        "bias_as_share_of_raw": float(bias / b) if (np.isfinite(bias) and b != 0)
        else np.nan,
        "hac_lag": int(hac_lag),
    }


def effective_obs_step_weights(weight, mask=None) -> dict:
    """With band rebalancing the regressor is a step function.

    Between rebalances w_t does not move, so those days carry no independent
    information about the regressor. The count of distinct steps is the honest
    effective sample size for anything driven by variation in w.
    """
    w = np.asarray(weight, float)
    if mask is not None:
        w = w[np.asarray(mask)]
    w = w[np.isfinite(w)]
    if len(w) < 2:
        return {"n_days": int(len(w)), "n_steps": 0, "median_hold_days": np.nan}
    ch = np.abs(np.diff(w)) > 1e-12
    n_steps = int(ch.sum()) + 1
    pos = np.flatnonzero(ch)
    holds = np.diff(np.concatenate(([0], pos, [len(w) - 1])))
    return {"n_days": int(len(w)), "n_steps": n_steps,
            "median_hold_days": float(np.median(holds[holds > 0])) if len(holds) else np.nan,
            "ratio_days_per_step": float(len(w) / max(n_steps, 1))}
