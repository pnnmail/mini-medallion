"""M1 -- target calculators. All targets look strictly FORWARD from t+1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import TRADING_DAYS


def forward_return(returns: pd.Series, horizon: int) -> pd.Series:
    """Sum of log returns over t+1 .. t+H, stamped at t."""
    fwd = returns.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    return fwd


def forward_direction(returns: pd.Series, horizon: int) -> pd.Series:
    fr = forward_return(returns, horizon)
    return np.sign(fr).replace(0.0, np.nan)


def forward_realized_vol(returns: pd.Series, horizon: int) -> pd.Series:
    """Annualised realised volatility over t+1 .. t+H, stamped at t."""
    v = np.sqrt(TRADING_DAYS * (returns ** 2).rolling(horizon, min_periods=horizon).mean())
    return v.shift(-horizon)


def spread_returns(prices: pd.DataFrame, a: str, b: str) -> pd.Series:
    """One-day return of the dollar-neutral pair: long A, short B."""
    from features import log_returns
    return log_returns(prices[a]) - log_returns(prices[b])


def build_targets(prices: pd.DataFrame, spec=None) -> dict[str, pd.Series]:
    """Every target in the frozen search space, plus the one-day traded series."""
    from config import PAIRS, RETURN_HORIZONS, RESIDUAL_SPECS, VOL_HORIZONS, VOL_SPILLOVER
    from features import log_returns

    tg: dict[str, pd.Series] = {}
    traded: dict[str, pd.Series] = {}

    for a, b in PAIRS:
        key = f"spread_{a}_{b}"
        r = spread_returns(prices, a, b)
        traded[key] = r
        for h in RETURN_HORIZONS:
            tg[f"fwdret_{key}_{h}"] = forward_return(r, h)
            tg[f"fwddir_{key}_{h}"] = forward_direction(r, h)

    for asset, _sid in RESIDUAL_SPECS:
        key = f"asset_{asset}"
        r = log_returns(prices[asset])
        traded[key] = r
        for h in RETURN_HORIZONS:
            tg[f"fwdret_{key}_{h}"] = forward_return(r, h)
            tg[f"fwddir_{key}_{h}"] = forward_direction(r, h)

    for _src, dst in VOL_SPILLOVER:
        r = log_returns(prices[dst])
        for h in VOL_HORIZONS:
            tg[f"fwdrvol_{dst}_{h}"] = forward_realized_vol(r, h)

    return tg, traded
