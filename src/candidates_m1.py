"""N3 -- candidate regressors, built once per target instrument.

Every candidate is tested ONLY as an increment to the HAR baseline. None of them
is ever evaluated as a standalone predictor -- that is the whole lesson of M0,
where a candidate that cleared the null threshold and would have cleared an
out-of-sample gate too turned out to be the target's own volatility clustering
reached through a correlated proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config_m1 import (RV_LOOKBACK, SEMIVAR_WINDOW, VIX_Z_WINDOWS, VOL_SOURCES)
from features import zscore_causal
from panel_m1 import calendar_for, instrument_ohlc, macro_t10y2y, vix_close
from rv_estimators import (daily_variance, downside_semivariance,
                           overnight_intraday_variance, trailing_mean_variance)


def _ann_vol(var_daily: pd.Series, window: int) -> pd.Series:
    return np.sqrt(252.0 * trailing_mean_variance(var_daily, window))


def calendar_forward_flags(flag: pd.Series, horizon: int) -> pd.Series:
    """Share of sessions t+1 .. t+H carrying the flag, stamped at t.

    Same alignment as any forward target, but this one is NOT look-ahead: the
    exchange calendar for the next H sessions is known today. `calendar_for`
    takes only a DatetimeIndex and never sees a price, which is the proof.
    """
    return flag.rolling(horizon, min_periods=horizon).mean().shift(-horizon)


def build_candidates(target: str, estimator: str, horizon: int):
    """Return (DataFrame of candidates aligned to the target's index, family map).

    The calendar family is FORWARD-looking by construction: at time t the exchange
    calendar for t+1..t+H is already known, so 'an OPEX falls in the next H days'
    uses no unavailable information. It is derived from the date index alone,
    which is the proof -- `calendar_for` never sees a price.
    """
    d = instrument_ohlc(target)
    idx = d.index
    out = pd.DataFrame(index=idx)
    fam: dict[str, str] = {}

    def add(name, series, family):
        out[name] = pd.Series(series).reindex(idx)
        fam[name] = family

    # ---- 1. VIX and the variance risk premium -------------------------
    vix = vix_close().reindex(idx).ffill(limit=5)
    spy = instrument_ohlc("SPY")
    spy_rv = _ann_vol(daily_variance(spy, estimator), RV_LOOKBACK).reindex(idx).ffill(limit=5)
    add("log_vix", np.log(vix), "vix")
    add("d_log_vix", np.log(vix).diff(), "vix")
    ratio = np.log((vix / 100.0) / spy_rv.replace(0.0, np.nan))
    add("log_vix_rv_ratio", ratio, "vix")
    for w in VIX_Z_WINDOWS:
        add(f"z_vix_rv_{w}", zscore_causal(ratio, w), "vix")

    # ---- 2. cross-asset volatility ------------------------------------
    for src in VOL_SOURCES:
        if src == target:
            continue
        sv = daily_variance(instrument_ohlc(src), estimator)
        add(f"log_rv{RV_LOOKBACK}_{src}",
            np.log(trailing_mean_variance(sv, RV_LOOKBACK)).reindex(idx).ffill(limit=5),
            "cross_asset_vol")

    # ---- 3. asymmetry / leverage effect --------------------------------
    r = np.log(d["adj_close"].astype(float)).diff()
    add("neg_ret_ind", (r < 0).astype(float).where(r.notna()), "asymmetry")
    add("signed_ret", r, "asymmetry")
    add("log_semivar20", np.log(downside_semivariance(d, SEMIVAR_WINDOW)), "asymmetry")

    # ---- 4. overnight / intraday decomposition -------------------------
    on, idv = overnight_intraday_variance(d)
    add("log_rv_overnight20", np.log(trailing_mean_variance(on, RV_LOOKBACK)),
        "overnight_intraday")
    add("log_rv_intraday20", np.log(trailing_mean_variance(idv, RV_LOOKBACK)),
        "overnight_intraday")

    # ---- 5. calendar (deterministic, forward window) -------------------
    cal = calendar_for(idx)
    for c in cal.columns:
        add(f"fwd_{c}_h{horizon}", calendar_forward_flags(cal[c], horizon), "calendar")

    # ---- 6. term structure ---------------------------------------------
    t10 = macro_t10y2y().reindex(idx).ffill(limit=5)
    add("t10y2y_level", t10, "macro")
    add("d_t10y2y", t10.diff(), "macro")

    return out, fam
