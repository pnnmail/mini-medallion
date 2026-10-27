"""The one comparison the whole of M2 is about, reproduced exactly as M1 built it.

HAR versus HAR+log(VIX), cap 1.0, sigma* 10%, band +-20%, sizing horizon 10d.
Nothing here is re-tuned: M2 questions this decision layer, and a layer that
moves while being questioned answers nothing.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from candidates_m1 import build_candidates
from config_m1 import (REBALANCE_BAND, RESULTS_DIR, SIZING_HORIZON, TRADING_DAYS,
                       VOL_TARGET_ANNUAL)
from har import har_forecast
from panel_m1 import instrument_ohlc, split_mask_m1
from rv_estimators import daily_variance
from voltarget import band_rebalance, strategy_pnl, target_weights

TARGET_EST = "cc"
CAP = 1.0
FAST = "HAR+log_vix"      # the more accurate forecast
SLOW = "HAR"              # the less accurate one that produced the better decision
_CACHE = {}


def chosen_estimator():
    with open(os.path.join(RESULTS_DIR, "m1_chosen_estimator.json")) as fh:
        return json.load(fh)["estimator"]


def build_pair(instrument: str, sigma_override=None):
    """Return a frame with both forecasts, both weight paths and both P&L series."""
    key = (instrument, id(sigma_override))
    if sigma_override is None and instrument in _CACHE:
        return _CACHE[instrument]

    est = chosen_estimator()
    d = instrument_ohlc(instrument)
    rv_reg = daily_variance(d, est)
    rv_tgt = daily_variance(d, TARGET_EST)
    C, _ = build_candidates(instrument, est, SIZING_HORIZON)

    f_slow = har_forecast(rv_reg, SIZING_HORIZON, rv_target=rv_tgt)
    f_fast = har_forecast(rv_reg, SIZING_HORIZON, extra=C[["log_vix"]],
                          rv_target=rv_tgt)

    r = np.log(d["adj_close"].astype(float)).diff().to_numpy(float)
    fwd = np.concatenate([r[1:], [np.nan]])

    out = pd.DataFrame(index=d.index)
    out["fwd_ret"] = fwd
    out["ret"] = r
    for tag, f in [(SLOW, f_slow), (FAST, f_fast)]:
        sig = np.sqrt(TRADING_DAYS * f["pred_var"].to_numpy(float))
        out[f"sigma_{tag}"] = sig
        out[f"target_var"] = f["target_var"].to_numpy(float)
        out[f"pred_var_{tag}"] = f["pred_var"].to_numpy(float)
    out["log_vix"] = C["log_vix"].reindex(d.index).to_numpy(float)
    if sigma_override is not None:
        for tag, s in sigma_override.items():
            out[f"sigma_{tag}"] = s
    if sigma_override is None:
        _CACHE[instrument] = out
    return out


def dev_mask(instrument: str, frame: pd.DataFrame, models=(SLOW, FAST)):
    """M1's development sample: TRAIN + VALIDATION, rows usable by BOTH models."""
    idx = frame.index
    m = (split_mask_m1(idx, instrument, "TRAIN") |
         split_mask_m1(idx, instrument, "VALIDATION"))
    m &= np.isfinite(frame["fwd_ret"].to_numpy(float))
    for t in models:
        m &= np.isfinite(frame[f"sigma_{t}"].to_numpy(float))
    return m


def weights_pnl(frame: pd.DataFrame, model: str, mask, cap: float = CAP):
    sig = frame[f"sigma_{model}"].to_numpy(float)
    wt = np.where(mask, target_weights(sig, cap, VOL_TARGET_ANNUAL), np.nan)
    w = np.where(mask, band_rebalance(wt, REBALANCE_BAND), np.nan)
    pnl, dw, _ = strategy_pnl(w, frame["fwd_ret"].to_numpy(float), 0.0)
    return w, pnl, dw


def paired_series(instrument: str, cap: float = CAP):
    """Aligned daily P&L of both strategies on the development sample."""
    f = build_pair(instrument)
    m = dev_mask(instrument, f)
    w_s, p_s, dw_s = weights_pnl(f, SLOW, m, cap)
    w_f, p_f, dw_f = weights_pnl(f, FAST, m, cap)
    sel = np.asarray(m)
    return {"frame": f, "mask": sel,
            "w_slow": w_s[sel], "w_fast": w_f[sel],
            "pnl_slow": p_s[sel], "pnl_fast": p_f[sel],
            "dw_slow": dw_s[sel], "dw_fast": dw_f[sel],
            "fwd_ret": f["fwd_ret"].to_numpy(float)[sel],
            "index": f.index[sel]}
