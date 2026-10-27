"""P0 -- the model universe, built once and shared by every M2 stage.

For each instrument: the HAR baseline plus HAR + X for EVERY M1 candidate.
Not the 30 that survived M1 -- restricting to survivors would import a selection
made under a different criterion and turn the selector race into a re-test of a
choice that was already made.

All models for one instrument are evaluated on ONE common mask (rows where every
model in the universe has a finite forecast). Models ranked on different samples
cannot be ranked against each other.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from candidates_m1 import build_candidates
from config_m2 import (BAND, CAP, HORIZON, INSTRUMENTS, RESULTS_DIR, SIGMA_TARGET,
                       SPLITS_M2, TRADING_DAYS)
from har import har_forecast
from panel_m1 import instrument_ohlc, split_mask_m1
from rv_estimators import daily_variance
from voltarget import band_rebalance, strategy_pnl, target_weights

TARGET_EST = "cc"
_CACHE = {}


def chosen_estimator():
    with open(os.path.join(RESULTS_DIR, "m1_chosen_estimator.json")) as fh:
        return json.load(fh)["estimator"]


def build_universe(instrument: str):
    """Return (index, fwd_ret, dict model_name -> forecast frame, families)."""
    if instrument in _CACHE:
        return _CACHE[instrument]
    est = chosen_estimator()
    d = instrument_ohlc(instrument)
    rv_reg = daily_variance(d, est)
    rv_tgt = daily_variance(d, TARGET_EST)
    C, fam = build_candidates(instrument, est, HORIZON)

    models = {"HAR": har_forecast(rv_reg, HORIZON, rv_target=rv_tgt)}
    families = {"HAR": "baseline"}
    for c in C.columns:
        models[f"HAR+{c}"] = har_forecast(rv_reg, HORIZON, extra=C[[c]],
                                          rv_target=rv_tgt)
        families[f"HAR+{c}"] = fam[c]

    r = np.log(d["adj_close"].astype(float)).diff().to_numpy(float)
    fwd = np.concatenate([r[1:], [np.nan]])

    out = (d.index, fwd, models, families, d)
    _CACHE[instrument] = out
    return out


def common_mask(instrument: str, split: str):
    """Rows usable by EVERY model in the universe, inside one split."""
    idx, fwd, models, _, d = build_universe(instrument)
    m = split_mask_m1(idx, instrument, split) & np.isfinite(fwd)
    for f in models.values():
        sig = np.sqrt(TRADING_DAYS * f["pred_var"].to_numpy(float))
        m = m & np.isfinite(sig)
    return m


def weights_and_pnl(instrument: str, model: str, mask, cap: float = CAP):
    idx, fwd, models, _, _ = build_universe(instrument)
    sig = np.sqrt(TRADING_DAYS * models[model]["pred_var"].to_numpy(float))
    wt = np.where(mask, target_weights(sig, cap, SIGMA_TARGET), np.nan)
    w = np.where(mask, band_rebalance(wt, BAND), np.nan)
    pnl, dw, _ = strategy_pnl(w, fwd, 0.0)
    return sig, w, pnl, dw


def sigma_of(instrument: str, model: str):
    _, _, models, _, _ = build_universe(instrument)
    return np.sqrt(TRADING_DAYS * models[model]["pred_var"].to_numpy(float))
