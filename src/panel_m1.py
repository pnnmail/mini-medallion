"""M1 panel: per-instrument OHLC on its own full history, plus VIX / macro / calendar.

Unlike the M0 panel there is no common-calendar intersection here: SPY is
allowed its 1993 start (declared in config_m1 before any result was seen), and
every instrument is evaluated on the history it actually has.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config_m1 import M1_SPLITS, SPY_EXTENDED_TRAIN_START
from data_loader import load_prices
from data_m1 import event_calendar, load_m1

_CACHE = {}


def instrument_ohlc(ticker: str) -> pd.DataFrame:
    """SPY comes from the M1 extension (verified against the M0 freeze in N0)."""
    if ticker in _CACHE:
        return _CACHE[ticker]
    if ticker == "SPY":
        d = load_m1("SPY_EXT").copy()
    else:
        d = load_prices()[ticker].copy()
    d = d[[c for c in ["open", "high", "low", "close", "adj_close", "volume"]
           if c in d.columns]].dropna(subset=["adj_close", "close"])
    _CACHE[ticker] = d
    return d


def vix_close() -> pd.Series:
    return load_m1("VIX")["close"].astype(float)


def macro_t10y2y() -> pd.Series:
    return load_m1("T10Y2Y_EXT")["value"].astype(float)


def calendar_for(index) -> pd.DataFrame:
    return event_calendar(index)


def split_bounds(ticker: str, split: str):
    """SPY's TRAIN is extended back to 1993 -- decided in config, before any result."""
    a, b = M1_SPLITS[split]
    if ticker == "SPY" and split == "TRAIN":
        a = SPY_EXTENDED_TRAIN_START
    return a, b


def split_mask_m1(index, ticker: str, split: str) -> np.ndarray:
    a, b = split_bounds(ticker, split)
    idx = pd.DatetimeIndex(index)
    return np.asarray((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)))
