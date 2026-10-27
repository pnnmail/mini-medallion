"""Shared construction of the research panel used by M1/M2/M3/M5.

One definition, imported everywhere, so that the null harness and the real run
cannot drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import load_macro, load_prices

# tickers actually consumed by the frozen M1 search space
FEATURE_TICKERS = ["GLD", "GDX", "XLE", "XOP", "XLF", "SPY", "IWM", "TLT"]
# additional tickers used only by M3 calibration / M4 anchors / M7
AUX_TICKERS = ["QQQ", "SHY", "SCHD"]


def research_panel(tickers=None, macro_id="T10Y2Y"):
    """Adjusted-close panel + macro level, aligned on the common trading calendar.

    Rows are kept only where every requested ticker has a price: no forward
    filling of prices, because a filled price fabricates a zero return.
    The macro level is forward-filled (max 5 days) across market holidays --
    that is causal, it only ever reuses the last published value.
    """
    tickers = list(tickers or FEATURE_TICKERS)
    px = load_prices()
    df = pd.DataFrame({t: px[t]["adj_close"].astype(float) for t in tickers}).sort_index()
    df = df.dropna(how="any")
    macro = load_macro()[macro_id].reindex(df.index).ffill(limit=5)
    return df, macro


def split_mask(index: pd.DatetimeIndex, split, splits_cfg=None):
    from config import SPLITS
    splits_cfg = splits_cfg or SPLITS
    a, b = splits_cfg[split]
    return (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
