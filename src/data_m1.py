"""M1 data extension: SPY back to 1993, VIX, and a deterministic event calendar.

Frozen separately from the M0 snapshot so that nothing in M0 is disturbed.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time

import numpy as np
import pandas as pd

from config_m1 import (M1_DOWNLOAD_END, M1_DOWNLOAD_START, M1_SNAPSHOT_DIR,
                       VIX_TICKER)

M1_TICKERS = {"SPY_EXT": "SPY", "VIX": VIX_TICKER}
META = os.path.join(M1_SNAPSHOT_DIR, "_metadata.json")


def _f(key):
    return os.path.join(M1_SNAPSHOT_DIR, f"{key}.parquet")


def download_m1_snapshot(force: bool = False) -> dict:
    import yfinance as yf

    os.makedirs(M1_SNAPSHOT_DIR, exist_ok=True)
    meta = json.load(open(META)) if os.path.exists(META) else {"tickers": {}}

    for key, tk in M1_TICKERS.items():
        p = _f(key)
        if os.path.exists(p) and not force:
            continue
        df = None
        for attempt in range(4):
            try:
                df = yf.download(tk, start=M1_DOWNLOAD_START, end=M1_DOWNLOAD_END,
                                 auto_adjust=False, actions=False, progress=False,
                                 threads=False)
                if df is not None and len(df):
                    break
            except Exception as exc:                       # noqa: BLE001
                print(f"  {tk} attempt {attempt}: {exc}")
            time.sleep(2 + 3 * attempt)
        if df is None or not len(df):
            raise RuntimeError(f"M1 download failed for {tk}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Adj Close": "adj_close", "Open": "open", "High": "high",
                                "Low": "low", "Close": "close", "Volume": "volume"})
        keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"]
                if c in df.columns]
        df = df[keep].copy()
        df.index.name = "date"
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df.to_parquet(p)
        meta["tickers"][key] = {
            "key": key, "yf_ticker": tk, "source": "yfinance",
            "download_timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "first_date": str(df.index.min().date()), "last_date": str(df.index.max().date()),
            "row_count": int(len(df)), "columns": keep,
        }
        print(f"  {key:8s} ({tk:6s}) {len(df):6d} rows  "
              f"{df.index.min().date()} .. {df.index.max().date()}")

    # T10Y2Y already reaches back to 1976 in the M0 snapshot start of 2000; extend it
    p = _f("T10Y2Y_EXT")
    if not os.path.exists(p) or force:
        from pandas_datareader import data as pdr
        ser = None
        for attempt in range(4):
            try:
                ser = pdr.DataReader("T10Y2Y", "fred", M1_DOWNLOAD_START, M1_DOWNLOAD_END)
                if ser is not None and len(ser):
                    break
            except Exception as exc:                       # noqa: BLE001
                print(f"  T10Y2Y attempt {attempt}: {exc}")
            time.sleep(2 + 3 * attempt)
        if ser is None or not len(ser):
            raise RuntimeError("FRED T10Y2Y extended download failed")
        ser.index.name = "date"
        ser.columns = ["value"]
        ser.to_parquet(p)
        meta["tickers"]["T10Y2Y_EXT"] = {
            "key": "T10Y2Y_EXT", "source": "FRED",
            "download_timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "first_date": str(ser.index.min().date()),
            "last_date": str(ser.index.max().date()), "row_count": int(len(ser))}
        print(f"  T10Y2Y_EXT        {len(ser):6d} rows  "
              f"{ser.index.min().date()} .. {ser.index.max().date()}")

    meta["snapshot_date"] = _dt.date.today().isoformat()
    json.dump(meta, open(META, "w"), indent=2)
    return meta


def load_m1(key: str) -> pd.DataFrame:
    p = _f(key)
    if not os.path.exists(p):
        raise FileNotFoundError(f"M1 snapshot missing: {key}")
    return pd.read_parquet(p)


# ------------------------------------------------------------------ calendar
def event_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic, verifiable date flags only.

    OPEX, month-end, quarter-end and the BLS first-Friday convention are rules
    that can be recomputed by anyone from the index alone.  FOMC and CPI release
    dates are NOT included -- see `CALENDAR_OMISSIONS` for why.
    """
    idx = pd.DatetimeIndex(index)
    df = pd.DataFrame(index=idx)
    dow = idx.dayofweek
    dom = idx.day

    # third Friday of the month
    is_friday = dow == 4
    friday_rank = pd.Series(is_friday, index=idx).groupby(
        [idx.year, idx.month]).cumsum()
    df["is_opex"] = (is_friday & (friday_rank.to_numpy() == 3)).astype(float)

    # first Friday of the month -- BLS Employment Situation convention
    df["is_nfp_friday"] = (is_friday & (friday_rank.to_numpy() == 1)).astype(float)

    # last three trading days of the month / quarter
    s = pd.Series(1, index=idx)
    df["is_month_end3"] = (s.groupby([idx.year, idx.month]).cumcount(
        ascending=False) < 3).astype(float).to_numpy()
    q = idx.to_period("Q")
    df["is_quarter_end3"] = (s.groupby(q).cumcount(ascending=False) < 3
                             ).astype(float).to_numpy()
    return df


CALENDAR_OMISSIONS = {
    "FOMC": "not included -- no verifiable, freezable list of the ~160 scheduled "
            "meeting dates over 1993-2026 was obtainable inside the frozen-data "
            "constraint, and there is no deterministic rule to derive them. "
            "Writing dates from memory would inject fabricated data into a "
            "milestone whose entire point is not trusting unverified inputs.",
    "CPI": "not included -- same reason. BLS release dates vary within the month "
           "and follow no rule recomputable from the index.",
    "consequence": "candidate family 5 (calendar) is therefore tested on OPEX, "
                   "month-end, quarter-end and the first-Friday NFP convention "
                   "only. This is a REDUCED version of the specified family and is "
                   "reported as such, not as the full test.",
}
