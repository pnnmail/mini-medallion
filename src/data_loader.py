"""M0 -- download once, freeze forever.

yfinance silently revises adjusted closes retroactively; reproducibility beats
freshness, so the snapshot is written once and then treated as immutable.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time

import numpy as np
import pandas as pd

from config import (ALL_TICKERS, DOWNLOAD_END, DOWNLOAD_START, MACRO_SERIES,
                    RAW_SNAPSHOT_DIR, SNAPSHOT_DATE)

META_PATH = os.path.join(RAW_SNAPSHOT_DIR, "_metadata.json")


def _ticker_file(t: str) -> str:
    return os.path.join(RAW_SNAPSHOT_DIR, f"{t.replace('=', '_')}.parquet")


def _macro_file(s: str) -> str:
    return os.path.join(RAW_SNAPSHOT_DIR, f"macro_{s}.parquet")


# ------------------------------------------------------------------ download
def download_snapshot(force: bool = False) -> dict:
    import yfinance as yf

    os.makedirs(RAW_SNAPSHOT_DIR, exist_ok=True)
    meta = {"snapshot_date": SNAPSHOT_DATE, "source": "yfinance", "tickers": {},
            "macro": {}, "download_timestamp": _dt.datetime.now().isoformat(timespec="seconds")}
    if os.path.exists(META_PATH) and not force:
        with open(META_PATH) as fh:
            meta = json.load(fh)

    for t in ALL_TICKERS:
        path = _ticker_file(t)
        if os.path.exists(path) and not force:
            continue
        df = None
        for attempt in range(4):
            try:
                df = yf.download(t, start=DOWNLOAD_START, end=DOWNLOAD_END,
                                 auto_adjust=False, actions=True, progress=False,
                                 threads=False)
                if df is not None and len(df):
                    break
            except Exception as exc:                       # noqa: BLE001
                print(f"  {t} attempt {attempt}: {exc}")
            time.sleep(2 + 3 * attempt)
        if df is None or not len(df):
            raise RuntimeError(f"download failed for {t}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Adj Close": "adj_close", "Open": "open", "High": "high",
                                "Low": "low", "Close": "close", "Volume": "volume",
                                "Dividends": "dividends", "Stock Splits": "splits"})
        keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume",
                            "dividends", "splits"] if c in df.columns]
        df = df[keep].copy()
        df.index.name = "date"
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df.to_parquet(path)
        meta["tickers"][t] = {
            "ticker": t, "file": os.path.basename(path), "source": "yfinance",
            "download_timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "first_date": str(df.index.min().date()), "last_date": str(df.index.max().date()),
            "row_count": int(len(df)), "columns": keep,
        }
        print(f"  {t:6s} {len(df):5d} rows  {df.index.min().date()} .. {df.index.max().date()}")

    for s in MACRO_SERIES:
        path = _macro_file(s)
        if os.path.exists(path) and not force:
            continue
        from pandas_datareader import data as pdr
        ser = None
        for attempt in range(4):
            try:
                ser = pdr.DataReader(s, "fred", DOWNLOAD_START, DOWNLOAD_END)
                if ser is not None and len(ser):
                    break
            except Exception as exc:                       # noqa: BLE001
                print(f"  {s} attempt {attempt}: {exc}")
            time.sleep(2 + 3 * attempt)
        if ser is None or not len(ser):
            raise RuntimeError(f"FRED download failed for {s}")
        ser.index.name = "date"
        ser.columns = ["value"]
        ser.to_parquet(path)
        meta["macro"][s] = {
            "series_id": s, "file": os.path.basename(path), "source": "FRED",
            "download_timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "first_date": str(ser.index.min().date()), "last_date": str(ser.index.max().date()),
            "row_count": int(len(ser)),
        }
        print(f"  {s:8s} {len(ser):5d} rows  {ser.index.min().date()} .. {ser.index.max().date()}")

    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


# ------------------------------------------------------------------ read side
def load_prices() -> dict[str, pd.DataFrame]:
    out = {}
    for t in ALL_TICKERS:
        p = _ticker_file(t)
        if not os.path.exists(p):
            raise FileNotFoundError(f"snapshot missing for {t}; run scripts/m0_freeze_data.py")
        out[t] = pd.read_parquet(p)
    return out


def load_macro() -> dict[str, pd.Series]:
    out = {}
    for s in MACRO_SERIES:
        p = _macro_file(s)
        if not os.path.exists(p):
            raise FileNotFoundError(f"snapshot missing for {s}")
        out[s] = pd.read_parquet(p)["value"]
    return out


def panel(field: str = "adj_close", tickers=None) -> pd.DataFrame:
    """Aligned wide panel on the union trading calendar (no forward fill)."""
    px = load_prices()
    tickers = tickers or ALL_TICKERS
    df = pd.DataFrame({t: px[t][field] for t in tickers if field in px[t].columns})
    return df.sort_index()
