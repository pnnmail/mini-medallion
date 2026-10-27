"""M0 -- freeze the dataset and produce data_quality_report.json."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config import (ALL_TICKERS, EQUITY_TICKERS, FUTURES_TICKERS, KNOWN_SPLITS,
                    MACRO_SERIES, PAIRS, RESULTS_DIR)
from data_loader import download_snapshot, load_macro, load_prices, panel


def roll_dates_clf(df, z_thresh=5.0):
    """Detect front-month roll gaps in CL=F.

    A roll shows up as a one-day jump that is large relative to local
    volatility and that is not accompanied by a comparable intraday range.
    """
    close = df["close"].astype(float)
    lr = np.log(close).diff()
    vol = lr.rolling(60).std()
    intraday = np.log(df["high"].astype(float) / df["low"].astype(float))
    z = lr.abs() / vol
    gap_vs_range = lr.abs() / intraday.replace(0, np.nan)
    flag = (z > z_thresh) & (gap_vs_range > 1.0)
    return flag.fillna(False)


def main():
    print("=" * 78)
    print("M0  DATA FREEZE")
    print("=" * 78)
    meta = download_snapshot(force=False)

    px = load_prices()
    macro = load_macro()
    con = db.connect()

    report = {"snapshot_date": meta["snapshot_date"], "tickers": {}, "macro": {},
              "issues": [], "total_return_consistency": {}, "clf_roll": {},
              "calendar": {}, "splits": {}}

    for t in ALL_TICKERS:
        d = px[t]
        close = d["close"].astype(float)
        adj = d["adj_close"].astype(float)
        lr = np.log(adj).diff()
        rec = {
            "first_date": str(d.index.min().date()),
            "last_date": str(d.index.max().date()),
            "row_count": int(len(d)),
            "n_nan_close": int(close.isna().sum()),
            "n_nan_adj_close": int(adj.isna().sum()),
            "n_zero_volume_days": int((d["volume"].fillna(0) == 0).sum()) if "volume" in d else None,
            "n_duplicate_dates": int(d.index.duplicated().sum()),
            "max_abs_daily_log_return": float(np.nanmax(np.abs(lr))) if lr.notna().any() else None,
            "n_abs_return_gt_20pct": int((lr.abs() > 0.20).sum()),
            "total_dividends_paid": float(d["dividends"].sum()) if "dividends" in d else 0.0,
            "n_dividend_events": int((d["dividends"] > 0).sum()) if "dividends" in d else 0,
        }
        if "splits" in d:
            sp = d["splits"][d["splits"] > 0]
            rec["splits_detected"] = {str(k.date()): float(v) for k, v in sp.items()}
        report["tickers"][t] = rec

        con.execute("DELETE FROM prices WHERE ticker = ?", [t])
        ins = d.reset_index()
        ins.columns = [str(c).lower() for c in ins.columns]
        ins["ticker"] = t
        for c in ["open", "high", "low", "close", "adj_close", "volume"]:
            if c not in ins:
                ins[c] = np.nan
        ins = ins[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
        con.register("ins_df", ins)
        con.execute("INSERT INTO prices SELECT ticker, CAST(date AS DATE), open, high, "
                    "low, close, adj_close, volume FROM ins_df")
        con.unregister("ins_df")

    for tk, evs in KNOWN_SPLITS.items():
        got = report["tickers"][tk].get("splits_detected", {})
        for date_s, ratio in evs:
            # yfinance stamps the EX-date; the spec quotes the announced date.
            # Accept a match within 3 calendar days and record what was observed.
            want = pd.Timestamp(date_s)
            hit = [(k, v) for k, v in got.items()
                   if abs((pd.Timestamp(k) - want).days) <= 3]
            found = len(hit) == 1 and abs(hit[0][1] - ratio) < 1e-9
            # the price series must already be back-adjusted: no 3x jump at the split
            px_tk = px[tk]["close"].astype(float)
            around = px_tk.loc[want - pd.Timedelta(days=6): want + pd.Timedelta(days=6)]
            jump = float(np.log(around).diff().abs().max()) if len(around) > 2 else None
            report["splits"][tk + "@" + date_s] = {
                "expected_ratio": ratio,
                "found_in_snapshot": found,
                "observed_ex_date": hit[0][0] if hit else None,
                "observed_ratio": hit[0][1] if hit else None,
                "max_abs_log_move_around_split": jump,
                "back_adjusted_ok": bool(jump is not None and jump < 0.10),
            }
            if not found:
                report["issues"].append(
                    "expected split " + tk + " " + date_s + " not found in snapshot actions")
            elif jump is not None and jump >= 0.10:
                report["issues"].append(
                    "split " + tk + " " + date_s + " leaves an unadjusted jump in close")

    for s in MACRO_SERIES:
        ser = macro[s]
        report["macro"][s] = {
            "first_date": str(ser.index.min().date()),
            "last_date": str(ser.index.max().date()),
            "row_count": int(len(ser)),
            "n_nan": int(ser.isna().sum()),
            "min": float(ser.min()), "max": float(ser.max()),
        }
        con.execute("DELETE FROM macro WHERE series_id = ?", [s])
        m = ser.reset_index()
        m.columns = ["date", "value"]
        m["series_id"] = s
        con.register("m_df", m[["series_id", "date", "value"]])
        con.execute("INSERT INTO macro SELECT series_id, CAST(date AS DATE), value FROM m_df")
        con.unregister("m_df")

    eq = panel("adj_close", EQUITY_TICKERS)
    common = eq.dropna(how="any")
    report["calendar"] = {
        "union_first_date": str(eq.index.min().date()),
        "union_last_date": str(eq.index.max().date()),
        "union_rows": int(len(eq)),
        "common_window_first_date": str(common.index.min().date()),
        "common_window_last_date": str(common.index.max().date()),
        "common_window_rows": int(len(common)),
        "rows_with_any_missing": int(len(eq) - len(common)),
        "per_ticker_first_date": {t: str(eq[t].first_valid_index().date()) for t in eq.columns},
    }
    research_start = max(pd.Timestamp(eq[t].first_valid_index())
                         for t in eq.columns if t != "SCHD")
    report["calendar"]["research_start_ex_SCHD"] = str(research_start.date())

    # Total-return consistency: both legs of every spread must use the SAME
    # adjustment scheme, otherwise the z-score measures accumulated dividends.
    adj_p = panel("adj_close")
    raw_p = panel("close")
    for a, b in list(PAIRS) + [("IWM", "SPY")]:
        sub_a = np.log(adj_p[a] / adj_p[b]).dropna()
        sub_r = np.log(raw_p[a] / raw_p[b]).dropna()
        idx = sub_a.index.intersection(sub_r.index)
        sub_a, sub_r = sub_a.loc[idx], sub_r.loc[idx]
        x = np.arange(len(idx), dtype=float)
        sl_a = float(np.polyfit(x, sub_a.values, 1)[0] * 252)
        sl_r = float(np.polyfit(x, sub_r.values, 1)[0] * 252)
        report["total_return_consistency"][a + "/" + b] = {
            "annual_drift_adjusted": sl_a,
            "annual_drift_unadjusted": sl_r,
            "difference": sl_a - sl_r,
            "n_obs": int(len(idx)),
            "comment": "difference is the dividend-yield gap between the legs; "
                       "the pipeline uses adj_close for BOTH legs everywhere",
        }

    for t in FUTURES_TICKERS:
        flag = roll_dates_clf(px[t])
        neg = px[t]["close"].astype(float) <= 0
        report["clf_roll"][t] = {
            "n_flagged_roll_gaps": int(flag.sum()),
            "flagged_dates_sample": [str(d.date()) for d in flag[flag].index[:15]],
            "n_nonpositive_close": int(neg.sum()),
            "nonpositive_dates": [str(d.date()) for d in neg[neg].index],
            "min_close": float(px[t]["close"].min()),
            "handling": "CL=F is NOT used by any feature in the M1 search space; kept in "
                        "the snapshot as a data asset only. Two independent blockers for "
                        "log-ratio spreads: (a) front-month roll gaps, (b) a negative "
                        "settlement price on 2020-04-20 that makes log() undefined. Any "
                        "future use must be built on returns with these dates excluded.",
        }
        if int(neg.sum()) > 0:
            report["issues"].append(
                t + " has non-positive closes (negative crude, 2020-04-20): log-ratio "
                    "spreads on this series are undefined; excluded from the search space")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "data_quality_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)

    db.log_experiment(con, "M0", "frozen data snapshot + integrity report",
                      {"tickers": ALL_TICKERS, "macro": MACRO_SERIES,
                       "start": str(eq.index.min().date()),
                       "end": str(eq.index.max().date())},
                      n_tests=0, seed=None, status="COMPLETED")
    con.close()

    print("\nwrote " + out)
    print("common window: " + report["calendar"]["common_window_first_date"] + " .. "
          + report["calendar"]["common_window_last_date"]
          + " (" + str(report["calendar"]["common_window_rows"]) + " rows)")
    print("research start (ex SCHD): " + report["calendar"]["research_start_ex_SCHD"])
    print("issues: " + str(report["issues"] or "none"))


if __name__ == "__main__":
    main()
