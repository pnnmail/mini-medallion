"""N0 -- extend the frozen snapshot for M1 and record what M1 deletes for good."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config_m1 import (M1_SNAPSHOT_DIR, REMOVED_FEATURES, RESULTS_DIR,
                       SPY_EXTENDED_TRAIN_START, VOL_TARGETS)
from data_m1 import CALENDAR_OMISSIONS, download_m1_snapshot, event_calendar, load_m1
from data_loader import load_prices


def main():
    print("=" * 78)
    print("N0  M1 DATA EXTENSION + REMOVALS")
    print("=" * 78)
    meta = download_m1_snapshot()
    con = db.connect()

    spy_ext = load_m1("SPY_EXT")
    vix = load_m1("VIX")
    t10 = load_m1("T10Y2Y_EXT")
    px = load_prices()

    # the extended SPY series must agree with the frozen M0 series where they overlap
    a = px["SPY"]["adj_close"].astype(float)
    b = spy_ext["adj_close"].astype(float)
    idx = a.index.intersection(b.index)
    rel = ((a.loc[idx] - b.loc[idx]).abs() / a.loc[idx]).replace(
        [np.inf, -np.inf], np.nan).dropna()
    overlap_ok = bool(rel.max() < 1e-4)   # 0.01%: below this is float noise, not a revision
    print(f"\nSPY extended vs M0 frozen: {len(idx)} overlapping rows, "
          f"max relative diff {rel.max():.2e} -> {'CONSISTENT' if overlap_ok else 'DRIFTED'}")
    print("   M1 uses SPY_EXT as its frozen SPY source throughout; that is legitimate "
          "only because the\n   overlap agrees with the M0 freeze to well inside float noise.")
    if not overlap_ok:
        raise SystemExit("SPY extension disagrees with the M0 freeze by more than 0.01% "
                         "-- yfinance revised the series; do not splice silently.")

    cal = event_calendar(px["SPY"].index)
    print(f"\nevent calendar flags on {len(cal)} sessions:")
    for c in cal.columns:
        print(f"   {c:18s} {int(cal[c].sum()):5d} days ({cal[c].mean() * 100:.1f}%)")
    print(f"   OMITTED: FOMC, CPI -- see CALENDAR_OMISSIONS")

    print("\nREMOVED FROM THE SEARCH SPACE FOR GOOD:")
    for k, why in REMOVED_FEATURES.items():
        print(f"   - {k}\n     {why}")
        db.log_experiment(con, "N0", f"REMOVED PERMANENTLY: {k}", {"reason": why},
                          n_tests=0, seed=None, status="REMOVED")

    report = {
        "m1_snapshot": meta,
        "spy_extended": {"first_date": str(spy_ext.index.min().date()),
                         "last_date": str(spy_ext.index.max().date()),
                         "rows": int(len(spy_ext)),
                         "declared_extended_train_start": SPY_EXTENDED_TRAIN_START,
                         "overlap_with_m0_consistent": overlap_ok,
                         "max_relative_diff_on_overlap": float(rel.max())},
        "vix": {"first_date": str(vix.index.min().date()),
                "last_date": str(vix.index.max().date()), "rows": int(len(vix)),
                "min": float(vix["close"].min()), "max": float(vix["close"].max()),
                "n_nan": int(vix["close"].isna().sum())},
        "t10y2y_extended": {"first_date": str(t10.index.min().date()),
                            "rows": int(len(t10))},
        "calendar_flags": {c: int(cal[c].sum()) for c in cal.columns},
        "calendar_omissions": CALENDAR_OMISSIONS,
        "removed_features": REMOVED_FEATURES,
        "vol_targets": VOL_TARGETS,
    }
    with open(os.path.join(RESULTS_DIR, "m1_data_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    db.log_experiment(con, "N0", "M1 snapshot extension (SPY 1993, VIX, T10Y2Y) "
                                 "+ deterministic event calendar",
                      {"spy_first": str(spy_ext.index.min().date()),
                       "vix_first": str(vix.index.min().date()),
                       "calendar_omissions": ["FOMC", "CPI"]},
                      n_tests=0, seed=None, status="COMPLETED")
    con.close()
    print(f"\nwrote results/m1_data_report.json")


if __name__ == "__main__":
    main()
