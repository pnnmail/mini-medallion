"""Test D -- was the baseline weak?

Not on the original list, but the data pointed at it. M1/N2 measured that HAR
loses to a one-parameter EWMA at h = 1 in 8 of 9 instruments. "Incremental over
HAR" then overstates what log(VIX) adds over a COMPETENT baseline, because part
of the increment is HAR's own weakness at that horizon.

Test: strengthen the baseline with the benchmark that beat it, then measure the
increment again. If log(VIX)'s t collapses, the result was a statement about HAR,
not about VIX.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config_m1 import RESULTS_DIR
from config_m2 import FIG_DIR, MASTER_SEED_M2
from har import ewma_forecast, har_regressors, har_target
from inference_m1 import nw_tstat_last
from panel_m1 import instrument_ohlc, split_mask_m1, vix_close
from rv_estimators import RV_FLOOR, daily_variance

INSTR = ["SPY", "SCHD", "TLT", "IWM"]


def cell(tk, horizon, strengthen, target_est="cc"):
    d = instrument_ohlc(tk)
    rv = daily_variance(d, "gk")
    Xb = har_regressors(rv)
    if strengthen:
        Xb = Xb.join(np.log(ewma_forecast(rv).clip(lower=RV_FLOOR)).rename("ewma94"))
    y = har_target(daily_variance(d, target_est), horizon)
    v = np.log(vix_close().reindex(d.index).ffill(limit=5)).rename("vix")
    m = split_mask_m1(d.index, tk, "TRAIN")
    df = pd.concat([y.rename("y"), Xb, v], axis=1)[m].dropna()
    if len(df) < 300:
        return None
    n = len(df)
    Xbase = np.column_stack([np.ones(n), df[Xb.columns].to_numpy(float)])
    Xfull = np.column_stack([Xbase, df["vix"].to_numpy(float)])
    yv = df["y"].to_numpy(float)
    b, t, p = nw_tstat_last(yv, Xfull, max(horizon - 1, 0))

    def r2(X):
        beta = np.linalg.lstsq(X, yv, rcond=None)[0]
        return 1.0 - np.var(yv - X @ beta) / np.var(yv)

    return {"instrument": tk, "horizon": horizon, "strengthened": strengthen,
            "n": n, "t": t, "r2_base": r2(Xbase), "delta_r2": r2(Xfull) - r2(Xbase)}


def main():
    print("=" * 78)
    print("D. WAS THE BASELINE WEAK?  HAR versus HAR + EWMA(0.94)")
    print("=" * 78)
    con = db.connect()
    rows = []
    for tk in INSTR:
        for h in [1, 5, 10]:
            for s in [False, True]:
                r = cell(tk, h, s)
                if r:
                    rows.append(r)
    df = pd.DataFrame(rows)

    print(f"\n   {'inst':5s} {'h':>3s} {'t | HAR':>9s} {'t | HAR+EWMA':>13s} "
          f"{'retained':>9s} {'R2 base HAR':>12s} {'R2 base +EWMA':>14s} "
          f"{'dR2 vix':>9s}")
    ratios = []
    for tk in INSTR:
        for h in [1, 5, 10]:
            a = df[(df.instrument == tk) & (df.horizon == h) & ~df.strengthened]
            b = df[(df.instrument == tk) & (df.horizon == h) & df.strengthened]
            if len(a) and len(b):
                ta, tb = float(a.t.iloc[0]), float(b.t.iloc[0])
                ratios.append(tb / ta if ta else np.nan)
                print(f"   {tk:5s} {h:3d} {ta:+9.2f} {tb:+13.2f} "
                      f"{(tb / ta if ta else np.nan):8.0%} "
                      f"{float(a.r2_base.iloc[0]):12.4f} "
                      f"{float(b.r2_base.iloc[0]):14.4f} "
                      f"{float(b.delta_r2.iloc[0]):+9.4f}")
    med = float(np.nanmedian(ratios))
    h1 = [r for (r, (tk, h)) in zip(ratios, [(t_, h_) for t_ in INSTR
                                             for h_ in [1, 5, 10]]) if h == 1]
    D_kills = bool(med < 0.5)
    print(f"\n   median t retained after strengthening the baseline: {med:.0%}")
    print(f"   at h=1 specifically: {np.nanmedian(h1):.0%}  "
          f"(h=1 is where M1/N2 found HAR losing to EWMA in 8 of 9)")
    print(f"   -> baseline weakness {'EXPLAINS IT' if D_kills else 'does not explain it'}")

    df.to_csv(os.path.join(RESULTS_DIR, "vix_baseline_strength.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "vix_baseline_strength.json"), "w") as fh:
        json.dump({"median_t_retained": med,
                   "median_t_retained_h1": float(np.nanmedian(h1)),
                   "explains": D_kills, "rows": df.to_dict("records")},
                  fh, indent=2, default=str)

    db.log_experiment(con, "FORENSICS", "test D: is the log(VIX) increment an artefact "
                                        "of a weak HAR baseline at short horizons?",
                      {"median_t_retained": med, "explains": D_kills},
                      n_tests=len(df), seed=MASTER_SEED_M2, status="COMPLETED")
    con.close()
    print("\nwrote results/vix_baseline_strength.csv/.json")


if __name__ == "__main__":
    main()
