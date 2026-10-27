"""Forensics on the one surviving result: is t = 14.13 a finding or an artefact?

The magnitude is the reason to look. |IC| ~ 0.2 at n ~ 5000 is larger than any
predictive effect in liquid instruments has a right to be, and twice the
strongest thing this project measured -- volatility persistence at t = 7.404,
which is known and untradeable.

Three candidate explanations, all cheap:

  A  MEASUREMENT OVERLAP. VIX is implied volatility ON SPY; HAR is built from
     realised volatility OF SPY; the target is that same quantity in the future.
     If the target estimator is close-to-close and VIX carries intraday-path
     information, log(VIX) may be supplying a better MEASUREMENT of the current
     state rather than a prediction of the next one.
     Test: swap the target onto Garman-Klass and GK+overnight.

  B  ALIGNMENT LEAK. VIX settles at 16:15 ET, fifteen minutes after the cash
     close. The forward target is measured from the 16:00 close of day t, so
     VIX_t contains a quarter hour of the target window.
     Test: lag VIX by one and two days. A sound feature loses a little; a broken
     one collapses. A forward-shifted VIX calibrates what a real leak looks like.

  C  HORIZON OVERLAP. VIX is a 30-day expectation. Against a 5- or 10-day target
     the windows intersect.
     Test: compare against 22- and 30-day targets. If the SHORT horizon is
     stronger, overlap is not the explanation.

Horizons 22 and 30 lie outside the frozen {1, 5, 10} set. They are diagnostics
for this question only, are logged as such, and enter the trial counter.
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
from har import har_regressors, har_target
from inference_m1 import nw_tstat_last
from panel_m1 import instrument_ohlc, split_mask_m1, vix_close
from rv_estimators import daily_variance

INSTR = ["SPY", "SCHD", "TLT", "IWM"]
REG_EST = "gk"


def cell(tk, target_est, horizon, vix_shift, split="TRAIN"):
    """Incremental t of log(VIX) over HAR, for one configuration."""
    d = instrument_ohlc(tk)
    Xb = har_regressors(daily_variance(d, REG_EST))
    y = har_target(daily_variance(d, target_est), horizon)
    v = np.log(vix_close().reindex(d.index).ffill(limit=5))
    if vix_shift:
        v = v.shift(vix_shift)
    m = split_mask_m1(d.index, tk, split)
    df = pd.concat([y.rename("y"), Xb, v.rename("vix")], axis=1)[m].dropna()
    if len(df) < 300:
        return None
    n = len(df)
    X_base = np.column_stack([np.ones(n), df[Xb.columns].to_numpy(float)])
    X_full = np.column_stack([X_base, df["vix"].to_numpy(float)])
    yv = df["y"].to_numpy(float)
    b, t, p = nw_tstat_last(yv, X_full, max(horizon - 1, 0))
    # incremental R2
    def r2(X):
        beta = np.linalg.lstsq(X, yv, rcond=None)[0]
        e = yv - X @ beta
        return 1.0 - np.var(e) / np.var(yv)
    return {"instrument": tk, "target_est": target_est, "horizon": horizon,
            "vix_shift": vix_shift, "n": n, "beta": b, "t": t, "p": p,
            "r2_base": r2(X_base), "r2_full": r2(X_full),
            "delta_r2": r2(X_full) - r2(X_base)}


def main():
    print("=" * 78)
    print("FORENSICS ON t = 14.13")
    print("=" * 78)
    con = db.connect()
    rows = []

    print("\nBASELINE -- reproducing M1/N3 (regressors GK, target CC, VIX at t)")
    print(f"   {'inst':5s} {'h':>3s} {'n':>5s} {'t':>8s} {'dR2':>8s}")
    for tk in INSTR:
        for h in [1, 5, 10]:
            r = cell(tk, "cc", h, 0)
            if r:
                r["variant"] = "baseline"
                rows.append(r)
                print(f"   {tk:5s} {h:3d} {r['n']:5d} {r['t']:+8.2f} "
                      f"{r['delta_r2']:+8.4f}")

    # ---------------------------------------------------------- A
    print("\nA. MEASUREMENT OVERLAP -- move the TARGET onto a better estimator")
    print(f"   {'inst':5s} {'h':>3s} {'t | cc':>10s} {'t | gk':>10s} "
          f"{'t | gk+on':>11s} {'dR2 cc':>8s} {'dR2 gk':>8s}")
    for tk in INSTR:
        for h in [1, 5, 10]:
            out = {}
            for est in ["cc", "gk", "gk_overnight"]:
                r = cell(tk, est, h, 0)
                if r:
                    r["variant"] = f"targetA_{est}"
                    rows.append(r)
                    out[est] = r
            if len(out) == 3:
                print(f"   {tk:5s} {h:3d} {out['cc']['t']:+10.2f} "
                      f"{out['gk']['t']:+10.2f} {out['gk_overnight']['t']:+11.2f} "
                      f"{out['cc']['delta_r2']:+8.4f} {out['gk']['delta_r2']:+8.4f}")

    # ---------------------------------------------------------- B
    print("\nB. ALIGNMENT -- lag VIX (shift +1, +2 = information removed);")
    print("   shift -1 uses tomorrow's VIX and calibrates what a real leak looks like")
    print(f"   {'inst':5s} {'h':>3s} {'t | -1 (leak)':>14s} {'t | 0':>9s} "
          f"{'t | +1':>9s} {'t | +2':>9s} {'retained +1':>12s}")
    for tk in INSTR:
        for h in [1, 5, 10]:
            out = {}
            for s in [-1, 0, 1, 2]:
                r = cell(tk, "cc", h, s)
                if r:
                    r["variant"] = f"shift_{s}"
                    rows.append(r)
                    out[s] = r
            if len(out) == 4:
                ret = out[1]["t"] / out[0]["t"] if out[0]["t"] else np.nan
                print(f"   {tk:5s} {h:3d} {out[-1]['t']:+14.2f} {out[0]['t']:+9.2f} "
                      f"{out[1]['t']:+9.2f} {out[2]['t']:+9.2f} {ret:11.0%}")

    # ---------------------------------------------------------- C
    print("\nC. HORIZON OVERLAP -- VIX is a 30-day expectation")
    print(f"   {'inst':5s} " + " ".join(f"{'h=' + str(h):>9s}" for h in
                                        [1, 5, 10, 22, 30]))
    for tk in INSTR:
        out = {}
        for h in [1, 5, 10, 22, 30]:
            r = cell(tk, "cc", h, 0)
            if r:
                r["variant"] = "horizonC"
                rows.append(r)
                out[h] = r
        if out:
            print(f"   {tk:5s} " + " ".join(
                f"{out[h]['t']:+9.2f}" if h in out else "        -"
                for h in [1, 5, 10, 22, 30]))
    print(f"   delta R2 by horizon (same cells):")
    for tk in INSTR:
        s = [r for r in rows if r["instrument"] == tk and r["variant"] == "horizonC"]
        if s:
            print(f"   {tk:5s} " + " ".join(
                f"{[r for r in s if r['horizon'] == h][0]['delta_r2']:+9.4f}"
                if any(r["horizon"] == h for r in s) else "        -"
                for h in [1, 5, 10, 22, 30]))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "vix_forensics.csv"), index=False)

    # ---------------------------------------------------------- verdicts
    def get(tk, variant, h, shift=0):
        s = df[(df.instrument == tk) & (df.variant == variant) & (df.horizon == h)
               & (df.vix_shift == shift)]
        return float(s.t.iloc[0]) if len(s) else np.nan

    print("\n" + "=" * 78)
    print("VERDICTS")
    print("=" * 78)
    a_rat, b_rat, c_short = [], [], []
    for tk in INSTR:
        for h in [1, 5, 10]:
            base = get(tk, "targetA_cc", h)
            gk = get(tk, "targetA_gk", h)
            if np.isfinite(base) and np.isfinite(gk) and base != 0:
                a_rat.append(gk / base)
            b0, b1 = get(tk, "shift_0", h, 0), get(tk, "shift_1", h, 1)
            if np.isfinite(b0) and np.isfinite(b1) and b0 != 0:
                b_rat.append(b1 / b0)
    for tk in INSTR:
        t5 = get(tk, "horizonC", 5)
        t30 = get(tk, "horizonC", 30)
        if np.isfinite(t5) and np.isfinite(t30):
            c_short.append(abs(t5) > abs(t30))

    a_med = float(np.nanmedian(a_rat)) if a_rat else np.nan
    b_med = float(np.nanmedian(b_rat)) if b_rat else np.nan
    A_kills = bool(a_med < 0.5)
    B_kills = bool(b_med < 0.5)
    C_kills = bool(c_short and sum(c_short) < len(c_short) / 2)

    print(f"A  target on GK retains {a_med:.0%} of the t-stat  -> measurement "
          f"overlap {'EXPLAINS IT' if A_kills else 'does not explain it'}")
    print(f"B  lagging VIX one day retains {b_med:.0%} of the t-stat  -> alignment "
          f"leak {'EXPLAINS IT' if B_kills else 'does not explain it'}")
    print(f"C  |t| larger at h=5 than at h=30 in {sum(c_short)}/{len(c_short)} "
          f"instruments  -> horizon overlap "
          f"{'EXPLAINS IT' if C_kills else 'does not explain it'}")

    survives = not (A_kills or B_kills or C_kills)
    print(f"\n  t = 14.13 {'SURVIVES all three' if survives else 'is EXPLAINED'}")

    payload = {"baseline_t_SPY_h5": get("SPY", "baseline", 5),
               "A_median_t_ratio_gk_over_cc": a_med, "A_explains": A_kills,
               "B_median_t_ratio_lag1_over_lag0": b_med, "B_explains": B_kills,
               "C_short_stronger_count": int(sum(c_short)),
               "C_n_instruments": len(c_short), "C_explains": C_kills,
               "survives_all_three": survives,
               "rows": df.to_dict("records")}
    with open(os.path.join(RESULTS_DIR, "vix_forensics.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    for tk in INSTR:
        s = df[(df.instrument == tk) & df.variant.str.startswith("targetA")]
        ax.plot(["cc", "gk", "gk+on"],
                [s[s.variant == f"targetA_{e}"].t.abs().max()
                 for e in ["cc", "gk", "gk_overnight"]], marker="o", label=tk)
    ax.set_ylabel("max |t| over horizons"); ax.set_xlabel("target estimator")
    ax.set_title("A  measurement overlap", fontsize=10); ax.legend(fontsize=7)
    ax = axes[1]
    for tk in INSTR:
        s = df[(df.instrument == tk) & df.variant.str.startswith("shift_")]
        ax.plot([-1, 0, 1, 2], [s[s.vix_shift == k].t.abs().max()
                                for k in [-1, 0, 1, 2]], marker="o", label=tk)
    ax.axvline(0, color="grey", ls=":", lw=1)
    ax.set_xlabel("VIX shift (days; -1 = leak)"); ax.set_ylabel("max |t|")
    ax.set_title("B  alignment", fontsize=10); ax.legend(fontsize=7)
    ax = axes[2]
    for tk in INSTR:
        s = df[(df.instrument == tk) & (df.variant == "horizonC")].sort_values("horizon")
        ax.plot(s.horizon, s.t.abs(), marker="o", label=tk)
    ax.set_xlabel("target horizon, days"); ax.set_ylabel("|t|")
    ax.set_title("C  horizon overlap", fontsize=10); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "vix_forensics.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "FORENSICS", "three-test forensics on the surviving "
                                        "log(VIX) incremental result; horizons 22 and "
                                        "30 are outside the frozen set and are "
                                        "diagnostics for this question only",
                      {"tests": ["measurement_overlap", "alignment", "horizon"],
                       "A_explains": A_kills, "B_explains": B_kills,
                       "C_explains": C_kills, "survives": survives},
                      n_tests=len(df), seed=MASTER_SEED_M2, status="COMPLETED")
    con.close()
    print("\nwrote results/vix_forensics.csv/.json, figures/vix_forensics.png")


if __name__ == "__main__":
    main()
