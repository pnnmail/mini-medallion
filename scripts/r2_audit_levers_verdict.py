"""Q2 + Q3 + Q4 -- power audit by class, the four levers, and the verdict.

Q2 is a retrospective power audit, not a re-evaluation: no M0-M3 verdict is
revisited. Q4 is derived from the numbers in Q1-Q3, not from a wish to continue.
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
from config_m2 import FIG_DIR

# verdict thresholds, declared here
FEASIBLE_N_MULTIPLIER = 4.0     # more than this is not obtainable on this panel


def interp_mde(curve, K, col):
    c = curve.dropna(subset=[col])
    return float(np.interp(np.log(K), np.log(c.K), c[col]))


def main():
    print("=" * 78)
    print("Q2  POWER AUDIT BY CLASS   |   Q3  LEVERS   |   Q4  VERDICT")
    print("=" * 78)
    con = db.connect()
    q1 = json.load(open(os.path.join(RESULTS_DIR, "q1_mde_curve.json")))
    curve = pd.DataFrame(q1["curve"])
    curve["MDE_IC_ret"] = curve["MDE_IC_returns"]
    curve["MDE_IC_vol"] = curve["MDE_IC_volatility"]

    # ------------------------------------------------------------ Q2
    sig = pd.read_csv(os.path.join(RESULTS_DIR, "signals_detail.csv"))
    n_m0_ret = int((sig._target_kind != "volatility").sum())
    n_m0_vol = int((sig._target_kind == "volatility").sum())

    classes = [
        {"cls": 1, "name": "feature -> return", "K": n_m0_ret + 3,
         "regime": "returns",
         "sources": f"M0 search ({n_m0_ret} return/direction targets) + "
                    f"M5-post confound (3)"},
        {"cls": 2, "name": "feature -> volatility", "K": n_m0_vol + 270 + 255 + 89,
         "regime": "volatility",
         "sources": f"M0 vol targets ({n_m0_vol}) + M1 N2 HAR baseline (270) + "
                    f"M1 N3 incremental (255) + M2sup P0 universe (89)"},
        {"cls": 3, "name": "decision comparisons by Sharpe",
         "K": 16 + 148 + 9 + 12 + 40 + 20 + 12 + 12 + 200, "regime": "decision",
         "sources": "M1 N4 (16) + M2sup P2 (148) P3 (9) + M2 N0 (12) N2 (40) "
                    "N3 (20) N4 (12) N5 (12) N6 (200)"},
        {"cls": 4, "name": "structural identities and decompositions", "K": 12,
         "regime": "algebraic", "sources": "M2 N1 (12)"},
        {"cls": 5, "name": "calibration facts", "K": 72, "regime": "volatility",
         "sources": "M0 M7 SCHD intraday legs (72)"},
    ]
    total = sum(c["K"] for c in classes)
    print(f"\nQ2  all {total} selection-relevant trials, by class")
    assert total == 1268, f"class counts sum to {total}, expected 1268"

    # effect sizes of interest per regime, from Q1's reference table
    ref = pd.DataFrame(q1["reference_effects"])
    ic_ret = float(ref[ref.regime == "returns"].IC.max())      # 0.0556, reversal
    ic_vol = float(ref[ref.regime == "volatility"].IC.min())   # 0.160, persistence

    for c in classes:
        if c["regime"] == "returns":
            mde = interp_mde(curve, c["K"], "MDE_IC_ret")
            target = ic_ret
            need = (mde / target) ** 2
            v = ("ANSWERABLE" if mde <= target else
                 "UNDERPOWERED" if need <= FEASIBLE_N_MULTIPLIER else
                 "UNANSWERABLE_IN_PRINCIPLE")
            c.update(mde=mde, target_effect=target, n_multiplier_needed=need,
                     verdict=v)
        elif c["regime"] == "volatility":
            mde = interp_mde(curve, c["K"], "MDE_IC_vol")
            target = ic_vol
            need = (mde / target) ** 2
            v = ("ANSWERABLE" if mde <= target else
                 "UNDERPOWERED" if need <= FEASIBLE_N_MULTIPLIER else
                 "UNANSWERABLE_IN_PRINCIPLE")
            c.update(mde=mde, target_effect=target, n_multiplier_needed=need,
                     verdict=v)
        elif c["regime"] == "decision":
            # M2 measured the paired Sharpe-difference SE directly
            se = 0.028                      # SPY, Ledoit-Wolf, the tightest of the four
            mde = (1.96 + 0.9945) * se
            target = 0.042                  # the largest gap M1 observed
            need = (mde / target) ** 2
            c.update(mde=mde, target_effect=target, n_multiplier_needed=need,
                     verdict="UNANSWERABLE_IN_PRINCIPLE" if need > FEASIBLE_N_MULTIPLIER
                     else "UNDERPOWERED")
        else:
            c.update(mde=0.0, target_effect=np.nan, n_multiplier_needed=0.0,
                     verdict="ANSWERABLE")

    cdf = pd.DataFrame(classes)
    print(f"   {'cls':>3s} {'name':40s} {'K':>5s} {'regime':11s} {'MDE':>7s} "
          f"{'target':>7s} {'n x':>7s} {'verdict':>26s}")
    for _, r in cdf.iterrows():
        tgt = f"{r.target_effect:7.4f}" if np.isfinite(r.target_effect) else "      -"
        print(f"   {int(r.cls):3d} {r['name']:40s} {int(r.K):5d} {r.regime:11s} "
              f"{r.mde:7.4f} {tgt} {r.n_multiplier_needed:7.1f} {r.verdict:>26s}")
    print(f"\n   class 4 is exact algebra: no sampling variability, nothing to power.")
    print(f"   class 3's MDE is the Ledoit-Wolf SE measured in M2 (0.028, SPY -- the")
    print(f"   tightest of the four), not an IC.")

    # ------------------------------------------------------------ Q3
    print(f"\nQ3  LEVERS, by effective n")
    mde_ret_1268 = interp_mde(curve, 1268, "MDE_IC_ret")
    mde_ret_1 = float(curve.loc[curve.K == 1, "MDE_IC_ret"].iloc[0])
    levers = [
        {"lever": "shrink the search space (1268 -> 1)",
         "cost": "free", "mde_multiplier": mde_ret_1268 / mde_ret_1,
         "mde_after_returns": mde_ret_1,
         "does_not_fix": "even a SINGLE pre-registered return hypothesis has "
                         f"MDE_IC = {mde_ret_1:.4f}, above every known return effect. "
                         "The binding constraint is n, not K."},
        {"lever": "longer history (SPY from 1993, n x 1.65)",
         "cost": "free, already used", "mde_multiplier": float(np.sqrt(1.65)),
         "mde_after_returns": mde_ret_1 / float(np.sqrt(1.65)),
         "does_not_fix": "0.0796 -> 0.0620, still above an IC of 0.05. And it adds "
                         "no crises: 1993-2006 contributes one (2000-2002)."},
        {"lever": "intraday data (nominally n x 390)",
         "cost": "free for 730 days only",
         "mde_multiplier": np.nan,
         "mde_after_returns": np.nan,
         "does_not_fix": "intraday autocorrelation and the U-shaped profile make the "
                         "effective multiplier far below sqrt(390). For SIZING it "
                         "gives nothing at all: the number of rebalancing episodes is "
                         "set by the band, not by data frequency -- 82-331 levels, "
                         "measured in M2."},
        {"lever": "wide cross-section (500 names)",
         "cost": "paid point-in-time constituents",
         "mde_multiplier": float(np.sqrt(500 / 2.0)),
         "mde_after_returns": mde_ret_1268 / float(np.sqrt(500 / 2.0)),
         "does_not_fix": "survivorship bias without point-in-time membership; and it "
                         "does not add a single crisis."},
    ]
    ldf = pd.DataFrame(levers)
    for _, r in ldf.iterrows():
        m = f"x{r.mde_multiplier:.2f}" if np.isfinite(r.mde_multiplier) else "unclear"
        a = f"{r.mde_after_returns:.4f}" if np.isfinite(r.mde_after_returns) else "-"
        print(f"   {r.lever:42s} {r.cost:34s} MDE {m:>8s} -> {a}")
        print(f"      does NOT fix: {r.does_not_fix}")

    print(f"\n   MANDATORY CONCLUSION: not one lever changes the crisis count "
          f"(n ~ 5).")
    print(f"   Everything regime-dependent -- which is every decision-level question "
          f"this")
    print(f"   program asked -- stays at that n permanently.")

    # ------------------------------------------------------------ Q4
    kstar = q1["K_star"]
    ret_kstar = kstar["returns_0.05"]
    print(f"\nQ4  VERDICT")
    print(f"   K* in the returns regime at IC <= 0.05 : {ret_kstar}")
    print(f"   K* in the returns regime at IC <= 0.03 : {kstar['returns_0.03']}")
    print(f"   K* in the volatility regime at IC <= 0.05 : {kstar['volatility_0.05']}")

    if ret_kstar == 0:
        verdict = "CONCLUDE"
        why = (
            "CONTINUE_NARROW is not available, and not because the budget would be "
            "small. K* = 0 in the returns regime: at K = 1 -- a single "
            "pre-registered hypothesis, the narrowest search that exists -- the "
            f"minimum detectable IC is {mde_ret_1:.4f}, above short-term reversal "
            "(0.0556), above the overnight premium (0.0357), above TSMOM (0.0242) "
            "and above the literature's 0.02-0.05. There is no hypothesis space "
            "small enough to make a return effect of realistic size visible on this "
            "data. 'A different kind of activity' would still be an activity that "
            "cannot resolve its own question.\n\n"
            "CONTINUE_AFTER_DATA has exactly one candidate -- a wide cross-section -- "
            "and it is a genuine lever for the returns regime "
            f"(MDE {mde_ret_1268:.4f} -> "
            f"{mde_ret_1268 / np.sqrt(250):.4f}). But it does not add a single "
            "crisis, so every decision-level question this program actually asked "
            "stays exactly where it is. Buying it would answer a question the "
            "program did not ask.\n\n"
            "The volatility regime is the one place the harness has power, and it "
            "used it: the two effects it detected (persistence at IC 0.160, VIX over "
            "HAR at 0.190) are both textbook results, and M1 showed the second does "
            "not improve a decision.")
    else:
        verdict = "CONTINUE_NARROW"
        why = f"K* = {ret_kstar} in the returns regime."

    print(f"\n   VERDICT: {verdict}")
    print("\n" + "\n".join("   " + l for l in why.split("\n")))

    payload = {"classes": cdf.to_dict("records"), "total_trials": int(total),
               "levers": ldf.to_dict("records"),
               "K_star": kstar, "verdict": verdict, "reasoning": why,
               "feasible_n_multiplier": FEASIBLE_N_MULTIPLIER}
    with open(os.path.join(RESULTS_DIR, "q2q3q4_audit_verdict.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    cdf.to_csv(os.path.join(RESULTS_DIR, "q2_power_audit.csv"), index=False)
    ldf.to_csv(os.path.join(RESULTS_DIR, "q3_levers.csv"), index=False)

    db.log_experiment(con, "Q2-Q4", "power audit by class, lever assessment, "
                                    "programme verdict",
                      {"total_trials": int(total), "verdict": verdict,
                       "K_star_returns_005": ret_kstar},
                      n_tests=0, seed=None, status=verdict)
    con.close()
    print("\nwrote results/q2q3q4_audit_verdict.json, q2_power_audit.csv, q3_levers.csv")


if __name__ == "__main__":
    main()
