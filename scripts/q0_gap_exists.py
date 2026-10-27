"""N0 -- establish first whether the gap exists at all.

The observed Sharpe gaps are 0.041 / 0.020 / 0.014 / 0.008. The standard error of
a single Sharpe over 20 years is about 0.24, so every one of them is under a
tenth of that. Only the pairing saves the test, and only if the two strategies
are correlated enough -- which is measured here, not assumed.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
import stats_engine as se
from config import BOOTSTRAP_MEAN_BLOCK, TRADING_DAYS
from config_m1 import RESULTS_DIR
from config_m2 import INSTRUMENTS, MASTER_SEED_M2
from m2pair import FAST, SLOW, paired_series
from provenance import estimator_fingerprint
from sharpe_inference import (paired_block_bootstrap_sharpe_diff,
                              sharpe_difference_lw, sign_test_corrected)

BOOT_REPS = 5000


def main():
    print("=" * 78)
    print("N0  DOES THE GAP EXIST?   (SLOW = HAR, FAST = HAR+log_vix)")
    print("=" * 78)
    con = db.connect()
    rng = np.random.default_rng(MASTER_SEED_M2 + 100)
    fp = estimator_fingerprint()["_joint"]
    print(f"estimator fingerprint: {fp}")

    # power arithmetic, computed rather than quoted
    print("\nPOWER ARITHMETIC")
    rows, diffs, series = [], {}, {}
    for tk in INSTRUMENTS:
        ps = paired_series(tk)
        a, b = ps["pnl_slow"], ps["pnl_fast"]
        lw = sharpe_difference_lw(a, b)
        if not lw:
            continue
        bs = paired_block_bootstrap_sharpe_diff(a, b, BOOTSTRAP_MEAN_BLOCK, rng,
                                                BOOT_REPS)
        d = a - b
        p_null, ci_null = se.bootstrap_mean_pvalue(d, BOOTSTRAP_MEAN_BLOCK, rng, 2000)
        n = lw["n"]
        sr = max(abs(lw["sharpe_1_ann"]), abs(lw["sharpe_2_ann"]))
        se_single = float(np.sqrt((1.0 + sr ** 2 / 2.0) / (n / TRADING_DAYS)))
        rho = lw["rho_strategies"]
        se_paired_theory = float(se_single * np.sqrt(2.0 * (1.0 - rho)))
        rows.append({
            "instrument": tk, "n_days": n,
            "sharpe_slow": lw["sharpe_1_ann"], "sharpe_fast": lw["sharpe_2_ann"],
            "sharpe_diff": lw["sharpe_diff_ann"],
            "rho_strategies": rho,
            "se_single_theory": se_single,
            "se_paired_theory": se_paired_theory,
            "se_diff_ledoit_wolf": lw["se_diff_ann"],
            "z_lw": lw["z"], "p_lw": lw["p_lw"],
            "ci_lo_lw": lw["ci_lo_ann"], "ci_hi_lw": lw["ci_hi_ann"],
            "p_bootstrap_paired": bs.get("p_bootstrap_paired", np.nan),
            "boot_se": bs.get("boot_se_ann", np.nan),
            "p_null_harness_mean_diff": p_null,
            "hac_lag": lw["hac_lag"],
        })
        diffs[tk] = lw["sharpe_diff_ann"]
        series[tk] = pd.Series(d, index=ps["index"])
    df = pd.DataFrame(rows)

    print(f"   {'inst':5s} {'n':>5s} {'SR slow':>8s} {'SR fast':>8s} {'gap':>7s} "
          f"{'rho':>6s} {'SEsingle':>9s} {'SEpair th':>10s} {'SE L-W':>8s} "
          f"{'z':>6s} {'p_LW':>6s} {'p_boot':>7s} {'p_null':>7s}")
    for _, r in df.iterrows():
        print(f"   {r.instrument:5s} {int(r.n_days):5d} {r.sharpe_slow:8.3f} "
              f"{r.sharpe_fast:8.3f} {r.sharpe_diff:+7.3f} {r.rho_strategies:6.3f} "
              f"{r.se_single_theory:9.3f} {r.se_paired_theory:10.3f} "
              f"{r.se_diff_ledoit_wolf:8.3f} {r.z_lw:+6.2f} {r.p_lw:6.3f} "
              f"{r.p_bootstrap_paired:7.3f} {r.p_null_harness_mean_diff:7.3f}")

    # dependence-corrected sign test
    al = pd.concat(series, axis=1).dropna()
    corr = al.corr().to_numpy()
    st = sign_test_corrected(np.array([diffs[t] for t in al.columns]), corr, True)
    print(f"\nCROSS-INSTRUMENT SIGN TEST")
    print(f"   correlation of the DIFFERENCE series between instruments "
          f"(n={len(al)} common days):")
    print(pd.DataFrame(corr, index=al.columns, columns=al.columns).round(3).to_string())
    print(f"   gap positive in {st['n_positive']}/{st['n_instruments']} instruments")
    print(f"   naive p = {st['p_naive']:.4f}   |   effective number of tests = "
          f"{st['effective_n_tests']:.2f}   ->   corrected p = "
          f"{st['p_dependence_corrected']:.4f}")

    n_sig_lw = int((df.p_lw < 0.05).sum())
    n_sig_boot = int((df.p_bootstrap_paired < 0.05).sum())
    if n_sig_lw >= 2 or (n_sig_lw >= 1 and st["p_dependence_corrected"] < 0.05):
        gap_status = "GAP_SIGNIFICANT"
    elif df.sharpe_diff.gt(0).all():
        gap_status = "GAP_UNDERPOWERED"
    else:
        gap_status = "GAP_ABSENT"

    print(f"\n   significant at 5%: Ledoit-Wolf {n_sig_lw}/{len(df)}, "
          f"paired bootstrap {n_sig_boot}/{len(df)}")
    print(f"\n   gap_status = {gap_status}")
    if gap_status == "GAP_UNDERPOWERED":
        print("   The gap has the same sign everywhere but no single instrument")
        print("   distinguishes it from zero. This does NOT close the milestone: the")
        print("   covariance term of N1 is estimated on ~5000 daily observations and")
        print("   has power the Sharpe difference does not. It does mean every later")
        print("   statement must be read against this.")

    payload = {
        "pair": {"slow": SLOW, "fast": FAST, "cap": 1.0,
                 "sample": "TRAIN + VALIDATION (M1 development sample)"},
        "estimator_fingerprint": fp,
        "per_instrument": df.to_dict("records"),
        "difference_series_correlation": pd.DataFrame(
            corr, index=al.columns, columns=al.columns).to_dict(),
        "sign_test": st,
        "n_significant_lw": n_sig_lw, "n_significant_bootstrap": n_sig_boot,
        "gap_status": gap_status,
        "power_note": "SE of a single Sharpe over this sample is about "
                      f"{df.se_single_theory.mean():.3f}; the largest observed gap is "
                      f"{df.sharpe_diff.abs().max():.3f}. Only the pairing gives any "
                      f"power, and the measured rho between the two strategies is "
                      f"{df.rho_strategies.min():.3f}-{df.rho_strategies.max():.3f}.",
    }
    with open(os.path.join(RESULTS_DIR, "m2_n0_gap.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    df.to_csv(os.path.join(RESULTS_DIR, "m2_n0_gap.csv"), index=False)

    db.log_experiment(con, "N0", "does the Sharpe gap exist: Ledoit-Wolf, paired block "
                                 "bootstrap, null harness on the difference, "
                                 "dependence-corrected sign test",
                      {"instruments": list(INSTRUMENTS), "boot_reps": BOOT_REPS,
                       "gap_status": gap_status, "fingerprint": fp},
                      n_tests=len(df) * 3, seed=MASTER_SEED_M2, status=gap_status)
    con.close()
    print(f"\nwrote results/m2_n0_gap.json, m2_n0_gap.csv")


if __name__ == "__main__":
    main()
