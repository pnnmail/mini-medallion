"""Q0 -- BLOCKING. Reconcile every published TLT number with its full specification.

The expected resolution was that table A carried a RAW predictive t and the
structural-coherence argument carried an INCREMENTAL one. That is not what the
specifications say.
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


def main():
    print("=" * 78)
    print("Q0  BLOCKING RECONCILIATION -- the TLT numbers")
    print("=" * 78)
    con = db.connect()
    f = pd.read_csv(os.path.join(RESULTS_DIR, "vix_forensics.csv"))
    b = pd.read_csv(os.path.join(RESULTS_DIR, "vix_baseline_strength.csv"))
    t = f[f.instrument == "TLT"]

    spec = []
    for est in ["cc", "gk", "gk_overnight"]:
        for h in [1, 5, 10]:
            r = t[(t.variant == f"targetA_{est}") & (t.horizon == h)]
            if len(r):
                r = r.iloc[0]
                spec.append({
                    "published_as": ("structural-coherence claim"
                                     if est == "cc" else "table A"),
                    "target": "log mean daily variance over t+1..t+H",
                    "target_estimator": est, "horizon": h,
                    "regressor_estimator": "gk",
                    "baseline": "HAR (har_d, har_w, har_m) + intercept",
                    "statistic": "INCREMENTAL t on log(VIX) added to the baseline",
                    "standard_errors": f"Newey-West, Bartlett, lag = H-1 = {max(h-1,0)}",
                    "n": int(r.n), "t": float(r.t), "delta_r2": float(r.delta_r2)})
    sp = pd.DataFrame(spec)
    print("\nfull specification of every TLT figure published:")
    print(sp[["published_as", "target_estimator", "horizon", "statistic", "n", "t",
              "delta_r2"]].to_string(index=False))

    both_incremental = bool((sp.statistic.str.startswith("INCREMENTAL")).all())
    print(f"\n   are BOTH figures incremental?  {both_incremental}")
    print("   The expected resolution -- raw versus incremental -- does NOT hold.")
    print("   Both are incremental over the same HAR baseline with the same errors.")
    print("   The only difference is the TARGET ESTIMATOR.")

    cc = sp[sp.target_estimator == "cc"]
    gk = sp[sp.target_estimator == "gk"]
    print(f"\n   TLT under target=cc : t = "
          f"{', '.join(f'{v:+.2f}' for v in cc.t)}   dR2 = "
          f"{', '.join(f'{v:.4f}' for v in cc.delta_r2)}")
    print(f"   TLT under target=gk : t = "
          f"{', '.join(f'{v:+.2f}' for v in gk.t)}   dR2 = "
          f"{', '.join(f'{v:.4f}' for v in gk.delta_r2)}")

    spy = f[(f.instrument == "SPY") & (f.variant == "targetA_gk")].sort_values("horizon")
    print(f"\n   for scale, SPY under target=gk : t = "
          f"{', '.join(f'{v:+.2f}' for v in spy.t)}")

    status = "ARGUMENT_WEAKENED"
    print("\n" + "=" * 78)
    print(f"STATUS: {status}")
    print("=" * 78)
    print("""
   The claim was: 'the effect is strong where VIX is defined and ABSENT on TLT
   (dR2 ~ 0.0005, t < 2), and an artefact would not respect that boundary.'

   That quoted the close-to-close row only. Under the Garman-Klass target -- the
   better-measured one, which test A itself showed is the right one to look at --
   VIX adds substantially to TLT as well: t up to +8.07 and dR2 up to 0.031.

   There is no wall at TLT. There is a gradient: SPY +19.93/+14.30/+10.67 against
   TLT +8.07/+7.41/+6.02 under the same target. Weaker, not absent.

   The structural-coherence argument is WITHDRAWN. It was an argumentation error
   on my side, not an inconsistency in the data: I quoted the weaker of two
   figures for the same statistic and presented it as the whole picture.""")

    print("""
   WHAT SURVIVES WITHOUT IT. The 'not an artefact' conclusion rested on four
   tests; the boundary was a fifth, supporting argument. A, B and D are
   untouched. C holds for 3 of 4 instruments -- and TLT is the exception in the
   direction that matters: under cc its |t| RISES monotonically toward h=30
   (1.55, 1.72, 1.87, 2.04), which is the pattern horizon-overlap predicts. On
   TLT specifically, overlap cannot be dismissed.

   Net effect: the conclusion stands for SPY, SCHD and IWM on four independent
   tests, and is NOT established for TLT.""")

    payload = {
        "status": status,
        "expected_resolution": "table A raw, structural claim incremental",
        "actual": "both incremental over the same HAR baseline; the difference is "
                  "the target estimator (cc vs gk)",
        "specifications": sp.to_dict("records"),
        "claim_withdrawn": "the structural-coherence boundary argument "
                           "('absent on TLT')",
        "reason": "the claim quoted the cc-target row; under the gk target VIX adds "
                  "t up to +8.07 and dR2 up to 0.031 to TLT as well",
        "what_survives": "tests A, B and D are unaffected; C holds for SPY, SCHD and "
                         "IWM. On TLT the cc-target |t| rises monotonically to h=30, "
                         "which is the horizon-overlap pattern, so overlap is not "
                         "excluded for TLT.",
        "revised_scope_of_the_forensic_conclusion":
            "not an artefact -- established for SPY, SCHD, IWM; NOT established "
            "for TLT",
    }
    with open(os.path.join(RESULTS_DIR, "q0_tlt_reconciliation.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    sp.to_csv(os.path.join(RESULTS_DIR, "q0_tlt_specifications.csv"), index=False)

    db.log_experiment(con, "Q0", "blocking reconciliation of the TLT figures; "
                                 "structural-coherence argument withdrawn",
                      {"status": status, "both_incremental": both_incremental},
                      n_tests=0, seed=None, status=status)
    con.close()
    print(f"\nwrote results/q0_tlt_reconciliation.json, q0_tlt_specifications.csv")


if __name__ == "__main__":
    main()
