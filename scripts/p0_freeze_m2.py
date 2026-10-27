"""P0 -- freeze the M2 universe and declare the sample constraint before any result."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config_m2 import (BAND, CAP, HORIZON, INSTRUMENTS, RESULTS_DIR, SELECTORS,
                       SIGMA_TARGET, SPEC_VERSION_M2)
from universe_m2 import build_universe, chosen_estimator, common_mask

MIN_RANKING_ROWS = 500      # declared here, before any result is inspected


def main():
    print("=" * 78)
    print("P0  M2 FREEZE -- universe, sample constraint, gates")
    print("=" * 78)
    con = db.connect()
    est = chosen_estimator()
    print(f"inherited from M1 unchanged: estimator={est}  sigma*={SIGMA_TARGET:.0%}  "
          f"cap={CAP}  band=+-{BAND:.0%}  horizon={HORIZON}d")

    rows, uni = [], {}
    for tk in INSTRUMENTS:
        _, _, models, fams, _ = build_universe(tk)
        counts = {s: int(common_mask(tk, s).sum())
                  for s in ["TRAIN", "VALIDATION", "TEST"]}
        uni[tk] = {"n_models": len(models), "models": list(models),
                   "families": fams, "rows": counts}
        rows.append({"instrument": tk, "n_models": len(models), **counts})
    df = pd.DataFrame(rows)
    print("\nmodel universe (all M1 candidates, NOT only the 30 that survived M1):")
    print(df.to_string(index=False))

    eligible_A = [t for t in INSTRUMENTS if uni[t]["rows"]["TRAIN"] >= MIN_RANKING_ROWS]
    excluded_A = [t for t in INSTRUMENTS if t not in eligible_A]
    print(f"\nSAMPLE CONSTRAINT (declared now, before any result):")
    print(f"   a selector needs at least {MIN_RANKING_ROWS} rows to rank on.")
    print(f"   Under the common mask every model in the universe must have a finite")
    print(f"   forecast, and the HAR expanding window alone needs 750 observations")
    print(f"   before it emits one. SCHD only starts in 2011-10.")
    for t in INSTRUMENTS:
        print(f"   {t:5s} TRAIN rows under the common mask: "
              f"{uni[t]['rows']['TRAIN']:5d}"
              f"{'   -> ELIGIBLE' if t in eligible_A else '   -> NOT ELIGIBLE for race A'}")

    design = {
        "spec_version": SPEC_VERSION_M2,
        "estimator": est,
        "inherited_unchanged": {"sigma_target": SIGMA_TARGET, "cap": CAP,
                                "band": BAND, "horizon": HORIZON},
        "universe": uni,
        "n_pairs_total": int(sum(u["n_models"] for u in uni.values())),
        "min_ranking_rows": MIN_RANKING_ROWS,
        "race_A": {"rank_on": "TRAIN", "evaluate_on": "VALIDATION",
                   "instruments": eligible_A, "excluded": excluded_A,
                   "reason_excluded": "fewer than 500 usable TRAIN rows under the "
                                      "common mask; SCHD's history starts 2011-10 and "
                                      "the HAR expanding window consumes 750 "
                                      "observations before its first forecast"},
        "race_B": {"rank_on": "TRAIN+VALIDATION", "evaluate_on": "TEST",
                   "instruments": list(INSTRUMENTS),
                   "note": "declared in P0, executed inside the single TEST access; "
                           "this is what lets SCHD participate at all"},
        "selectors": SELECTORS,
        "common_mask_rule": "all models in an instrument's universe must have a finite "
                            "forecast on a row for that row to enter any comparison. "
                            "Models ranked on different samples cannot be ranked "
                            "against each other.",
    }
    with open(os.path.join(RESULTS_DIR, "m2_design.json"), "w") as fh:
        json.dump(design, fh, indent=2, default=str)

    db.log_experiment(con, "P0", "M2 universe frozen; sample constraint and two-race "
                                 "design declared before any result",
                      {"n_pairs": design["n_pairs_total"], "selectors": SELECTORS,
                       "race_A_instruments": eligible_A,
                       "race_B_instruments": list(INSTRUMENTS)},
                      n_tests=design["n_pairs_total"], seed=None, status="COMPLETED")
    con.close()
    print(f"\nrace A: rank on TRAIN -> evaluate on VALIDATION   {eligible_A}")
    print(f"race B: rank on TRAIN+VALIDATION -> evaluate on TEST   {list(INSTRUMENTS)}")
    print(f"\n{design['n_pairs_total']} (instrument, model) pairs total")
    print("wrote results/m2_design.json")


if __name__ == "__main__":
    main()
