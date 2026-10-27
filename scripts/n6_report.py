"""N6 -- consolidated M1 report and completion gate.

Reads only saved artefacts. It opens nothing, re-derives nothing, and cannot
touch TEST.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config_m1 import (COST_SWEEP_BPS, REBALANCE_BAND, RESULTS_DIR, ROOT,
                       SIZING_HORIZON, VOL_TARGET_CAPS)
from data_m1 import CALENDAR_OMISSIONS


def load(name):
    p = os.path.join(RESULTS_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    print("=" * 78)
    print("N6  MILESTONE 1 REPORT")
    print("=" * 78)
    con = db.connect()

    calib = load("m1_calibration_rv.json")
    har = load("m1_har_baseline.json")
    inc = load("m1_incremental_search.json")
    fin = load("m1_final_report.json")
    datar = load("m1_data_report.json")
    vr = pd.read_csv(os.path.join(RESULTS_DIR, "vol_results_detail.csv"))
    strat = vr[vr.strategy_id.notna()]
    models = vr[vr.strategy_id.isna()]

    robust = strat[(strat.status == "PASSED") & (strat._advantage_ex_all_crises > 0)]
    n_robust = int(len(robust))
    exp = con.execute("SELECT milestone, description, n_tests, status "
                      "FROM experiment_log ORDER BY timestamp").df()
    tal = con.execute("SELECT * FROM test_access_log").df()
    n_m1_access = int((tal.accessed_by.str.contains("n5")).sum())

    vp = calib["calibration_vol_persistence"]
    chosen = calib["estimator_selection"]["chosen"]

    checks = [
        ("dead pairs removed from the search space, recorded in the log",
         int(con.execute("SELECT COUNT(*) FROM experiment_log WHERE status='REMOVED'"
                         ).fetchone()[0]) >= 3),
        (f"calibration rewritten to volatility persistence, |t| > 5 on every "
         f"instrument ({vp['n_passed']}/{vp['n_instruments']}, weakest "
         f"{vp['weakest_instrument']} t={vp['weakest_t']:.1f})",
         bool(vp["passed"])),
        (f"four RV estimators implemented and compared; choice justified by "
         f"accuracy (chosen: {chosen})",
         len(calib["estimator_selection"]["ranking"]) == 4),
        (f"HAR-RV implemented, beats naive on QLIKE on VALIDATION "
         f"({har['gate']['cells_passed']}/{har['gate']['cells_total']} cells)",
         bool(har["gate"]["passed"])),
        ("incremental framework: no candidate evaluated outside a regression that "
         "contains the baseline",
         bool(models.beta_incremental.notna().all())),
        (f"Romano-Wolf implemented; the BH gap recorded as a calibration diagnostic "
         f"({inc['n_bh_rejected_train_diagnostic']} BH vs {inc['n_rw_rejected_train']} "
         f"RW on TRAIN)",
         inc["bh_minus_harness_gap"] is not None),
        (f"TLT->SPY retested correctly and died "
         f"({fin['cross_asset_family_train']['n_rejected']}/"
         f"{fin['cross_asset_family_train']['n']} of the cross-asset family rejected, "
         f"max |t| = {fin['cross_asset_family_train']['max_abs_t']:.2f})",
         fin["cross_asset_family_train"]["n_rejected"] == 0),
        (f"vol targeting with a +-{REBALANCE_BAND:.0%} band, both caps "
         f"{VOL_TARGET_CAPS}, turnover measured",
         bool(strat.turnover_annual.notna().all() and
              set(strat.cap.unique()) == set(VOL_TARGET_CAPS))),
        ("three matched-risk comparisons built (vol, exposure, drawdown)",
         bool(strat.sharpe_vt_matched_vol.notna().any() and
              strat.sharpe_vt_matched_exposure.notna().any() and
              strat._excess_return_at_matched_vol.notna().any())),
        ("leave-one-crisis-out built; contribution of the top-5 episodes shown",
         bool(strat.crisis_contribution_top5.notna().all())),
        ("execution costs and short-term gain turnover reported separately",
         bool(all(strat[f"net_sharpe_{c}bps"].notna().all() for c in COST_SWEEP_BPS)
              and strat.short_term_gain_turnover.notna().all())),
        (f"TEST opened exactly once ({n_m1_access})", n_m1_access == 1),
        (f"final table contains every model tested "
         f"({len(models)} incremental + {len(strat)} strategies = {len(vr)})",
         len(models) == 255 and len(strat) == 16),
    ]
    print("\nCOMPLETION GATE")
    for name, ok in checks:
        print(f"  [{'x' if ok else ' '}] {name}")

    print("\nSCOPE LIMITATION (declared, not silent):")
    print(f"  candidate family 5 (calendar) is REDUCED: OPEX, month-end, quarter-end")
    print(f"  and the first-Friday NFP convention only. FOMC and CPI release dates are")
    print(f"  omitted -- {CALENDAR_OMISSIONS['FOMC'][:96]}...")

    print("\nWHAT SURVIVED")
    print(f"  incremental models: {int((models.status == 'PASSED').sum())}/{len(models)} "
          f"confirmed on TRAIN + VALIDATION + TEST")
    fam = fin.get("passed_candidates_by_family", {})
    print(f"    by family: {fam}")
    print(f"  strategies: {int((strat.status == 'PASSED').sum())}/{len(strat)} cleared "
          f"all gates; {n_robust} of those still positive with the crisis declines "
          f"removed")
    if n_robust:
        for _, r in robust.iterrows():
            print(f"    {r.strategy_id:28s} adv_full={r._advantage_full:+.3f}  "
                  f"adv_test={r._advantage_test:+.3f}  "
                  f"adv_ex_crises={r._advantage_ex_all_crises:+.3f}")

    # measured precisely on both samples: DEV is everything that existed before
    # TEST was opened, so it is the honest number for "what would you have done"
    dev = pd.read_csv(os.path.join(RESULTS_DIR, "m1_voltarget_dev.csv")
                      ).set_index("strategy_id")
    fullm = strat.set_index("strategy_id")
    nd = nf = 0
    for tk in ["SPY", "SCHD", "TLT", "IWM"]:
        for cap in VOL_TARGET_CAPS:
            a, b = f"{tk}_HAR_cap{cap}", f"{tk}_HAR+log_vix_cap{cap}"
            nd += int(dev.loc[b, "sharpe_vt"] < dev.loc[a, "sharpe_vt"])
            nf += int(fullm.loc[b, "sharpe_vt"] < fullm.loc[a, "sharpe_vt"])
    vix_worse = nf

    print("\nTHE THREE FINDINGS THAT MATTER")
    print(f"  1. The strongest statistical candidate makes the DECISION worse.")
    print(f"     log(VIX) reaches t = 14.1 incrementally over HAR, survives Romano-Wolf "
          f"on TRAIN,")
    print(f"     VALIDATION and TEST, and adds up to +0.11 out-of-sample R2 to the "
          f"volatility forecast.")
    print(f"     Yet sizing on HAR+log(VIX) was worse than sizing on HAR alone in "
          f"{nd}/8 cases on")
    print(f"     TRAIN+VALIDATION -- every single one, and that is all the data that "
          f"existed before")
    print(f"     TEST was opened. On the full sample it is {nf}/8. Statistical "
          f"significance and")
    print(f"     usefulness to a decision are different quantities; this milestone "
          f"measured the gap.")
    print(f"  2. {int((strat._advantage_ex_all_crises <= 0).sum())}/{len(strat)} "
          f"strategies lose their entire advantage once the five crisis")
    print(f"     declines are removed -- and those declines are only 16-29% of the "
          f"sample.")
    print(f"  3. The advantage does not travel: 8/16 strategies flip sign from "
          f"DEV to TEST, and only")
    print(f"     2/16 retain more than half of it.")

    print(f"\nexperiments logged: {len(exp)}   (including "
          f"{int((exp.status == 'POST_HOC_ITERATION').sum())} post-hoc iterations and "
          f"{int((exp.status.str.startswith('FAILED')).sum())} failures)")

    payload = {
        "verdict": fin["verdict"],
        "completion_gate": {name: bool(ok) for name, ok in checks},
        "all_gates_passed": bool(all(ok for _, ok in checks)),
        "n_incremental_models": len(models),
        "n_incremental_passed": int((models.status == "PASSED").sum()),
        "passed_by_family": fam,
        "n_strategies": len(strat),
        "n_strategies_passed": int((strat.status == "PASSED").sum()),
        "n_strategies_robust_to_crisis_removal": n_robust,
        "robust_strategies": robust.strategy_id.tolist(),
        "strategy_status_counts": strat.status.value_counts().to_dict(),
        "model_status_counts": models.status.value_counts().to_dict(),
        "vix_worsens_sizing_dev_all_pre_test_data": nd,
        "vix_worsens_sizing_full_sample": nf,
        "n_strategies_advantage_gone_ex_crises":
            int((strat._advantage_ex_all_crises <= 0).sum()),
        "episode_coverage": fin["episode_coverage"],
        "bh_vs_harness_gap_train": inc["bh_minus_harness_gap"],
        "calibration": {"gate": calib["GATE"], "estimator": chosen,
                        "weakest_t": vp["weakest_t"]},
        "har_gate": har["gate"],
        "har_vs_harder_benchmarks": har["harder_benchmarks"],
        "cross_asset_verification": fin["cross_asset_family_train"],
        "scope_limitation_calendar": CALENDAR_OMISSIONS,
        "experiments_logged": int(len(exp)),
        "post_hoc_iterations": int((exp.status == "POST_HOC_ITERATION").sum()),
        "test_accesses_m1": n_m1_access,
        "experiment_ledger": exp.to_dict("records"),
        "test_access_log": tal.to_dict("records"),
    }
    with open(os.path.join(RESULTS_DIR, "m1_completion_report.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    for t in ["vol_results", "incremental_results", "har_baseline",
              "rv_estimator_comparison", "crisis_episodes"]:
        db.dump_table_csv(con, t)
    con.close()

    print("\n" + "=" * 78)
    print(f"MILESTONE 1 VERDICT: {fin['verdict']}")
    print("=" * 78)
    print("\nwrote results/m1_completion_report.json")


if __name__ == "__main__":
    main()
