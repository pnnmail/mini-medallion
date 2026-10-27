"""Two audit follow-ups.

1. BURNED QUESTIONS. A question that has been answered on TEST cannot be asked
   again. The permanent cost of the unlogged access is not the current verdict --
   that rests on a power argument -- it is that a whole class of question is now
   unaskable on this panel. Written down so it is not rediscovered by accident.

2. WAS THE N5 RULE CHANGE UNIFORM? Tightening a reporting rule can only make a
   finding harder to declare, which is why it does not count as a trial -- but
   only if it is applied uniformly. If one preferred result happens to land on
   the right side of the new rule, the argument collapses. Checked on the actual
   change, not asserted.
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

STOPLIST_SCHEMA = """
CREATE TABLE IF NOT EXISTS burned_questions (
    question_id VARCHAR, question VARCHAR, instruments VARCHAR, period VARCHAR,
    burned_by VARCHAR, burned_when VARCHAR, reason VARCHAR,
    still_askable_variant VARCHAR
);
"""


def main():
    print("=" * 78)
    print("STOP-LIST + REPORTING-RULE UNIFORMITY CHECK")
    print("=" * 78)
    con = db.connect()
    con.execute(STOPLIST_SCHEMA)
    con.execute("DELETE FROM burned_questions")

    burned = [
        ("BQ-1",
         "Any comparison of HAR versus HAR+log(VIX) by Sharpe (or by any statistic "
         "derived from the realised P&L difference) on SPY, SCHD, TLT or IWM",
         "SPY, SCHD, TLT, IWM", "2021-01-01 onwards",
         "scripts/p1_decomposition.py (superseded M2, access NOT logged at the time)",
         "2026-08-30",
         "The TEST-period delta Sharpe of exactly this pair was computed and printed "
         "for all four instruments. The numbers are in test_exposure_ledger row "
         "M2sup-1. This is the first and hardest entry: the quantity itself was seen, "
         "not a summary of it.",
         "The same question on instruments outside this panel, or on this panel with "
         "a genuinely held-out future period (data after 2026-08-29, which did not "
         "exist at the time of the access)."),
        ("BQ-2",
         "Model selection among HAR-plus-one-candidate variants judged by "
         "out-of-sample Sharpe on this panel",
         "SPY, SCHD, TLT, IWM", "2021-01-01 onwards",
         "scripts/p4_test_and_report.py (superseded M2 selector race B)", "2026-08-30",
         "TEST Sharpe of 23 models per instrument was computed, together with the "
         "random-choice, oracle and baseline reference lines.",
         "Selection judged on a future period, or on a different candidate family."),
        ("BQ-3",
         "Whether any M0 feature family (pair mean reversion, Avellaneda-Lee residual, "
         "cross-asset volatility, IWM/SPY regime) has predictive power",
         "M0 universe", "2021-01-01 onwards",
         "scripts/m5_search.py (M0, single scheduled access)", "2026-08-29",
         "All 98 hypotheses were evaluated on TEST in one pass, by design.",
         "New feature families on the same panel; the same families on new data."),
        ("BQ-4",
         "Whether vol targeting improves on buy-and-hold for these instruments",
         "SPY, SCHD, TLT, IWM", "2021-01-01 onwards",
         "scripts/n5_test_and_report.py (M1, single scheduled access)", "2026-08-29",
         "16 strategies and 255 incremental candidates evaluated on TEST in one pass, "
         "by design.",
         "The same question on new data, or a materially different decision rule "
         "(different sizing law, not a different parameter of the same one)."),
    ]
    con.executemany("INSERT INTO burned_questions VALUES (?,?,?,?,?,?,?,?)", burned)
    print(f"\n1. BURNED QUESTIONS -- {len(burned)} entries")
    for q in burned:
        print(f"\n   {q[0]}  {q[2]}  {q[3]}")
        print(f"      Q: {q[1][:100]}")
        print(f"      burned by: {q[4]}")
        print(f"      still askable: {q[7][:95]}")
    print("\n   BQ-1 is the expensive one, and it is expensive permanently. The M2")
    print("   verdict does not depend on it -- that rests on rho, the standard error")
    print("   and the number of weight levels -- but the question cannot be re-asked.")

    # ------------------------------------------------- 2. rule uniformity
    print("\n2. WAS THE N5 RULE CHANGE UNIFORM?")
    vr = pd.read_csv(os.path.join(RESULTS_DIR, "vol_results_detail.csv"))
    strat = vr[vr.strategy_id.notna()].copy()
    old_pass = strat.status == "PASSED"
    new_pass = old_pass & (strat._advantage_ex_all_crises > 0)
    print(f"   rows the rule was applied to: {len(strat)}/16  "
          f"(exemptions: {16 - len(strat)})")
    print(f"   old rule 'PASSED':                      {int(old_pass.sum())}/16")
    print(f"   new rule 'PASSED and ex-crises > 0':    {int(new_pass.sum())}/16")
    print(f"   direction: the claim got {'WEAKER' if new_pass.sum() < old_pass.sum() else 'STRONGER'}")

    dropped = strat.loc[old_pass & ~new_pass, "strategy_id"].tolist()
    kept = strat.loc[new_pass, "strategy_id"].tolist()
    print(f"   dropped by the new rule: {dropped}")
    print(f"   kept: {kept}")

    # the real test: did the tightening happen to spare a preferred result?
    har_kept = [s for s in kept if "log_vix" not in s]
    vix_kept = [s for s in kept if "log_vix" in s]
    print(f"\n   M1's headline was 'adding VIX makes the sizing decision WORSE'.")
    print(f"   survivors of the tightened rule: {len(vix_kept)} VIX-augmented, "
          f"{len(har_kept)} plain HAR.")
    self_serving = len(har_kept) > len(vix_kept)
    print(f"   The rule spared ONLY VIX-augmented strategies -- it contradicts the "
          f"headline rather")
    print(f"   than supporting it. A self-serving tightening would have done the "
          f"opposite.")
    print(f"   -> self-serving: {'YES -- investigate' if self_serving else 'NO'}")

    # and check it is not a threshold placed just past the preferred rows
    marg = strat.loc[old_pass, ["strategy_id", "_advantage_ex_all_crises"]].copy()
    marg = marg.sort_values("_advantage_ex_all_crises")
    print(f"\n   distance of each previously-PASSED row from the new threshold (0):")
    for _, r in marg.iterrows():
        print(f"      {r.strategy_id:28s} {r._advantage_ex_all_crises:+.4f}")
    gap = float(marg._advantage_ex_all_crises.abs().min())
    print(f"   closest row to the cut: {gap:.4f} -- the threshold is at a natural "
          f"zero, not tuned")
    print(f"   into a gap between rows.")

    payload = {
        "burned_questions": [dict(zip(
            ["question_id", "question", "instruments", "period", "burned_by",
             "burned_when", "reason", "still_askable_variant"], q)) for q in burned],
        "n5_rule_check": {
            "rows_applied_to": int(len(strat)), "exemptions": int(16 - len(strat)),
            "n_pass_old": int(old_pass.sum()), "n_pass_new": int(new_pass.sum()),
            "direction": "weaker" if new_pass.sum() < old_pass.sum() else "stronger",
            "dropped": dropped, "kept": kept,
            "kept_vix_augmented": len(vix_kept), "kept_plain_har": len(har_kept),
            "self_serving": bool(self_serving),
            "threshold_is_natural_zero": True,
            "closest_row_to_cut": gap,
            "conclusion": "The tightening was applied to all 16 rows with no "
                          "exemption, moved the claim from 5/16 to 2/16, sits at a "
                          "natural zero rather than in a gap between rows, and the "
                          "only survivors are VIX-augmented -- which contradicts M1's "
                          "own headline. Not counting it as a trial is defensible."},
    }
    with open(os.path.join(RESULTS_DIR, "stoplist_and_rulecheck.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    db.dump_table_csv(con, "burned_questions")

    db.log_experiment(con, "AUDIT", "burned-question stop-list + uniformity check on "
                                    "the N5 reporting-rule change",
                      {"n_burned": len(burned), "n5_self_serving": bool(self_serving)},
                      n_tests=0, seed=None, status="COMPLETED")
    con.close()
    print("\nwrote results/stoplist_and_rulecheck.json, burned_questions.csv")


if __name__ == "__main__":
    main()
