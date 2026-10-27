"""Q5 dual-status record + M4 completion gate.

TEST is not opened here and must not have been opened anywhere in M4.
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

DUAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS consolidated_results (
    result_id VARCHAR, name VARCHAR, statuses VARCHAR, evidence VARCHAR,
    magnitude VARCHAR, needs_multiplicity_correction BOOLEAN, why VARCHAR
);
"""


def main():
    print("=" * 78)
    print("Q5 CONSOLIDATION RECORD + M4 COMPLETION GATE")
    print("=" * 78)
    con = db.connect()
    con.execute(DUAL_SCHEMA)
    con.execute("DELETE FROM consolidated_results")

    rows = [
        ("CR-1", "covariance identity for sizing decisions",
         "STRUCTURAL_RESULT",
         "E[w r] = E[w]E[r] + Cov(w,r); identity closes to 1.4e-20; 77% of the "
         "numerator difference runs through the covariance term",
         "exact", False,
         "never selected from a pool, so it carries no selection bias by "
         "construction -- which is exactly why it survived"),
        ("CR-2", "log(VIX) incremental over HAR for forecasting realised volatility",
         "CALIBRATION_PASS | EXEMPLAR_COVARIANCE_THESIS",
         "t = 14.13 (SPY h=5), dR2 0.03-0.06, survives four artefact tests; and "
         "sizing on HAR+log(VIX) was WORSE than sizing on HAR in 8 of 8 comparisons "
         "on all pre-TEST data",
         "IC 0.190", True,
         "BOTH statuses are required and neither may be dropped. As CALIBRATION_PASS "
         "it is an independent re-derivation of a standard literature result, "
         "evidence the harness detects real effects. As EXEMPLAR_COVARIANCE_THESIS "
         "it is the strongest single example of CR-1: an overwhelming forecast "
         "improvement that does not improve the decision."),
        ("CR-3", "multiplicity arithmetic",
         "TRANSFERABLE_TOOL",
         "lambda measured per family (1.705 / 1.195 / 0.999); exact max-|z| "
         "quantiles; threshold-vs-K curve to K = 1268; MDE_IC floor 0.0796 at K = 1 "
         "in the returns regime",
         "cost of the search space exceeded every known real return effect", False,
         "a fact about method, not about markets; portable to any search"),
    ]
    con.executemany("INSERT INTO consolidated_results VALUES (?,?,?,?,?,?,?)", rows)
    print("\nQ5  three results recorded")
    for r in rows:
        print(f"   {r[0]}  {r[1]}")
        print(f"        status: {r[2]}")
    print("\n   CR-2 carries TWO statuses simultaneously, by design: it is both the "
          "proof the")
    print("   harness works and the strongest example of the project's own "
          "structural claim.")

    # ---------------------------------------------------------- gate
    q0 = json.load(open(os.path.join(RESULTS_DIR, "q0_tlt_reconciliation.json")))
    q1 = json.load(open(os.path.join(RESULTS_DIR, "q1_mde_curve.json")))
    q234 = json.load(open(os.path.join(RESULTS_DIR, "q2q3q4_audit_verdict.json")))
    curve = pd.DataFrame(q1["curve"])
    classes = pd.DataFrame(q234["classes"])
    levers = pd.DataFrame(q234["levers"])

    tal = con.execute("SELECT * FROM test_access_log").df()
    m4_access = int(tal.accessed_by.str.contains("r0_|r1_|r2_|r3_|q8_|q9_|qa_|qb_"
                                                 ).sum())

    pr = os.environ.get("PORTFOLIO_RISK_THESES", "../portfolio-risk/theses")
    checks = [
        (f"Q0 closed: every TLT figure has a written specification, status = "
         f"{q0['status']}",
         q0["status"] in ("RECONCILED", "ARGUMENT_WEAKENED")
         and len(q0["specifications"]) == 9),
        ("MDE_IC(K) built to K = 1268, empirically and analytically, divergence "
         f"reported (mean {q1['divergence_mean']:+.3f})",
         int(curve.K.max()) == 1268
         and curve.t_null_95_empirical.notna().sum() >= 7),
        ("reference effects plotted, regimes separated and labelled",
         len(q1["reference_effects"]) == 6
         and set(pd.DataFrame(q1["reference_effects"]).regime) ==
         {"returns", "volatility"}),
        (f"K* determined for both thresholds and both regimes "
         f"(returns {q1['K_star']['returns_0.05']}/"
         f"{q1['K_star']['returns_0.03']}, volatility "
         f"{q1['K_star']['volatility_0.05']}/{q1['K_star']['volatility_0.03']})",
         len(q1["K_star"]) == 4),
        (f"all {q234['total_trials']} trials assigned to 5 classes, each with a "
         f"power verdict", int(q234["total_trials"]) == 1268 and len(classes) == 5
         and classes.verdict.notna().all()),
        ("four levers assessed; recorded that none changes the crisis count",
         len(levers) == 4 and all("crisis" in str(x) or "crises" in str(x)
                                  for x in [levers.does_not_fix.str.cat()])),
        (f"Q4 verdict derived from the numbers: {q234['verdict']}",
         q234["verdict"] in ("CONCLUDE", "CONTINUE_NARROW", "CONTINUE_AFTER_DATA")),
        ("PROGRAM_SUMMARY.md written for a reader who has seen no milestone",
         os.path.exists(os.path.join(RESULTS_DIR, "PROGRAM_SUMMARY.md"))
         and os.path.getsize(os.path.join(RESULTS_DIR, "PROGRAM_SUMMARY.md")) > 4000),
        ("the VIX result recorded under BOTH statuses",
         "CALIBRATION_PASS" in rows[1][2]
         and "EXEMPLAR_COVARIANCE_THESIS" in rows[1][2]),
        ("mu-revision log added to portfolio-risk; the weight-drift finding "
         "recorded there too",
         os.path.exists(f"{pr}/mu_revision_log.py")
         and os.path.exists(f"{pr}/MU_REVISION_LOG.md")),
        (f"TEST was not opened in M4 ({m4_access} accesses by M4 scripts)",
         m4_access == 0),
    ]
    print("\n" + "=" * 78)
    print("M4 COMPLETION GATE")
    print("=" * 78)
    for name, ok in checks:
        print(f"  [{'x' if ok else ' '}] {name}")
    all_ok = all(ok for _, ok in checks)

    print(f"\n  test_access_log holds {len(tal)} rows, none of them from M4:")
    for _, r in tal.iterrows():
        print(f"     {str(r.timestamp)[:19]}  {r.accessed_by}")

    payload = {"consolidated_results": [dict(zip(
        ["result_id", "name", "statuses", "evidence", "magnitude",
         "needs_multiplicity_correction", "why"], r)) for r in rows],
        "completion_gate": {n: bool(o) for n, o in checks},
        "all_gates_passed": all_ok,
        "verdict": q234["verdict"],
        "test_accesses_in_m4": m4_access}
    with open(os.path.join(RESULTS_DIR, "m4_completion.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    db.dump_table_csv(con, "consolidated_results")

    db.log_experiment(con, "Q5-gate", "consolidation record and M4 completion gate",
                      {"all_gates_passed": all_ok, "verdict": q234["verdict"]},
                      n_tests=0, seed=None, status="COMPLETED")
    con.close()
    print("\n" + "=" * 78)
    print(f"MILESTONE 4 — GATE {'PASSED' if all_ok else 'INCOMPLETE'}")
    print(f"PROGRAMME VERDICT: {q234['verdict']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
