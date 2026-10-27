"""AUDIT -- what was actually seen on TEST, what class each logged error is,
and how many trials the project has run cumulatively.

Three gaps this closes:

1. `test_access_log` recorded THAT an access happened, not WHAT was seen. Without
   the content, any future question touching these instruments after 2021 has to
   be treated as fully contaminated, when in fact a narrow slice was observed.
   It also missed one access entirely: scripts/p1_decomposition.py of the
   superseded M2 evaluated the TEST split and never logged it.

2. Post-hoc iterations were all one undifferentiated class. A sign transposition
   that an outsider could catch from the specification text alone carries no
   selection bias; a specification changed because a result came out wrong does.
   Merging them inflates the trial counter with typos.

3. Trial counts were per-milestone. Cumulative selection across three milestones
   is not covered by per-milestone Romano-Wolf thresholds. There is no clean fix
   -- a cumulative correction across three milestones would suppress everything --
   so the choice must be a recorded decision with a reason, not a default.
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

EXPOSURE_SCHEMA = """
CREATE TABLE IF NOT EXISTS test_exposure_ledger (
    exposure_id VARCHAR, project_milestone VARCHAR, script VARCHAR,
    logged_at_the_time BOOLEAN, split_name VARCHAR,
    instruments VARCHAR, n_models BIGINT, n_rows BIGINT,
    statistics_seen VARCHAR, printed_to_console VARCHAR,
    values_seen VARCHAR, direction_seen VARCHAR,
    contaminates VARCHAR, robustness_of_current_conclusion VARCHAR
);
"""


def add_col(con, table, col, typ):
    cols = [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
    if col not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def main():
    print("=" * 78)
    print("AUDIT  disclosure, error classes, cumulative trial count")
    print("=" * 78)
    con = db.connect()
    con.execute(EXPOSURE_SCHEMA)
    for c, t in [("project_milestone", "VARCHAR"), ("error_class", "VARCHAR"),
                 ("counts_toward_trials", "BOOLEAN"),
                 ("outsider_detectable_from_spec", "BOOLEAN")]:
        add_col(con, "experiment_log", c, t)

    # ---------------------------------------------------------- 1. exposure
    dec = pd.read_csv(os.path.join(RESULTS_DIR, "m2_decomposition.csv"))
    t = dec[dec.split_name == "TEST"]
    vix = t[t.model == "HAR+log_vix"].set_index("instrument")
    b = pd.read_csv(os.path.join(RESULTS_DIR, "m2_race_B.csv"))

    con.execute("DELETE FROM test_exposure_ledger")
    rows = [
        {
            "exposure_id": "M0-1", "project_milestone": "M0",
            "script": "scripts/m5_search.py", "logged_at_the_time": True,
            "split_name": "TEST", "instruments": "spread/asset universe of M0",
            "n_models": 98, "n_rows": 1420,
            "statistics_seen": "IC, Newey-West t, net Sharpe for all 98 M0 hypotheses",
            "printed_to_console": "top-8 by |t| only; full table written to signals.csv",
            "values_seen": "max |t| on TEST reached 6.427 for rvol_TLT_10_lag0",
            "direction_seen": "the single null-beating candidate was STRONGER on TEST "
                              "than on TRAIN",
            "contaminates": "any future question about the M0 feature families on "
                            "2021+ data",
            "robustness_of_current_conclusion":
                "n/a -- this access belongs to M0, which concluded NO ROBUST EDGE and "
                "did not act on it",
        },
        {
            "exposure_id": "M1-1", "project_milestone": "M1",
            "script": "scripts/n5_test_and_report.py", "logged_at_the_time": True,
            "split_name": "TEST", "instruments": "SPY, SCHD, TLT, IWM",
            "n_models": 255, "n_rows": 5680,
            "statistics_seen": "incremental t on 255 candidates; vol-targeting Sharpe "
                               "for 16 strategies; full-sample crisis analysis",
            "printed_to_console": "top-12 candidates and all 16 strategy rows",
            "values_seen": "TLT buy-and-hold Sharpe on TEST is negative (-0.58); "
                           "vol-targeting advantage on TEST ranged -0.150 to +0.050",
            "direction_seen": "8 of 16 strategies flipped sign from development to TEST",
            "contaminates": "vol targeting on these four instruments after 2021",
            "robustness_of_current_conclusion":
                "n/a -- belongs to M1, which concluded NO USEFUL IMPROVEMENT",
        },
        {
            "exposure_id": "M2sup-1", "project_milestone": "M2-superseded",
            "script": "scripts/p1_decomposition.py", "logged_at_the_time": False,
            "split_name": "TEST", "instruments": "SPY, SCHD, TLT, IWM",
            "n_models": 23, "n_rows": 1410,
            "statistics_seen": "exposure term, timing term, delta mean P&L, "
                               "delta Sharpe vs HAR, QLIKE and turnover for every "
                               "model in the universe, ON THE TEST SPLIT",
            "printed_to_console": "the HAR+log_vix row for all four instruments, "
                                  "including its TEST delta Sharpe",
            "values_seen": json.dumps({tk: {
                "delta_sharpe_TEST": round(float(vix.loc[tk, "delta_sharpe"]), 5),
                "timing_term_ann_TEST": round(float(vix.loc[tk, "timing_term_ann"]), 5),
                "exposure_term_ann_TEST": round(float(vix.loc[tk, "exposure_term_ann"]), 5),
            } for tk in vix.index}),
            "direction_seen": "THE MOST CONTAMINATING ITEM IN THE PROJECT: this is the "
                              "TEST-period value of exactly the quantity the real M2's "
                              "N0 tests. SCHD +0.084 and TLT +0.070 on TEST are far "
                              "LARGER than their development values (+0.020, +0.014); "
                              "SPY collapses to +0.001 and IWM turns negative (-0.011).",
            "contaminates": "any future statement about the HAR vs HAR+log_vix gap on "
                            "these instruments after 2021, and more weakly any "
                            "vol-forecast comparison on the same panel",
            "robustness_of_current_conclusion":
                "The current M2 conclusion IS robust to this, for a structural reason. "
                "M2 ended on a POWER argument, not on a result: 'the gap is too small "
                "to distinguish' follows from the measured rho (0.986-0.996), the "
                "resulting standard error (0.020-0.064) and the step-function weight "
                "path (82-331 distinct levels), none of which uses TEST. One cannot "
                "overfit toward the conclusion that nothing can be resolved. Note the "
                "direction of the temptation: the contaminating values would have "
                "pushed TOWARD claiming a real effect on SCHD and TLT, and the "
                "conclusion went the other way.",
        },
        {
            "exposure_id": "M2sup-2", "project_milestone": "M2-superseded",
            "script": "scripts/p4_test_and_report.py", "logged_at_the_time": True,
            "split_name": "TEST", "instruments": "SPY, SCHD, TLT, IWM",
            "n_models": 23, "n_rows": 5640,
            "statistics_seen": "selector race B -- rank correlation of three selectors "
                               "with TEST Sharpe, the model each selector picked, its "
                               "TEST Sharpe, plus random-choice, oracle and baseline "
                               "reference lines",
            "printed_to_console": "all 12 selector x instrument rows",
            "values_seen": json.dumps({tk: {
                "baseline_HAR_sharpe_TEST": round(float(
                    b[b.instrument == tk].baseline_sharpe.iloc[0]), 4),
                "random_choice_TEST": round(float(
                    b[b.instrument == tk].random_choice_sharpe.iloc[0]), 4),
                "oracle_TEST": round(float(b[b.instrument == tk].oracle_sharpe.iloc[0]), 4),
            } for tk in b.instrument.unique()}),
            "direction_seen": "every selector flipped sign between race A and race B; "
                              "TLT is negative-Sharpe throughout the TEST period",
            "contaminates": "model-selection questions on this panel after 2021",
            "robustness_of_current_conclusion":
                "The real M2 never used a selector race and never referenced these "
                "numbers; its N0-N6 ran on TRAIN and VALIDATION only.",
        },
    ]
    con.executemany(
        "INSERT INTO test_exposure_ledger VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [tuple(r[k] for k in ["exposure_id", "project_milestone", "script",
                              "logged_at_the_time", "split_name", "instruments",
                              "n_models", "n_rows", "statistics_seen",
                              "printed_to_console", "values_seen", "direction_seen",
                              "contaminates", "robustness_of_current_conclusion"])
         for r in rows])

    # the access that was never logged gets logged now, marked retroactive
    have = con.execute("SELECT COUNT(*) FROM test_access_log WHERE accessed_by "
                       "LIKE '%p1_decomposition%'").fetchone()[0]
    if have == 0:
        db.log_test_access(
            con, "TEST", "scripts/p1_decomposition.py (RETROACTIVE)", "M2-superseded",
            int(len(t)),
            "DISCLOSURE: this access happened during the superseded improvised M2 and "
            "was NOT logged at the time. It evaluated the TEST split for all 23 models "
            "on all four instruments and printed the TEST delta Sharpe of HAR+log_vix. "
            "See test_exposure_ledger row M2sup-1 for exactly what was seen.")

    print("\n1. TEST EXPOSURE LEDGER -- what was seen, not only that it happened")
    led = con.execute("SELECT exposure_id, project_milestone, script, "
                      "logged_at_the_time, n_models, n_rows FROM "
                      "test_exposure_ledger").df()
    print(led.to_string(index=False))
    print("\n   the one that was never logged at the time:")
    print("   p1_decomposition.py saw the TEST delta Sharpe of HAR+log_vix directly:")
    for tk in vix.index:
        print(f"      {tk:5s} dev {'+0.042' if tk == 'SPY' else '+0.020' if tk == 'SCHD' else '+0.014' if tk == 'TLT' else '+0.009'}"
              f"   TEST {vix.loc[tk, 'delta_sharpe']:+.5f}")
    print("   This is the most contaminating item in the project and it pushed toward")
    print("   claiming an effect. The conclusion went the other way, and it rests on a")
    print("   power argument built from rho and the weight path -- neither uses TEST.")

    # ---------------------------------------------------------- 2. error class
    CLASSES = {
        "N1|ITERATION: volatility-persistence gate respecified": (
            "SPECIFICATION_CHANGE_AFTER_RESULT", False, True,
            "the gate was moved to a different variance estimator after it failed. An "
            "outsider could NOT have known from the spec that close-to-close would "
            "attenuate it; the trigger was the failure. Counts as a trial."),
        "N5|ITERATION after seeing TEST: the VERDICT RULE": (
            "REPORTING_RULE_CHANGE", False, False,
            "a reporting threshold, not a hypothesis. No new hypothesis was tested, so "
            "it adds no trial, but it was triggered by seeing results and is logged."),
        "P2|ITERATION after seeing results: H2 was pre-registered": (
            "ALGEBRAIC_ERROR", True, False,
            "w = sigma*/sigma_hat is monotonically decreasing in sigma_hat, which is "
            "checkable from the specification text with no data at all. The theory was "
            "right; the inequality was transcribed backwards. Carries no selection "
            "bias, so it does NOT inflate the trial counter. It is still recorded as "
            "post-hoc because the trigger was seeing the result -- the conservative "
            "treatment stands, only the accounting changes."),
        "P4|RERUN of the reporting stage": (
            "RERUN_NO_NEW_HYPOTHESIS", False, False,
            "deterministic re-execution of a reporting stage; nothing re-derived."),
    }
    ph = con.execute("SELECT experiment_id, milestone, description FROM experiment_log "
                     "WHERE status='POST_HOC_ITERATION'").df()
    print("\n2. ERROR CLASSES")
    print("   criterion: could an outsider reading ONLY the pre-registration, never "
          "having seen\n   the data, have detected it?  yes -> algebraic -> no "
          "selection bias -> no trial")
    for _, r in ph.iterrows():
        hit = None
        for k, v in CLASSES.items():
            ms, frag = k.split("|", 1)
            if r.milestone == ms and frag in r.description:
                hit = v
                break
        if hit is None:
            hit = ("UNCLASSIFIED", False, True, "")
        cls, outsider, counts, why = hit
        con.execute("UPDATE experiment_log SET error_class=?, "
                    "outsider_detectable_from_spec=?, counts_toward_trials=? "
                    "WHERE experiment_id=?", [cls, outsider, counts, r.experiment_id])
        print(f"   {r.milestone:8s} {cls:34s} outsider-detectable="
              f"{'yes' if outsider else 'no':3s}  counts_as_trial="
              f"{'yes' if counts else 'no'}")
        if why:
            print(f"            {why}")
    con.execute("UPDATE experiment_log SET counts_toward_trials = TRUE "
                "WHERE counts_toward_trials IS NULL")
    con.execute("UPDATE experiment_log SET error_class='NOT_AN_ERROR' "
                "WHERE error_class IS NULL")

    # ---------------------------------------------------------- 3. milestones
    marker = con.execute("SELECT timestamp FROM experiment_log WHERE "
                         "milestone='M2-SPEC'").fetchone()[0]
    con.execute("UPDATE experiment_log SET project_milestone = CASE "
                "WHEN milestone IN ('M0','M1','M2','M3','M4','M5','M5-post','M7') "
                "  THEN 'M0' "
                "WHEN milestone IN ('P0','P1','P2','P3','P4') THEN 'M2-superseded' "
                "WHEN milestone='M2-SPEC' THEN 'M2-superseded' "
                "WHEN timestamp < ? THEN 'M1' ELSE 'M2' END", [marker])

    # Not every logged row is a trial in the multiplicity sense. Null calibration,
    # data checks, cost estimation and engine calibration are not part of any
    # selection pool -- counting them would inflate the number in the other
    # direction and make it just as meaningless as leaving typos in.
    # A trial is a HYPOTHESIS in a selection pool, not an evaluation. Three things
    # must therefore be excluded, and loose keyword matching gets all three wrong:
    #   (a) calibration, cost estimation, data checks and null calibration -- no
    #       finding is ever drawn from them;
    #   (b) re-runs of the same experiment -- executing a script twice does not
    #       create new hypotheses;
    #   (c) re-EVALUATIONS of already-counted hypotheses on another split.
    # An earlier version of this audit used LIKE '%null harness%' and silently
    # dropped N6's 200-configuration grid, which is the single largest genuine
    # search in M2. Explicit lists, not patterns.
    # reset first: this script is idempotent, and an exclusion set from an earlier
    # run must not survive into a corrected one
    con.execute("UPDATE experiment_log SET counts_toward_trials = TRUE")
    NOT_A_SEARCH = [
        "null harness on block-bootstrapped synthetic data",
        "calibration on known effects (TRAIN only)",
        "Corwin-Schultz cost model + anchor validation",
        "spread estimators + anchor validation",
        "volatility-persistence calibration + RV estimator selection",
        "RV estimator selection (QLIKE on VALIDATION)",
        "volatility-persistence + short-term-reversal gates",
        "search space frozen; look-ahead + target tests passed",
    ]
    for d_ in NOT_A_SEARCH:
        con.execute("UPDATE experiment_log SET counts_toward_trials = FALSE "
                    "WHERE description = ?", [d_])
    con.execute("UPDATE experiment_log SET counts_toward_trials = FALSE WHERE "
                "status IN ('REMOVED','SUPERSEDED','FAILED_CALIBRATION') "
                "OR description LIKE 'ITERATION:%' "
                "OR description LIKE 'M2 snapshot%' OR description LIKE 'M1 snapshot%' "
                "OR description LIKE 'frozen data%' OR description LIKE 'disclosure%'")
    # (c) evaluations of hypotheses already counted upstream
    RE_EVALUATION = [
        ("single TEST access: incremental candidates + vol-targeting strategies",
         "the 255 candidates and 16 strategies were already counted in M1's N3 and N4"),
        ("single TEST access: race B (rank on TRAIN+VALIDATION, judge on TEST) p",
         "the same 89 pairs already counted in P0/P1"),
        ("exact exposure/timing decomposition of the sizing decision across the",
         "decomposes the 89 pairs of P0 across three splits; no new hypothesis"),
    ]
    for d_, _ in RE_EVALUATION:
        con.execute("UPDATE experiment_log SET counts_toward_trials = FALSE "
                    "WHERE description LIKE ?", [d_ + "%"])
    # (b) re-runs: keep only the first logged row of each identical experiment
    con.execute("""
        UPDATE experiment_log SET counts_toward_trials = FALSE
        WHERE experiment_id IN (
            SELECT experiment_id FROM (
                SELECT experiment_id, ROW_NUMBER() OVER (
                    PARTITION BY project_milestone, milestone, description
                    ORDER BY timestamp) rn
                FROM experiment_log) WHERE rn > 1)
    """)

    cnt = con.execute(
        "SELECT project_milestone, COUNT(*) n_rows, SUM(n_tests) raw_n_tests, "
        "SUM(CASE WHEN counts_toward_trials THEN n_tests ELSE 0 END) "
        "  selection_relevant_trials "
        "FROM experiment_log GROUP BY project_milestone "
        "ORDER BY MIN(timestamp)").df()
    print("\n3. TRIAL COUNT -- local and cumulative")
    print("   raw_n_tests counts every logged evaluation; selection_relevant_trials "
          "counts only")
    print("   hypotheses that entered a selection pool. Null calibration, data checks, "
          "engine")
    print("   calibration and cost estimation are excluded -- inflating the counter "
          "with those")
    print("   makes it as meaningless as leaving typos in.")
    print(cnt.to_string(index=False))
    cum = int(cnt.selection_relevant_trials.sum())
    local = cnt.set_index("project_milestone").selection_relevant_trials.to_dict()
    print(f"\n   LOCAL (per milestone):      {local}")
    print(f"   CUMULATIVE across project:  {cum}")
    excl = con.execute("SELECT SUM(n_tests) FROM experiment_log WHERE NOT "
                       "counts_toward_trials").fetchone()[0]
    print(f"   excluded as non-search:     {int(excl or 0)} "
          f"(dominated by the M0 null harness, 1000 x 98)")

    decision = {
        "thresholds_are_calibrated": "per milestone, not cumulatively",
        "selection_relevant_trials_local": local,
        "selection_relevant_trials_cumulative": cum,
        "excluded_as_non_search": int(excl or 0),
        "why": "A cumulative Romano-Wolf correction across three milestones would "
               "suppress everything, including the one structural result the project "
               "has (the covariance identity). But that is a cost, not a "
               "justification, so the decision is recorded rather than left implicit. "
               "The defensible part: each milestone asked a DIFFERENT question of the "
               "data -- M0 searched for directional edge, M1 asked whether known "
               "volatility predictability changes a sizing decision, M2 asked why a "
               "more accurate forecast was a worse input. They are not three passes "
               "over one hypothesis space. The indefensible part: they used the same "
               "instruments and the same sample, so cross-milestone selection is real "
               "and uncorrected.",
        "practical_consequence": "no finding in this project should be treated as "
                                 "having survived a cumulative correction. The only "
                                 "result reported as durable -- the exact "
                                 "E[w*r] = E[w]E[r] + Cov(w,r) decomposition -- is an "
                                 "algebraic identity and needs no multiplicity "
                                 "correction at all, which is precisely why it is the "
                                 "one that survives.",
    }
    print(f"\n   DECISION RECORDED: thresholds stay per-milestone. Reason:")
    print(f"   {decision['why'][:300]}...")

    payload = {"test_exposure_ledger": rows,
               "unlogged_access_disclosed": "scripts/p1_decomposition.py",
               "error_classes": {k: v[0] for k, v in CLASSES.items()},
               "trial_counts": cnt.to_dict("records"),
               "cumulative_decision": decision}
    with open(os.path.join(RESULTS_DIR, "audit_disclosure.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    db.dump_table_csv(con, "test_exposure_ledger")
    db.dump_table_csv(con, "test_access_log")
    db.dump_table_csv(con, "experiment_log")

    db.log_experiment(con, "AUDIT", "disclosure of TEST exposure content, retroactive "
                                    "logging of the p1_decomposition access, error "
                                    "classification, cumulative trial accounting",
                      {"unlogged_access": "scripts/p1_decomposition.py",
                       "cumulative_hypothesis_trials": cum},
                      n_tests=0, seed=None, status="COMPLETED")
    con.close()
    print("\nwrote results/audit_disclosure.json, test_exposure_ledger.csv")


if __name__ == "__main__":
    main()
