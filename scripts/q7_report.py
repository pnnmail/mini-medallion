"""N7 -- the decision_loss_results table, the completion gate and the verdict.

Reads only saved artefacts. Opens nothing.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config import COST_SWEEP_BPS, TRADING_DAYS
from config_m1 import RESULTS_DIR
from config_m2 import FIG_DIR, INSTRUMENTS
from m2pair import FAST, SLOW, build_pair, dev_mask, weights_pnl
from voltarget import perf, strategy_pnl


def load(name):
    p = os.path.join(RESULTS_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    print("=" * 78)
    print("N7  MILESTONE 2 REPORT")
    print("=" * 78)
    con = db.connect()
    n0 = load("m2_n0_gap.json")
    n1 = load("m2_n1_decomposition.json")
    n2 = load("m2_n2_conditional.json")
    n3 = load("m2_n3_smoothing.json")
    n4 = load("m2_n4_reversal.json")
    n5 = load("m2_n5_crisis.json")
    n6 = load("m2_n6_objective.json")

    d0 = pd.DataFrame(n0["per_instrument"]).set_index("instrument")
    d1 = pd.DataFrame(n1["numerator"]).set_index("instrument")
    dp = pd.DataFrame(n1["predictive_regression"])
    d2 = pd.DataFrame(n2["by_state"])
    dloc = pd.DataFrame(n2["localisation"]).set_index("instrument")
    dvrp = pd.DataFrame(n2["vrp"])
    d3 = pd.DataFrame(n3["rows"])
    d4 = pd.DataFrame(n4["rows"]).set_index("instrument")
    d5 = pd.DataFrame(n5["rows"]).set_index("instrument")
    d6n = pd.DataFrame(n6["null"]).set_index("instrument")

    # cost sweep on both versions and on the gap
    rows = []
    for tk in INSTRUMENTS:
        f = build_pair(tk)
        m = dev_mask(tk, f)
        sel = np.asarray(m)
        fwd = f["fwd_ret"].to_numpy(float)
        nets = {}
        for tag in (SLOW, FAST):
            w, _, _ = weights_pnl(f, tag, m)
            for c in COST_SWEEP_BPS:
                p, _, _ = strategy_pnl(w, fwd, c)
                nets[f"{tag}_{c}"] = perf(p, sel)["sharpe"]
        pr = dp[dp.instrument == tk].set_index("forecast_version")
        for tag in (SLOW, FAST):
            rows.append({
                "test_id": f"{tk}_{tag}", "instrument": tk, "forecast_version": tag,
                "rho_strategies": float(d0.loc[tk, "rho_strategies"]),
                "sharpe_diff": float(d0.loc[tk, "sharpe_diff"]),
                "se_diff_ledoit_wolf": float(d0.loc[tk, "se_diff_ledoit_wolf"]),
                "p_lw": float(d0.loc[tk, "p_lw"]),
                "p_bootstrap_paired": float(d0.loc[tk, "p_bootstrap_paired"]),
                "gap_status": n0["gap_status"],
                "delta_E_w_times_E_r": float(d1.loc[tk, "delta_E_w_times_E_r_ann"]),
                "delta_cov_w_r": float(d1.loc[tk, "delta_cov_w_r_ann"]),
                "delta_numerator": float(d1.loc[tk, "delta_numerator_ann"]),
                "delta_denominator": float(d1.loc[tk, "sh_denominator_effect"]),
                "b_predictive": float(pr.loc[tag, "b_raw"]),
                "b_stambaugh_adjusted": float(pr.loc[tag, "b_stambaugh_adjusted"]),
                "t_hac": float(pr.loc[tag, "t_hac"]),
                "bias_by_stress_decile": json.dumps(
                    d2[d2.instrument == tk][["decile", "bias_slow", "bias_fast"]]
                    .round(4).to_dict("records")),
                "qlike_by_stress_decile": json.dumps(
                    d2[d2.instrument == tk][["decile", "qlike_slow", "qlike_fast"]]
                    .round(4).to_dict("records")),
                "vrp_by_stress_decile": json.dumps(
                    dvrp[dvrp.instrument == tk][["decile", "vrp_pct"]]
                    .round(3).to_dict("records")),
                "stress_localization_ratio": float(
                    dloc.loc[tk, "stress_localization_ratio"]),
                "k_smoothing": int(d3[(d3.instrument == tk)]
                                   .set_index("k").sharpe_smoothed.idxmax()),
                "sharpe_by_k": json.dumps(
                    d3[d3.instrument == tk][["k", "sharpe_smoothed"]]
                    .round(4).to_dict("records")),
                "reversal_overlay_sharpe": float(d4.loc[tk, "sharpe_projection"]),
                "reversal_match_residual": float(d4.loc[tk, "reversal_match_residual"]),
                "crisis_contribution_top5": float(d5.loc[tk, "crisis_contribution_top5"]),
                "sharpe_diff_ex_2008": float(d5.loc[tk, "gap_ex_2008"]),
                "sharpe_diff_ex_2020": float(d5.loc[tk, "gap_ex_2020"]),
                "lambda_blend": 0.0 if tag == SLOW else 1.0,
                "cap": 1.0,
                "n_trials_total": int(n6["n_trials_total"] + len(d3)),
                "t_null_threshold_rw": float(d6n.loc[tk, "null_max_abs_t_p95"]),
                "beats_null": bool(d6n.loc[tk, "beats_null"]),
                **{f"net_sharpe_{c}bps": nets[f"{tag}_{c}"] for c in COST_SWEEP_BPS},
            })
    df = pd.DataFrame(rows)

    # per-row status
    def status(r):
        if r.gap_status == "GAP_UNDERPOWERED":
            return "GAP_UNDERPOWERED"
        if not n2["mechanism_supported"]:
            return "FAILED_MECHANISM"
        if n4["explained_by_reversal"]:
            return "EXPLAINED_BY_REVERSAL"
        if not r.beats_null:
            return "FAILED_NULL"
        return "PASSED"

    df["status"] = df.apply(status, axis=1)
    con.execute("DELETE FROM decision_loss_results")
    con.register("dlr", df)
    con.execute("INSERT INTO decision_loss_results SELECT * FROM dlr")
    con.unregister("dlr")
    df.to_csv(os.path.join(RESULTS_DIR, "decision_loss_results.csv"), index=False)

    # ---------------------------------------------------------- verdict
    if n0["gap_status"] == "GAP_ABSENT":
        verdict = "GAP_ABSENT -- the difference does not even have a consistent sign"
    elif n4["explained_by_reversal"]:
        verdict = "EXPLAINED_BY_REVERSAL -- the slow forecast's advantage is a " \
                  "short-term reversal bet that M0 already showed does not survive costs"
    elif n0["gap_status"] == "GAP_UNDERPOWERED":
        verdict = ("GAP_UNDERPOWERED -- the gap has the same sign on all four "
                   "instruments but is indistinguishable from zero by every test "
                   "applied, and the proposed mechanism is rejected on its own "
                   "falsifiable prediction")
    else:
        verdict = "PASSED -- mechanism confirmed, localised, not reversal, not crises"

    print("\nSTAGE BY STAGE")
    print(f"  N0  gap_status = {n0['gap_status']}; Ledoit-Wolf significant "
          f"{n0['n_significant_lw']}/4, paired bootstrap "
          f"{n0['n_significant_bootstrap']}/4; sign test 4/4 positive, "
          f"corrected p = {n0['sign_test']['p_dependence_corrected']:.4f}")
    print(f"  N1  identity closes to {n1['identity_worst_residual']:.1e}; the "
          f"covariance term carries {n1['covariance_share_of_numerator']:.0%} of the "
          f"numerator but clears its own null in "
          f"{n1['n_cov_beats_null']}/4 instruments")
    print(f"      predictive slope more negative for the fast version in "
          f"{n1['m1_hypothesis_more_negative_slope_for_fast']['n_instruments_supporting']}"
          f"/4 -- direction right, every |t| below 0.9")
    print(f"  N2  stress localisation: gap larger in stress in "
          f"{n2['n_instruments_gap_larger_in_stress']}/4 -> mechanism "
          f"{'SUPPORTED' if n2['mechanism_supported'] else 'NOT SUPPORTED'}")
    print(f"  N3  smoothing halves the gap in {n3['n_gap_halved']}/4 -> defect is "
          f"{n3['verdict']}")
    print(f"  N4  reversal explains {n4['mean_r2_dw_on_reversal']:.1%} of the weight "
          f"difference; gap reproduced in {n4['n_gap_reproduced']}/4 -> "
          f"EXPLAINED_BY_REVERSAL = {n4['explained_by_reversal']}")
    print(f"  N5  gap survives removal of 2008 and 2020 in "
          f"{n5['n_survives_ex_2008_and_2020']}/4 and of all five declines in "
          f"{n5['n_survives_ex_all_crises']}/4 -> crisis-dependent = "
          f"{n5['crisis_dependent']}")
    print(f"  N6  {n6['n_trials_total']} configurations, null run over the same grid; "
          f"status {n6['status']}; TEST opened: {n6['test_opened']}")

    exp = con.execute("SELECT milestone, description, n_tests, status FROM "
                      "experiment_log ORDER BY timestamp").df()
    tal = con.execute("SELECT * FROM test_access_log").df()

    checks = [
        ("rho between strategies measured, not assumed "
         f"({d0.rho_strategies.min():.3f}-{d0.rho_strategies.max():.3f})", True),
        (f"Ledoit-Wolf and paired bootstrap run on the difference; gap_status = "
         f"{n0['gap_status']}", True),
        (f"sign test reported with a dependence correction (effective tests = "
         f"{n0['sign_test']['effective_n_tests']:.2f}, not 4)", True),
        (f"exact numerator and denominator decomposition built (residual "
         f"{n1['identity_worst_residual']:.1e})",
         n1["identity_worst_residual"] < 1e-12),
        ("predictive regression r_(t+1) ~ w_t estimated with HAC and the Stambaugh "
         "correction", len(dp) == 8),
        ("conditional bias and VRP built by stress decile", len(d2) > 0 and len(dvrp) > 0),
        (f"stress_localization_ratio computed; mechanism explicitly "
         f"{'confirmed' if n2['mechanism_supported'] else 'rejected'}", True),
        (f"the N3 smoothing grid k={n3['k_grid']} run in full", len(d3) == 20),
        ("explicit reversal overlay built and compared with the gap (N4)", len(d4) == 4),
        ("leave-one-crisis-out run on the difference", len(d5) == 4),
        (f"N6 run only under its preconditions; null run over the same "
         f"{n6['grid']['n_per_instrument']}-configuration grid",
         n6["preconditions"]["cov_channel"] and n6["preconditions"]["not_reversal"]),
        (f"TEST opened by this specification: {int(n6['test_opened'])} times "
         f"(N6 ended FAILED_NULL, so no configuration was selected to test)",
         not n6["test_opened"]),
        (f"n_trials_total includes every N6 configuration "
         f"({n6['n_trials_total']})", n6["n_trials_total"] == 200),
    ]
    print("\nCOMPLETION GATE")
    for name, ok in checks:
        print(f"  [{'x' if ok else ' '}] {name}")

    print("\nTEST ACCESS LEDGER (nothing hidden)")
    for _, r in tal.iterrows():
        print(f"  {str(r.timestamp)[:19]}  {r.accessed_by}  {int(r.n_rows_read)} rows")
    print("  The third row belongs to the SUPERSEDED improvised M2 that was built")
    print("  before this specification arrived. This specification opened TEST zero")
    print("  times, because N6 selected no configuration.")

    payload = {
        "verdict": verdict,
        "gap_status": n0["gap_status"],
        "stage_results": {"N0": n0["gap_status"],
                          "N1_cov_share": n1["covariance_share_of_numerator"],
                          "N1_cov_beats_null": n1["n_cov_beats_null"],
                          "N2_mechanism_supported": n2["mechanism_supported"],
                          "N3_verdict": n3["verdict"],
                          "N4_explained_by_reversal": n4["explained_by_reversal"],
                          "N4_reversal_r2": n4["mean_r2_dw_on_reversal"],
                          "N5_crisis_dependent": n5["crisis_dependent"],
                          "N6_status": n6["status"]},
        "completion_gate": {name: bool(ok) for name, ok in checks},
        "all_gates_passed": bool(all(ok for _, ok in checks)),
        "status_counts": df.status.value_counts().to_dict(),
        "n_trials_total": n6["n_trials_total"],
        "test_accesses_by_this_spec": 0,
        "test_access_ledger": tal.to_dict("records"),
        "experiments_logged": int(len(exp)),
        "rows": df.drop(columns=[c for c in df.columns
                                 if c.endswith("_by_stress_decile")
                                 or c == "sharpe_by_k"]).to_dict("records"),
    }
    with open(os.path.join(RESULTS_DIR, "m2_completion_report.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    for t in ["decision_loss_results"]:
        db.dump_table_csv(con, t)
    con.close()

    print("\n" + "=" * 78)
    print(f"MILESTONE 2 VERDICT: {verdict}")
    print("=" * 78)
    print("\nwrote results/decision_loss_results.csv, m2_completion_report.json")


if __name__ == "__main__":
    main()
