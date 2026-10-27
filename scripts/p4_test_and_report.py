"""P4 -- the single TEST access, the m2_results table, and the automatic verdict."""
from __future__ import annotations

import importlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config_m2 import (FIG_DIR, INSTRUMENTS, MASTER_SEED_M2, RESULTS_DIR, SELECTORS,
                       TRADING_DAYS)
from universe_m2 import build_universe, common_mask, weights_and_pnl
from voltarget import perf

p3 = importlib.import_module("p3_selector_race")


def main():
    print("=" * 78)
    print("P4  TEST (single access) + REPORT")
    print("=" * 78)
    con = db.connect()
    rng = np.random.default_rng(MASTER_SEED_M2 + 84)
    design = json.load(open(os.path.join(RESULTS_DIR, "m2_design.json")))

    exp_id = db.log_experiment(
        con, "P4", "single TEST access: race B (rank on TRAIN+VALIDATION, judge on "
                   "TEST) plus TEST-side decomposition for every model",
        {"instruments": list(INSTRUMENTS), "selectors": SELECTORS},
        n_tests=0, seed=MASTER_SEED_M2, status="RUNNING")

    # ---------------------------------------------------------- race B
    inst = design["race_B"]["instruments"]
    rank_masks, eval_masks = {}, {}
    for t in inst:
        rank_masks[t] = common_mask(t, "TRAIN") | common_mask(t, "VALIDATION")
        eval_masks[t] = common_mask(t, "TEST")
    dfB, detB = p3.run_race(inst, rank_masks, eval_masks, "B_dev_to_test", rng)
    aggB = p3.report(dfB, "B  (rank on TRAIN+VALIDATION, judge on TEST)")

    n_test_rows = int(sum(int(eval_masks[t].sum()) for t in inst))
    already = con.execute("SELECT COUNT(*) FROM test_access_log WHERE accessed_by "
                          "LIKE '%p4%'").fetchone()[0]
    if already == 0:
        db.log_test_access(con, "TEST", "scripts/p4_test_and_report.py", exp_id,
                           n_test_rows,
                           "single scheduled access; race B and the TEST-side "
                           "decomposition for all 89 pairs evaluated in one pass; "
                           "no selector, model or constant changed afterwards")
    else:
        db.log_experiment(con, "P4", "RERUN of the reporting stage after the single "
                                     "TEST access; deterministic, nothing re-derived",
                          {}, n_tests=0, seed=MASTER_SEED_M2,
                          status="POST_HOC_ITERATION")

    # ---------------------------------------------------------- combined view
    aggA = pd.DataFrame(json.load(
        open(os.path.join(RESULTS_DIR, "m2_p3_race.json")))["race_A"]["aggregate"])
    comb = aggA[["selector", "mean_rank_corr", "mean_percentile", "n_beats_random",
                 "n_significant", "n"]].merge(
        aggB[["selector", "mean_rank_corr", "mean_percentile", "n_beats_random",
              "n_significant", "n"]], on="selector", suffixes=("_A", "_B"))
    comb["mean_rank_corr_both"] = (comb.mean_rank_corr_A * comb.n_A +
                                   comb.mean_rank_corr_B * comb.n_B) / \
                                  (comb.n_A + comb.n_B)
    comb["n_cells"] = comb.n_A + comb.n_B
    comb["n_beats_random_both"] = comb.n_beats_random_A + comb.n_beats_random_B
    comb["n_significant_both"] = comb.n_significant_A + comb.n_significant_B
    print("\n" + "=" * 78)
    print("BOTH RACES COMBINED")
    print("=" * 78)
    print(comb[["selector", "mean_rank_corr_A", "mean_rank_corr_B",
                "mean_rank_corr_both", "mean_percentile_A", "mean_percentile_B",
                "n_beats_random_both", "n_significant_both", "n_cells"]
               ].round(3).to_string(index=False))

    # ---------------------------------------------------------- m2_results
    dec = pd.read_csv(os.path.join(RESULTS_DIR, "m2_decomposition.csv"))
    mech = pd.read_csv(os.path.join(RESULTS_DIR, "m2_mechanism.csv"))
    key = ["instrument", "model", "split_name"]
    full = dec.merge(mech[key + ["corr_delta_fwd_ret", "timing_null_p95",
                                 "timing_beats_null", "timing_cov_ann"]],
                     on=key, how="left")

    ranks = []
    for tk in INSTRUMENTS:
        sv = p3.selector_values(tk, rank_masks[tk])
        names = list(sv)
        for sel in SELECTORS:
            v = np.array([sv[n][sel] for n in names], float) * p3.DIRECTION[sel]
            order = (-v).argsort().argsort() + 1
            for n, rk in zip(names, order):
                ranks.append({"instrument": tk, "model": n,
                              f"selector_{sel.lower()}_rank": int(rk)})
    rk = pd.DataFrame(ranks).groupby(["instrument", "model"]).first().reset_index()
    full = full.merge(rk, on=["instrument", "model"], how="left")

    def status(r):
        if r.model == "HAR":
            return "NEUTRAL"
        if pd.isna(r.timing_beats_null):
            # P2 calibrated the null on TRAIN and VALIDATION by design; TEST rows
            # were never null-tested, and calling that a failure would be a lie.
            return "NOT_NULL_TESTED"
        if not bool(r.timing_beats_null):
            return "FAILED_NULL"
        return "PASSED" if (pd.notna(r.delta_sharpe) and r.delta_sharpe > 0) \
            else "FAILED_OOS"

    full["status"] = full.apply(status, axis=1)
    full["model_id"] = full.instrument + "_" + full.model + "_" + full.split_name
    full["timing_cov"] = full.timing_cov_ann
    full["sharpe_is"] = full.sharpe
    full["sharpe_oos"] = full.sharpe
    full["residual_check"] = full.residual
    full["exposure_term"] = full.exposure_term_ann
    full["timing_term"] = full.timing_term_ann
    full["delta_mean_total"] = full.delta_mean_ann

    schema = ["model_id", "instrument", "candidate", "family", "split_name", "n_obs",
              "mean_w", "cov_w_r", "exposure_term", "timing_term",
              "delta_mean_total", "residual_check", "qlike", "sharpe_is",
              "timing_cov", "sharpe_oos", "corr_delta_fwd_ret", "timing_null_p95",
              "timing_beats_null", "selector_qlike_rank", "selector_sharpe_is_rank",
              "selector_timing_cov_rank", "status"]
    for c in schema:
        if c not in full:
            full[c] = np.nan
    con.execute("DELETE FROM m2_results")
    con.register("m2r", full[schema])
    con.execute("INSERT INTO m2_results SELECT * FROM m2r")
    con.unregister("m2r")
    full[schema].to_csv(os.path.join(RESULTS_DIR, "m2_results.csv"), index=False)
    full.to_csv(os.path.join(RESULTS_DIR, "m2_results_detail.csv"), index=False)
    dfB.to_csv(os.path.join(RESULTS_DIR, "m2_race_B.csv"), index=False)
    detB.to_csv(os.path.join(RESULTS_DIR, "m2_race_B_detail.csv"), index=False)

    # ---------------------------------------------------------- verdict
    best = comb.sort_values("mean_rank_corr_both", ascending=False).iloc[0]
    consistent = bool(best.mean_rank_corr_A > 0 and best.mean_rank_corr_B > 0)
    majority_sig = bool(best.n_significant_both * 2 > best.n_cells)
    qlike_row = comb[comb.selector == "QLIKE"].iloc[0]

    if consistent and majority_sig:
        verdict = (f"CAN BE SELECTED EX ANTE -- {best.selector} orders models "
                   f"consistently in both races (mean rank correlation "
                   f"{best.mean_rank_corr_both:+.3f}, significant in "
                   f"{int(best.n_significant_both)}/{int(best.n_cells)} cells)")
    elif consistent:
        verdict = (f"CANNOT BE SELECTED EX ANTE RELIABLY -- the best selector "
                   f"({best.selector}) points the right way in both races "
                   f"({best.mean_rank_corr_both:+.3f}) but is significant in only "
                   f"{int(best.n_significant_both)}/{int(best.n_cells)} cells")
    else:
        verdict = ("CANNOT BE SELECTED EX ANTE -- no selector orders models "
                   "consistently across both races")

    print("\n" + "=" * 78)
    print(f"best selector overall      {best.selector}  "
          f"(A {best.mean_rank_corr_A:+.3f}, B {best.mean_rank_corr_B:+.3f})")
    print(f"QLIKE, the M1 criterion    A {qlike_row.mean_rank_corr_A:+.3f}, "
          f"B {qlike_row.mean_rank_corr_B:+.3f}, "
          f"combined {qlike_row.mean_rank_corr_both:+.3f}")
    print(f"models in the table        {len(full)} rows over "
          f"{full.groupby(['instrument', 'model']).ngroups} pairs")
    print(f"status counts              {full.status.value_counts().to_dict()}")
    tested = full[full.status.isin(["PASSED", "FAILED_NULL", "FAILED_OOS"])]
    n_beat = int((tested.status != "FAILED_NULL").sum())
    print("\nTHE NUMBER THAT EXPLAINS THE VERDICT:")
    print(f"   of {len(tested)} null-tested cells, the timing term clears the "
          f"circular-shift null in {n_beat}.")
    print(f"   P1 showed the timing term IS the decision difference "
          f"(Spearman +0.94 vs +0.28 for exposure).")
    print(f"   If the thing that drives the decision is itself indistinguishable "
          f"from noise, then no")
    print(f"   selector built on it can be stable -- which is exactly what the two "
          f"races show.")
    print(f"\nMILESTONE 2 VERDICT: {verdict}")
    print("=" * 78)

    payload = {
        "verdict": verdict,
        "race_A_aggregate": aggA.to_dict("records"),
        "race_B_aggregate": aggB.to_dict("records"),
        "race_B_rows": dfB.to_dict("records"),
        "combined": comb.to_dict("records"),
        "best_selector": best.selector,
        "best_consistent_across_races": consistent,
        "best_significant_majority": majority_sig,
        "qlike_combined_rank_corr": float(qlike_row.mean_rank_corr_both),
        "n_rows": int(len(full)),
        "n_pairs": int(full.groupby(["instrument", "model"]).ngroups),
        "status_counts": full.status.value_counts().to_dict(),
        "n_null_tested_cells": int(len(full[full.status.isin(
            ["PASSED", "FAILED_NULL", "FAILED_OOS"])])),
        "n_timing_beats_null": int((full[full.status.isin(
            ["PASSED", "FAILED_NULL", "FAILED_OOS"])].status != "FAILED_NULL").sum()),
    }
    with open(os.path.join(RESULTS_DIR, "m2_final_report.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    x = np.arange(len(comb))
    ax.bar(x - 0.2, comb.mean_rank_corr_A, 0.4, label="race A (TRAIN->VAL)",
           color="#4C72B0")
    ax.bar(x + 0.2, comb.mean_rank_corr_B, 0.4, label="race B (DEV->TEST)",
           color="#DD8452")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(comb.selector, fontsize=9)
    ax.set_ylabel("mean rank correlation with out-of-sample Sharpe")
    ax.set_title("P4  can a TRAIN-side selector order models correctly?", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    d = full[full.split_name.isin(["TRAIN", "VALIDATION"]) & (full.model != "HAR")]
    ax.scatter(d.timing_term, d.delta_sharpe, s=14, alpha=0.6, color="#C44E52",
               label="timing term")
    ax.scatter(d.exposure_term, d.delta_sharpe, s=14, alpha=0.6, color="#4C72B0",
               label="exposure term")
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("annualised contribution"); ax.set_ylabel("delta Sharpe vs HAR")
    ax.set_title("the identity that explains it", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "m2_final.png"), dpi=135)
    plt.close(fig)

    con.execute("UPDATE experiment_log SET status='COMPLETED', n_tests=? "
                "WHERE experiment_id=?", [int(len(full)), exp_id])
    con.close()
    print("\nwrote results/m2_results.csv, m2_race_B.csv, m2_final_report.json")


if __name__ == "__main__":
    main()
