"""N3 -- what adds information ON TOP OF the HAR baseline.

No candidate is ever evaluated as a standalone predictor.  Each is tested as
`y ~ HAR(own history) + beta*X`, H0: beta = 0, with the null calibrated by a
restricted residual block bootstrap and multiplicity handled by Romano-Wolf
stepdown within each block.

TRAIN and VALIDATION only.  TEST is opened once, later, in n5.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
import stats_engine as se
from candidates_m1 import build_candidates
from config_m1 import (BH_ALPHA_DIAGNOSTIC, BOOTSTRAP_MEAN_BLOCK, FIG_DIR,
                       FORECAST_HORIZONS, MASTER_SEED_M1, RESULTS_DIR,
                       RW_ALPHA, RW_REPLICATIONS, VOL_TARGETS)
from har import har_forecast, har_regressors, har_target
from inference_m1 import (group_candidates_by_sample, incremental_null_bootstrap,
                          romano_wolf)
from panel_m1 import instrument_ohlc, split_mask_m1
from rv_estimators import daily_variance, qlike, r2_oos

TARGET_EST = "cc"


def load_chosen():
    with open(os.path.join(RESULTS_DIR, "m1_chosen_estimator.json")) as fh:
        return json.load(fh)["estimator"]


def oos_improvement(rv_reg, rv_tgt, cand_df, horizon, mask):
    """Expanding-window forecast comparison: baseline vs baseline + candidate."""
    base = har_forecast(rv_reg, horizon, rv_target=rv_tgt)
    out = {}
    al = base["target_log"].to_numpy(float)[mask]
    av = base["target_var"].to_numpy(float)[mask]
    bl = base["pred_log"].to_numpy(float)[mask]
    bv = base["pred_var"].to_numpy(float)[mask]
    ok0 = np.isfinite(al) & np.isfinite(bl)
    mean_bench = np.full(int(ok0.sum()), float(np.nanmean(al[ok0]))) if ok0.sum() else []
    base_r2 = r2_oos(al[ok0], bl[ok0], mean_bench)
    base_q = qlike(av, bv)
    for c in cand_df.columns:
        f = har_forecast(rv_reg, horizon, extra=cand_df[[c]], rv_target=rv_tgt)
        ml = f["pred_log"].to_numpy(float)[mask]
        mv = f["pred_var"].to_numpy(float)[mask]
        ok = np.isfinite(al) & np.isfinite(ml) & np.isfinite(bl)
        out[c] = {
            "baseline_r2_oos": base_r2,
            "model_r2_oos": r2_oos(al[ok], ml[ok],
                                   np.full(int(ok.sum()), float(np.nanmean(al[ok])))),
            "delta_r2": r2_oos(al[ok], ml[ok], bl[ok]),
            "qlike_baseline": base_q, "qlike_model": qlike(av, mv),
        }
    return out


def run_split(target, horizon, split, chosen, rng, recs):
    d = instrument_ohlc(target)
    rv_reg = daily_variance(d, chosen)
    rv_tgt = daily_variance(d, TARGET_EST)
    Xb = har_regressors(rv_reg)
    y = har_target(rv_tgt, horizon)
    C, fam = build_candidates(target, chosen, horizon)
    mask = split_mask_m1(d.index, target, split)
    oos = oos_improvement(rv_reg, rv_tgt, C, horizon, mask)

    per_rep_max = []
    for block_name, ok, cols in group_candidates_by_sample(y, Xb, C, mask, fam):
        Y = y.to_numpy(float)[ok]
        XB = np.column_stack([np.ones(int(ok.sum())),
                              Xb.to_numpy(float)[ok]])
        CM = C[cols].to_numpy(float)[ok]
        B, T, P, TS = incremental_null_bootstrap(
            Y, XB, CM, horizon - 1, RW_REPLICATIONS, BOOTSTRAP_MEAN_BLOCK, rng)
        rej, adj_p, thr = romano_wolf(T, TS, RW_ALPHA)
        with np.errstate(invalid="ignore"):
            per_rep_max.append(np.nanmax(np.abs(TS), axis=1))
        for j, c in enumerate(cols):
            recs.append({
                "target_instrument": target, "horizon": horizon, "split_name": split,
                "block": block_name, "family": fam[c], "candidate": c,
                "n_obs": int(ok.sum()), "beta_incremental": B[j],
                "t_incremental": T[j], "p_analytic": P[j],
                "t_null_threshold_rw": thr[j], "p_romano_wolf": adj_p[j],
                "rw_rejected": bool(rej[j]),
                "null_max_t_p95_block": float(np.nanquantile(
                    np.nanmax(np.abs(TS), axis=1), 0.95)),
                **oos[c]})
    return per_rep_max


def main():
    print("=" * 78)
    print("N3  INCREMENTAL SEARCH ON TOP OF HAR-RV")
    print("=" * 78)
    chosen = load_chosen()
    con = db.connect()
    rng = np.random.default_rng(MASTER_SEED_M1 + 31)
    recs, pooled = [], {}

    t0 = time.time()
    for split in ["TRAIN", "VALIDATION"]:
        pooled[split] = []
        for tk in VOL_TARGETS:
            for h in FORECAST_HORIZONS:
                pooled[split] += run_split(tk, h, split, chosen, rng, recs)
        print(f"   {split:11s} done  ({time.time() - t0:.0f}s elapsed)")
    df = pd.DataFrame(recs)

    # conservative pooled FWER threshold across every family in the milestone
    global_thr = {}
    for split, mats in pooled.items():
        n = min(len(m) for m in mats)
        M = np.vstack([m[:n] for m in mats])
        global_thr[split] = float(np.nanquantile(np.nanmax(M, axis=0), 0.95))
    print(f"\nglobal pooled FWER threshold (conservative, treats families as "
          f"independent):")
    for k, v in global_thr.items():
        print(f"   {k:11s} |t| > {v:.3f}")

    # BH on the analytic p-values -- diagnostic only, never a criterion
    for split in ["TRAIN", "VALIDATION"]:
        m = df.split_name == split
        rej, adj = se.benjamini_hochberg(df.loc[m, "p_analytic"].to_numpy(float),
                                         BH_ALPHA_DIAGNOSTIC)
        df.loc[m, "bh_rejected"] = rej
        df.loc[m, "p_bh_adjusted"] = adj
    df["beats_global_threshold"] = df.apply(
        lambda r: bool(np.isfinite(r.t_incremental) and
                       abs(r.t_incremental) > global_thr[r.split_name]), axis=1)

    tr = df[df.split_name == "TRAIN"].set_index(["target_instrument", "horizon",
                                                 "candidate"])
    va = df[df.split_name == "VALIDATION"].set_index(["target_instrument", "horizon",
                                                      "candidate"])
    j = tr.index.intersection(va.index)
    status, rows = {}, []
    for k in j:
        a, b = tr.loc[k], va.loc[k]
        same_sign = (np.isfinite(a.beta_incremental) and np.isfinite(b.beta_incremental)
                     and np.sign(a.beta_incremental) == np.sign(b.beta_incremental))
        if not a.rw_rejected:
            st = "FAILED_NULL"
        elif not (b.rw_rejected and same_sign):
            st = "FAILED_OOS"
        else:
            st = "CANDIDATE_TO_TEST"
        status[k] = st
        rows.append({"target_instrument": k[0], "horizon": k[1], "candidate": k[2],
                     "family": a.family, "status": st,
                     "t_train": a.t_incremental, "t_val": b.t_incremental,
                     "rw_p_train": a.p_romano_wolf, "rw_p_val": b.p_romano_wolf,
                     "beta_train": a.beta_incremental, "beta_val": b.beta_incremental,
                     "same_sign": same_sign,
                     "delta_r2_train": a.delta_r2, "delta_r2_val": b.delta_r2,
                     "qlike_baseline_val": b.qlike_baseline,
                     "qlike_model_val": b.qlike_model})
    surv = pd.DataFrame(rows)
    df["status"] = df.apply(
        lambda r: status.get((r.target_instrument, r.horizon, r.candidate), "UNPAIRED"),
        axis=1)

    n_tests = len(j)
    print(f"\n{n_tests} incremental hypotheses (unique target x horizon x candidate)")
    print(f"   Romano-Wolf rejected on TRAIN      "
          f"{int(tr.loc[j].rw_rejected.sum())}/{n_tests}")
    print(f"   BH at a={BH_ALPHA_DIAGNOSTIC} on TRAIN (DIAGNOSTIC)  "
          f"{int(tr.loc[j].bh_rejected.sum())}/{n_tests}")
    print(f"   survive TRAIN + VALIDATION + sign   "
          f"{int((surv.status == 'CANDIDATE_TO_TEST').sum())}/{n_tests}")
    print(f"   status: {surv.status.value_counts().to_dict()}")

    gap = int(tr.loc[j].bh_rejected.sum()) - int(tr.loc[j].rw_rejected.sum())
    print(f"\n   BH-vs-harness gap on TRAIN: {gap:+d} hypotheses. Tracked as a "
          f"calibration diagnostic:")
    print(f"   BH controls FDR GIVEN correct p-values; the harness independently "
          f"measures whether\n   that condition holds. In M0 the same gap was "
          f"56 vs 1 on 98 tests.")

    print("\nby family (TRAIN):")
    fam = tr.loc[j].reset_index().groupby("family").agg(
        n=("candidate", "size"),
        max_abs_t=("t_incremental", lambda s: float(s.abs().max())),
        n_rw=("rw_rejected", "sum"), n_bh=("bh_rejected", "sum"),
        median_delta_r2=("delta_r2", "median")).reset_index()
    print(fam.to_string(index=False))

    print("\nsurvivors carried to TEST:")
    s = surv[surv.status == "CANDIDATE_TO_TEST"].sort_values(
        "t_train", key=lambda x: x.abs(), ascending=False)
    if len(s) == 0:
        print("   none")
    else:
        for _, r in s.iterrows():
            print(f"   {r.target_instrument:5s} h={r.horizon:<3d} {r.candidate:22s} "
                  f"t_tr={r.t_train:+7.2f} t_va={r.t_val:+7.2f}  "
                  f"dR2_va={r.delta_r2_val:+.4f}  {r.family}")

    con.execute("DELETE FROM incremental_results")
    ins = df.copy()
    ins["model_id"] = (ins.target_instrument + "_h" + ins.horizon.astype(str) + "_"
                       + ins.candidate + "_" + ins.split_name)
    cols = ["model_id", "target_instrument", "horizon", "family", "candidate",
            "split_name", "beta_incremental", "t_incremental", "p_analytic",
            "t_null_threshold_rw", "p_romano_wolf", "p_bh_adjusted", "rw_rejected",
            "bh_rejected", "baseline_r2_oos", "model_r2_oos", "delta_r2",
            "qlike_baseline", "qlike_model", "n_obs", "status"]
    con.register("ir", ins[cols])
    con.execute("INSERT INTO incremental_results SELECT * FROM ir")
    con.unregister("ir")
    df.to_csv(os.path.join(RESULTS_DIR, "m1_incremental_results.csv"), index=False)
    surv.to_csv(os.path.join(RESULTS_DIR, "m1_incremental_survivors.csv"), index=False)

    payload = {
        "n_hypotheses": n_tests, "rw_alpha": RW_ALPHA,
        "rw_replications": RW_REPLICATIONS,
        "global_pooled_threshold": global_thr,
        "n_rw_rejected_train": int(tr.loc[j].rw_rejected.sum()),
        "n_bh_rejected_train_diagnostic": int(tr.loc[j].bh_rejected.sum()),
        "bh_minus_harness_gap": gap,
        "status_counts": surv.status.value_counts().to_dict(),
        "by_family_train": fam.to_dict("records"),
        "survivors": s.to_dict("records"),
        "tlt_to_spy_retest": df[(df.target_instrument == "SPY") &
                                (df.candidate == "log_rv20_TLT")].to_dict("records"),
    }
    with open(os.path.join(RESULTS_DIR, "m1_incremental_search.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    db.log_experiment(con, "N3", "incremental candidate search on top of HAR-RV, "
                                 "Romano-Wolf stepdown, TRAIN + VALIDATION",
                      {"targets": VOL_TARGETS, "horizons": FORECAST_HORIZONS,
                       "rw_replications": RW_REPLICATIONS, "estimator": chosen},
                      n_tests=n_tests, seed=MASTER_SEED_M1, status="COMPLETED")
    con.close()
    print(f"\nwrote results/m1_incremental_results.csv, m1_incremental_survivors.csv, "
          f"m1_incremental_search.json   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
