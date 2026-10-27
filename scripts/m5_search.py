"""M5 -- the first and only look at real data.

Order of operations is fixed:
  TRAIN  -> full search, sign fitted, compared against the M2 null thresholds
  VALID  -> survivors carried forward with the sign FROZEN
  TEST   -> opened exactly once, for every row at once, and never again
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
from config import (BASE_COST_BPS, BH_FDR_ALPHA, BOOTSTRAP_CI_REPLICATIONS,
                    BOOTSTRAP_MEAN_BLOCK, COST_SWEEP_BPS, HORIZON_HALF_LIFE_MULTIPLE,
                    MASTER_SEED, MAX_ADMISSIBLE_HALF_LIFE, RESULTS_DIR, TRADING_DAYS)
from discovery import enumerate_search_space, evaluate_spec
from features import build_features, half_life
from panel import research_panel, split_mask
from provenance import estimator_fingerprint
from targets import build_targets


def load_null():
    p = os.path.join(RESULTS_DIR, "null_thresholds.json")
    with open(p) as fh:
        return json.load(fh)


def main():
    print("=" * 78)
    print("M5  SEARCH ON REAL DATA")
    print("=" * 78)
    con = db.connect()
    null = load_null()

    fp_now = estimator_fingerprint()["_joint"]
    fp_null = null["estimator_fingerprint"]["_joint"]
    print(f"estimator fingerprint  now={fp_now}  null={fp_null}  "
          f"{'MATCH' if fp_now == fp_null else 'MISMATCH'}")
    if fp_now != fp_null:
        db.log_experiment(con, "M5", "aborted: estimator changed since the null harness ran",
                          {"fingerprint_now": fp_now, "fingerprint_null": fp_null},
                          0, MASTER_SEED, "ABORTED_FINGERPRINT_MISMATCH")
        con.close()
        raise SystemExit(
            "ABORT: the estimator code changed after M2. The null thresholds were "
            "produced by different code and are not comparable. Re-run M2.")

    t_thr_95 = null["percentiles"]["max_abs_t_stat_nw"]["p95"]
    t_thr_99 = null["percentiles"]["max_abs_t_stat_nw"]["p99"]
    ic_thr_95 = null["percentiles"]["max_abs_ic"]["p95"]
    sh_thr_95 = null["percentiles"]["max_net_sharpe"]["p95"]
    tval_thr_95 = null["percentiles"]["max_abs_t_val_selected"]["p95"]
    print(f"null thresholds (N={null['n_replications']}): |t|>{t_thr_95:.3f} (p95), "
          f"{t_thr_99:.3f} (p99) | |IC|>{ic_thr_95:.3f} | netSharpe>{sh_thr_95:.3f} | "
          f"val |t|>{tval_thr_95:.3f}")

    prices, macro = research_panel()
    specs = enumerate_search_space()
    feats = build_features(prices, macro, None)
    tgts, traded = build_targets(prices)
    tr = split_mask(prices.index, "TRAIN")
    va = split_mask(prices.index, "VALIDATION")
    te = split_mask(prices.index, "TEST")
    print(f"{len(specs)} hypotheses | TRAIN {tr.sum()} | VAL {va.sum()} | TEST {te.sum()}")

    exp_id = db.log_experiment(
        con, "M5", "full frozen search space on real data, TRAIN stage",
        {"n_specs": len(specs), "cost_bps": BASE_COST_BPS,
         "null_t_p95": t_thr_95, "fingerprint": fp_now},
        n_tests=len(specs), seed=MASTER_SEED, status="RUNNING")

    rng = np.random.default_rng(MASTER_SEED + 7)

    # ------------------------------------------------------------ TRAIN
    print("\n-- TRAIN --")
    t0 = time.time()
    rows = []
    for sp in specs:
        r = evaluate_spec(sp, feats, tgts, traded, tr, sign=None,
                          cost_bps=BASE_COST_BPS, cost_sweep=True)
        hl, hl_note = half_life(feats[sp.feature][tr])
        pb, ci = se.bootstrap_slope_pvalue(
            tgts[sp.target].to_numpy(float)[tr], feats[sp.feature].to_numpy(float)[tr],
            BOOTSTRAP_MEAN_BLOCK, rng, BOOTSTRAP_CI_REPLICATIONS)
        r.update(spec=sp, half_life=hl, half_life_note=hl_note,
                 p_bootstrap=pb, boot_ci_lo=ci[0], boot_ci_hi=ci[1])
        rows.append(r)
    print(f"   {len(rows)} hypotheses evaluated in {time.time() - t0:.0f}s "
          f"(incl. {BOOTSTRAP_CI_REPLICATIONS} bootstrap reps each)")

    tr_abs_t = np.array([abs(r["t_stat_nw"]) if np.isfinite(r["t_stat_nw"]) else np.nan
                         for r in rows])
    print(f"   observed max |t_NW| on TRAIN = {np.nanmax(tr_abs_t):.3f}   "
          f"null p95 = {t_thr_95:.3f}")

    # Benjamini-Hochberg over the WHOLE set of tests, using the bootstrap p-value
    # (more conservative than Newey-West wherever they disagree)
    p_for_fdr = np.array([r["p_bootstrap"] if np.isfinite(r.get("p_bootstrap", np.nan))
                          else r["p_nw"] for r in rows], float)
    rej, p_adj = se.benjamini_hochberg(p_for_fdr, BH_FDR_ALPHA)

    # Deflated Sharpe: trial variance measured across the trials actually run
    daily_sr = np.array([r["net_sharpe"] / np.sqrt(TRADING_DAYS)
                         if np.isfinite(r["net_sharpe"]) else np.nan for r in rows])
    sr_var = float(np.nanvar(daily_sr, ddof=1))
    n_trials = len(specs)

    # ------------------------------------------------------- VALIDATION
    print("\n-- VALIDATION (sign frozen from TRAIN) --")
    val = []
    for r in rows:
        sp = r["spec"]
        sign = r["sign"] if np.isfinite(r["sign"]) else None
        val.append(evaluate_spec(sp, feats, tgts, traded, va, sign=sign,
                                 cost_bps=BASE_COST_BPS, cost_sweep=False))
    print(f"   {len(val)} hypotheses carried to VALIDATION")

    # ------------------------------------------------------------- TEST
    # One access, all rows at once, at the very end. Nothing is changed after.
    print("\n-- TEST (single access, all rows) --")
    test = []
    for r in rows:
        sp = r["spec"]
        sign = r["sign"] if np.isfinite(r["sign"]) else None
        test.append(evaluate_spec(sp, feats, tgts, traded, te, sign=sign,
                                  cost_bps=BASE_COST_BPS, cost_sweep=False))
    db.log_test_access(con, "TEST", "scripts/m5_search.py", exp_id, int(te.sum()),
                       "single scheduled access; all 98 hypotheses evaluated in one "
                       "pass; no parameter, window, target or pair changed afterwards")
    print(f"   TEST opened once, {int(te.sum())} rows, {len(test)} hypotheses")

    # ------------------------------------------------------------ status
    out = []
    for i, r in enumerate(rows):
        sp = r["spec"]
        v, t_ = val[i], test[i]
        beats_null = bool(np.isfinite(r["t_stat_nw"]) and abs(r["t_stat_nw"]) > t_thr_95)
        fdr_ok = bool(rej[i])
        tradable = sp.position_mode != "none"

        net5 = r.get(f"net_sharpe_{BASE_COST_BPS}bps", np.nan)
        costs_ok = (not tradable) or (np.isfinite(net5) and net5 > 0)
        oos_ok = bool(np.isfinite(v["t_stat_nw"]) and
                      abs(v["t_stat_nw"]) > tval_thr_95 and
                      np.isfinite(v["ic"]) and np.isfinite(r["ic"]) and
                      np.sign(v["ic"]) == np.sign(r["ic"]))
        hl = r["half_life"]
        decay_ok = bool(np.isfinite(hl) and hl <= MAX_ADMISSIBLE_HALF_LIFE and
                        sp.horizon <= HORIZON_HALF_LIFE_MULTIPLE * hl)

        if not beats_null or not fdr_ok:
            status = "FAILED_NULL"
        elif not costs_ok:
            status = "FAILED_COSTS"
        elif not oos_ok:
            status = "FAILED_OOS"
        elif not decay_ok:
            status = "FAILED_DECAY"
        else:
            status = "PASSED"

        dsr = se.deflated_sharpe_ratio(
            daily_sr[i], n_trials, r["n_obs"], r["skew"], r["kurtosis"], sr_var) \
            if tradable else np.nan

        out.append({
            "signal_id": sp.signal_id, "description": sp.description, "pair": sp.pair,
            "feature": sp.feature, "feature_window": sp.feature_window,
            "target": sp.target, "horizon": sp.horizon, "n_obs": r["n_obs"],
            "train_ic": r["ic"], "val_ic": v["ic"], "test_ic": t_["ic"],
            "gross_sharpe": r["gross_sharpe"],
            "net_sharpe_2bps": r.get("net_sharpe_2bps"),
            "net_sharpe_5bps": r.get("net_sharpe_5bps"),
            "net_sharpe_10bps": r.get("net_sharpe_10bps"),
            "net_sharpe_20bps": r.get("net_sharpe_20bps"),
            "turnover_annual": r["turnover_annual"], "half_life": hl,
            "half_life_note": r["half_life_note"], "t_stat_nw": r["t_stat_nw"],
            "p_bootstrap": r["p_bootstrap"], "p_bh_adjusted": float(p_adj[i]),
            "deflated_sharpe": dsr, "null_t_threshold_95": t_thr_95,
            "beats_null": beats_null, "max_drawdown": r["max_drawdown"],
            "skew": r["skew"], "kurtosis": r["kurtosis"], "status": status,
            # auxiliary diagnostics (not part of the mandated schema)
            "_family": sp.family, "_target_kind": sp.target_kind,
            "_sign_fitted_on_train": r["sign"], "_p_nw": r["p_nw"],
            "_fdr_rejected": fdr_ok, "_beats_null_p99":
                bool(np.isfinite(r["t_stat_nw"]) and abs(r["t_stat_nw"]) > t_thr_99),
            "_val_t_stat_nw": v["t_stat_nw"], "_val_net_sharpe": v["net_sharpe"],
            "_test_t_stat_nw": t_["t_stat_nw"], "_test_net_sharpe": t_["net_sharpe"],
            "_boot_ci_lo": r["boot_ci_lo"], "_boot_ci_hi": r["boot_ci_hi"],
            "_costs_ok": costs_ok, "_oos_ok": oos_ok, "_decay_ok": decay_ok,
            "_tradable": tradable,
        })

    df = pd.DataFrame(out)
    schema_cols = [
        "signal_id", "description", "pair", "feature", "feature_window", "target",
        "horizon", "n_obs", "train_ic", "val_ic", "test_ic", "gross_sharpe",
        "net_sharpe_2bps", "net_sharpe_5bps", "net_sharpe_10bps", "net_sharpe_20bps",
        "turnover_annual", "half_life", "half_life_note", "t_stat_nw", "p_bootstrap",
        "p_bh_adjusted", "deflated_sharpe", "null_t_threshold_95", "beats_null",
        "max_drawdown", "skew", "kurtosis", "status"]
    con.execute("DELETE FROM signal_results")
    con.register("sr", df[schema_cols])
    con.execute("INSERT INTO signal_results SELECT * FROM sr")
    con.unregister("sr")
    df[schema_cols].to_csv(os.path.join(RESULTS_DIR, "signals.csv"), index=False)
    df.to_csv(os.path.join(RESULTS_DIR, "signals_detail.csv"), index=False)

    summary = {
        "n_tests": len(df),
        "n_beats_null_p95": int(df.beats_null.sum()),
        "n_beats_null_p99": int(df._beats_null_p99.sum()),
        "n_fdr_rejected": int(df._fdr_rejected.sum()),
        "n_passed": int((df.status == "PASSED").sum()),
        "status_counts": df.status.value_counts().to_dict(),
        "observed_max_abs_t_train": float(np.nanmax(tr_abs_t)),
        "observed_max_abs_ic_train": float(np.nanmax(np.abs(df.train_ic))),
        "observed_max_net_sharpe_train": float(np.nanmax(df.net_sharpe_5bps)),
        "null_thresholds_used": {"t_p95": t_thr_95, "t_p99": t_thr_99,
                                 "ic_p95": ic_thr_95, "net_sharpe_p95": sh_thr_95,
                                 "val_t_p95": tval_thr_95},
        "deflated_sharpe_trial_variance": sr_var,
        "n_trials_for_dsr": n_trials,
        "bh_alpha": BH_FDR_ALPHA,
        "min_p_bootstrap": float(np.nanmin(df.p_bootstrap)),
        "min_p_bh_adjusted": float(np.nanmin(df.p_bh_adjusted)),
    }
    with open(os.path.join(RESULTS_DIR, "search_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    con.execute("UPDATE experiment_log SET status='COMPLETED' WHERE experiment_id=?",
                [exp_id])
    con.close()

    print("\n" + "-" * 78)
    print(f"observed max |t_NW| TRAIN      {summary['observed_max_abs_t_train']:.3f}"
          f"   (null p95 {t_thr_95:.3f}, p99 {t_thr_99:.3f})")
    print(f"observed max |IC|   TRAIN      {summary['observed_max_abs_ic_train']:.3f}"
          f"   (null p95 {ic_thr_95:.3f})")
    print(f"observed max net Sharpe TRAIN  {summary['observed_max_net_sharpe_train']:.3f}"
          f"   (null p95 {sh_thr_95:.3f})")
    print(f"beats null p95: {summary['n_beats_null_p95']}/{len(df)}   "
          f"FDR rejected: {summary['n_fdr_rejected']}/{len(df)}   "
          f"PASSED: {summary['n_passed']}")
    print(f"status: {summary['status_counts']}")
    print("\ntop 8 by |t_NW| on TRAIN:")
    top = df.reindex(df.t_stat_nw.abs().sort_values(ascending=False).index).head(8)
    for _, r in top.iterrows():
        print(f"   {r.signal_id:34s} t={r.t_stat_nw:+7.3f}  IC={r.train_ic:+.4f}  "
              f"net5={r.net_sharpe_5bps if pd.notna(r.net_sharpe_5bps) else float('nan'):+6.3f}  "
              f"pBH={r.p_bh_adjusted:.4f}  {r.status}")


if __name__ == "__main__":
    main()
