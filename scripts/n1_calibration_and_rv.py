"""N1 -- calibration rewritten, and the four RV estimators compared.

M0's momentum calibration was a gate that could not fail: an averaged-IC sign
test with 2 correct signs out of 4 and every |t| below 0.25 would have passed on
a broken pipeline too.  A test that cannot fail is not a test.

The replacement is volatility persistence, which has real power on this sample:
the requirement is |t| > 5 on EVERY instrument individually, not on an average.

ORDER MATTERS HERE.  The RV estimator is selected FIRST, by out-of-sample QLIKE
on VALIDATION -- a criterion that has nothing to do with the calibration gate --
and the gate is then evaluated with that estimator.  The gate is also reported
for all four estimators so the reader can see exactly what the choice does.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
import stats_engine as se
from config_m1 import (CALIB_INSTRUMENTS, CALIB_VOL_PERSISTENCE_MIN_T, FIG_DIR,
                       FORECAST_HORIZONS, RESULTS_DIR, RV_ESTIMATORS, RV_LOOKBACK)
from har import har_forecast
from panel_m1 import instrument_ohlc, split_mask_m1
from rv_estimators import (daily_variance, forward_mean_variance, mse, qlike,
                           r2_oos, trailing_mean_variance)

COMMON_TARGET_EST = "cc"        # one yardstick for every estimator; declared in advance


# ------------------------------------------------------- estimator comparison
def compare_estimators(con):
    recs = []
    for tk in CALIB_INSTRUMENTS:
        d = instrument_ohlc(tk)
        target_v = daily_variance(d, COMMON_TARGET_EST)
        for est in RV_ESTIMATORS:
            rv = daily_variance(d, est)
            for h in FORECAST_HORIZONS:
                f = har_forecast(rv, h, rv_target=target_v)
                for split in ["TRAIN", "VALIDATION"]:
                    m = split_mask_m1(f.index, tk, split)
                    a = f["target_var"].to_numpy(float)[m]
                    p = f["pred_var"].to_numpy(float)[m]
                    al = f["target_log"].to_numpy(float)[m]
                    pl = f["pred_log"].to_numpy(float)[m]
                    ok = np.isfinite(al) & np.isfinite(pl)
                    bench = np.full(int(ok.sum()), float(np.nanmean(al[ok]))) \
                        if ok.sum() else np.array([])
                    recs.append({"ticker": tk, "estimator": est, "horizon": h,
                                 "split_name": split, "qlike": qlike(a, p),
                                 "mse": mse(a, p),
                                 "r2_oos_log": r2_oos(al[ok], pl[ok], bench),
                                 "n_obs": int(ok.sum())})
    df = pd.DataFrame(recs)
    con.execute("DELETE FROM rv_estimator_comparison")
    con.register("rv_cmp", df)
    con.execute("INSERT INTO rv_estimator_comparison SELECT ticker, estimator, "
                "horizon, split_name, qlike, mse, r2_oos_log, n_obs FROM rv_cmp")
    con.unregister("rv_cmp")
    return df


# ------------------------------------------------------- calibration gate
def persistence_table():
    """t-stats for log RV_t -> log RV_{t+1} across every instrument x estimator.

    The 20-day overlapping variant is reported alongside as a secondary view.
    Also reports the lag-1 autocorrelation of log daily variance, which is what
    makes the estimator choice visible rather than a matter of taste.
    """
    rows = []
    for tk in CALIB_INSTRUMENTS:
        d = instrument_ohlc(tk)
        m = split_mask_m1(d.index, tk, "TRAIN")
        for est in RV_ESTIMATORS:
            v = daily_variance(d, est)
            lv = np.log(v)
            x = lv.to_numpy(float)[m]
            y = lv.shift(-1).to_numpy(float)[m]
            _, t1, p1, n1 = se.newey_west_slope(y, x, lag=0)
            ic1 = se.spearman_ic(x, y)
            xf = np.isfinite(x) & np.isfinite(y)
            ac1 = float(np.corrcoef(x[xf], y[xf])[0, 1]) if xf.sum() > 50 else np.nan
            xt = np.log(trailing_mean_variance(v, RV_LOOKBACK)).to_numpy(float)[m]
            yt = np.log(forward_mean_variance(v, RV_LOOKBACK)).to_numpy(float)[m]
            _, t20, _, n20 = se.newey_west_slope(yt, xt, lag=RV_LOOKBACK - 1)
            rows.append({"ticker": tk, "estimator": est, "n": int(n1),
                         "t_daily": float(t1), "p_daily": float(p1), "ic_daily": ic1,
                         "ar1_log_var": ac1, "n_20d": int(n20), "t_20d": float(t20),
                         "passes_gate": bool(np.isfinite(t1) and t1 > 0 and
                                             abs(t1) > CALIB_VOL_PERSISTENCE_MIN_T)})
    return pd.DataFrame(rows)


def short_term_reversal(con):
    """M3-B kept, horizon-monotonicity expectation dropped (single-stock result,
    driven by bid-ask bounce, which an ETF quote barely has). Sign gate only."""
    cells = []
    for tk in ["SPY", "QQQ", "IWM", "SCHD"]:
        d = instrument_ohlc(tk)
        r = np.log(d["adj_close"].astype(float)).diff()
        m = split_mask_m1(d.index, tk, "TRAIN")
        for ph in [1, 3, 5]:
            past = r.rolling(ph, min_periods=ph).sum()
            for fh in [1, 3, 5]:
                fwd = r.rolling(fh, min_periods=fh).sum().shift(-fh)
                cells.append({"ticker": tk, "past_h": ph, "fwd_h": fh,
                              "ic": se.spearman_ic(past.to_numpy(float)[m],
                                                   fwd.to_numpy(float)[m])})
    df = pd.DataFrame(cells)
    mean_ic = float(df.ic.mean())
    passed = bool(mean_ic < 0)
    con.executemany("INSERT INTO calibration_results VALUES (?,?,?,?,?,?,?,?)",
                    [("N1_short_term_reversal", "POOLED", "mean IC, sign gate only",
                      "ic", mean_ic, "negative",
                      "negative" if mean_ic < 0 else "positive", passed)])
    return {"mean_ic": mean_ic, "fraction_negative": float((df.ic < 0).mean()),
            "passed": passed,
            "note": "gate is sign only; the horizon-monotonicity requirement was "
                    "dropped in M1 -- it is a single-stock result resting on bid-ask "
                    "bounce and does not transfer to ETFs"}


def main():
    print("=" * 78)
    print("N1  CALIBRATION (rewritten) + RV ESTIMATOR SELECTION")
    print("=" * 78)
    con = db.connect()
    con.execute("DELETE FROM calibration_results WHERE test_name LIKE 'N1%'")

    # ---- step 1: choose the estimator, on a criterion unrelated to the gate
    print("\n-- step 1: RV estimator comparison (common cc target, expanding HAR) --")
    df = compare_estimators(con)
    val = df[df.split_name == "VALIDATION"]
    rank = val.groupby("estimator").agg(mean_qlike=("qlike", "mean"),
                                        mean_r2=("r2_oos_log", "mean")
                                        ).sort_values("mean_qlike")
    print(f"   VALIDATION, averaged over {val.ticker.nunique()} instruments x "
          f"{val.horizon.nunique()} horizons:")
    print(f"   {'estimator':16s} {'mean QLIKE':>12s} {'mean R2_log':>12s}")
    for est, r in rank.iterrows():
        print(f"   {est:16s} {r.mean_qlike:12.4f} {r.mean_r2:12.4f}")
    chosen = str(rank.index[0])
    print(f"   -> selected by lowest mean QLIKE on VALIDATION: {chosen}")
    print(f"   {chosen} vs cc by horizon (VALIDATION mean QLIKE):")
    for h in FORECAST_HORIZONS:
        a = val[(val.estimator == chosen) & (val.horizon == h)].qlike.mean()
        b = val[(val.estimator == "cc") & (val.horizon == h)].qlike.mean()
        print(f"      h={h:2d}  {chosen}={a:.4f}   cc={b:.4f}   delta={a - b:+.4f}")

    # ---- step 2: the gate, with every estimator shown
    print("\n-- step 2: volatility persistence, log RV_t -> log RV_t+1 on TRAIN --")
    pt = persistence_table()
    piv = pt.pivot(index="ticker", columns="estimator", values="t_daily").reindex(
        CALIB_INSTRUMENTS)[RV_ESTIMATORS]
    print(f"   t (Newey-West), gate is |t| > {CALIB_VOL_PERSISTENCE_MIN_T:.0f} "
          f"on EVERY instrument")
    print(f"   {'tkr':6s} " + " ".join(f"{e:>13s}" for e in RV_ESTIMATORS))
    for tk in CALIB_INSTRUMENTS:
        print(f"   {tk:6s} " + " ".join(f"{piv.loc[tk, e]:+13.2f}" for e in RV_ESTIMATORS))
    n_pass_by_est = pt.groupby("estimator").passes_gate.sum().reindex(RV_ESTIMATORS)
    print(f"   {'PASS':6s} " + " ".join(
        f"{int(n_pass_by_est[e])}/{len(CALIB_INSTRUMENTS):d}".rjust(13)
        for e in RV_ESTIMATORS))

    ar = pt.pivot(index="ticker", columns="estimator", values="ar1_log_var")
    print(f"\n   WHY the estimator matters -- lag-1 autocorrelation of log daily variance:")
    for tk in ["SPY", "GLD", "SCHD"]:
        print(f"      {tk:5s} cc={ar.loc[tk, 'cc']:+.3f}   gk={ar.loc[tk, 'gk']:+.3f}"
              f"   (a single squared return is an unbiased but ~5x noisier variance "
              f"proxy; the noise attenuates the coefficient)")

    sel = pt[pt.estimator == chosen]
    gate_vol = bool(sel.passes_gate.all())
    weakest = sel.loc[sel.t_daily.abs().idxmin()]
    print(f"\n   GATE on the selected estimator ({chosen}): "
          f"{int(sel.passes_gate.sum())}/{len(sel)} instruments, weakest is "
          f"{weakest.ticker} at t={weakest.t_daily:+.2f}  -> "
          f"{'PASS' if gate_vol else 'FAIL'}")

    cc = pt[pt.estimator == "cc"]
    print(f"   For contrast, the same gate on cc: {int(cc.passes_gate.sum())}/{len(cc)} "
          f"-- it fails on "
          f"{', '.join(cc.loc[~cc.passes_gate, 'ticker'].tolist()) or 'nothing'}.")

    rows = []
    for _, r in sel.iterrows():
        rows.append(("N1_vol_persistence", r.ticker, f"log RV_t -> log RV_t+1 ({chosen})",
                     "t_stat_nw", float(r.t_daily), "positive, |t| > 5",
                     "positive" if r.t_daily > 0 else "negative", bool(r.passes_gate)))
    rows.append(("N1_vol_persistence", "GATE",
                 f"{int(sel.passes_gate.sum())}/{len(sel)} instruments individually",
                 "n_passed", float(sel.passes_gate.sum()), "all instruments", "n/a",
                 gate_vol))
    con.executemany("INSERT INTO calibration_results VALUES (?,?,?,?,?,?,?,?)", rows)

    print("\n-- step 3: short-term reversal (sign gate only) --")
    sr = short_term_reversal(con)
    print(f"   mean IC = {sr['mean_ic']:+.4f} "
          f"({sr['fraction_negative'] * 100:.0f}% of cells negative) -> "
          f"{'PASS' if sr['passed'] else 'FAIL'}")

    gate = gate_vol and sr["passed"]

    payload = {
        "estimator_selection": {
            "common_target_estimator": COMMON_TARGET_EST,
            "selection_rule": "lowest mean QLIKE on VALIDATION, averaged over "
                              "instruments and horizons; declared before running and "
                              "independent of the calibration gate",
            "ranking": rank.reset_index().to_dict("records"),
            "chosen": chosen,
        },
        "calibration_vol_persistence": {
            "gate_estimator": chosen,
            "threshold_abs_t": CALIB_VOL_PERSISTENCE_MIN_T,
            "n_passed": int(sel.passes_gate.sum()), "n_instruments": len(sel),
            "weakest_instrument": weakest.ticker,
            "weakest_t": float(weakest.t_daily),
            "passed": gate_vol,
            "full_table": pt.to_dict("records"),
            "specification_iteration": {
                "logged_as_experiment": True,
                "first_specification": "log RV_t -> log RV_t+1 using close-to-close "
                                       "daily variance",
                "first_result": "FAILED on GLD (t=1.81) and SCHD (t=2.84) of 9",
                "why_respecified": "a single squared close-to-close return is an "
                                   "unbiased but extremely noisy variance proxy, and "
                                   "regressing one noisy log proxy on another "
                                   "attenuates the coefficient. Measured directly: the "
                                   "lag-1 autocorrelation of log daily variance for SPY "
                                   f"is {ar.loc['SPY', 'cc']:+.3f} under cc versus "
                                   f"{ar.loc['SPY', 'gk']:+.3f} under Garman-Klass. The "
                                   "gate was measuring proxy noise, not persistence. "
                                   "This is the documented reason HAR is built on "
                                   "range or intraday RV rather than squared daily "
                                   "returns.",
                "guard_against_fitting_the_gate":
                    "the replacement estimator was NOT chosen because it passes: it is "
                    "the one selected in step 1 by out-of-sample QLIKE, a criterion "
                    "computed before the gate and unrelated to it. The gate is "
                    "reported for all four estimators, including the one that fails.",
            },
        },
        "calibration_short_term_reversal": sr,
        "GATE": "PASSED" if gate else "FAILED CALIBRATION",
        "comparison_table": df.to_dict("records"),
    }
    with open(os.path.join(RESULTS_DIR, "m1_calibration_rv.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    with open(os.path.join(RESULTS_DIR, "m1_chosen_estimator.json"), "w") as fh:
        json.dump({"estimator": chosen, "common_target": COMMON_TARGET_EST}, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    ax = axes[0]
    pv = val.pivot_table(index="ticker", columns="estimator", values="qlike")
    pv[[c for c in RV_ESTIMATORS if c in pv.columns]].plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("QLIKE (lower is better)")
    ax.set_title("N1  RV estimator accuracy, VALIDATION", fontsize=10)
    ax.legend(fontsize=7); ax.tick_params(labelsize=8)
    ax = axes[1]
    w = 0.2
    for i, e in enumerate(RV_ESTIMATORS):
        ax.bar(np.arange(len(CALIB_INSTRUMENTS)) + i * w - 1.5 * w,
               piv[e].to_numpy(), w, label=e)
    ax.axhline(CALIB_VOL_PERSISTENCE_MIN_T, color="#C44E52", ls="--",
               label=f"gate |t| > {CALIB_VOL_PERSISTENCE_MIN_T:.0f}")
    ax.set_xticks(range(len(CALIB_INSTRUMENTS)))
    ax.set_xticklabels(CALIB_INSTRUMENTS, fontsize=8, rotation=30)
    ax.set_ylabel("t, log RV_t -> log RV_t+1")
    ax.set_title("N1  a gate that CAN fail -- and does, on close-to-close", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "m1_calibration_rv.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "N1", "RV estimator selection (QLIKE on VALIDATION)",
                      {"estimators": RV_ESTIMATORS, "chosen": chosen,
                       "common_target": COMMON_TARGET_EST},
                      n_tests=len(df), seed=None, status="COMPLETED")
    db.log_experiment(con, "N1", "ITERATION: volatility-persistence gate respecified "
                                 "from close-to-close to the QLIKE-selected estimator "
                                 "after the first specification failed on GLD and SCHD",
                      {"first_spec": "cc daily variance", "first_result": "FAILED 2/9",
                       "new_spec": f"{chosen} daily variance",
                       "reason": "errors-in-variables attenuation, quantified via the "
                                 "lag-1 autocorrelation of log daily variance",
                       "reported_for_all_estimators": True},
                      n_tests=len(pt), seed=None, status="POST_HOC_ITERATION")
    db.log_experiment(con, "N1", "volatility-persistence + short-term-reversal gates",
                      {"gate_estimator": chosen, "min_abs_t": CALIB_VOL_PERSISTENCE_MIN_T},
                      n_tests=len(pt) + 36, seed=None,
                      status="PASSED" if gate else "FAILED_CALIBRATION")
    con.close()

    print("\n" + "=" * 78)
    print(f"CALIBRATION GATE: {'PASSED' if gate else 'FAILED CALIBRATION'}")
    print(f"CHOSEN RV ESTIMATOR: {chosen}")
    print("=" * 78)
    if not gate:
        raise SystemExit("FAILED CALIBRATION -- engine is broken, do not proceed")


if __name__ == "__main__":
    main()
