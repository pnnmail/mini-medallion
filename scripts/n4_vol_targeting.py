"""N4 + N5 traps -- volatility targeting, on TRAIN + VALIDATION only.

TEST stays closed here; n5 opens it once for everything at the same time.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from candidates_m1 import build_candidates
from config_m1 import (COST_SWEEP_BPS, CRISIS_BOOTSTRAP_REPS, FIG_DIR,
                       MASTER_SEED_M1, REBALANCE_BAND, RESULTS_DIR, SIZING_HORIZON,
                       TRADING_DAYS, VOL_TARGET_ANNUAL, VOL_TARGET_CAPS, VOL_TARGETS)
from har import har_forecast
from panel_m1 import instrument_ohlc, split_mask_m1
from rv_estimators import daily_variance
from voltarget import (band_rebalance, crisis_block_bootstrap, drawdown_episodes,
                       leave_one_crisis_out, matched_comparisons, perf,
                       short_term_gain_turnover, strategy_pnl, target_weights)

TARGET_EST = "cc"
FORECAST_MODELS = ["HAR", "HAR+log_vix"]


def load_chosen():
    with open(os.path.join(RESULTS_DIR, "m1_chosen_estimator.json")) as fh:
        return json.load(fh)["estimator"]


def sigma_forecast(target, chosen, model):
    d = instrument_ohlc(target)
    rv_reg = daily_variance(d, chosen)
    rv_tgt = daily_variance(d, TARGET_EST)
    extra = None
    if model == "HAR+log_vix":
        C, _ = build_candidates(target, chosen, SIZING_HORIZON)
        extra = C[["log_vix"]]
    f = har_forecast(rv_reg, SIZING_HORIZON, extra=extra, rv_target=rv_tgt)
    return d, np.sqrt(TRADING_DAYS * f["pred_var"].to_numpy(float))


def build_rows(dev_only=True, splits=("TRAIN", "VALIDATION"), tag="DEV"):
    chosen = load_chosen()
    rng = np.random.default_rng(MASTER_SEED_M1 + 41)
    rows, curves, episodes_out = [], {}, []
    seen_eps, coverage = set(), {}
    for tk in VOL_TARGETS:
        for model in FORECAST_MODELS:
            d, sig = sigma_forecast(tk, chosen, model)
            r = np.log(d["adj_close"].astype(float)).diff().to_numpy(float)
            fwd = np.concatenate([r[1:], [np.nan]])          # return from t to t+1
            px = d["adj_close"].astype(float).to_numpy(float)
            mask = np.zeros(len(d), bool)
            for s in splits:
                mask |= split_mask_m1(d.index, tk, s)
            mask &= np.isfinite(sig) & np.isfinite(fwd)

            pnl_bh, _, _ = strategy_pnl(np.where(mask, 1.0, np.nan), fwd, 0.0)
            eps = drawdown_episodes(np.where(mask, pnl_bh, 0.0), d.index, span="decline")
            if tk not in seen_eps:
                seen_eps.add(tk)
                cov = sum(e["length_sessions"] for e in eps) / max(int(mask.sum()), 1)
                cov_full = sum(e["full_span_sessions"] for e in eps) / max(int(mask.sum()), 1)
                coverage[tk] = {"decline_leg_share_of_sample": float(cov),
                                "peak_to_recovery_share_of_sample": float(cov_full)}
                for i, e in enumerate(eps):
                    episodes_out.append({"instrument": tk, "episode_id": i,
                                         "peak_date": pd.Timestamp(e["peak_date"]).date(),
                                         "trough_date": pd.Timestamp(e["trough_date"]).date(),
                                         "recovery_date": pd.Timestamp(e["recovery_date"]).date(),
                                         "depth": e["depth"],
                                         "length_days": e["length_sessions"]})
            for cap in VOL_TARGET_CAPS:
                wt = target_weights(sig, cap)
                wt = np.where(mask, wt, np.nan)
                w = band_rebalance(wt, REBALANCE_BAND)
                w = np.where(mask, w, np.nan)
                pnl, dw, ok = strategy_pnl(w, fwd, 0.0)
                pv, pb = perf(pnl, mask), perf(pnl_bh, mask)
                mc = matched_comparisons(pnl, pnl_bh, w, mask)

                nyears = max(mask.sum() / TRADING_DAYS, 1e-9)
                n_reb = int((np.abs(np.diff(np.concatenate(([np.nan], w)))) > 1e-12
                             )[mask].sum())
                nets = {}
                for c in COST_SWEEP_BPS:
                    pc, _, _ = strategy_pnl(w, fwd, c)
                    nets[f"net_sharpe_{c}bps"] = perf(pc, mask)["sharpe"]

                loo = leave_one_crisis_out(pnl, pnl_bh, eps, len(d))
                # the two episodes the spec names explicitly, by calendar year
                yrs = d.index.year.to_numpy()
                ex = {}
                for label, drop in [("2008", {2008}), ("2020", {2020}),
                                    ("2008_and_2020", {2008, 2020})]:
                    k = mask & ~np.isin(yrs, list(drop))
                    a, b = perf(pnl, k)["sharpe"], perf(pnl_bh, k)["sharpe"]
                    ex[f"sharpe_vt_ex_{label}"] = a
                    ex[f"sharpe_bh_ex_{label}"] = b
                    ex[f"advantage_ex_{label}"] = (a - b) if np.isfinite(a) and \
                        np.isfinite(b) else np.nan
                keep_all = mask.copy()
                for e in eps:
                    keep_all[e["start"]:e["end"] + 1] = False
                sh_ex_all = perf(pnl, keep_all)["sharpe"]
                sb_ex_all = perf(pnl_bh, keep_all)["sharpe"]
                cb = crisis_block_bootstrap(np.where(mask, pnl, np.nan),
                                            np.where(mask, pnl_bh, np.nan), eps,
                                            len(d), CRISIS_BOOTSTRAP_REPS, rng)
                stg = short_term_gain_turnover(w, px, d.index, mask)

                sid = f"{tk}_{model}_cap{cap}"
                rows.append({
                    "period": tag, "strategy_id": sid, "target_instrument": tk,
                    "forecast_model": model, "cap": cap, "band": REBALANCE_BAND,
                    "horizon": SIZING_HORIZON, "n_days": int(mask.sum()),
                    "turnover_annual": float(dw[mask].sum() / nyears),
                    "n_rebalances_yr": float(n_reb / nyears),
                    "mean_exposure": mc.get("mean_exposure_vt"),
                    "sharpe_bh": pb["sharpe"], "sharpe_vt": pv["sharpe"],
                    "ann_return_bh": pb["ann_return"], "ann_return_vt": pv["ann_return"],
                    "ann_vol_bh": pb["ann_vol"], "ann_vol_vt": pv["ann_vol"],
                    "maxdd_bh": pb["max_drawdown"], "maxdd_vt": pv["max_drawdown"],
                    "sharpe_vt_matched_vol": mc.get("sharpe_vt_matched_vol"),
                    "ann_return_bh_at_matched_vol": mc.get("ann_return_bh_at_matched_vol"),
                    "excess_return_at_matched_vol": mc.get("excess_return_at_matched_vol"),
                    "sharpe_vt_matched_exposure": mc.get("sharpe_vt_matched_exposure"),
                    "excess_return_at_matched_dd": mc.get("excess_return_at_matched_dd"),
                    "sharpe_vt_ex_all_crises": sh_ex_all,
                    "sharpe_bh_ex_all_crises": sb_ex_all,
                    "advantage_full": (pv["sharpe"] - pb["sharpe"]),
                    "advantage_ex_all_crises": (sh_ex_all - sb_ex_all)
                    if np.isfinite(sh_ex_all) and np.isfinite(sb_ex_all) else np.nan,
                    "loo_min_advantage": float(np.nanmin([x["advantage"] for x in loo]))
                    if loo else np.nan,
                    "loo_detail": loo,
                    "crisis_block_p_advantage_le_0": cb["p_advantage_le_0"],
                    "crisis_block_ci_lo": cb["ci_lo"], "crisis_block_ci_hi": cb["ci_hi"],
                    "crisis_n_blocks": cb["n_blocks"],
                    "short_term_gain_annual_pct": stg["short_term_gain_annual_pct"],
                    "long_term_gain_annual_pct": stg["long_term_gain_annual_pct"],
                    **ex, **nets})
                curves[sid] = {"index": d.index[mask], "vt": np.cumsum(pnl[mask]),
                               "bh": np.cumsum(pnl_bh[mask]), "w": w[mask]}
    return pd.DataFrame(rows), curves, pd.DataFrame(episodes_out), coverage


def assign_status(df):
    st = []
    for _, r in df.iterrows():
        if not np.isfinite(r.advantage_full) or r.advantage_full <= 0:
            st.append("FAILED_BASELINE")
        elif np.isfinite(r.loo_min_advantage) and r.loo_min_advantage <= 0:
            st.append("FAILED_CRISIS_DEPENDENCE")
        elif not np.isfinite(r.net_sharpe_10bps) or \
                r.net_sharpe_10bps <= r.sharpe_bh:
            st.append("FAILED_COSTS")
        else:
            st.append("PASSED_DEV")
        # crisis-block bootstrap is reported but does not override the LOO gate
    df = df.copy()
    df["status"] = st
    return df


def main():
    print("=" * 78)
    print("N4  VOLATILITY TARGETING  (TRAIN + VALIDATION only)")
    print("=" * 78)
    con = db.connect()
    df, curves, eps, coverage = build_rows()
    df = assign_status(df)

    print(f"\nsigma_target={VOL_TARGET_ANNUAL:.0%}  caps={VOL_TARGET_CAPS}  "
          f"band=+-{REBALANCE_BAND:.0%}  sizing horizon={SIZING_HORIZON}d")
    print(f"\n{'strategy':28s} {'expo':>5s} {'turn':>6s} {'reb/y':>6s} "
          f"{'ShBH':>6s} {'ShVT':>6s} {'adv':>6s} {'advXcr':>7s} {'LOOmin':>7s} "
          f"{'net10':>6s} {'STgain%':>7s}")
    for _, r in df.iterrows():
        print(f"{r.strategy_id:28s} {r.mean_exposure:5.2f} {r.turnover_annual:6.2f} "
              f"{r.n_rebalances_yr:6.1f} {r.sharpe_bh:6.3f} {r.sharpe_vt:6.3f} "
              f"{r.advantage_full:+6.3f} {r.advantage_ex_all_crises:+7.3f} "
              f"{r.loo_min_advantage:+7.3f} {r.net_sharpe_10bps:6.3f} "
              f"{r.short_term_gain_annual_pct:7.2f}")

    print(f"\nstatus: {df.status.value_counts().to_dict()}")

    print("\nCRISIS EPISODE COVERAGE -- why the decline leg is the episode, not "
          "peak-to-recovery:")
    for tk, c in coverage.items():
        a = c["decline_leg_share_of_sample"]
        b = c["peak_to_recovery_share_of_sample"]
        print(f"   {tk:5s} the five decline legs cover {a:5.1%} of the sample; "
              f"peak-to-recovery spans would cover {b:5.1%}")
    print("   Removing >90% of the sample would leave only stretches at new highs, "
          "where holding")
    print("   less than 100% must underperform by construction. That would measure the "
          "definition.")

    print("\nTHE COMPARISON THE SPEC NAMES BY NAME -- Sharpe advantage of VT over BH:")
    print(f"   {'strategy':28s} {'full':>7s} {'ex2008':>8s} {'ex2020':>8s} "
          f"{'ex both':>8s} {'ex all 5 declines':>19s}")
    for _, r in df.iterrows():
        print(f"   {r.strategy_id:28s} {r.advantage_full:+7.3f} "
              f"{r.advantage_ex_2008:+8.3f} {r.advantage_ex_2020:+8.3f} "
              f"{r.advantage_ex_2008_and_2020:+8.3f} "
              f"{r.advantage_ex_all_crises:+19.3f}")

    print("\nMATCHED-RISK COMPARISONS (the Sharpe column is scale invariant, so the")
    print("return columns are what carry information):")
    print(f"{'strategy':28s} {'annRetVT':>9s} {'BH@matchVol':>12s} {'excess':>8s} "
          f"{'ShVT@matchExp':>14s} {'excess@matchDD':>15s}")
    for _, r in df.iterrows():
        print(f"{r.strategy_id:28s} {r.ann_return_vt:9.4f} "
              f"{r.ann_return_bh_at_matched_vol:12.4f} "
              f"{r.excess_return_at_matched_vol:+8.4f} "
              f"{r.sharpe_vt_matched_exposure:14.3f} "
              f"{r.excess_return_at_matched_dd:+15.4f}")

    print("\nLEAVE-ONE-CRISIS-OUT (Sharpe advantage of VT over BH, episode removed):")
    for _, r in df.iterrows():
        if not r.loo_detail:
            continue
        s = "  ".join(f"{x['trough_date']}:{x['advantage']:+.3f}" for x in r.loo_detail)
        print(f"   {r.strategy_id:28s} full={r.advantage_full:+.3f}  {s}")

    print("\nCRISIS-BLOCK BOOTSTRAP (resampling episodes, not days):")
    for _, r in df.iterrows():
        print(f"   {r.strategy_id:28s} blocks={int(r.crisis_n_blocks):3d}  "
              f"P(advantage<=0)={r.crisis_block_p_advantage_le_0:.3f}  "
              f"95% CI [{r.crisis_block_ci_lo:+.3f}, {r.crisis_block_ci_hi:+.3f}]")

    print("\nCOSTS, reported separately as N5.3 requires:")
    print(f"{'strategy':28s} {'gross':>6s} " +
          " ".join(f"{'net' + str(c):>7s}" for c in COST_SWEEP_BPS) +
          f" {'STgain%/yr':>11s} {'LTgain%/yr':>11s}")
    for _, r in df.iterrows():
        print(f"{r.strategy_id:28s} {r.sharpe_vt:6.3f} " +
              " ".join(f"{r['net_sharpe_' + str(c) + 'bps']:7.3f}" for c in COST_SWEEP_BPS) +
              f" {r.short_term_gain_annual_pct:11.2f} {r.long_term_gain_annual_pct:11.2f}")

    con.execute("DELETE FROM crisis_episodes")
    con.register("ce", eps)
    con.execute("INSERT INTO crisis_episodes SELECT instrument, episode_id, "
                "CAST(peak_date AS DATE), CAST(trough_date AS DATE), "
                "CAST(recovery_date AS DATE), depth, length_days FROM ce")
    con.unregister("ce")

    out = df.drop(columns=["loo_detail"])
    out.to_csv(os.path.join(RESULTS_DIR, "m1_voltarget_dev.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "m1_voltarget_dev.json"), "w") as fh:
        json.dump({"config": {"sigma_target": VOL_TARGET_ANNUAL,
                              "caps": VOL_TARGET_CAPS, "band": REBALANCE_BAND,
                              "sizing_horizon": SIZING_HORIZON,
                              "forecast_models": FORECAST_MODELS},
                   "rows": df.to_dict("records"),
                   "episodes": eps.to_dict("records"),
                   "episode_coverage": coverage}, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
    for ax, tk in zip(axes.ravel(), VOL_TARGETS):
        sid = f"{tk}_HAR_cap1.0"
        c = curves[sid]
        ax.plot(c["index"], np.expm1(c["bh"]) * 100, label="buy & hold", lw=1.2,
                color="#666666")
        ax.plot(c["index"], np.expm1(c["vt"]) * 100, label="vol targeted (cap 1.0)",
                lw=1.2, color="#4C72B0")
        ax2 = ax.twinx()
        ax2.plot(c["index"], c["w"], color="#C44E52", lw=0.6, alpha=0.5)
        ax2.set_ylabel("weight", fontsize=7, color="#C44E52")
        ax2.tick_params(labelsize=6)
        ax.set_title(tk, fontsize=10)
        ax.set_ylabel("cumulative %", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.suptitle("N4  vol targeting vs buy-and-hold, TRAIN + VALIDATION "
                 "(NOT risk-matched -- see the table)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "m1_voltarget_dev.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "N4", "vol targeting on TRAIN+VALIDATION with matched-risk "
                                 "comparisons, leave-one-crisis-out, cost sweep and "
                                 "short-term gain accounting",
                      {"instruments": VOL_TARGETS, "models": FORECAST_MODELS,
                       "caps": VOL_TARGET_CAPS, "band": REBALANCE_BAND},
                      n_tests=len(df), seed=MASTER_SEED_M1, status="COMPLETED")
    con.close()
    print("\nwrote results/m1_voltarget_dev.csv / .json, figures/m1_voltarget_dev.png")


if __name__ == "__main__":
    main()
