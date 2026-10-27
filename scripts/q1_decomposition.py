"""N1 -- the exact decomposition, and the covariance term measured directly.

The methodological move of the milestone: the mechanism is tested on the
covariance term, which is estimated from ~5000 daily observations, not on the
Sharpe difference, which N0 just showed has no power.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config import BOOTSTRAP_MEAN_BLOCK, TRADING_DAYS
from config_m1 import RESULTS_DIR
from config_m2 import FIG_DIR, INSTRUMENTS, MASTER_SEED_M2
from decomposition import circular_shift_null, null_summary, pop_cov, sharpe_attribution
from m2pair import FAST, SLOW, paired_series
from stambaugh import effective_obs_step_weights, predictive_regression


def main():
    print("=" * 78)
    print("N1  EXACT DECOMPOSITION + DIRECT ESTIMATE OF THE COVARIANCE TERM")
    print("=" * 78)
    con = db.connect()
    rng = np.random.default_rng(MASTER_SEED_M2 + 110)
    rows, reg_rows = [], []

    for tk in INSTRUMENTS:
        ps = paired_series(tk)
        r = ps["fwd_ret"]
        ws, wf = ps["w_slow"], ps["w_fast"]
        ok = np.isfinite(r) & np.isfinite(ws) & np.isfinite(wf)
        r, ws, wf = r[ok], ws[ok], wf[ok]
        n = len(r)

        mean_r = float(np.mean(r))
        d_mean_w = float(np.mean(ws) - np.mean(wf))
        cov_s, cov_f = pop_cov(ws, r), pop_cov(wf, r)
        exposure = d_mean_w * mean_r
        cov_term = cov_s - cov_f
        total = float(np.mean(ws * r) - np.mean(wf * r))

        att = sharpe_attribution(ps["pnl_slow"], ps["pnl_fast"],
                                 np.ones(len(ps["pnl_slow"]), bool))

        # null for the covariance term: circular shift of the weight DIFFERENCE
        dw = ws - wf
        nd, _ = circular_shift_null(dw, r, 2000, rng, "cov")
        ns = null_summary(nd, cov_term, 95)

        eff_s = effective_obs_step_weights(ws)
        eff_f = effective_obs_step_weights(wf)
        hac_lag = int(max(10, round(eff_s["median_hold_days"] or 10)))

        rows.append({
            "instrument": tk, "n": n, "mean_r_ann": mean_r * TRADING_DAYS,
            "mean_w_slow": float(np.mean(ws)), "mean_w_fast": float(np.mean(wf)),
            "delta_E_w": d_mean_w,
            "delta_E_w_times_E_r_ann": exposure * TRADING_DAYS,
            "cov_slow_ann": cov_s * TRADING_DAYS, "cov_fast_ann": cov_f * TRADING_DAYS,
            "delta_cov_w_r_ann": cov_term * TRADING_DAYS,
            "delta_numerator_ann": total * TRADING_DAYS,
            "identity_residual": float(total - exposure - cov_term),
            "cov_term_null_p95_ann": ns["p95_abs"] * TRADING_DAYS,
            "cov_term_p": ns["p_two_sided"],
            "cov_term_beats_null": ns["beats_null"],
            "sd_slow_ann": att["sd_a"] * np.sqrt(TRADING_DAYS),
            "sd_fast_ann": att["sd_b"] * np.sqrt(TRADING_DAYS),
            "delta_sharpe": att["delta_sharpe"],
            "sh_numerator_effect": att["mean_effect"],
            "sh_denominator_effect": att["vol_effect"],
            "sh_interaction": att["interaction"],
            "sh_residual": att["residual"],
            "n_steps_slow": eff_s["n_steps"], "n_steps_fast": eff_f["n_steps"],
            "days_per_step_slow": eff_s["ratio_days_per_step"],
            "hac_lag_used": hac_lag,
        })

        for tag, w in [(SLOW, ws), (FAST, wf)]:
            pr = predictive_regression(r, w, hac_lag)
            pr.update({"instrument": tk, "forecast_version": tag})
            reg_rows.append(pr)

    df = pd.DataFrame(rows)
    rg = pd.DataFrame(reg_rows)

    print(f"\nIDENTITY: worst |residual| = {df.identity_residual.abs().max():.3e}   "
          f"Sharpe attribution worst |residual| = {df.sh_residual.abs().max():.3e}")

    print("\nNUMERATOR -- annualised, SLOW minus FAST")
    print(f"   {'inst':5s} {'dE[w]':>8s} {'E[r]ann':>9s} {'dE[w]E[r]':>10s} "
          f"{'dCov(w,r)':>10s} {'total':>9s} {'nullp95':>9s} {'p':>6s} {'beats':>6s}")
    for _, r_ in df.iterrows():
        print(f"   {r_.instrument:5s} {r_.delta_E_w:+8.4f} {r_.mean_r_ann:+9.4f} "
              f"{r_.delta_E_w_times_E_r_ann:+10.5f} {r_.delta_cov_w_r_ann:+10.5f} "
              f"{r_.delta_numerator_ann:+9.5f} {r_.cov_term_null_p95_ann:9.5f} "
              f"{r_.cov_term_p:6.3f} {'yes' if r_.cov_term_beats_null else 'no':>6s}")

    print("\nSHARPE -- numerator vs denominator (exact, interaction shown)")
    print(f"   {'inst':5s} {'dSharpe':>8s} {'numer':>8s} {'denom':>8s} "
          f"{'interact':>9s} {'sdSlow':>8s} {'sdFast':>8s}")
    for _, r_ in df.iterrows():
        print(f"   {r_.instrument:5s} {r_.delta_sharpe:+8.3f} "
              f"{r_.sh_numerator_effect:+8.3f} {r_.sh_denominator_effect:+8.3f} "
              f"{r_.sh_interaction:+9.3f} {r_.sd_slow_ann:8.4f} {r_.sd_fast_ann:8.4f}")

    share_cov = float(np.mean(np.abs(df.delta_cov_w_r_ann) /
                              (np.abs(df.delta_cov_w_r_ann) +
                               np.abs(df.delta_E_w_times_E_r_ann))))
    n_cov_beats = int(df.cov_term_beats_null.sum())
    print(f"\n   the covariance term carries {share_cov:.0%} of the absolute "
          f"numerator difference on average")
    print(f"   it clears its own circular-shift null in {n_cov_beats}/{len(df)} "
          f"instruments")

    print("\nPREDICTIVE REGRESSION  r_(t+1) = a + b*w_t   (HAC + Stambaugh)")
    print(f"   {'inst':5s} {'version':12s} {'b_raw':>10s} {'t_HAC':>7s} {'phi':>6s} "
          f"{'bias':>10s} {'b_adj':>10s} {'bias/raw':>9s}")
    for _, r_ in rg.iterrows():
        print(f"   {r_.instrument:5s} {r_.forecast_version:12s} {r_.b_raw:+10.5f} "
              f"{r_.t_hac:+7.2f} {r_.phi_ar1_regressor:6.3f} "
              f"{r_.stambaugh_bias:+10.6f} {r_.b_stambaugh_adjusted:+10.5f} "
              f"{r_.bias_as_share_of_raw:+9.3f}")

    piv = rg.pivot(index="instrument", columns="forecast_version",
                   values="b_stambaugh_adjusted")
    more_neg = int((piv[FAST] < piv[SLOW]).sum())
    print(f"\n   M1's hypothesis predicts a MORE NEGATIVE slope for the fast "
          f"(VIX) version.")
    print(f"   corrected slope more negative for {FAST} in {more_neg}/{len(piv)} "
          f"instruments")
    for tk in piv.index:
        print(f"      {tk:5s} slow {piv.loc[tk, SLOW]:+.5f}   fast "
              f"{piv.loc[tk, FAST]:+.5f}   diff "
              f"{piv.loc[tk, FAST] - piv.loc[tk, SLOW]:+.5f}")

    print("\nEFFECTIVE SAMPLE with band rebalancing (the weight is a step function):")
    for _, r_ in df.iterrows():
        print(f"   {r_.instrument:5s} {int(r_.n):5d} days but only "
              f"{int(r_.n_steps_slow):4d} distinct weight levels "
              f"({r_.days_per_step_slow:.1f} days per step); HAC lag "
              f"{int(r_.hac_lag_used)}")

    df.to_csv(os.path.join(RESULTS_DIR, "m2_n1_decomposition.csv"), index=False)
    rg.to_csv(os.path.join(RESULTS_DIR, "m2_n1_predictive.csv"), index=False)
    payload = {
        "identity_worst_residual": float(df.identity_residual.abs().max()),
        "sharpe_attribution_worst_residual": float(df.sh_residual.abs().max()),
        "numerator": df.to_dict("records"),
        "predictive_regression": rg.to_dict("records"),
        "covariance_share_of_numerator": share_cov,
        "n_cov_beats_null": n_cov_beats,
        "m1_hypothesis_more_negative_slope_for_fast": {
            "n_instruments_supporting": more_neg, "n_total": int(len(piv))},
    }
    with open(os.path.join(RESULTS_DIR, "m2_n1_decomposition.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    ax = axes[0]
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.delta_E_w_times_E_r_ann * 100, 0.4, label="exposure term",
           color="#4C72B0")
    ax.bar(x + 0.2, df.delta_cov_w_r_ann * 100, 0.4, label="covariance term",
           color="#C44E52")
    ax.errorbar(x + 0.2, df.delta_cov_w_r_ann * 100,
                yerr=df.cov_term_null_p95_ann * 100, fmt="none", ecolor="black",
                capsize=3, lw=1, label="null p95")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(df.instrument)
    ax.set_ylabel("annualised contribution, %")
    ax.set_title("N1  SLOW minus FAST: numerator split", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(x - 0.2, df.sh_numerator_effect, 0.4, label="numerator effect",
           color="#4C72B0")
    ax.bar(x + 0.2, df.sh_denominator_effect, 0.4, label="denominator effect",
           color="#DD8452")
    ax.plot(x, df.delta_sharpe, "ko", label="total delta Sharpe")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(df.instrument)
    ax.set_ylabel("Sharpe units")
    ax.set_title("numerator or denominator?", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "m2_n1_decomposition.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "N1", "exact numerator/denominator decomposition + "
                                 "predictive regression with HAC and Stambaugh "
                                 "correction",
                      {"instruments": list(INSTRUMENTS),
                       "cov_share": share_cov, "n_cov_beats_null": n_cov_beats,
                       "more_negative_slope_for_fast": more_neg},
                      n_tests=len(df) + len(rg), seed=MASTER_SEED_M2,
                      status="COMPLETED")
    con.close()
    print("\nwrote results/m2_n1_decomposition.csv/.json, m2_n1_predictive.csv")


if __name__ == "__main__":
    main()
