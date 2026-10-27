"""P1 -- exposure versus timing. The analytic core of M2.

mean(w*r) = mean(w)*mean(r) + cov(w,r) is an identity. Applied to two forecasts
it says exactly how much of their P&L difference comes from holding MORE and how
much from holding more AT THE RIGHT MOMENTS. M1 could measure the difference but
not attribute it; this is the missing step.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config_m2 import (DECOMP_TOLERANCE, FIG_DIR, INSTRUMENTS, RESULTS_DIR,
                       TRADING_DAYS)
from decomposition import decompose_pair, sharpe_attribution
from rv_estimators import qlike
from universe_m2 import build_universe, common_mask, weights_and_pnl


def main():
    print("=" * 78)
    print("P1  DECOMPOSITION -- exposure vs timing")
    print("=" * 78)
    con = db.connect()
    recs, worst_resid = [], 0.0

    for tk in INSTRUMENTS:
        idx, fwd, models, fams, d = build_universe(tk)
        for split in ["TRAIN", "VALIDATION", "TEST"]:
            m = common_mask(tk, split)
            if m.sum() < 60:
                continue
            _, w_b, pnl_b, _ = weights_and_pnl(tk, "HAR", m)
            for name in models:
                _, w_a, pnl_a, dw = weights_and_pnl(tk, name, m)
                dec = decompose_pair(w_a, w_b, fwd, m)
                if not dec:
                    continue
                att = sharpe_attribution(pnl_a, pnl_b, m)
                f = models[name]
                q = qlike(f["target_var"].to_numpy(float)[m],
                          f["pred_var"].to_numpy(float)[m])
                worst_resid = max(worst_resid, abs(dec["residual"]))
                recs.append({
                    "instrument": tk, "model": name,
                    "candidate": name.replace("HAR+", "") if name != "HAR" else "",
                    "family": fams[name], "split_name": split, "n_obs": dec["n"],
                    "mean_w": dec["mean_w_a"], "cov_w_r": dec["cov_a"],
                    "exposure_term_ann": dec["exposure_term_ann"],
                    "timing_term_ann": dec["timing_term_ann"],
                    "delta_mean_ann": dec["delta_mean_ann"],
                    "residual": dec["residual"],
                    "qlike": q,
                    "sharpe": att.get("sharpe_a", np.nan),
                    "sharpe_baseline": att.get("sharpe_b", np.nan),
                    "delta_sharpe": att.get("delta_sharpe", np.nan),
                    "sh_mean_effect": att.get("mean_effect", np.nan),
                    "sh_vol_effect": att.get("vol_effect", np.nan),
                    "sh_interaction": att.get("interaction", np.nan),
                    "sh_residual": att.get("residual", np.nan),
                    "turnover_annual": float(dw[m].sum() * TRADING_DAYS / m.sum()),
                })
    df = pd.DataFrame(recs)

    print(f"\nIDENTITY CHECK: worst |residual| over {len(df)} decompositions = "
          f"{worst_resid:.3e}  (tolerance {DECOMP_TOLERANCE:.0e})")
    ok_ident = bool(worst_resid <= DECOMP_TOLERANCE)
    print(f"   -> {'the decomposition closes exactly' if ok_ident else 'IDENTITY BROKEN'}")
    sh_res = float(df.sh_residual.abs().max())
    print(f"   Sharpe attribution worst |residual| = {sh_res:.3e}")

    # ---- H1: is the VIX loss in the timing term or the exposure term? ----
    print("\nH1 -- HAR+log_vix vs HAR, annualised mean-P&L decomposition")
    print(f"   {'inst':5s} {'split':11s} {'exposure':>10s} {'timing':>10s} "
          f"{'total':>10s}   {'dSharpe':>8s} {'meanEff':>8s} {'volEff':>8s} "
          f"{'interact':>9s}")
    h1 = df[df.model == "HAR+log_vix"]
    for _, r in h1.iterrows():
        print(f"   {r.instrument:5s} {r.split_name:11s} {r.exposure_term_ann:+10.5f} "
              f"{r.timing_term_ann:+10.5f} {r.delta_mean_ann:+10.5f}   "
              f"{r.delta_sharpe:+8.3f} {r.sh_mean_effect:+8.3f} "
              f"{r.sh_vol_effect:+8.3f} {r.sh_interaction:+9.3f}")

    dev = h1[h1.split_name.isin(["TRAIN", "VALIDATION"])]
    n_timing_neg = int((dev.timing_term_ann < 0).sum())
    n_expo_neg = int((dev.exposure_term_ann < 0).sum())
    timing_share = float(np.nanmean(
        np.abs(dev.timing_term_ann) /
        (np.abs(dev.timing_term_ann) + np.abs(dev.exposure_term_ann))))
    print(f"\n   on TRAIN and VALIDATION ({len(dev)} cells): timing term negative in "
          f"{n_timing_neg}, exposure term negative in {n_expo_neg}")
    print(f"   timing accounts for {timing_share:.0%} of the absolute difference on "
          f"average")
    h1_supported = bool(n_timing_neg > n_expo_neg and timing_share > 0.5)
    print(f"   H1 ({'supported' if h1_supported else 'NOT supported'}): the loss sits "
          f"in {'timing' if h1_supported else 'the exposure term'}")

    # ---- the same question across the whole universe ----
    print("\nAcross ALL models (TRAIN+VALIDATION), correlation between the two terms")
    print("and the Sharpe difference they are supposed to explain:")
    sub = df[df.split_name.isin(["TRAIN", "VALIDATION"]) & (df.model != "HAR")]
    from scipy import stats as sps
    for col, lab in [("exposure_term_ann", "exposure"), ("timing_term_ann", "timing")]:
        ok = sub[col].notna() & sub.delta_sharpe.notna()
        rho = float(sps.spearmanr(sub.loc[ok, col], sub.loc[ok, "delta_sharpe"]).statistic)
        print(f"   Spearman({lab:8s}, delta_sharpe) = {rho:+.3f}")

    print("\nby family (TRAIN+VALIDATION, mean annualised terms):")
    fam = sub.groupby("family").agg(
        n=("model", "size"),
        exposure=("exposure_term_ann", "mean"),
        timing=("timing_term_ann", "mean"),
        d_sharpe=("delta_sharpe", "mean"),
        d_qlike=("qlike", "mean")).reset_index()
    print(fam.round(5).to_string(index=False))

    df.to_csv(os.path.join(RESULTS_DIR, "m2_decomposition.csv"), index=False)
    payload = {
        "identity_check": {"worst_abs_residual": worst_resid,
                           "tolerance": DECOMP_TOLERANCE, "closes": ok_ident,
                           "sharpe_attribution_worst_residual": sh_res},
        "H1": {"statement": "the HAR+log_vix loss sits in the timing term, not the "
                            "exposure term",
               "n_cells_dev": len(dev),
               "timing_negative_cells": n_timing_neg,
               "exposure_negative_cells": n_expo_neg,
               "mean_timing_share_of_absolute_difference": timing_share,
               "supported": h1_supported},
        "by_family": fam.to_dict("records"),
        "vix_cells": h1.to_dict("records"),
    }
    with open(os.path.join(RESULTS_DIR, "m2_p1_decomposition.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    cells = h1[h1.split_name.isin(["TRAIN", "VALIDATION"])]
    lab = [f"{r.instrument}\n{r.split_name[:3]}" for _, r in cells.iterrows()]
    x = np.arange(len(cells))
    ax.bar(x - 0.2, cells.exposure_term_ann * 100, 0.4, label="exposure term",
           color="#4C72B0")
    ax.bar(x + 0.2, cells.timing_term_ann * 100, 0.4, label="timing term",
           color="#C44E52")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylabel("annualised contribution, %")
    ax.set_title("P1  HAR+log(VIX) minus HAR: where the difference comes from",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.scatter(sub.timing_term_ann * 100, sub.delta_sharpe, s=12, alpha=0.6,
               color="#C44E52", label="timing")
    ax.scatter(sub.exposure_term_ann * 100, sub.delta_sharpe, s=12, alpha=0.6,
               color="#4C72B0", label="exposure")
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("annualised term, %"); ax.set_ylabel("delta Sharpe vs HAR")
    ax.set_title("which term actually moves the decision", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "m2_decomposition.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "P1", "exact exposure/timing decomposition of the sizing "
                                 "decision across the whole model universe",
                      {"n_decompositions": len(df), "identity_closes": ok_ident,
                       "worst_residual": worst_resid, "H1_supported": h1_supported},
                      n_tests=len(df), seed=None,
                      status="COMPLETED" if ok_ident else "IDENTITY_BROKEN")
    con.close()
    print(f"\nwrote results/m2_decomposition.csv, m2_p1_decomposition.json")
    if not ok_ident:
        raise SystemExit("P1 identity does not close -- stop and fix")


if __name__ == "__main__":
    main()
