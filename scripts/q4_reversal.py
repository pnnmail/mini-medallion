"""N4 -- is the slow forecast's advantage a disguised reversal bet?

The most important test in the milestone, and it closes a loop back to M0.

If the M1 mechanism is right, the slow forecast wins not because it forecasts
better but because lagging accidentally creates a short-term reversal bet: the
position is not cut on the way down and is therefore larger just before the
bounce. M0 already measured that reversal -- correct sign in 36 of 36 cells,
mean IC -0.056 -- and already showed it does not survive costs.

Two constructions, both declared before running:

  PROJECTION  regress the actual weight difference on reversal signals, then
              rebuild the strategy from the FITTED part alone. If that
              reproduces the gap, the gap is reversal.
  CALIBRATED  an explicit reversal overlay on the fast forecast, scaled so its
              weight movement matches the observed weight difference.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config import BOOTSTRAP_MEAN_BLOCK
from config_m1 import RESULTS_DIR
from config_m2 import FIG_DIR, INSTRUMENTS, MASTER_SEED_M2
from features import zscore_causal
from m2pair import FAST, SLOW, build_pair, dev_mask, weights_pnl
from sharpe_inference import sharpe_difference_lw
from stats_engine import stationary_bootstrap_index
from voltarget import perf, strategy_pnl

M0_MEAN_IC = -0.0556        # measured in M0, 36/36 cells correct sign
REV_HORIZONS = [1, 5]
STD_WINDOW = 252


def reversal_signals(frame, mask):
    r = frame["ret"]
    sig = {}
    for h in REV_HORIZONS:
        past = r.rolling(h, min_periods=h).sum()
        sig[f"rev{h}"] = (-zscore_causal(past, STD_WINDOW)).to_numpy(float)
    return sig


def main():
    print("=" * 78)
    print("N4  IS THE ADVANTAGE A DISGUISED REVERSAL BET?")
    print("=" * 78)
    con = db.connect()
    rng = np.random.default_rng(MASTER_SEED_M2 + 140)
    rows = []

    for tk in INSTRUMENTS:
        f = build_pair(tk)
        m = dev_mask(tk, f)
        sel = np.asarray(m)
        w_s, p_s, _ = weights_pnl(f, SLOW, m)
        w_f, p_f, _ = weights_pnl(f, FAST, m)
        fwd = f["fwd_ret"].to_numpy(float)

        sig = reversal_signals(f, m)
        X_cols = [sig[k] for k in sorted(sig)]
        dw = w_s - w_f

        ok = sel & np.isfinite(dw)
        for c in X_cols:
            ok &= np.isfinite(c)
        X = np.column_stack([np.ones(int(ok.sum()))] + [c[ok] for c in X_cols])
        y = dw[ok]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fit = X @ beta
        resid = y - fit
        r2 = float(1.0 - np.var(resid) / np.var(y)) if np.var(y) > 0 else np.nan

        # PROJECTION: fast weight plus only the reversal-explained part
        w_proj = np.full(len(dw), np.nan)
        w_res = np.full(len(dw), np.nan)
        w_proj[ok] = w_f[ok] + fit
        w_res[ok] = w_f[ok] + resid
        p_proj, _, _ = strategy_pnl(w_proj, fwd, 0.0)
        p_res, _, _ = strategy_pnl(w_res, fwd, 0.0)

        # CALIBRATED: explicit overlay scaled to the same weight movement
        rev = sig["rev1"]
        okc = sel & np.isfinite(rev)
        theta = float(np.nanstd(dw[ok]) / np.nanstd(rev[okc])) if np.nanstd(rev[okc]) > 0 \
            else np.nan
        w_cal = np.full(len(dw), np.nan)
        w_cal[okc] = w_f[okc] + theta * rev[okc]
        p_cal, _, _ = strategy_pnl(w_cal, fwd, 0.0)

        sh_slow = perf(p_s, sel)["sharpe"]
        sh_fast = perf(p_f, sel)["sharpe"]
        sh_proj = perf(p_proj, ok)["sharpe"]
        sh_res = perf(p_res, ok)["sharpe"]
        sh_cal = perf(p_cal, okc)["sharpe"]

        gap_slow = sh_slow - sh_fast
        gap_proj = sh_proj - sh_fast
        gap_res = sh_res - sh_fast
        gap_cal = sh_cal - sh_fast
        lw = sharpe_difference_lw(p_s[ok], p_proj[ok])

        rows.append({
            "instrument": tk, "n": int(ok.sum()),
            "r2_dw_on_reversal": r2,
            "beta_rev1": float(beta[1]), "beta_rev5": float(beta[2]),
            "sharpe_fast": sh_fast, "sharpe_slow": sh_slow,
            "sharpe_projection": sh_proj, "sharpe_residual_part": sh_res,
            "sharpe_calibrated_overlay": sh_cal,
            "gap_slow": gap_slow, "gap_projection": gap_proj,
            "gap_residual_part": gap_res, "gap_calibrated": gap_cal,
            "reversal_match_residual": gap_slow - gap_proj,
            "share_of_gap_explained": (gap_proj / gap_slow) if gap_slow != 0 else np.nan,
            "p_lw_slow_vs_projection": lw.get("p_lw", np.nan),
            "theta_calibrated": theta,
        })

    df = pd.DataFrame(rows)

    print(f"\nHow much of the weight difference IS a reversal signal?")
    print(f"   {'inst':5s} {'R2':>7s} {'b_rev1':>9s} {'b_rev5':>9s}")
    for _, r in df.iterrows():
        print(f"   {r.instrument:5s} {r.r2_dw_on_reversal:7.4f} {r.beta_rev1:+9.5f} "
              f"{r.beta_rev5:+9.5f}")
    mean_r2 = float(df.r2_dw_on_reversal.mean())
    print(f"   mean R2 = {mean_r2:.4f}  -- the reversal signals explain "
          f"{mean_r2:.1%} of the variance of the weight difference")

    print(f"\nDoes an EXPLICIT reversal tilt reproduce the gap?")
    print(f"   {'inst':5s} {'gap slow':>9s} {'projection':>11s} {'residual':>9s} "
          f"{'calibrated':>11s} {'explained':>10s} {'p(slow=proj)':>13s}")
    for _, r in df.iterrows():
        print(f"   {r.instrument:5s} {r.gap_slow:+9.4f} {r.gap_projection:+11.4f} "
              f"{r.gap_residual_part:+9.4f} {r.gap_calibrated:+11.4f} "
              f"{r.share_of_gap_explained:+10.1%} {r.p_lw_slow_vs_projection:13.3f}")

    n_explained = int(((df.share_of_gap_explained > 0.5) &
                       (df.share_of_gap_explained < 2.0)).sum())
    n_indist = int((df.p_lw_slow_vs_projection > 0.05).sum())
    explained = bool(n_explained >= 3 and n_indist >= 3)
    print(f"\n   the projection reproduces 50-200% of the gap in "
          f"{n_explained}/{len(df)} instruments")
    print(f"   the slow strategy is statistically indistinguishable from the "
          f"reversal-only projection in {n_indist}/{len(df)}")
    print(f"   EXPLAINED_BY_REVERSAL: {'YES' if explained else 'NO'}")

    if mean_r2 < 0.05:
        print(f"\n   BUT the premise is weak at the source: the reversal signals "
              f"explain only")
        print(f"   {mean_r2:.1%} of the weight difference. Whatever the projected "
              f"strategy does, most")
        print(f"   of what separates the two forecasts is NOT a reversal bet.")

    print(f"\n   M0 measured short-term reversal at mean IC {M0_MEAN_IC:+.4f} "
          f"(36/36 cells correct sign)")
    print(f"   and showed it does not survive costs. If the gap were reversal, the "
          f"right conclusion")
    print(f"   would not be 'use the worse forecast' but 'separate the two decisions'.")

    df.to_csv(os.path.join(RESULTS_DIR, "m2_n4_reversal.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "m2_n4_reversal.json"), "w") as fh:
        json.dump({"rows": df.to_dict("records"), "m0_mean_ic": M0_MEAN_IC,
                   "mean_r2_dw_on_reversal": mean_r2,
                   "n_gap_reproduced": n_explained,
                   "n_indistinguishable": n_indist,
                   "explained_by_reversal": explained,
                   "constructions": {
                       "projection": "regress w_slow - w_fast on causal reversal "
                                     "z-scores at 1 and 5 days, rebuild from the "
                                     "fitted part",
                       "calibrated": "explicit overlay on the fast forecast, theta "
                                     "scaled so the overlay's weight movement matches "
                                     "the observed difference"}},
                  fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(df))
    ax.bar(x - 0.3, df.gap_slow, 0.2, label="actual gap (slow - fast)", color="#4C72B0")
    ax.bar(x - 0.1, df.gap_projection, 0.2, label="reversal-explained part",
           color="#C44E52")
    ax.bar(x + 0.1, df.gap_residual_part, 0.2, label="non-reversal part",
           color="#55A868")
    ax.bar(x + 0.3, df.gap_calibrated, 0.2, label="explicit calibrated overlay",
           color="#DD8452")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(df.instrument)
    ax.set_ylabel("Sharpe gap vs the fast forecast")
    ax.set_title("N4  can an explicit reversal tilt reproduce the advantage?",
                 fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "m2_n4_reversal.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "N4", "is the slow forecast's advantage a disguised "
                                 "short-term reversal bet?",
                      {"mean_r2": mean_r2, "n_explained": n_explained,
                       "explained_by_reversal": explained},
                      n_tests=len(df) * 3, seed=MASTER_SEED_M2,
                      status="EXPLAINED_BY_REVERSAL" if explained else "COMPLETED")
    con.close()
    print("\nwrote results/m2_n4_reversal.csv/.json")


if __name__ == "__main__":
    main()
