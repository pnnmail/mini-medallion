"""N5 -- crisis dependence, applied to the DIFFERENCE.

M1 established that the effective sample size for vol targeting is the number of
crises, not the number of days. That applies harder here: the hypothesised
mechanism operates in stress, and the sample contains five stress episodes.
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
from config_m2 import FIG_DIR, INSTRUMENTS, MASTER_SEED_M2
from m2pair import FAST, SLOW, build_pair, dev_mask, weights_pnl
from voltarget import (crisis_block_bootstrap, drawdown_episodes,
                       leave_one_crisis_out, perf)

BOOT = 2000


def main():
    print("=" * 78)
    print("N5  CRISIS DEPENDENCE OF THE GAP")
    print("=" * 78)
    con = db.connect()
    rng = np.random.default_rng(MASTER_SEED_M2 + 150)
    rows, loo_rows = [], []

    for tk in INSTRUMENTS:
        f = build_pair(tk)
        m = dev_mask(tk, f)
        sel = np.asarray(m)
        _, p_s, _ = weights_pnl(f, SLOW, m)
        _, p_f, _ = weights_pnl(f, FAST, m)
        bh = np.where(sel, f["fwd_ret"].to_numpy(float), 0.0)
        eps = drawdown_episodes(bh, f.index, span="decline")

        gap_full = perf(p_s, sel)["sharpe"] - perf(p_f, sel)["sharpe"]
        loo = leave_one_crisis_out(p_s, p_f, eps, len(f))
        keep = sel.copy()
        for e in eps:
            keep[e["start"]:e["end"] + 1] = False
        gap_ex_all = perf(p_s, keep)["sharpe"] - perf(p_f, keep)["sharpe"]

        yrs = f.index.year.to_numpy()
        ex = {}
        for lab, drop in [("2008", {2008}), ("2020", {2020}),
                          ("2008_and_2020", {2008, 2020})]:
            k = sel & ~np.isin(yrs, list(drop))
            ex[f"gap_ex_{lab}"] = perf(p_s, k)["sharpe"] - perf(p_f, k)["sharpe"]

        cb = crisis_block_bootstrap(np.where(sel, p_s, np.nan),
                                    np.where(sel, p_f, np.nan), eps, len(f), BOOT, rng)
        rows.append({
            "instrument": tk, "gap_full": gap_full, "gap_ex_all_crises": gap_ex_all,
            "crisis_contribution_top5": gap_full - gap_ex_all,
            **ex,
            "loo_min_gap": float(np.nanmin([x["advantage"] for x in loo])) if loo else np.nan,
            "loo_max_gap": float(np.nanmax([x["advantage"] for x in loo])) if loo else np.nan,
            "n_episodes": len(eps),
            "crisis_block_p_gap_le_0": cb["p_advantage_le_0"],
            "crisis_block_ci_lo": cb["ci_lo"], "crisis_block_ci_hi": cb["ci_hi"],
            "crisis_n_blocks": cb["n_blocks"],
        })
        for x in loo:
            loo_rows.append({"instrument": tk, **x})

    df = pd.DataFrame(rows)
    loo_df = pd.DataFrame(loo_rows)

    print(f"\n{'inst':5s} {'gap full':>9s} {'ex2008':>8s} {'ex2020':>8s} "
          f"{'ex both':>8s} {'ex all 5':>9s} {'top5 contrib':>13s} {'LOOmin':>8s}")
    for _, r in df.iterrows():
        print(f"{r.instrument:5s} {r.gap_full:+9.4f} {r.gap_ex_2008:+8.4f} "
              f"{r.gap_ex_2020:+8.4f} {r.gap_ex_2008_and_2020:+8.4f} "
              f"{r.gap_ex_all_crises:+9.4f} {r.crisis_contribution_top5:+13.4f} "
              f"{r.loo_min_gap:+8.4f}")

    print(f"\nCRISIS-BLOCK BOOTSTRAP on the difference (resampling episodes, not days)")
    for _, r in df.iterrows():
        print(f"   {r.instrument:5s} blocks={int(r.crisis_n_blocks):3d}  "
              f"P(gap<=0)={r.crisis_block_p_gap_le_0:.3f}  95% CI "
              f"[{r.crisis_block_ci_lo:+.4f}, {r.crisis_block_ci_hi:+.4f}]")

    n_survives = int((df.gap_ex_all_crises > 0).sum())
    n_loo = int((df.loo_min_gap > 0).sum())
    n_ex_both = int((df.gap_ex_2008_and_2020 > 0).sum())
    print(f"\n   gap still positive with all five declines removed: "
          f"{n_survives}/{len(df)}")
    print(f"   gap still positive under every leave-one-crisis-out: {n_loo}/{len(df)}")
    print(f"   gap still positive with 2008 and 2020 both removed: "
          f"{n_ex_both}/{len(df)}")
    crisis_dependent = bool(n_survives < 3)
    print(f"   -> {'CRISIS-DEPENDENT' if crisis_dependent else 'not crisis-dependent'}")

    df.to_csv(os.path.join(RESULTS_DIR, "m2_n5_crisis.csv"), index=False)
    loo_df.to_csv(os.path.join(RESULTS_DIR, "m2_n5_loo.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "m2_n5_crisis.json"), "w") as fh:
        json.dump({"rows": df.to_dict("records"), "loo": loo_df.to_dict("records"),
                   "n_survives_ex_all_crises": n_survives,
                   "n_survives_loo": n_loo,
                   "n_survives_ex_2008_and_2020": n_ex_both,
                   "crisis_dependent": crisis_dependent}, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(df))
    ax.bar(x - 0.3, df.gap_full, 0.2, label="full sample", color="#4C72B0")
    ax.bar(x - 0.1, df.gap_ex_2008, 0.2, label="ex 2008", color="#DD8452")
    ax.bar(x + 0.1, df.gap_ex_2020, 0.2, label="ex 2020", color="#55A868")
    ax.bar(x + 0.3, df.gap_ex_all_crises, 0.2, label="ex all five declines",
           color="#C44E52")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(df.instrument)
    ax.set_ylabel("Sharpe gap, HAR minus HAR+VIX")
    ax.set_title("N5  how much of the gap is the crises?", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "m2_n5_crisis.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "N5", "crisis dependence of the Sharpe gap: leave-one-out "
                                 "and crisis-block bootstrap on the difference",
                      {"n_survives_ex_all": n_survives, "n_survives_loo": n_loo,
                       "crisis_dependent": crisis_dependent},
                      n_tests=len(df) * 3, seed=MASTER_SEED_M2,
                      status="FAILED_CRISIS_DEPENDENCE" if crisis_dependent
                      else "COMPLETED")
    con.close()
    print("\nwrote results/m2_n5_crisis.csv/.json, m2_n5_loo.csv")


if __name__ == "__main__":
    main()
