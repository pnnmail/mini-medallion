"""P3 -- can you tell IN ADVANCE which forecast will make the better decision?

Three selectors, all computed on TRAIN only, all declared in P0:
  QLIKE       forecast accuracy      (the criterion M1 used)
  SHARPE_IS   in-sample decision performance
  TIMING_COV  cov(w, r_next) alone   (only the term P1 showed matters)

Each ranks the whole model universe on TRAIN; the ranking is then confronted
with what actually happened on VALIDATION.

Two metrics, because they answer different questions:
  rank correlation  -- does the selector order models correctly? uses all 22.
  top-1 percentile  -- if you had to pick one, where would you land? this is
                       the practical question and it is far noisier.

Reference lines are mandatory: a selector that cannot beat picking at random has
no content, however plausible its motivation.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
from scipy import stats as sps

import db
from config_m2 import (FIG_DIR, MASTER_SEED_M2, RESULTS_DIR, SELECTORS, TRADING_DAYS)
from decomposition import pop_cov
from rv_estimators import qlike
from universe_m2 import build_universe, common_mask, weights_and_pnl
from voltarget import perf

PERM_REPS = 5000


def selector_values(tk, split_mask_rank):
    """All three selector values for every model, on one sample."""
    idx, fwd, models, fams, _ = build_universe(tk)
    out = {}
    for name in models:
        _, w, pnl, _ = weights_and_pnl(tk, name, split_mask_rank)
        f = models[name]
        q = qlike(f["target_var"].to_numpy(float)[split_mask_rank],
                  f["pred_var"].to_numpy(float)[split_mask_rank])
        ww = np.asarray(w, float)[split_mask_rank]
        rr = np.asarray(fwd, float)[split_mask_rank]
        ok = np.isfinite(ww) & np.isfinite(rr)
        out[name] = {
            "QLIKE": q,                                   # lower is better
            "SHARPE_IS": perf(pnl, split_mask_rank)["sharpe"],   # higher is better
            "TIMING_COV": pop_cov(ww[ok], rr[ok]) * TRADING_DAYS,  # higher is better
            "family": fams[name],
        }
    return out


def oos_sharpe(tk, split_mask_eval):
    idx, fwd, models, _, _ = build_universe(tk)
    return {name: perf(weights_and_pnl(tk, name, split_mask_eval)[2],
                       split_mask_eval)["sharpe"] for name in models}


DIRECTION = {"QLIKE": -1, "SHARPE_IS": +1, "TIMING_COV": +1}   # +1 = higher is better


def run_race(instruments, rank_masks, eval_masks, label, rng):
    rows, detail = [], []
    for tk in instruments:
        sv = selector_values(tk, rank_masks[tk])
        oos = oos_sharpe(tk, eval_masks[tk])
        names = [n for n in sv if np.isfinite(oos.get(n, np.nan))]
        if len(names) < 5:
            continue
        oos_arr = np.array([oos[n] for n in names])
        for sel in SELECTORS:
            vals = np.array([sv[n][sel] for n in names], float)
            ok = np.isfinite(vals) & np.isfinite(oos_arr)
            if ok.sum() < 5:
                continue
            v, o = vals[ok] * DIRECTION[sel], oos_arr[ok]
            nm = [n for n, k in zip(names, ok) if k]
            rho = float(sps.spearmanr(v, o).statistic)
            perm = np.array([sps.spearmanr(rng.permutation(v), o).statistic
                             for _ in range(PERM_REPS // 10)])
            p_perm = float((np.abs(perm) >= abs(rho)).mean())
            pick = nm[int(np.argmax(v))]
            pick_sh = float(oos[pick])
            pct = float((o < pick_sh).mean() * 100)
            rows.append({
                "race": label, "selector": sel, "instrument": tk,
                "n_models": int(ok.sum()),
                "rank_corr": rho, "rank_corr_p_perm": p_perm,
                "picked_model": pick, "picked_sharpe_oos": pick_sh,
                "picked_percentile": pct,
                "random_choice_sharpe": float(np.mean(o)),
                "random_choice_sd": float(np.std(o, ddof=1)),
                "oracle_sharpe": float(np.max(o)),
                "baseline_sharpe": float(oos["HAR"]),
                "beats_random": bool(pick_sh > np.mean(o)),
                "beats_baseline": bool(pick_sh > oos["HAR"]),
                "oracle_gap": float(np.max(o) - pick_sh),
                "picked_family": sv[pick]["family"],
            })
        for n in names:
            detail.append({"race": label, "instrument": tk, "model": n,
                           "family": sv[n]["family"], "sharpe_oos": oos[n],
                           **{f"sel_{s}": sv[n][s] for s in SELECTORS}})
    return pd.DataFrame(rows), pd.DataFrame(detail)


def report(df, label):
    print(f"\n{'=' * 78}\nRACE {label}\n{'=' * 78}")
    print(f"{'selector':11s} {'inst':5s} {'rankCorr':>9s} {'p_perm':>7s} "
          f"{'picked':26s} {'ShOOS':>7s} {'pctile':>7s} {'random':>7s} "
          f"{'oracle':>7s} {'HAR':>7s}")
    for _, r in df.iterrows():
        print(f"{r.selector:11s} {r.instrument:5s} {r.rank_corr:+9.3f} "
              f"{r.rank_corr_p_perm:7.3f} {r.picked_model:26s} "
              f"{r.picked_sharpe_oos:7.3f} {r.picked_percentile:7.0f} "
              f"{r.random_choice_sharpe:7.3f} {r.oracle_sharpe:7.3f} "
              f"{r.baseline_sharpe:7.3f}")
    agg = df.groupby("selector").agg(
        mean_rank_corr=("rank_corr", "mean"),
        n_rank_corr_positive=("rank_corr", lambda s: int((s > 0).sum())),
        n_significant=("rank_corr_p_perm", lambda s: int((s < 0.05).sum())),
        mean_percentile=("picked_percentile", "mean"),
        n_beats_random=("beats_random", "sum"),
        n_beats_baseline=("beats_baseline", "sum"),
        n=("instrument", "size")).reset_index()
    print(f"\naggregate over {df.instrument.nunique()} instruments:")
    print(agg.round(3).to_string(index=False))
    return agg


def main():
    print("=" * 78)
    print("P3  SELECTOR RACE -- rank on TRAIN, judge on VALIDATION")
    print("=" * 78)
    con = db.connect()
    design = json.load(open(os.path.join(RESULTS_DIR, "m2_design.json")))
    rng = np.random.default_rng(MASTER_SEED_M2 + 73)

    inst = design["race_A"]["instruments"]
    rank_masks = {t: common_mask(t, "TRAIN") for t in inst}
    eval_masks = {t: common_mask(t, "VALIDATION") for t in inst}
    dfA, detA = run_race(inst, rank_masks, eval_masks, "A_train_to_validation", rng)
    aggA = report(dfA, "A  (rank on TRAIN, judge on VALIDATION)")

    best = aggA.sort_values("mean_rank_corr", ascending=False).iloc[0]
    all_same = bool(dfA.groupby("instrument").rank_corr.nunique().eq(1).all())
    none_beats_random = bool(aggA.n_beats_random.sum() == 0)
    premise_dead = bool(all_same or none_beats_random)

    print(f"\nGATE P3: if every selector orders models identically, or none beats a "
          f"random pick,\n   the premise of the milestone is wrong.")
    print(f"   identical orderings: {all_same}   |   none beats random: "
          f"{none_beats_random}   ->  premise "
          f"{'FAILED' if premise_dead else 'holds'}")
    print(f"\n   best selector by mean rank correlation: {best.selector} "
          f"({best.mean_rank_corr:+.3f}, significant in "
          f"{int(best.n_significant)}/{int(best.n)} instruments)")
    qs = aggA[aggA.selector == "QLIKE"].iloc[0]
    print(f"   QLIKE -- the criterion M1 used -- scores "
          f"{qs.mean_rank_corr:+.3f}, mean percentile {qs.mean_percentile:.0f} "
          f"(50 = no ability)")

    con.execute("DELETE FROM m2_selector_race")
    ins = dfA[["selector", "instrument", "picked_model", "picked_sharpe_oos",
               "random_choice_sharpe", "oracle_sharpe", "baseline_sharpe",
               "beats_random", "beats_baseline", "oracle_gap"]].copy()
    ins.insert(2, "split_name", "VALIDATION")
    con.register("sr", ins)
    con.execute("INSERT INTO m2_selector_race SELECT selector, instrument, split_name, "
                "picked_model, picked_sharpe_oos, random_choice_sharpe, oracle_sharpe, "
                "baseline_sharpe, beats_random, beats_baseline, oracle_gap FROM sr")
    con.unregister("sr")
    dfA.to_csv(os.path.join(RESULTS_DIR, "m2_race_A.csv"), index=False)
    detA.to_csv(os.path.join(RESULTS_DIR, "m2_race_A_detail.csv"), index=False)

    payload = {"race_A": {"rows": dfA.to_dict("records"),
                          "aggregate": aggA.to_dict("records"),
                          "instruments": inst,
                          "n_models_per_instrument": int(dfA.n_models.max()),
                          "permutation_reps": PERM_REPS // 10},
               "gate_P3": {"all_selectors_identical": all_same,
                           "none_beats_random": none_beats_random,
                           "premise_failed": premise_dead},
               "best_selector": best.selector,
               "best_mean_rank_corr": float(best.mean_rank_corr)}
    with open(os.path.join(RESULTS_DIR, "m2_p3_race.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, tk in zip(axes, inst):
        s = detA[detA.instrument == tk]
        ax.scatter(-s.sel_QLIKE, s.sharpe_oos, s=22, label="QLIKE (neg)",
                   color="#937860", alpha=0.8)
        ax2 = ax.twiny()
        ax2.scatter(s.sel_TIMING_COV, s.sharpe_oos, s=22, label="TIMING_COV",
                    color="#C44E52", alpha=0.8, marker="^")
        ax2.tick_params(labelsize=6, colors="#C44E52")
        h = s[s.model == "HAR"]
        ax.axhline(float(h.sharpe_oos.iloc[0]), color="black", ls="--", lw=1,
                   label="HAR baseline")
        ax.axhline(float(s.sharpe_oos.mean()), color="grey", ls=":", lw=1,
                   label="random pick")
        ax.set_title(tk, fontsize=10)
        ax.set_ylabel("Sharpe on VALIDATION", fontsize=8)
        ax.set_xlabel("selector value on TRAIN", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="lower right")
    fig.suptitle("P3  does a TRAIN-side selector order models the way VALIDATION does?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "m2_selector_race.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "P3", "selector race: QLIKE vs in-sample Sharpe vs timing "
                                 "covariance, ranked on TRAIN and judged on VALIDATION",
                      {"instruments": inst, "selectors": SELECTORS,
                       "best": best.selector,
                       "premise_failed": premise_dead},
                      n_tests=len(dfA), seed=MASTER_SEED_M2, status="COMPLETED")
    con.close()
    print("\nwrote results/m2_race_A.csv, m2_race_A_detail.csv, m2_p3_race.json")


if __name__ == "__main__":
    main()
