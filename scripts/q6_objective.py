"""N6 -- the constructive section, and the only place in the project where the
search touches the decision objective directly.

Preconditions checked at run time: N1 must have shown the gap runs through the
covariance term, and N4 must NOT have explained it away as reversal.

Grid declared in the specification and not extended under any outcome:
    lambda in {0, 0.25, 0.5, 0.75, 1.0}   blend of HAR and HAR+VIX
    k      in {1, 3, 5, 10, 20}           EWMA smoothing from N3
    cap    in {1.0, 1.5}
50 configurations per instrument, 200 in total, every one of them logged.

The null harness is run over THE SAME 50-configuration grid. A threshold
calibrated on anything narrower would be comparing a search against a
non-search.
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
from inference_m1 import romano_wolf
from m2pair import FAST, SLOW, build_pair, weights_pnl
from panel_m1 import split_mask_m1
from stats_engine import stationary_bootstrap_index
from voltarget import perf

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
K_GRID = [1, 3, 5, 10, 20]
CAPS = [1.0, 1.5]
NULL_REPS = 1000
BASELINE = (0.0, 1, 1.0)     # plain HAR, unsmoothed, cap 1.0


def blended_sigma(f, lam, k):
    s = (1.0 - lam) * f["sigma_" + SLOW].to_numpy(float) + \
        lam * f["sigma_" + FAST].to_numpy(float)
    if k > 1:
        s = pd.Series(s, index=f.index).ewm(span=k, adjust=False,
                                            min_periods=1).mean().to_numpy()
    return s


def check_preconditions():
    n1 = json.load(open(os.path.join(RESULTS_DIR, "m2_n1_decomposition.json")))
    n4 = json.load(open(os.path.join(RESULTS_DIR, "m2_n4_reversal.json")))
    cov_channel = n1["covariance_share_of_numerator"] > 0.5
    not_reversal = not n4["explained_by_reversal"]
    return cov_channel, not_reversal, n1, n4


def main():
    print("=" * 78)
    print("N6  DECISION-ALIGNED OBJECTIVE  (the only tuned section in the project)")
    print("=" * 78)
    con = db.connect()
    rng = np.random.default_rng(MASTER_SEED_M2 + 160)

    cov_ok, rev_ok, n1, n4 = check_preconditions()
    print(f"\nPRECONDITIONS")
    print(f"   N1: the gap runs through the covariance term "
          f"({n1['covariance_share_of_numerator']:.0%} of the numerator)  -> "
          f"{'met' if cov_ok else 'NOT met'}")
    print(f"   N4: not explained by reversal (reversal explains "
          f"{n4['mean_r2_dw_on_reversal']:.1%} of the weight difference)  -> "
          f"{'met' if rev_ok else 'NOT met'}")
    if not (cov_ok and rev_ok):
        db.log_experiment(con, "N6", "skipped: preconditions not met",
                          {"cov_channel": cov_ok, "not_reversal": rev_ok},
                          n_tests=0, seed=None, status="SKIPPED")
        con.close()
        raise SystemExit("N6 preconditions not met -- section skipped by design")

    rows, diff_store = [], {}
    for tk in INSTRUMENTS:
        f = build_pair(tk)
        base_mask = {}
        for split in ["TRAIN", "VALIDATION"]:
            m = split_mask_m1(f.index, tk, split)
            m &= np.isfinite(f["fwd_ret"].to_numpy(float))
            m &= np.isfinite(f["sigma_" + SLOW].to_numpy(float))
            m &= np.isfinite(f["sigma_" + FAST].to_numpy(float))
            base_mask[split] = m
        f2 = f.copy()
        f2["sigma_cfg"] = blended_sigma(f, *BASELINE[:2])
        _, p_base_all, _ = weights_pnl(f2, "cfg", base_mask["VALIDATION"], BASELINE[2])
        diffs = {}
        for lam in LAMBDAS:
            for k in K_GRID:
                sig = blended_sigma(f, lam, k)
                for cap in CAPS:
                    f3 = f.copy()
                    f3["sigma_cfg"] = sig
                    res = {}
                    for split in ["TRAIN", "VALIDATION"]:
                        w, p, dw = weights_pnl(f3, "cfg", base_mask[split], cap)
                        pf = perf(p, np.asarray(base_mask[split]))
                        res[split] = pf["sharpe"]
                        if split == "VALIDATION":
                            sel = np.asarray(base_mask["VALIDATION"])
                            diffs[(lam, k, cap)] = (p - p_base_all)[sel]
                            turn = float(dw[sel].sum() / (sel.sum() / TRADING_DAYS))
                    rows.append({"instrument": tk, "lambda": lam, "k": k, "cap": cap,
                                 "sharpe_train": res["TRAIN"],
                                 "sharpe_val": res["VALIDATION"],
                                 "turnover_annual": turn,
                                 "is_baseline": (lam, k, cap) == BASELINE})
        diff_store[tk] = diffs
    df = pd.DataFrame(rows)
    print(f"\n{len(df)} configurations evaluated "
          f"({len(LAMBDAS)}x{len(K_GRID)}x{len(CAPS)} per instrument x "
          f"{len(INSTRUMENTS)} instruments)")

    # ---- null over the SAME grid ----------------------------------------
    print(f"\nNULL HARNESS over the same {len(LAMBDAS) * len(K_GRID) * len(CAPS)}-"
          f"configuration grid, {NULL_REPS} block-bootstrap replications")
    null_rows = []
    for tk in INSTRUMENTS:
        keys = [k for k in diff_store[tk] if k != BASELINE]
        D = np.column_stack([diff_store[tk][k] for k in keys])
        D = D[np.isfinite(D).all(1)]
        n = len(D)
        # Configurations that are numerically identical to the baseline produce a
        # zero difference series. Dividing by its standard deviation manufactures
        # an astronomical t and an astronomical null threshold, which would make
        # "does not beat the null" true for the wrong reason. Drop them and say
        # how many. On instruments where the cap never binds, the cap = 1.5 half
        # of the grid IS the cap = 1.0 half.
        scale = float(np.median(np.abs(D))) or 1.0
        sd0 = D.std(0, ddof=1)
        live = sd0 > max(1e-10, 1e-8 * scale)
        n_dropped = int((~live).sum())
        keys = [k for k, ok in zip(keys, live) if ok]
        D = D[:, live]
        Dc = D - D.mean(0)
        obs_t = D.mean(0) / (D.std(0, ddof=1) / np.sqrt(n))
        T_star = np.empty((NULL_REPS, D.shape[1]))
        for r in range(NULL_REPS):
            idx = stationary_bootstrap_index(n, BOOTSTRAP_MEAN_BLOCK, rng)
            B = Dc[idx]
            sd = B.std(0, ddof=1)
            T_star[r] = np.where(sd > max(1e-10, 1e-8 * scale),
                                 B.mean(0) / (sd / np.sqrt(n)), np.nan)
        thr = float(np.nanquantile(np.nanmax(np.abs(T_star), axis=1), 0.95))
        rej, adj_p, _ = romano_wolf(obs_t, T_star, 0.05)
        best = int(np.nanargmax(np.abs(obs_t)))
        null_rows.append({
            "instrument": tk, "n_configs": len(keys),
            "n_configs_dropped_as_duplicates": n_dropped, "n_obs": n,
            "null_max_abs_t_p95": thr,
            "observed_max_abs_t": float(np.nanmax(np.abs(obs_t))),
            "best_config": str(keys[best]),
            "best_rw_p": float(adj_p[best]),
            "n_rw_rejected": int(rej.sum()),
            "beats_null": bool(np.nanmax(np.abs(obs_t)) > thr),
        })
    nl = pd.DataFrame(null_rows)
    print(f"   {'inst':5s} {'live':>5s} {'dup':>4s} {'obs max|t|':>11s} {'null p95':>9s} "
          f"{'RW rejected':>12s} {'best cfg':>18s} {'RW p':>7s} {'beats':>6s}")
    for _, r in nl.iterrows():
        print(f"   {r.instrument:5s} {int(r.n_configs):5d} "
              f"{int(r.n_configs_dropped_as_duplicates):4d} {r.observed_max_abs_t:11.3f} "
              f"{r.null_max_abs_t_p95:9.3f} {int(r.n_rw_rejected):12d} "
              f"{r.best_config:>18s} {r.best_rw_p:7.3f} "
              f"{'yes' if r.beats_null else 'no':>6s}")

    n_beat = int(nl.beats_null.sum())
    total_rw = int(nl.n_rw_rejected.sum())
    passed = bool(n_beat >= 1 and total_rw >= 1)

    print(f"\n   configurations clearing the grid-calibrated null: "
          f"{n_beat}/{len(nl)} instruments; Romano-Wolf rejects {total_rw} "
          f"configurations in total")
    status = "PASSED" if passed else "FAILED_NULL"
    print(f"   N6 status: {status}")
    if not passed:
        print("   No configuration is recommended, and TEST is NOT opened: the")
        print("   specification calls for a single TEST access AFTER a configuration")
        print("   is finally chosen, and none was.")

    print(f"\n   best configuration per instrument by VALIDATION Sharpe "
          f"(reported, not recommended):")
    for tk in INSTRUMENTS:
        s = df[df.instrument == tk]
        b = s.loc[s.sharpe_val.idxmax()]
        base = s[s.is_baseline].iloc[0]
        print(f"      {tk:5s} lambda={b['lambda']:.2f} k={int(b.k):2d} "
              f"cap={b.cap:.1f}  ValSharpe={b.sharpe_val:.4f} vs baseline "
              f"{base.sharpe_val:.4f}  (+{b.sharpe_val - base.sharpe_val:.4f})")

    df.to_csv(os.path.join(RESULTS_DIR, "m2_n6_grid.csv"), index=False)
    nl.to_csv(os.path.join(RESULTS_DIR, "m2_n6_null.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "m2_n6_objective.json"), "w") as fh:
        json.dump({"grid": {"lambdas": LAMBDAS, "k": K_GRID, "caps": CAPS,
                            "n_per_instrument": len(LAMBDAS) * len(K_GRID) * len(CAPS),
                            "n_total": len(df)},
                   "baseline": {"lambda": BASELINE[0], "k": BASELINE[1],
                                "cap": BASELINE[2]},
                   "null": nl.to_dict("records"),
                   "rows": df.to_dict("records"),
                   "n_trials_total": len(df),
                   "status": status,
                   "test_opened": False,
                   "preconditions": {"cov_channel": cov_ok, "not_reversal": rev_ok}},
                  fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, tk in zip(axes, INSTRUMENTS):
        s = df[(df.instrument == tk) & (df.cap == 1.0)]
        p = s.pivot(index="k", columns="lambda", values="sharpe_val")
        im = ax.imshow(p.to_numpy(), aspect="auto", cmap="RdYlGn",
                       origin="lower")
        ax.set_xticks(range(len(LAMBDAS)))
        ax.set_xticklabels([f"{l:.2f}" for l in LAMBDAS], fontsize=7)
        ax.set_yticks(range(len(K_GRID)))
        ax.set_yticklabels(K_GRID, fontsize=7)
        ax.set_xlabel("lambda (0 = HAR, 1 = +VIX)", fontsize=8)
        ax.set_ylabel("smoothing k", fontsize=8)
        ax.set_title(f"{tk}  VALIDATION Sharpe, cap 1.0", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("N6  the 50-configuration grid -- none of it clears a null "
                 "calibrated on the same grid", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "m2_n6_grid.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "N6", "decision-aligned objective: 50-configuration grid per "
                                 "instrument, null harness run over the same grid, "
                                 "Romano-Wolf across it",
                      {"lambdas": LAMBDAS, "k_grid": K_GRID, "caps": CAPS,
                       "n_trials_total": len(df), "n_beat_null": n_beat,
                       "test_opened": False},
                      n_tests=len(df), seed=MASTER_SEED_M2, status=status)
    con.close()
    print("\nwrote results/m2_n6_grid.csv, m2_n6_null.csv, m2_n6_objective.json")


if __name__ == "__main__":
    main()
