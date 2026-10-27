"""M2 -- run the frozen search space on synthetic data and build the thresholds."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config import (BOOTSTRAP_MEAN_BLOCK, FIG_DIR, MASTER_SEED,
                    NULL_PERCENTILES, NULL_REPLICATIONS_MIN,
                    NULL_REPLICATIONS_TARGET, NULL_TIME_BUDGET_SECONDS, RESULTS_DIR)
from discovery import enumerate_search_space
from null_harness import derive_seed, run_replication
from panel import research_panel, split_mask
from provenance import estimator_fingerprint

METRICS = ["max_abs_t_stat_nw", "max_abs_ic", "max_sharpe", "max_net_sharpe",
           "max_abs_t_val_selected"]


def main():
    print("=" * 78)
    print("M2  NULL HARNESS")
    print("=" * 78)
    con = db.connect()
    prices, macro = research_panel()
    specs = enumerate_search_space()
    tr = split_mask(prices.index, "TRAIN")
    va = split_mask(prices.index, "VALIDATION")
    print(f"search space: {len(specs)} hypotheses | TRAIN rows {tr.sum()} | "
          f"VAL rows {va.sum()}")

    # ---- calibrate N against the time budget --------------------------
    t0 = time.time()
    for i in range(3):
        run_replication(prices, macro, specs, tr, va, 900_000 + i)
    per_rep = (time.time() - t0) / 3.0
    n_budget = int(NULL_TIME_BUDGET_SECONDS / max(per_rep, 1e-6))
    n_rep = int(min(NULL_REPLICATIONS_TARGET, max(NULL_REPLICATIONS_MIN, n_budget)))
    print(f"timing probe: {per_rep:.3f}s/replication -> N = {n_rep} "
          f"(budget {NULL_TIME_BUDGET_SECONDS}s allows {n_budget})")

    # ---- run ----------------------------------------------------------
    recs = []
    start = time.time()
    for i in range(n_rep):
        rt0 = time.time()
        m = run_replication(prices, macro, specs, tr, va, i)
        m["elapsed_seconds"] = time.time() - rt0
        recs.append(m)
        if (i + 1) % 100 == 0:
            print(f"   {i + 1}/{n_rep}  elapsed {time.time() - start:.0f}s")
    total = time.time() - start
    print(f"done: {n_rep} replications in {total:.0f}s")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "seeds"} for r in recs])
    df["n_tests"] = len(specs)

    con.execute("DELETE FROM null_distribution")
    ins = df[["replication_id", "seed", "n_tests", "max_abs_t_stat_nw", "max_abs_ic",
              "max_sharpe", "max_net_sharpe", "max_abs_t_val_selected",
              "elapsed_seconds"]].copy()
    con.register("nd", ins)
    con.execute("INSERT INTO null_distribution SELECT * FROM nd")
    con.unregister("nd")
    df.to_csv(os.path.join(RESULTS_DIR, "null_distribution.csv"), index=False)

    # full seed ledger -- a null distribution nobody can reproduce is not evidence
    seed_rows = []
    for r in recs:
        for name, s in r["seeds"].items():
            seed_rows.append({"replication_id": r["replication_id"],
                              "series": name, "seed": s})
    pd.DataFrame(seed_rows).to_csv(os.path.join(RESULTS_DIR, "null_seed_ledger.csv"),
                                   index=False)

    # ---- thresholds ---------------------------------------------------
    thr = {"master_seed": MASTER_SEED, "n_replications": n_rep,
           "n_tests_per_replication": len(specs),
           "mean_block": BOOTSTRAP_MEAN_BLOCK,
           "bootstrap": "stationary block bootstrap (Politis-Romano), per-series independent",
           "seconds_per_replication": round(per_rep, 4),
           "total_seconds": round(total, 1),
           "estimator_fingerprint": estimator_fingerprint(),
           "seed_derivation": "master_seed + replication_id*1000003 + crc32(series_name), mod 2^31-1",
           "percentiles": {}}
    for m in METRICS:
        v = df[m].to_numpy(float)
        v = v[np.isfinite(v)]
        rec = {"n_finite": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
               "max": float(v.max())}
        for p in NULL_PERCENTILES:
            if p == 99 and n_rep < 1000:
                rec["p99"] = None
                rec["p99_note"] = "not reported: N < 1000, the 99th percentile would be noise"
                continue
            rec[f"p{p}"] = float(np.percentile(v, p))
        thr["percentiles"][m] = rec

    with open(os.path.join(RESULTS_DIR, "null_thresholds.json"), "w") as fh:
        json.dump(thr, fh, indent=2)

    # ---- figures ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.4))
    for ax, m in zip(axes, METRICS):
        v = df[m].to_numpy(float)
        v = v[np.isfinite(v)]
        ax.hist(v, bins=40, color="#4C72B0", edgecolor="white")
        p95 = np.percentile(v, 95)
        ax.axvline(p95, color="#C44E52", lw=2, label=f"p95={p95:.2f}")
        if n_rep >= 1000:
            p99 = np.percentile(v, 99)
            ax.axvline(p99, color="#8172B2", lw=1.5, ls="--", label=f"p99={p99:.2f}")
        ax.set_title(m, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"M2 null distribution -- {n_rep} replications x {len(specs)} hypotheses",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "null_distributions.png"), dpi=130)
    plt.close(fig)

    db.log_experiment(con, "M2", "null harness on block-bootstrapped synthetic data",
                      {"n_replications": n_rep, "mean_block": BOOTSTRAP_MEAN_BLOCK,
                       "seconds_per_replication": round(per_rep, 4),
                       "fingerprint": estimator_fingerprint()["_joint"]},
                      n_tests=len(specs) * n_rep, seed=MASTER_SEED, status="COMPLETED")
    con.close()

    print("\nNULL THRESHOLDS")
    for m in METRICS:
        r = thr["percentiles"][m]
        p99 = r.get("p99")
        print(f"   {m:26s} median={r['median']:6.3f}  p95={r['p95']:6.3f}  "
              f"p99={p99 if p99 is None else round(p99, 3)}  max={r['max']:6.3f}")
    print(f"\nwrote results/null_thresholds.json, null_distribution.csv, "
          f"null_seed_ledger.csv, figures/null_distributions.png")


if __name__ == "__main__":
    main()
