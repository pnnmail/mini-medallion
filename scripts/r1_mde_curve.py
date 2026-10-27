"""Q1 -- the MDE curve. The one computation worth sitting down for.

The harness is built; what is needed is the null threshold as a function of the
SIZE of the hypothesis space, and the effect size that threshold implies.

Empirically: re-run M0's null harness keeping the full per-hypothesis |t| matrix
(1000 x 98), then subsample K columns for K <= 98.
Analytically: lambda * c_95(K) with lambda measured at K = 98.
The divergence between the two is itself diagnostic.

REGIMES ARE KEPT SEPARATE. Predicting returns has a baseline R^2 of about 0.005;
predicting log variance about 0.49. A magnitude impossible in the first is
ordinary in the second, and mixing them is exactly the error this project made
once already.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sps

import db
from config import MASTER_SEED
from config_m1 import RESULTS_DIR
from config_m2 import FIG_DIR
from discovery import enumerate_search_space, run_search
from features import build_features
from null_harness import synthetic_panel
from panel import research_panel, split_mask
from targets import build_targets

K_GRID = [1, 3, 5, 10, 25, 50, 98, 200, 500, 1268]
N_REP = 1000
Z_POWER = float(sps.norm.ppf(0.84))          # 84% power
SUBSAMPLES = 200

REGIMES = {
    "returns": {"n_eff": 2399, "baseline_r2": 0.005,
                "note": "M0 TRAIN rows; forward return targets"},
    "volatility": {"n_eff": 5753, "baseline_r2": 0.49,
                   "note": "SPY TRAIN rows in the VIX forensics; log-variance targets"},
}

REFERENCE = [
    ("short-term reversal (M0)", "returns", 0.0556,
     "mean IC over 36 cells, correct sign 36/36"),
    ("overnight premium, SPY (M0/M7)", "returns", 1.79 / np.sqrt(2516),
     "t = 1.79 on n = 2516, converted to IC"),
    ("TSMOM 12-1 (M0)", "returns", 0.25 / np.sqrt(107),
     "|t| < 0.25 on 107 monthly observations"),
    ("typical return predictor (literature)", "returns", 0.035,
     "IC 0.02-0.05"),
    ("volatility persistence, SPY (M2)", "volatility", 7.404 / np.sqrt(2138),
     "t = 7.404 on n = 2138"),
    ("VIX over HAR, SPY h=5 (M1/forensics)", "volatility", 14.38 / np.sqrt(5753),
     "t = 14.38 on n = 5753, dR2 0.046"),
]


def c95(K, alpha=0.05):
    tgt = (1.0 - alpha) ** (1.0 / K)
    return float(optimize.brentq(lambda c: (2 * sps.norm.cdf(c) - 1) - tgt, 0.3, 12.0))


def build_null_matrix(n_rep=N_REP):
    """Full per-hypothesis |t| matrix under the M0 null. Same seeds as M0."""
    prices, macro = research_panel()
    specs = enumerate_search_space()
    tr = split_mask(prices.index, "TRAIN")
    T = np.full((n_rep, len(specs)), np.nan)
    t0 = time.time()
    for i in range(n_rep):
        syn, syn_macro, _ = synthetic_panel(prices, macro, i)
        feats = build_features(syn, syn_macro, None)
        tg, traded = build_targets(syn)
        rows = run_search(feats, tg, traded, tr, specs)
        T[i] = [abs(r["t_stat_nw"]) if np.isfinite(r["t_stat_nw"]) else np.nan
                for r in rows]
        if (i + 1) % 250 == 0:
            print(f"      {i + 1}/{n_rep}  ({time.time() - t0:.0f}s)")
    return T, len(specs)


def main():
    print("=" * 78)
    print("Q1  MDE CURVE")
    print("=" * 78)
    con = db.connect()

    cache = os.path.join(RESULTS_DIR, "null_t_matrix.npy")
    if os.path.exists(cache):
        T = np.load(cache)
        K_max = T.shape[1]
        print(f"\n   loaded cached null matrix {T.shape}")
    else:
        print(f"\n   rebuilding the M0 null keeping every hypothesis "
              f"({N_REP} replications)")
        T, K_max = build_null_matrix()
        np.save(cache, T)

    rng = np.random.default_rng(MASTER_SEED + 400)
    emp_98 = float(np.nanpercentile(np.nanmax(T, axis=1), 95))
    print(f"   self-check: empirical p95 of max|t| over all {K_max} hypotheses = "
          f"{emp_98:.3f}")
    print(f"   M0 reported 5.913 -> "
          f"{'REPRODUCED' if abs(emp_98 - 5.913) < 0.01 else 'MISMATCH'}")
    lam = emp_98 / c95(K_max)
    print(f"   lambda measured at K={K_max}: {lam:.3f}")

    rows = []
    for K in K_GRID:
        if K <= K_max:
            vals = []
            for _ in range(SUBSAMPLES):
                cols = rng.choice(K_max, size=K, replace=False)
                vals.append(np.nanpercentile(np.nanmax(T[:, cols], axis=1), 95))
            emp = float(np.mean(vals))
            emp_sd = float(np.std(vals))
        else:
            emp, emp_sd = np.nan, np.nan
        ana = lam * c95(K)
        rows.append({"K": K, "t_null_95_empirical": emp,
                     "t_null_95_empirical_sd": emp_sd,
                     "t_null_95_analytic": ana,
                     "divergence": (emp - ana) if np.isfinite(emp) else np.nan})
    df = pd.DataFrame(rows)

    for reg, cfg in REGIMES.items():
        df[f"MDE_t"] = df.t_null_95_empirical.fillna(df.t_null_95_analytic) + Z_POWER
        df[f"MDE_IC_{reg}"] = df["MDE_t"] / np.sqrt(cfg["n_eff"])

    print(f"\n   {'K':>6s} {'t_null95 emp':>13s} {'+-sd':>6s} {'analytic':>9s} "
          f"{'diverg':>8s} {'MDE_t':>7s} {'MDE_IC ret':>11s} {'MDE_IC vol':>11s}")
    for _, r in df.iterrows():
        e = f"{r.t_null_95_empirical:13.3f}" if np.isfinite(r.t_null_95_empirical) \
            else "            -"
        s = f"{r.t_null_95_empirical_sd:6.3f}" if np.isfinite(r.t_null_95_empirical_sd) \
            else "     -"
        dv = f"{r.divergence:+8.3f}" if np.isfinite(r.divergence) else "       -"
        print(f"   {int(r.K):6d} {e} {s} {r.t_null_95_analytic:9.3f} {dv} "
              f"{r.MDE_t:7.3f} {r['MDE_IC_returns']:11.4f} "
              f"{r['MDE_IC_volatility']:11.4f}")

    div = df.divergence.dropna()
    print(f"\n   empirical minus analytic: mean {div.mean():+.3f}, "
          f"range [{div.min():+.3f}, {div.max():+.3f}]")
    print(f"   the analytic curve is calibrated at K={K_max} by construction; the "
          f"divergence at small K")
    print(f"   shows how far the per-test statistic departs from normality once the "
          f"max is not taken.")

    # K* for each threshold and regime
    kstar = {}
    Kfine = np.unique(np.concatenate([np.arange(1, 2001), K_GRID]))
    ana_fine = lam * np.array([c95(int(k)) for k in Kfine]) + Z_POWER
    for reg, cfg in REGIMES.items():
        mde = ana_fine / np.sqrt(cfg["n_eff"])
        for thr in [0.05, 0.03]:
            ok = Kfine[mde <= thr]
            kstar[f"{reg}_{thr}"] = int(ok.max()) if len(ok) else 0
    print(f"\n   K* -- the largest hypothesis space at which an effect is still "
          f"detectable with 84% power:")
    for reg in REGIMES:
        print(f"      {reg:12s} MDE_IC <= 0.05 : K* = {kstar[f'{reg}_0.05']:5d}     "
              f"MDE_IC <= 0.03 : K* = {kstar[f'{reg}_0.03']:5d}")

    print(f"\n   REFERENCE EFFECTS, by regime")
    ref_rows = []
    for name, reg, ic, note in REFERENCE:
        kmax_det = Kfine[(ana_fine / np.sqrt(REGIMES[reg]["n_eff"])) <= ic]
        k_det = int(kmax_det.max()) if len(kmax_det) else 0
        ref_rows.append({"effect": name, "regime": reg, "IC": ic, "note": note,
                         "K_detectable": k_det})
        print(f"      {name:38s} {reg:11s} IC={ic:.4f}  detectable up to K = "
              f"{k_det if k_det else '<1'}")
    ref = pd.DataFrame(ref_rows)

    df.to_csv(os.path.join(RESULTS_DIR, "q1_mde_curve.csv"), index=False)
    ref.to_csv(os.path.join(RESULTS_DIR, "q1_reference_effects.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "q1_mde_curve.json"), "w") as fh:
        json.dump({"K_grid": K_GRID, "n_replications": int(T.shape[0]),
                   "lambda": lam, "self_check_p95_at_98": emp_98,
                   "power": 0.84, "z_power": Z_POWER,
                   "regimes": REGIMES, "curve": df.to_dict("records"),
                   "K_star": kstar, "reference_effects": ref.to_dict("records"),
                   "divergence_mean": float(div.mean())}, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, reg in zip(axes, REGIMES):
        n_eff = REGIMES[reg]["n_eff"]
        ax.plot(Kfine, ana_fine / np.sqrt(n_eff), color="#4C72B0", lw=1.8,
                label=f"MDE_IC, analytic (lambda={lam:.2f})")
        e = df.dropna(subset=["t_null_95_empirical"])
        ax.plot(e.K, (e.t_null_95_empirical + Z_POWER) / np.sqrt(n_eff), "o",
                color="#C44E52", label="empirical")
        for thr, ls in [(0.05, "--"), (0.03, ":")]:
            ax.axhline(thr, color="grey", ls=ls, lw=1)
            ax.annotate(f"IC={thr}", (1.2, thr), fontsize=7, color="grey",
                        va="bottom")
        for name, r_, ic, _ in REFERENCE:
            if r_ != reg:
                continue
            ax.axhline(ic, color="#55A868", lw=1.1, alpha=0.75)
            ax.annotate(name, (1.2, ic), fontsize=6.5, color="#2f6b46", va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("K -- size of the hypothesis space")
        ax.set_ylabel("minimum detectable IC (84% power)")
        ax.set_title(f"{reg} regime -- n_eff = {n_eff}, baseline R2 "
                     f"~ {REGIMES[reg]['baseline_r2']}", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Q1  what the harness can see, as a function of how much it looks at",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q1_mde_curve.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "Q1", "MDE curve: empirical and analytic null threshold "
                                 "versus K, with regimes separated",
                      {"K_grid": K_GRID, "lambda": lam, "K_star": kstar},
                      n_tests=0, seed=MASTER_SEED, status="COMPLETED")
    con.close()
    print("\nwrote results/q1_mde_curve.csv/.json, q1_reference_effects.csv, "
          "figures/q1_mde_curve.png")


if __name__ == "__main__":
    main()
