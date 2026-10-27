"""Pay the cumulative multiplicity correction and delete the caveat.

The audit ended with "no finding in this project should be treated as having
survived a cumulative correction". That caveat is true but EMPTY: nothing
survived the local correction either, because the local threshold already sat
above any realistic effect. The multiplicity penalty grows like sqrt(2 ln K) --
logarithmically, not in K -- so going from the local K to the cumulative 1268
moves the bar by about a fifth and changes no verdict.

Cheaper to pay it than to carry a permanent asterisk.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sps

import db
from config_m1 import RESULTS_DIR
from config_m2 import FIG_DIR

ALPHA = 0.05


def c_two_sided_max(K: int, alpha: float = ALPHA) -> float:
    """Threshold c with P(max_K |z| < c) = 1 - alpha for K independent normals."""
    target = (1.0 - alpha) ** (1.0 / K)
    return float(optimize.brentq(lambda c: (2 * sps.norm.cdf(c) - 1) - target,
                                 0.5, 12.0))


def main():
    print("=" * 78)
    print("CUMULATIVE MULTIPLICITY CORRECTION")
    print("=" * 78)
    con = db.connect()
    audit = json.load(open(os.path.join(RESULTS_DIR, "audit_disclosure.json")))
    K_CUM = int(audit["cumulative_decision"]["selection_relevant_trials_cumulative"])

    # ---- the arithmetic, computed rather than quoted --------------------
    print(f"\nthe penalty grows logarithmically in K, which is why this is cheap")
    print(f"   {'K':>6s} {'sqrt(2 ln K)':>13s} {'exact max-|z| p95':>18s}")
    for K in [98, 200, 255, 1268]:
        print(f"   {K:6d} {np.sqrt(2 * np.log(K)):13.3f} {c_two_sided_max(K):18.3f}")

    # ---- per-milestone: measured inflation, then rescale to K_CUM -------
    null = json.load(open(os.path.join(RESULTS_DIR, "null_thresholds.json")))
    m0_thr = null["percentiles"]["max_abs_t_stat_nw"]["p95"]
    m1 = json.load(open(os.path.join(RESULTS_DIR, "m2_p3_race.json"))) \
        if os.path.exists(os.path.join(RESULTS_DIR, "m2_p3_race.json")) else None
    inc = json.load(open(os.path.join(RESULTS_DIR, "m1_incremental_search.json")))
    m1_thr = float(inc["global_pooled_threshold"]["TRAIN"])
    n6 = json.load(open(os.path.join(RESULTS_DIR, "m2_n6_objective.json")))
    m2_thr = float(max(r["null_max_abs_t_p95"] for r in n6["null"]))

    rows = []
    for name, K_local, thr_local, n_obs in [
            ("M0 search", 98, m0_thr, 2399),
            ("M1 incremental", 255, m1_thr, 2517),
            ("M2 N6 grid", 49, m2_thr, 1259)]:
        c_loc = c_two_sided_max(K_local)
        lam = thr_local / c_loc                       # measured inflation factor
        thr_cum = lam * c_two_sided_max(K_CUM)
        rows.append({
            "family": name, "K_local": K_local, "n_obs": n_obs,
            "threshold_local_empirical": thr_local,
            "c_normal_local": c_loc, "inflation_lambda": lam,
            "c_normal_cumulative": c_two_sided_max(K_CUM),
            "threshold_cumulative": thr_cum,
            "increase_pct": 100.0 * (thr_cum / thr_local - 1.0),
            "mde_ic_local": thr_local / np.sqrt(n_obs),
            "mde_ic_cumulative": thr_cum / np.sqrt(n_obs),
        })
    df = pd.DataFrame(rows)
    print(f"\nrescaling each family's OWN empirical null to K = {K_CUM}")
    print(f"   the inflation factor lambda = empirical threshold / normal quantile")
    print(f"   captures Newey-West on fat tails, and is carried over unchanged")
    print(f"   {'family':16s} {'K_loc':>6s} {'thr_loc':>8s} {'lambda':>7s} "
          f"{'thr_cum':>8s} {'change':>8s} {'MDE_IC loc':>11s} {'MDE_IC cum':>11s}")
    for _, r in df.iterrows():
        print(f"   {r.family:16s} {int(r.K_local):6d} {r.threshold_local_empirical:8.3f} "
              f"{r.inflation_lambda:7.3f} {r.threshold_cumulative:8.3f} "
              f"{r.increase_pct:+7.1f}% {r.mde_ic_local:11.4f} "
              f"{r.mde_ic_cumulative:11.4f}")

    # ---- does paying it change a single verdict? ------------------------
    sig = pd.read_csv(os.path.join(RESULTS_DIR, "signals_detail.csv"))
    m0_max = float(sig.t_stat_nw.abs().max())
    incr = pd.read_csv(os.path.join(RESULTS_DIR, "m1_incremental_results.csv"))
    m1_max = float(incr.loc[incr.split_name == "TRAIN", "t_incremental"].abs().max())
    n6max = float(max(r["observed_max_abs_t"] for r in n6["null"]))
    thr = df.set_index("family").threshold_cumulative

    verdicts = []
    for name, obs, key, local in [
            ("M0 search", m0_max, "M0 search", m0_thr),
            ("M1 incremental", m1_max, "M1 incremental", m1_thr),
            ("M2 N6 grid", n6max, "M2 N6 grid", m2_thr)]:
        verdicts.append({
            "family": name, "observed_max_abs_t": obs,
            "threshold_local": local, "beats_local": obs > local,
            "threshold_cumulative": float(thr[key]),
            "beats_cumulative": obs > float(thr[key])})
    v = pd.DataFrame(verdicts)
    print(f"\ndoes paying it change anything?")
    print(f"   {'family':16s} {'observed':>9s} {'thr local':>10s} {'beats':>6s} "
          f"{'thr cum':>8s} {'beats':>6s}")
    for _, r in v.iterrows():
        print(f"   {r.family:16s} {r.observed_max_abs_t:9.3f} "
              f"{r.threshold_local:10.3f} {'yes' if r.beats_local else 'no':>6s} "
              f"{r.threshold_cumulative:8.3f} "
              f"{'yes' if r.beats_cumulative else 'no':>6s}")

    flips = v[v.beats_local != v.beats_cumulative]
    print(f"\n   verdicts changed by paying the cumulative correction: {len(flips)}")
    if len(flips):
        print(flips.to_string(index=False))
    survives = v[v.beats_cumulative]
    print(f"   families still clearing the CUMULATIVE bar: {len(survives)}/{len(v)}"
          f"{'  -> ' + ', '.join(survives.family) if len(survives) else ''}")

    # ---- threshold-vs-K curve, endpoint at the real cumulative K --------
    Ks = np.unique(np.concatenate([np.arange(10, 1300, 10), [98, 200, 255, 1268]]))
    curve = pd.DataFrame({"K": Ks, "c_normal": [c_two_sided_max(int(k)) for k in Ks]})
    for _, r in df.iterrows():
        curve[r.family] = r.inflation_lambda * curve.c_normal
    curve.to_csv(os.path.join(RESULTS_DIR, "threshold_vs_K.csv"), index=False)

    payload = {
        "K_cumulative": K_CUM,
        "arithmetic": {str(K): {"sqrt_2lnK": float(np.sqrt(2 * np.log(K))),
                                "exact_max_abs_z_p95": c_two_sided_max(K)}
                       for K in [98, 200, 255, 1268]},
        "per_family": df.to_dict("records"),
        "verdicts": v.to_dict("records"),
        "n_verdicts_changed": int(len(flips)),
        "decision": "PAID. Thresholds are now reported at the cumulative K = "
                    f"{K_CUM}, not per milestone. The caveat 'no finding survived a "
                    "cumulative correction' is withdrawn as empty: it was true but "
                    "carried no information, because nothing survived the local "
                    "correction either. Paying it costs "
                    f"{df.increase_pct.mean():.0f}% on the threshold and changes "
                    f"{len(flips)} verdicts.",
        "supersedes": "audit_disclosure.json -> cumulative_decision",
    }
    with open(os.path.join(RESULTS_DIR, "cumulative_correction.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for _, r in df.iterrows():
        ax.plot(curve.K, curve[r.family], lw=1.6, label=f"{r.family} "
                f"(lambda={r.inflation_lambda:.2f})")
    ax.plot(curve.K, curve.c_normal, "k--", lw=1.2, label="normal max |z| (lambda=1)")
    for K, lab in [(98, "M0  K=98"), (255, "M1  K=255"), (200, "N6 grid  K=200"),
                   (K_CUM, f"cumulative  K={K_CUM}")]:
        ax.axvline(K, color="grey", lw=0.8, ls=":")
        ax.annotate(lab, (K, ax.get_ylim()[0]), rotation=90, fontsize=7,
                    va="bottom", ha="right", color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("K -- number of hypotheses in the selection pool")
    ax.set_ylabel("95% null threshold on max |t|")
    ax.set_title("The multiplicity penalty is logarithmic: K=98 to K=1268 costs "
                 f"about {df.increase_pct.mean():.0f}%", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "threshold_vs_K.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "AUDIT", "cumulative multiplicity correction paid; "
                                    "per-milestone caveat withdrawn as empty",
                      {"K_cumulative": K_CUM, "n_verdicts_changed": int(len(flips)),
                       "mean_threshold_increase_pct": float(df.increase_pct.mean())},
                      n_tests=0, seed=None, status="COMPLETED")
    con.close()
    print(f"\nwrote results/cumulative_correction.json, threshold_vs_K.csv, "
          f"figures/threshold_vs_K.png")


if __name__ == "__main__":
    main()
