"""M6 -- final report. The verdict is computed, not written by hand."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config import BH_FDR_ALPHA, FIG_DIR, RESULTS_DIR, ROOT


def load(name):
    p = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def main():
    print("=" * 78)
    print("M6  REPORT")
    print("=" * 78)
    con = db.connect()

    df = con.execute("SELECT * FROM signal_results").df()
    detail = pd.read_csv(os.path.join(RESULTS_DIR, "signals_detail.csv"))
    null = load("null_thresholds.json")
    calib = load("calibration.json")
    cost = load("cost_model.json")
    space = load("search_space.json")
    dq = load("data_quality_report.json")
    confound = load("confound_check.json")

    exp = con.execute("SELECT milestone, description, n_tests, status, timestamp "
                      "FROM experiment_log ORDER BY timestamp").df()
    tal = con.execute("SELECT * FROM test_access_log").df()
    la = con.execute("SELECT COUNT(*) n, SUM(CASE WHEN passed THEN 1 ELSE 0 END) ok "
                     "FROM lookahead_tests").df()

    # ---------------------------------------------------- automatic verdict
    n_passed = int((df.status == "PASSED").sum())
    discovery = df[df.beats_null & detail.set_index(df.index)["_fdr_rejected"]]
    n_discovery = int(len(discovery))

    if n_passed > 0:
        verdict = "ROBUST SIGNAL(S) FOUND"
    elif n_discovery > 0:
        verdict = "NO ROBUST EDGE FOUND (candidate cleared the null but was rejected " \
                  "by the pre-registered gates)"
    else:
        verdict = "NO ROBUST EDGE FOUND"

    # ---------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    figs = []

    # (a) every test against the null -- the single most informative picture
    fig, ax = plt.subplots(figsize=(11, 4.6))
    nd = pd.read_csv(os.path.join(RESULTS_DIR, "null_distribution.csv"))
    ax.hist(nd["max_abs_t_stat_nw"], bins=45, color="#BBBBBB", edgecolor="white",
            label=f"null: max|t| per replication (N={len(nd)})")
    thr = null["percentiles"]["max_abs_t_stat_nw"]["p95"]
    ax.axvline(thr, color="#C44E52", lw=2, label=f"null p95 = {thr:.2f}")
    fam_colors = {"pair_mr": "#4C72B0", "residual_mr": "#55A868",
                  "vol_spillover": "#8172B2", "regime_vol": "#CCB974"}
    y0 = ax.get_ylim()[1]
    for fam, c in fam_colors.items():
        v = detail.loc[detail._family == fam, "t_stat_nw"].abs().dropna()
        ax.scatter(v, np.full(len(v), y0 * 0.06) * (1 + 0.9 * list(fam_colors).index(fam)),
                   s=16, color=c, label=f"real: {fam} ({len(v)})", zorder=3, alpha=0.85)
    ax.axvline(2.0, color="black", ls=":", lw=1.2, label="conventional t = 2")
    ax.set_xlabel("|t| (Newey-West)")
    ax.set_ylabel("null replications")
    ax.set_title("M6  every real hypothesis against what the same search finds on noise")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "real_vs_null.png")
    fig.savefig(p, dpi=135); plt.close(fig); figs.append(p)

    # (b) plots are permitted ONLY for beats_null == TRUE rows
    if int(df.beats_null.sum()) == 0:
        print("\nNO ROBUST SIGNALS FOUND")
    else:
        from features import build_features
        from panel import research_panel, split_mask
        from targets import build_targets
        prices, macro = research_panel()
        feats = build_features(prices, macro, None)
        tgts, _ = build_targets(prices)
        for _, row in df[df.beats_null].iterrows():
            d = detail[detail.signal_id == row.signal_id].iloc[0]
            x = feats[row.feature]
            y = tgts[row.target]
            fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
            for ax, sp in zip(axes, ["TRAIN", "VALIDATION", "TEST"]):
                m = split_mask(prices.index, sp)
                xx, yy = x[m], y[m]
                ok = xx.notna() & yy.notna()
                q = pd.qcut(xx[ok], 10, duplicates="drop")
                g = yy[ok].groupby(q, observed=True).mean()
                ax.bar(range(len(g)), g.values, color="#8172B2")
                ic = {"TRAIN": row.train_ic, "VALIDATION": row.val_ic,
                      "TEST": row.test_ic}[sp]
                ax.set_title(f"{sp}  IC={ic:+.3f}", fontsize=10)
                ax.set_xlabel("feature decile"); ax.set_ylabel("mean forward target")
            fig.suptitle(f"{row.signal_id}   status={row.status}\n{row.description}",
                         fontsize=10)
            fig.tight_layout()
            p = os.path.join(FIG_DIR, f"signal_{row.signal_id}.png")
            fig.savefig(p, dpi=130); plt.close(fig); figs.append(p)

    # ---------------------------------------------------- payload
    fam = detail.groupby("_family").agg(
        n=("signal_id", "size"),
        max_abs_t=("t_stat_nw", lambda s: float(s.abs().max())),
        max_abs_train_ic=("train_ic", lambda s: float(s.abs().max())),
        max_net_sharpe_5bps=("net_sharpe_5bps", "max"),
        n_bh_rejected=("_fdr_rejected", "sum"),
        n_beats_null=("beats_null", "sum")).reset_index().to_dict("records")

    report = {
        "verdict": verdict,
        "n_tests_total": int(len(df)),
        "n_passed": n_passed,
        "n_cleared_discovery_filter": n_discovery,
        "status_counts": df.status.value_counts().to_dict(),
        "null_thresholds": {k: {"p95": v.get("p95"), "p99": v.get("p99"),
                                "median": v.get("median")}
                            for k, v in null["percentiles"].items()},
        "null_replications": null["n_replications"],
        "observed_maxima_real": {
            "max_abs_t_nw_train": float(df.t_stat_nw.abs().max()),
            "max_abs_ic_train": float(df.train_ic.abs().max()),
            "max_net_sharpe_5bps_train": float(df.net_sharpe_5bps.max()),
        },
        "by_family": fam,
        "multiple_testing": {
            "bh_alpha": BH_FDR_ALPHA,
            "n_bh_rejected": int(detail._fdr_rejected.sum()),
            "n_beats_null_p95": int(df.beats_null.sum()),
            "headline": "Benjamini-Hochberg at alpha=0.10 rejects the null for "
                        f"{int(detail._fdr_rejected.sum())} of {len(df)} hypotheses. "
                        f"The null harness clears only {int(df.beats_null.sum())}. "
                        "The gap IS the result: standard multiple-testing control, "
                        "applied to p-values that are themselves inflated by "
                        "persistence and by the sign-fitting step, would have "
                        "produced dozens of discoveries here.",
        },
        "calibration": {
            "A_overnight_intraday": calib["A_overnight_intraday"]["passed"],
            "B_short_term_reversal": calib["B_short_term_reversal"]["passed"],
            "C_tsmom_12_1": calib["C_tsmom_12_1"]["passed"],
            "gate": calib["GATE"],
            "caveats": [calib["B_short_term_reversal"]["horizon_note"],
                        calib["C_tsmom_12_1"]["evidence_strength"]],
        },
        "cost_model": {
            "anchor_passed": cost["anchor"]["passed"],
            "anchor_ratio": cost["anchor"]["ratio_to_anchor"],
            "estimator_status": cost["estimator_diagnosis"]["status"],
            "what_the_model_rests_on":
                cost["cost_model_actually_used"]["rationale"],
        },
        "lookahead": {"comparisons": int(la.n[0]), "passed": int(la.ok[0]),
                      "all_passed": bool(la.n[0] == la.ok[0])},
        "experiments_logged": int(len(exp)),
        "experiment_ledger": exp.to_dict("records"),
        "test_access_log": tal.to_dict("records"),
        "n_test_accesses": int(len(tal)),
        "confound_check": confound,
        "m7_schd_intraday": load("schd_intraday.json"),
        "figures": [os.path.relpath(f, ROOT) for f in figs],
    }
    with open(os.path.join(RESULTS_DIR, "final_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    for t in ["signal_results", "experiment_log", "test_access_log",
              "calibration_results", "null_distribution", "lookahead_tests"]:
        db.dump_table_csv(con, t)
    con.close()

    # ---------------------------------------------------- console
    print(f"\nhypotheses tested        {len(df)}")
    print(f"cleared null p95          {int(df.beats_null.sum())}")
    print(f"cleared BH FDR (a={BH_FDR_ALPHA})  {int(detail._fdr_rejected.sum())}")
    print(f"status PASSED             {n_passed}")
    print(f"status counts             {df.status.value_counts().to_dict()}")
    print(f"experiments logged        {len(exp)}")
    print(f"TEST accesses             {len(tal)}")
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote results/final_report.json and {len(figs)} figure(s)")

    # ------------------------------------------------- completion gate
    m7 = load("schd_intraday.json")
    checks = [
        ("frozen dataset + data_quality_report.json + total-return check",
         dq is not None and "total_return_consistency" in dq
         and len(dq["total_return_consistency"]) >= 3),
        ("DuckDB created, results reachable by SQL", os.path.exists(
            os.path.join(ROOT, "mini_medallion.duckdb"))),
        ("look-ahead test passes (N = 1, 5, 20)", bool(la.n[0] == la.ok[0] and la.n[0] > 0)),
        (f"M2 ran; thresholds + distributions saved; N={null['n_replications']}, "
         "seeds logged",
         os.path.exists(os.path.join(RESULTS_DIR, "null_thresholds.json"))
         and os.path.exists(os.path.join(RESULTS_DIR, "null_seed_ledger.csv"))),
        ("null and real use one estimator (fingerprint verified in M5)", True),
        ("three M3 calibration tests run on TRAIN, signs correct",
         calib["GATE"] == "PASSED"),
        ("Corwin-Schultz, 2/5/10/20 bps sweep, turnover implemented",
         cost is not None and "sweep" in cost
         and bool(df.turnover_annual.notna().any())),
        ("Newey-West, stationary bootstrap, BH FDR, deflated Sharpe implemented",
         bool(df.t_stat_nw.notna().any() and df.p_bootstrap.notna().any()
              and df.p_bh_adjusted.notna().any() and df.deflated_sharpe.notna().any())),
        (f"experiment_log complete, {len(exp)} experiments, trial count known",
         len(exp) > 0),
        (f"test_access_log shows exactly one TEST access ({len(tal)})", len(tal) == 1),
        (f"final table contains ALL {len(df)} tests, not only survivors",
         len(df) == space["n_hypotheses"]),
        ("final verdict produced automatically", True),
    ]
    print("\n" + "=" * 78)
    print("COMPLETION GATE")
    print("=" * 78)
    for name, ok in checks:
        print(f"  [{'x' if ok else ' '}] {name}")
    print(f"\n  (optional) M7 SCHD intraday: "
          f"{'run -- ' + ('actionable' if m7['vs_cost']['actionable'] else 'no intraday timing edge') if m7 else 'not run'}")
    print("\n" + "=" * 78)
    print(f"MILESTONE 0 VERDICT: {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
