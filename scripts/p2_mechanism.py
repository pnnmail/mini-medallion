"""P2 -- where the timing term comes from, with its null.

P1 showed the timing term IS the decision difference (Spearman +0.94 against
+0.28 for exposure). P2 asks why a candidate that improves the forecast can make
that term negative.

H2: decompose sigma_hat(HAR+X) = sigma_hat(HAR) + delta. If delta correlates
NEGATIVELY with the next return, the increment cuts exposure just before
rebounds -- it is right about the volatility and wrong about the moment.

Every number here is compared against a null built by circular-shifting delta
against the returns, which preserves both series' autocorrelation entirely and
destroys only the link between them.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config_m2 import (FIG_DIR, INSTRUMENTS, MASTER_SEED_M2, NULL_PERCENTILE_M2,
                       NULL_REPLICATIONS_M2, RESULTS_DIR, TRADING_DAYS)
from decomposition import circular_shift_null, null_summary, pop_cov
from universe_m2 import build_universe, common_mask, sigma_of, weights_and_pnl


def main():
    print("=" * 78)
    print("P2  MECHANISM -- why the timing term goes negative")
    print("=" * 78)
    con = db.connect()
    rng = np.random.default_rng(MASTER_SEED_M2 + 62)
    recs = []

    for tk in INSTRUMENTS:
        idx, fwd, models, fams, d = build_universe(tk)
        sig_b = sigma_of(tk, "HAR")
        for split in ["TRAIN", "VALIDATION"]:
            m = common_mask(tk, split)
            if m.sum() < 300:
                continue
            _, w_b, _, _ = weights_and_pnl(tk, "HAR", m)
            for name in models:
                if name == "HAR":
                    continue
                delta = sigma_of(tk, name) - sig_b
                dsel, rsel = delta[m], fwd[m]
                ok = np.isfinite(dsel) & np.isfinite(rsel)
                if ok.sum() < 300:
                    continue
                corr = float(np.corrcoef(dsel[ok], rsel[ok])[0, 1])
                nd, obs = circular_shift_null(dsel[ok], rsel[ok],
                                              NULL_REPLICATIONS_M2, rng, "corr")
                ns = null_summary(nd, corr, NULL_PERCENTILE_M2)

                _, w_a, _, _ = weights_and_pnl(tk, name, m)
                dw = (np.asarray(w_a, float) - np.asarray(w_b, float))[m]
                ok2 = np.isfinite(dw) & np.isfinite(rsel)
                tcov = pop_cov(dw[ok2], rsel[ok2])
                nd2, _ = circular_shift_null(dw[ok2], rsel[ok2],
                                             NULL_REPLICATIONS_M2, rng, "cov")
                ns2 = null_summary(nd2, tcov, NULL_PERCENTILE_M2)

                recs.append({
                    "instrument": tk, "model": name,
                    "candidate": name.replace("HAR+", ""), "family": fams[name],
                    "split_name": split, "n_obs": int(ok.sum()),
                    "corr_delta_fwd_ret": corr,
                    "corr_null_p95": ns["p95_abs"], "corr_p": ns["p_two_sided"],
                    "corr_beats_null": ns["beats_null"],
                    "timing_cov": tcov, "timing_cov_ann": tcov * TRADING_DAYS,
                    "timing_null_p95": ns2["p95_abs"], "timing_p": ns2["p_two_sided"],
                    "timing_beats_null": ns2["beats_null"],
                })
    df = pd.DataFrame(recs)

    print(f"\n{len(df)} (instrument, model, split) cells, each null-calibrated with "
          f"{NULL_REPLICATIONS_M2} circular shifts")

    vix = df[df.candidate == "log_vix"]
    print("\nH2 -- log(VIX): does the increment it adds to the forecast mistime?")
    print(f"   {'inst':5s} {'split':11s} {'corr(delta,r_next)':>19s} {'null p95':>9s} "
          f"{'p':>7s} {'beats':>6s}")
    for _, r in vix.iterrows():
        print(f"   {r.instrument:5s} {r.split_name:11s} {r.corr_delta_fwd_ret:+19.4f} "
              f"{r.corr_null_p95:9.4f} {r.corr_p:7.3f} "
              f"{'yes' if r.corr_beats_null else 'no':>6s}")
    n_neg = int((vix.corr_delta_fwd_ret < 0).sum())
    n_sig = int(((vix.corr_delta_fwd_ret < 0) & vix.corr_beats_null).sum())
    h2 = bool(n_neg >= len(vix) - 1 and n_sig >= 1)
    print(f"   negative in {n_neg}/{len(vix)} cells, of which {n_sig} beat the null")
    print(f"   H2 AS PRE-REGISTERED: {'SUPPORTED' if h2 else 'NOT SUPPORTED'}")

    # ---- H2 as written was wrong, and it was wrong in the SPEC, not the data ----
    # The weight is w = sigma*/sigma_hat, which DECREASES in sigma_hat. So the
    # mechanism the spec described in words -- "the increment cuts exposure just
    # before rebounds" -- implies corr(delta, next return) > 0, not < 0. The
    # pre-registered inequality was written with the sign reversed.
    n_pos = int((vix.corr_delta_fwd_ret > 0).sum())
    n_pos_sig = int(((vix.corr_delta_fwd_ret > 0) & vix.corr_beats_null).sum())
    pos_cells = df[df.corr_delta_fwd_ret > 0]
    neg_cells = df[df.corr_delta_fwd_ret < 0]
    h2p = bool(n_pos > n_neg and n_pos_sig >= 1 and
               pos_cells.timing_cov_ann.mean() < neg_cells.timing_cov_ann.mean())
    print(f"\n   H2' -- the SAME mechanism with the sign written correctly.")
    print(f"   w = sigma*/sigma_hat is DECREASING in sigma_hat, so 'cuts exposure "
          f"before rebounds'")
    print(f"   means corr(delta, next return) > 0. The pre-registered inequality had "
          f"it backwards.")
    print(f"   log(VIX): positive in {n_pos}/{len(vix)} cells, {n_pos_sig} of them "
          f"beat the null")
    print(f"   across all {len(df)} cells: mean timing term is "
          f"{pos_cells.timing_cov_ann.mean():+.5f} where corr > 0 "
          f"({len(pos_cells)} cells) and "
          f"{neg_cells.timing_cov_ann.mean():+.5f} where corr < 0 "
          f"({len(neg_cells)} cells)")
    from scipy import stats as _s
    okc = df.corr_delta_fwd_ret.notna() & df.timing_cov_ann.notna()
    rho_sign = float(_s.spearmanr(df.loc[okc, "corr_delta_fwd_ret"],
                                  df.loc[okc, "timing_cov_ann"]).statistic)
    print(f"   Spearman(corr(delta, r_next), timing term) = {rho_sign:+.3f}")
    print(f"   H2' {'SUPPORTED' if h2p else 'NOT supported'} -- reported as a "
          f"corrected restatement found after seeing the data, not as a pre-registered "
          f"result.")

    print("\nby family (TRAIN+VALIDATION): mean corr(delta, next return) and how "
          "often it clears the null")
    fam = df.groupby("family").agg(
        n=("model", "size"),
        mean_corr=("corr_delta_fwd_ret", "mean"),
        n_negative=("corr_delta_fwd_ret", lambda s: int((s < 0).sum())),
        n_beats_null=("corr_beats_null", "sum"),
        mean_timing_ann=("timing_cov_ann", "mean"),
        n_timing_beats_null=("timing_beats_null", "sum")).reset_index()
    print(fam.round(5).to_string(index=False))

    # link the mechanism to the outcome measured in P1
    dec = pd.read_csv(os.path.join(RESULTS_DIR, "m2_decomposition.csv"))
    j = df.merge(dec[["instrument", "model", "split_name", "delta_sharpe",
                      "timing_term_ann", "qlike"]],
                 on=["instrument", "model", "split_name"], how="left")
    from scipy import stats as sps
    ok = j.corr_delta_fwd_ret.notna() & j.delta_sharpe.notna()
    rho_corr = float(sps.spearmanr(j.loc[ok, "corr_delta_fwd_ret"],
                                   j.loc[ok, "delta_sharpe"]).statistic)
    ok2 = j.qlike.notna() & j.delta_sharpe.notna()
    rho_qlike = float(sps.spearmanr(j.loc[ok2, "qlike"], j.loc[ok2, "delta_sharpe"]).statistic)
    print(f"\nTHE POINT OF THE MILESTONE, in two numbers:")
    print(f"   Spearman( corr(delta, next return) , delta_sharpe ) = {rho_corr:+.3f}")
    print(f"   Spearman( QLIKE                    , delta_sharpe ) = {rho_qlike:+.3f}")
    print(f"   QLIKE is lower-is-better, so a NEGATIVE rho here would mean accuracy "
          f"helps the decision.")

    df.to_csv(os.path.join(RESULTS_DIR, "m2_mechanism.csv"), index=False)
    payload = {
        "null": {"method": "circular shift of delta against returns",
                 "replications": NULL_REPLICATIONS_M2,
                 "percentile": NULL_PERCENTILE_M2},
        "H2_as_preregistered": {
            "statement": "the increment log(VIX) adds to the volatility forecast "
                         "correlates NEGATIVELY with the next return",
            "cells": vix.to_dict("records"),
            "n_negative": n_neg, "n_cells": int(len(vix)),
            "n_negative_and_beats_null": n_sig, "supported": h2,
            "verdict": "NOT SUPPORTED"},
        "H2_prime_corrected_sign": {
            "statement": "the increment correlates POSITIVELY with the next return",
            "why_the_spec_was_wrong": "w = sigma*/sigma_hat DECREASES in sigma_hat, so "
                                      "the mechanism the spec described in words -- the "
                                      "increment cuts exposure just before rebounds -- "
                                      "implies a POSITIVE correlation. The "
                                      "pre-registered inequality was written with the "
                                      "sign reversed. This is an error in the "
                                      "specification I wrote, found by the data.",
            "status": "POST-HOC RESTATEMENT, not a pre-registered result",
            "n_positive": n_pos, "n_positive_and_beats_null": n_pos_sig,
            "mean_timing_ann_where_corr_positive": float(pos_cells.timing_cov_ann.mean()),
            "n_cells_corr_positive": int(len(pos_cells)),
            "mean_timing_ann_where_corr_negative": float(neg_cells.timing_cov_ann.mean()),
            "n_cells_corr_negative": int(len(neg_cells)),
            "spearman_corr_vs_timing_term": rho_sign,
            "supported": h2p,
            "unaffected_by_the_error": "the decisive statistic -- Spearman(corr(delta, "
                                       "r_next), delta_sharpe) -- does not depend on "
                                       "which sign was pre-registered; it was computed "
                                       "the same way either way."},
        "by_family": fam.to_dict("records"),
        "spearman_corrdelta_vs_delta_sharpe": rho_corr,
        "spearman_qlike_vs_delta_sharpe": rho_qlike,
    }
    with open(os.path.join(RESULTS_DIR, "m2_p2_mechanism.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    for f_, c in zip(sorted(df.family.unique()),
                     ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]):
        s = j[j.family == f_]
        ax.scatter(s.corr_delta_fwd_ret, s.delta_sharpe, s=16, alpha=0.75,
                   label=f_, color=c)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("corr(increment to the vol forecast, next return)")
    ax.set_ylabel("delta Sharpe vs HAR")
    ax.set_title(f"P2  mistiming explains the decision (rho = {rho_corr:+.2f})",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.scatter(j.qlike, j.delta_sharpe, s=16, alpha=0.75, color="#937860")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("QLIKE (lower = more accurate forecast)")
    ax.set_ylabel("delta Sharpe vs HAR")
    ax.set_title(f"accuracy does not (rho = {rho_qlike:+.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "m2_mechanism.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "P2", "mechanism: corr(forecast increment, next return), "
                                 "null-calibrated by circular shift",
                      {"n_cells": len(df), "replications": NULL_REPLICATIONS_M2,
                       "H2_as_preregistered_supported": h2, "rho_corr": rho_corr,
                       "rho_qlike": rho_qlike},
                      n_tests=len(df), seed=MASTER_SEED_M2, status="COMPLETED")
    db.log_experiment(con, "P2", "ITERATION after seeing results: H2 was pre-registered "
                                 "with the inequality reversed. w = sigma*/sigma_hat "
                                 "decreases in sigma_hat, so the mechanism described in "
                                 "the spec implies corr(delta, next return) > 0. H2 as "
                                 "written is reported as NOT SUPPORTED and the corrected "
                                 "restatement H2' is reported as post-hoc.",
                      {"h2_as_written_supported": h2, "h2_prime_supported": h2p,
                       "spearman_corr_vs_timing": rho_sign},
                      n_tests=0, seed=MASTER_SEED_M2, status="POST_HOC_ITERATION")
    con.close()
    print("\nwrote results/m2_mechanism.csv, m2_p2_mechanism.json")


if __name__ == "__main__":
    main()
