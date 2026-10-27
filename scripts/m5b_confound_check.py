"""M5-post -- POST-HOC diagnostic on the single hypothesis that cleared the null.

This runs AFTER the results were seen, so it is a new experiment and is logged
as one.  It cannot promote anything: statuses assigned by the pre-registered
rule in m5_search.py are not touched.  It can only ask whether the surviving
candidate survives an obvious confound.

The claim under test is 'TLT volatility predicts SPY volatility'.  The obvious
competing explanation is that SPY volatility predicts its own future -- which
nobody disputes -- and that TLT vol is simply a noisy proxy for it.
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
import stats_engine as se
from config import MASTER_SEED, RESULTS_DIR, STD_WINDOW
from features import log_returns, realized_vol, zscore_causal
from panel import research_panel, split_mask
from targets import forward_realized_vol


def nw_multi(y, X, lag):
    """Newey-West for a multivariate OLS. Returns (betas, t-stats)."""
    m = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[m], X[m]
    n = len(y)
    X = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        G = u[l:].T @ u[:-l]
        S = S + w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return beta, beta / np.sqrt(np.diag(V)), n


def main():
    print("=" * 78)
    print("M5-post  CONFOUND CHECK (post-hoc, logged, cannot change any status)")
    print("=" * 78)
    con = db.connect()
    prices, macro = research_panel()
    tr = split_mask(prices.index, "TRAIN")

    r_tlt = log_returns(prices["TLT"])
    r_spy = log_returns(prices["SPY"])
    x_tlt = zscore_causal(realized_vol(r_tlt, 10), STD_WINDOW)
    x_spy = zscore_causal(realized_vol(r_spy, 10), STD_WINDOW)
    y = forward_realized_vol(r_spy, 5)

    xt = x_tlt.to_numpy(float)[tr]
    xs = x_spy.to_numpy(float)[tr]
    yv = y.to_numpy(float)[tr]

    _, t_tlt, n1 = nw_multi(yv, xt[:, None], 4)
    _, t_spy, _ = nw_multi(yv, xs[:, None], 4)
    beta, tt, n2 = nw_multi(yv, np.column_stack([xt, xs]), 4)

    m = np.isfinite(xt) & np.isfinite(xs)
    corr_x = float(sps.spearmanr(xt[m], xs[m]).statistic)
    ic_tlt = se.spearman_ic(xt, yv)
    ic_spy = se.spearman_ic(xs, yv)

    print(f"\nTRAIN, n={n2}")
    print(f"   univariate   TLT vol  -> SPY fwd vol :  t_NW = {t_tlt[1]:+6.3f}   "
          f"IC = {ic_tlt:+.4f}")
    print(f"   univariate   SPY vol  -> SPY fwd vol :  t_NW = {t_spy[1]:+6.3f}   "
          f"IC = {ic_spy:+.4f}")
    print(f"   joint        TLT | controlling for SPY vol : t_NW = {tt[1]:+6.3f}  "
          f"beta = {beta[1]:+.5f}")
    print(f"   joint        SPY | controlling for TLT vol : t_NW = {tt[2]:+6.3f}  "
          f"beta = {beta[2]:+.5f}")
    print(f"   Spearman(TLT vol, SPY vol) = {corr_x:+.3f}")

    killed = bool(abs(tt[1]) < 2.0)
    print(f"\n   -> once SPY's own volatility is in the regression, the TLT term "
          f"{'collapses' if killed else 'survives'} "
          f"({t_tlt[1]:+.2f} -> {tt[1]:+.2f}).")

    payload = {
        "post_hoc": True,
        "cannot_change_status": True,
        "candidate": "rvol_TLT_10_lag0__fwdrvol5",
        "n_train": int(n2),
        "t_univariate_TLT": float(t_tlt[1]), "ic_univariate_TLT": ic_tlt,
        "t_univariate_SPY_own_vol": float(t_spy[1]), "ic_univariate_SPY_own_vol": ic_spy,
        "t_TLT_controlling_for_SPY_vol": float(tt[1]),
        "t_SPY_controlling_for_TLT_vol": float(tt[2]),
        "spearman_between_regressors": corr_x,
        "conclusion": ("The cross-asset reading of this hypothesis does not survive the "
                       "obvious control. SPY's own realised volatility predicts SPY's "
                       f"forward volatility with t = {t_spy[1]:+.2f}; adding TLT "
                       f"volatility moves its own t from {t_tlt[1]:+.2f} to {tt[1]:+.2f}. "
                       "What the search found is volatility clustering in SPY, reached "
                       "through a correlated proxy -- not a TLT-to-SPY spillover.")
        if killed else
        ("The TLT term retains an independent contribution after controlling for SPY's "
         "own volatility. This does NOT change its status: it failed the "
         "pre-registered out-of-sample gate and is reported as FAILED_OOS."),
    }
    with open(os.path.join(RESULTS_DIR, "confound_check.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    db.log_experiment(con, "M5-post",
                      "POST-HOC confound check on the single null-beating candidate; "
                      "run after results were seen; changes no status",
                      {"candidate": "rvol_TLT_10_lag0__fwdrvol5",
                       "controls": ["rvol_SPY_10"], "split": "TRAIN"},
                      n_tests=3, seed=MASTER_SEED, status="COMPLETED_POST_HOC")
    con.close()
    print(f"\nwrote results/confound_check.json")


if __name__ == "__main__":
    main()
