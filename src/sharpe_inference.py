"""N0 -- does the gap exist at all?

Building an explanation for an effect indistinguishable from zero is exactly the
failure M0 and M1 were built to prevent, so this runs first and its answer
qualifies everything after it.

Contains: Ledoit-Wolf (2008) HAC test for the difference of two dependent Sharpe
ratios, the studentised paired block bootstrap they recommend alongside it, and
an effective-number-of-tests correction for the cross-instrument sign test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from config import TRADING_DAYS
from stats_engine import stationary_bootstrap_index


# ------------------------------------------------------------------ HAC
def _nw_lag(n: int) -> int:
    """Newey-West automatic bandwidth, floor(4*(T/100)^(2/9))."""
    return max(int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))), 1)


def hac_cov(V: np.ndarray, lag: int | None = None) -> np.ndarray:
    """Bartlett-kernel HAC covariance of the columns of V (already demeaned)."""
    n, k = V.shape
    L = _nw_lag(n) if lag is None else lag
    S = V.T @ V / n
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = V[l:].T @ V[:-l] / n
        S = S + w * (G + G.T)
    return S


# ---------------------------------------------------- Ledoit-Wolf (2008)
def sharpe_difference_lw(r1, r2, lag: int | None = None):
    """Ledoit-Wolf robust test for SR1 - SR2 on two DEPENDENT return series.

    Sharpe is a smooth function of (mu, gamma) with gamma = E[r^2]; the delta
    method over a HAC covariance of (r1, r2, r1^2, r2^2) gives a standard error
    that is valid under autocorrelation, heteroskedasticity and -- the point
    here -- correlation between the two strategies.
    """
    a = np.asarray(r1, float)
    b = np.asarray(r2, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 100:
        return {}
    mu1, mu2 = a.mean(), b.mean()
    g1, g2 = np.mean(a ** 2), np.mean(b ** 2)
    s1, s2 = np.sqrt(g1 - mu1 ** 2), np.sqrt(g2 - mu2 ** 2)
    if s1 <= 0 or s2 <= 0:
        return {}
    sr1, sr2 = mu1 / s1, mu2 / s2
    diff = sr1 - sr2

    grad = np.array([g1 / s1 ** 3, -g2 / s2 ** 3,
                     -mu1 / (2 * s1 ** 3), mu2 / (2 * s2 ** 3)])
    V = np.column_stack([a - mu1, b - mu2, a ** 2 - g1, b ** 2 - g2])
    S = hac_cov(V, lag)
    var = float(grad @ S @ grad / n)
    if not np.isfinite(var) or var <= 0:
        return {}
    se = np.sqrt(var)
    z = diff / se
    k = np.sqrt(TRADING_DAYS)
    return {
        "n": int(n),
        "sharpe_1_ann": float(sr1 * k), "sharpe_2_ann": float(sr2 * k),
        "sharpe_diff_ann": float(diff * k),
        "se_diff_ann": float(se * k),
        "z": float(z), "p_lw": float(2.0 * (1.0 - sps.norm.cdf(abs(z)))),
        "rho_strategies": float(np.corrcoef(a, b)[0, 1]),
        "hac_lag": int(_nw_lag(n) if lag is None else lag),
        "ci_lo_ann": float((diff - 1.96 * se) * k),
        "ci_hi_ann": float((diff + 1.96 * se) * k),
    }


# ------------------------------------------- paired block bootstrap
def paired_block_bootstrap_sharpe_diff(r1, r2, mean_block, rng, n_rep=5000):
    """Bootstrap on the PAIRED series, resampling the same block indices for both.

    Resampling each strategy independently would destroy the correlation that is
    the only reason this test has any power, and would give a standard error
    several times too wide.
    """
    a = np.asarray(r1, float)
    b = np.asarray(r2, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 100:
        return {}
    k = np.sqrt(TRADING_DAYS)

    def sd(x, y):
        s1, s2 = x.std(ddof=1), y.std(ddof=1)
        if s1 <= 0 or s2 <= 0:
            return np.nan
        return (x.mean() / s1 - y.mean() / s2) * k

    obs = sd(a, b)
    boot = np.empty(n_rep)
    for i in range(n_rep):
        idx = stationary_bootstrap_index(n, mean_block, rng)
        boot[i] = sd(a[idx], b[idx])
    boot = boot[np.isfinite(boot)]
    if len(boot) < 200:
        return {}
    centred = boot - obs
    p = float((np.abs(centred) >= abs(obs)).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"sharpe_diff_ann": float(obs),
            "p_bootstrap_paired": max(p, 1.0 / len(boot)),
            "boot_se_ann": float(np.std(boot, ddof=1)),
            "boot_ci_lo": float(lo), "boot_ci_hi": float(hi),
            "n_boot": int(len(boot))}


# --------------------------------------------- dependence-corrected sign test
def effective_number_of_tests(corr_matrix: np.ndarray) -> float:
    """Cheverud-Nyholt effective number of independent tests from eigenvalues.

    M_eff = 1 + (M - 1) * (1 - Var(lambda) / M)

    Reporting a naive p = 1/2^4 across SPY, SCHD, IWM and TLT would be a lie:
    the three equity ETFs move together, so the sample contains roughly two
    independent bets, not four.
    """
    C = np.asarray(corr_matrix, float)
    M = C.shape[0]
    if M < 2:
        return float(M)
    lam = np.linalg.eigvalsh(C)
    lam = np.clip(lam, 0, None)
    var_l = float(np.var(lam, ddof=0))
    return float(1.0 + (M - 1) * (1.0 - var_l / M))


def sign_test_corrected(signs, corr_matrix, expected_positive=True):
    """Binomial sign test using the effective, not the nominal, number of tests."""
    s = np.asarray(signs)
    n_pos = int((s > 0).sum())
    M = len(s)
    m_eff = effective_number_of_tests(corr_matrix)
    k_eff = n_pos * m_eff / M
    p_naive = float(sps.binomtest(n_pos, M, 0.5,
                                  alternative="greater" if expected_positive
                                  else "less").pvalue)
    k_round = int(round(k_eff))
    m_round = max(int(round(m_eff)), 1)
    k_round = min(max(k_round, 0), m_round)
    p_eff = float(sps.binomtest(k_round, m_round, 0.5,
                                alternative="greater" if expected_positive
                                else "less").pvalue)
    return {"n_instruments": M, "n_positive": n_pos,
            "effective_n_tests": m_eff,
            "p_naive": p_naive, "p_dependence_corrected": p_eff,
            "note": "the naive p treats four correlated instruments as four "
                    "independent bets; the corrected one uses the eigenvalue-based "
                    "effective count"}
