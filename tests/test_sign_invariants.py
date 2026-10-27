"""Sign invariants, asserted numerically.

This exists because the project has now had TWO sign collisions:

  M0/M1  the Avellaneda-Lee s-score reduces to -m/sigma_eq only because the OLS
         residual sums to zero over its own estimation window -- easy to get
         backwards, and the naive version measures nothing.
  M2     H2 was pre-registered as corr(delta, next return) < 0 when the mechanism
         it described implies > 0, because w = sigma*/sigma_hat DECREASES in
         sigma_hat.

Both were detectable from the text with no data at all. A failure surface that
recurs deserves a gate, not vigilance. Run this before any pre-registration is
written; a reversed sign should break the build, not survive into a hypothesis.

    python tests/test_sign_invariants.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

FAILURES = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


# ------------------------------------------------------------------ 1
def test_weight_decreasing_in_sigma():
    """d w / d sigma_hat < 0 everywhere below the cap. THE M2 invariant."""
    from voltarget import target_weights
    sig = np.linspace(0.05, 0.80, 400)
    w = target_weights(sig, cap=1.0, sigma_target=0.10)
    d = np.diff(w)
    below_cap = w[:-1] < 0.999
    check("dw/dsigma < 0 below the cap",
          np.all(d[below_cap] < 0),
          f"max diff below cap = {d[below_cap].max():.3e}")
    check("weights respect the cap", np.all(w <= 1.0 + 1e-12),
          f"max w = {w.max():.4f}")
    check("cap binds only at low sigma",
          w[0] >= 0.999 and w[-1] < 0.999,
          f"w(0.05) = {w[0]:.3f}, w(0.80) = {w[-1]:.3f}")


# ------------------------------------------------------------------ 2
def test_timing_sign_flips_through_the_weight():
    """cov(delta_w, r) must carry the OPPOSITE sign to cov(delta_sigma, r).

    This is the exact relation the M2 pre-registration got backwards.
    """
    from decomposition import pop_cov
    from voltarget import target_weights
    rng = np.random.default_rng(7)
    n = 4000
    base = 0.15 + 0.03 * rng.standard_normal(n).cumsum() / np.sqrt(n)
    base = np.clip(base, 0.06, 0.60)
    r = rng.standard_normal(n) * 0.01
    for target_sign in (+1.0, -1.0):
        # build an increment deliberately correlated with the next return
        delta = target_sign * 0.02 * (r - r.mean()) / r.std()
        w0 = target_weights(base, 1.0, 0.10)
        w1 = target_weights(base + delta, 1.0, 0.10)
        cov_sigma = pop_cov(delta, r)
        cov_w = pop_cov(w1 - w0, r)
        check(f"sign(cov(dw, r)) == -sign(cov(dsigma, r))  [dsigma sign "
              f"{'+' if target_sign > 0 else '-'}]",
              np.sign(cov_w) == -np.sign(cov_sigma),
              f"cov(dsigma,r)={cov_sigma:+.3e}  cov(dw,r)={cov_w:+.3e}")


# ------------------------------------------------------------------ 3
def test_qlike_is_lower_is_better():
    from rv_estimators import qlike
    rng = np.random.default_rng(11)
    actual = np.abs(rng.standard_normal(3000)) * 1e-4 + 1e-5
    perfect = qlike(actual, actual)
    over = qlike(actual, actual * 2.0)
    under = qlike(actual, actual * 0.5)
    check("QLIKE penalises over-forecasting", over > perfect,
          f"{over:.4f} > {perfect:.4f}")
    check("QLIKE penalises under-forecasting", under > perfect,
          f"{under:.4f} > {perfect:.4f}")


# ------------------------------------------------------------------ 4
def test_forward_targets_look_forward():
    from targets import forward_return
    from candidates_m1 import calendar_forward_flags
    r = pd.Series(np.arange(20, dtype=float),
                  index=pd.bdate_range("2020-01-01", periods=20))
    for h in (1, 3, 5):
        fr = forward_return(r, h)
        expect = r.to_numpy()[1:1 + h].sum()
        check(f"forward_return(h={h}) at t=0 sums t+1..t+{h}",
              abs(fr.iloc[0] - expect) < 1e-12,
              f"{fr.iloc[0]} vs {expect}")
    flag = pd.Series([0, 0, 1, 0, 0, 0, 1, 0, 0, 0.],
                     index=pd.bdate_range("2020-01-01", periods=10))
    f3 = calendar_forward_flags(flag, 3)
    check("calendar_forward_flags(h=3) at t=0 averages t+1..t+3",
          abs(f3.iloc[0] - flag.to_numpy()[1:4].mean()) < 1e-12,
          f"{f3.iloc[0]:.4f} vs {flag.to_numpy()[1:4].mean():.4f}")


# ------------------------------------------------------------------ 5
def test_reversal_signal_orientation():
    """A reversal signal must be POSITIVE after a fall (buy the dip)."""
    from features import zscore_causal
    r = pd.Series(np.concatenate([np.zeros(300), [-0.05]]),
                  index=pd.bdate_range("2019-01-01", periods=301))
    rev = -zscore_causal(r.rolling(1).sum(), 252)
    check("reversal signal is positive after a negative return",
          rev.iloc[-1] > 0, f"rev = {rev.iloc[-1]:+.3f}")


# ------------------------------------------------------------------ 6
def test_decomposition_identity():
    from decomposition import decompose_pair, pop_cov
    rng = np.random.default_rng(3)
    n = 2000
    r = rng.standard_normal(n) * 0.01
    wa = 0.6 + 0.1 * rng.standard_normal(n)
    wb = 0.6 + 0.1 * rng.standard_normal(n)
    d = decompose_pair(wa, wb, r, np.ones(n, bool))
    check("E[w r] = E[w]E[r] + cov(w, r) closes to machine precision",
          abs(d["residual"]) < 1e-15, f"residual = {d['residual']:.3e}")


# ------------------------------------------------------------------ 7
def test_newey_west_sign_matches_slope():
    from stats_engine import newey_west_slope
    rng = np.random.default_rng(5)
    x = rng.standard_normal(2000)
    for beta in (+0.4, -0.4):
        y = beta * x + rng.standard_normal(2000)
        b, t, _, _ = newey_west_slope(y, x, lag=3)
        check(f"NW t has the same sign as the slope (beta={beta:+.1f})",
              np.sign(b) == np.sign(t), f"b={b:+.3f}, t={t:+.2f}")


def main():
    print("=" * 78)
    print("SIGN INVARIANTS")
    print("=" * 78)
    for fn in [test_weight_decreasing_in_sigma,
               test_timing_sign_flips_through_the_weight,
               test_qlike_is_lower_is_better,
               test_forward_targets_look_forward,
               test_reversal_signal_orientation,
               test_decomposition_identity,
               test_newey_west_sign_matches_slope]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} invariant(s): {FAILURES}")
        sys.exit(1)
    print("ALL SIGN INVARIANTS HOLD")
    print("=" * 78)


if __name__ == "__main__":
    main()
