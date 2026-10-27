"""M4 -- Corwin-Schultz high/low spread estimator.

Costs are part of the metric from the start.  Gross performance is kept only
as a diagnostic column and is never used for ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_K = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz_daily(high: pd.Series, low: pd.Series, close: pd.Series,
                         gap_adjust: bool = True) -> pd.Series:
    """Two-day high/low spread estimate, as a fraction of price.

    `gap_adjust` applies the overnight-move correction from the original paper:
    when the whole range gaps between the two days, the gap is removed before
    the two-day range is formed, otherwise the estimator reads the overnight
    return as spread.
    """
    h = high.astype(float).copy()
    l = low.astype(float).copy()
    c = close.astype(float)

    h1, l1 = h.shift(1), l.shift(1)
    if gap_adjust:
        gap_up = (l - h1).clip(lower=0.0)
        gap_dn = (l1 - h).clip(lower=0.0)
        h = h - gap_up + gap_dn
        l = l - gap_up + gap_dn

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(h / l) ** 2 + np.log(h1 / l1) ** 2
        h2 = pd.concat([h, h1], axis=1).max(axis=1)
        l2 = pd.concat([l, l1], axis=1).min(axis=1)
        gamma = np.log(h2 / l2) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def monthly_spread_bps(px: dict, tickers, gap_adjust: bool = True) -> pd.DataFrame:
    """Monthly average Corwin-Schultz spread in basis points, per instrument.

    Negative daily estimates are set to zero (standard treatment: the estimator
    is noisy and can go negative for a strictly positive quantity).
    """
    out = []
    for t in tickers:
        d = px[t]
        s = corwin_schultz_daily(d["high"], d["low"], d["close"], gap_adjust)
        s = s.clip(lower=0.0)
        g = s.groupby(pd.Grouper(freq="ME"))
        m = g.mean() * 1e4
        n = g.count()
        for dt, v in m.items():
            if np.isfinite(v) and n.loc[dt] >= 5:
                out.append({"ticker": t, "month": dt.date(),
                            "cs_spread_bps": float(v), "n_days": int(n.loc[dt])})
    return pd.DataFrame(out)


def cost_per_unit_turnover(cost_bps_round_trip: float, n_legs: int) -> float:
    """A full open-and-close cycle accumulates sum|dW| = 2, so one unit of
    one-way turnover is charged half the round-trip cost, per leg."""
    return cost_bps_round_trip / 2.0 / 1e4 * n_legs


def abdi_ranaldo_monthly_bps(d: pd.DataFrame) -> pd.Series:
    """Abdi & Ranaldo (2017) close-high-low estimator, aggregated per month.

    The squared-spread term is averaged over the month BEFORE the square root,
    which avoids the flooring bias that destroys the daily Corwin-Schultz
    estimate on low-spread instruments.
    """
    c = np.log(d["close"].astype(float))
    h = np.log(d["high"].astype(float))
    l = np.log(d["low"].astype(float))
    eta = (h + l) / 2.0
    s2 = 4.0 * (c - eta) * (c - eta.shift(-1))
    g = s2.groupby(pd.Grouper(freq="ME"))
    m, n = g.mean(), g.count()
    return (np.sqrt(m.clip(lower=0)) * 1e4).where(n >= 5)


def roll_monthly_bps(d: pd.DataFrame) -> pd.Series:
    """Roll (1984) serial-covariance estimator; zero when the autocovariance
    is positive (which is most of the time for liquid ETFs)."""
    dp = np.log(d["close"].astype(float)).diff()

    def f(s):
        x = s.dropna().to_numpy(float)
        if len(x) < 10:
            return np.nan
        cv = np.cov(x[1:], x[:-1])[0, 1]
        return 2.0 * np.sqrt(-cv) * 1e4 if cv < 0 else 0.0

    return dp.groupby(pd.Grouper(freq="ME")).apply(f)


def cs_monthly_bps(d: pd.DataFrame, floor: str = "per_day") -> pd.Series:
    """Corwin-Schultz per month. `floor` is 'per_day' or 'after_average'.

    The two configurations bracket the estimator's behaviour: flooring each
    daily estimate turns symmetric noise into a large positive bias, while
    flooring the monthly average returns exactly zero whenever the true spread
    sits below the estimator's resolution.
    """
    s = corwin_schultz_daily(d["high"], d["low"], d["close"])
    if floor == "per_day":
        s = s.clip(lower=0.0)
        g = s.groupby(pd.Grouper(freq="ME"))
        return (g.mean() * 1e4).where(g.count() >= 5)
    g = s.groupby(pd.Grouper(freq="ME"))
    return (g.mean().clip(lower=0.0) * 1e4).where(g.count() >= 5)


def fraction_negative_daily(d: pd.DataFrame) -> float:
    s = corwin_schultz_daily(d["high"], d["low"], d["close"])
    s = s[np.isfinite(s)]
    return float((s < 0).mean()) if len(s) else np.nan
