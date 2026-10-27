"""N4 -- the decision layer, and N5's traps.

M0 tested statistics. This tests a DECISION: does a volatility forecast change
how much to hold, in a way that survives matched-risk comparison, the removal of
a couple of crises, and costs?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config_m1 import (CRISIS_MIN_DEPTH, CRISIS_TOP_N, MIN_SIGMA, REBALANCE_BAND,
                       SHORT_TERM_DAYS, TRADING_DAYS, VOL_TARGET_ANNUAL)


# ------------------------------------------------------------------ sizing
def target_weights(sigma_hat: np.ndarray, cap: float,
                   sigma_target: float = VOL_TARGET_ANNUAL) -> np.ndarray:
    s = np.maximum(np.asarray(sigma_hat, float), MIN_SIGMA)
    w = sigma_target / s
    return np.clip(np.where(np.isfinite(w), w, np.nan), 0.0, cap)


def band_rebalance(w_target: np.ndarray, band: float = REBALANCE_BAND) -> np.ndarray:
    """Hold the position until it drifts more than `band` from target.

    Daily rebalancing generates turnover that M0 already showed will eat any
    edge these instruments offer, so the band is part of the construction, not
    an optimisation applied afterwards.
    """
    w = np.asarray(w_target, float)
    held = np.full(len(w), np.nan)
    cur = np.nan
    for i, wt in enumerate(w):
        if not np.isfinite(wt):
            held[i] = cur
            continue
        if not np.isfinite(cur):
            cur = wt
        elif abs(wt - cur) > band * max(abs(wt), 1e-12):
            cur = wt
        held[i] = cur
    return held


# ------------------------------------------------------------------ P&L
def strategy_pnl(weights: np.ndarray, fwd_ret: np.ndarray, cost_bps: float = 0.0):
    """weights[t] applies to the return from t to t+1. cost_bps is round-trip."""
    w = np.asarray(weights, float)
    r = np.asarray(fwd_ret, float)
    m = np.isfinite(w) & np.isfinite(r)
    w = np.where(m, w, 0.0)
    r = np.where(m, r, 0.0)
    dw = np.abs(np.diff(np.concatenate(([0.0], w))))
    cost = dw * (cost_bps / 2.0 / 1e4)
    return w * r - cost, dw, m


def perf(pnl: np.ndarray, mask: np.ndarray | None = None) -> dict:
    p = np.asarray(pnl, float)
    if mask is not None:
        p = p[mask]
    p = p[np.isfinite(p)]
    if len(p) < 60:
        return {"n": len(p), "sharpe": np.nan, "ann_return": np.nan,
                "ann_vol": np.nan, "max_drawdown": np.nan}
    sd = p.std(ddof=1)
    eq = np.cumsum(p)
    dd = eq - np.maximum.accumulate(eq)
    return {"n": int(len(p)),
            "sharpe": float(p.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
            "ann_return": float(p.mean() * TRADING_DAYS),
            "ann_vol": float(sd * np.sqrt(TRADING_DAYS)),
            "max_drawdown": float(dd.min())}


# --------------------------------------------------- matched-risk comparisons
def matched_comparisons(pnl_vt, pnl_bh, w_vt, mask):
    """Three comparisons at equal risk. Sharpe is scale invariant, so scaling
    alone can never change it -- which is exactly why the RETURN at matched risk
    has to be reported next to it, or the comparison says nothing."""
    v = np.asarray(pnl_vt, float)[mask]
    b = np.asarray(pnl_bh, float)[mask]
    w = np.asarray(w_vt, float)[mask]
    v = v[np.isfinite(v)]
    b = b[np.isfinite(b)]
    n = min(len(v), len(b))
    v, b = v[:n], b[:n]
    if n < 60:
        return {}
    sv, sb = v.std(ddof=1), b.std(ddof=1)
    mean_w = float(np.nanmean(w))
    out = {
        "mean_exposure_vt": mean_w,
        "sharpe_vt": float(v.mean() / sv * np.sqrt(TRADING_DAYS)) if sv > 0 else np.nan,
        "sharpe_bh": float(b.mean() / sb * np.sqrt(TRADING_DAYS)) if sb > 0 else np.nan,
    }
    # 1. matched ex-post realised volatility: lever buy-and-hold up to VT's vol
    k = sv / sb if sb > 0 else np.nan
    out["matched_vol_scale_on_bh"] = float(k)
    out["ann_return_vt"] = float(v.mean() * TRADING_DAYS)
    out["ann_return_bh_at_matched_vol"] = float(b.mean() * k * TRADING_DAYS)
    out["excess_return_at_matched_vol"] = out["ann_return_vt"] - \
        out["ann_return_bh_at_matched_vol"]
    out["sharpe_vt_matched_vol"] = out["sharpe_vt"]      # invariant, stated explicitly

    # 2. matched average exposure: scale VT so it holds 1.0 on average
    ke = 1.0 / mean_w if mean_w > 0 else np.nan
    vv = v * ke
    svv = vv.std(ddof=1)
    out["matched_exposure_scale_on_vt"] = float(ke)
    out["sharpe_vt_matched_exposure"] = float(
        vv.mean() / svv * np.sqrt(TRADING_DAYS)) if svv > 0 else np.nan
    out["ann_return_vt_matched_exposure"] = float(vv.mean() * TRADING_DAYS)
    out["ann_vol_vt_matched_exposure"] = float(svv * np.sqrt(TRADING_DAYS))
    out["ann_vol_bh"] = float(sb * np.sqrt(TRADING_DAYS))

    # 3. matched maximum drawdown
    ev, eb = np.cumsum(v), np.cumsum(b)
    ddv = float((ev - np.maximum.accumulate(ev)).min())
    ddb = float((eb - np.maximum.accumulate(eb)).min())
    kd = ddv / ddb if ddb < 0 else np.nan
    out["maxdd_vt"] = ddv
    out["maxdd_bh"] = ddb
    out["matched_dd_scale_on_bh"] = float(kd)
    out["ann_return_bh_at_matched_dd"] = float(b.mean() * kd * TRADING_DAYS)
    out["excess_return_at_matched_dd"] = out["ann_return_vt"] - \
        out["ann_return_bh_at_matched_dd"]
    return out


# ------------------------------------------------------------------ crises
def drawdown_episodes(returns: np.ndarray, index, top_n: int = CRISIS_TOP_N,
                      min_depth: float = CRISIS_MIN_DEPTH, span: str = "decline"):
    """Buy-and-hold drawdown episodes, ranked by depth.

    `span` controls what counts as the episode:
      "decline"  peak -> trough. This is the crisis itself, and the default.
      "full"     peak -> recovery to the prior peak.

    The distinction is not cosmetic. On this sample the peak-to-recovery spans
    cover 92-126% of the data (they overlap), so removing "all crises" under
    that definition removes almost the entire sample and leaves only stretches
    when the instrument was at a new high -- a subsample in which holding less
    than 100% must underperform by construction. That statistic would measure
    the definition, not the strategy.
    """
    r = np.asarray(returns, float)
    r = np.where(np.isfinite(r), r, 0.0)
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    eps, i, n = [], 0, len(r)
    while i < n:
        if dd[i] < 0:
            j = i
            while j < n and dd[j] < 0:
                j += 1
            seg = dd[i:j]
            k = int(np.argmin(seg))
            trough = i + k
            end = trough if span == "decline" else min(j, n - 1)
            eps.append({"start": i, "trough": trough, "end": end,
                        "depth": float(seg[k]),
                        "peak_date": index[max(i - 1, 0)], "trough_date": index[trough],
                        "recovery_date": index[min(j, n - 1)],
                        "length_sessions": int(end - i + 1),
                        "full_span_sessions": int(j - i)})
            i = j
        else:
            i += 1
    eps = [e for e in eps if -e["depth"] >= min_depth]
    eps.sort(key=lambda e: e["depth"])
    return eps[:top_n]


def leave_one_crisis_out(pnl_vt, pnl_bh, episodes, n):
    """Sharpe advantage recomputed with each episode removed in turn."""
    out = []
    for e in episodes:
        keep = np.ones(n, bool)
        keep[e["start"]:e["end"] + 1] = False
        a = perf(pnl_vt, keep)["sharpe"]
        b = perf(pnl_bh, keep)["sharpe"]
        out.append({"trough_date": str(pd.Timestamp(e["trough_date"]).date()),
                    "depth": e["depth"], "days_removed": int(e["end"] - e["start"] + 1),
                    "sharpe_vt": a, "sharpe_bh": b,
                    "advantage": (a - b) if np.isfinite(a) and np.isfinite(b) else np.nan})
    return out


def crisis_block_bootstrap(pnl_vt, pnl_bh, episodes, n, n_rep, rng):
    """Resample at the level of crisis / non-crisis blocks, not days.

    Twenty years of daily data contain about five volatility crises. Bootstrapping
    days pretends there are 5000 independent observations of the thing that
    actually drives the result.
    """
    marks = np.zeros(n, bool)
    for e in episodes:
        marks[e["start"]:e["end"] + 1] = True
    bounds = np.flatnonzero(np.diff(marks.astype(int)) != 0) + 1
    seg = np.split(np.arange(n), bounds)
    seg = [s for s in seg if len(s)]
    v = np.asarray(pnl_vt, float)
    b = np.asarray(pnl_bh, float)
    adv = np.empty(n_rep)
    for i in range(n_rep):
        pick = rng.integers(0, len(seg), size=len(seg))
        idx = np.concatenate([seg[k] for k in pick])
        a = perf(v[idx])["sharpe"]
        c = perf(b[idx])["sharpe"]
        adv[i] = a - c if np.isfinite(a) and np.isfinite(c) else np.nan
    adv = adv[np.isfinite(adv)]
    if len(adv) < 50:
        return {"n_blocks": len(seg), "p_advantage_le_0": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}
    lo, hi = np.percentile(adv, [2.5, 97.5])
    return {"n_blocks": len(seg), "p_advantage_le_0": float((adv <= 0).mean()),
            "ci_lo": float(lo), "ci_hi": float(hi), "mean_advantage": float(adv.mean())}


# ------------------------------------------------------- short-term gains
def short_term_gain_turnover(weights, prices, index, mask,
                             short_days: int = SHORT_TERM_DAYS):
    """FIFO lot accounting on the weight path: how much SHORT-TERM gain does the
    rebalancing realise per year, as a fraction of average exposure?

    Reported as a magnitude only. No tax rate is applied and no recommendation
    is made -- the point is that in a taxable account this line can dominate the
    5 bps spread, and it is invisible unless measured.
    """
    w = np.asarray(weights, float)
    p = np.asarray(prices, float)
    pos = np.flatnonzero(np.asarray(mask))
    if len(pos) < 60:
        return {"short_term_gain_annual_pct": np.nan, "long_term_gain_annual_pct": np.nan,
                "n_sales": 0}
    lots, st_gain, lt_gain, n_sales = [], 0.0, 0.0, 0
    prev = 0.0
    for i in pos:
        if not (np.isfinite(w[i]) and np.isfinite(p[i])):
            continue
        d = w[i] - prev
        if d > 1e-12:
            lots.append([d, p[i], index[i]])
        elif d < -1e-12:
            n_sales += 1
            to_close = -d
            while to_close > 1e-12 and lots:
                q, ep, ed = lots[0]
                take = min(q, to_close)
                gain = take * (p[i] / ep - 1.0)
                held = (pd.Timestamp(index[i]) - pd.Timestamp(ed)).days
                if gain > 0:
                    if held < short_days:
                        st_gain += gain
                    else:
                        lt_gain += gain
                to_close -= take
                if take >= q - 1e-12:
                    lots.pop(0)
                else:
                    lots[0][0] = q - take
        prev = w[i]
    years = max((pd.Timestamp(index[pos[-1]]) - pd.Timestamp(index[pos[0]])).days / 365.25,
                1e-9)
    mean_w = float(np.nanmean(w[pos])) or 1.0
    return {"short_term_gain_annual_pct": float(st_gain / years / max(mean_w, 1e-9) * 100),
            "long_term_gain_annual_pct": float(lt_gain / years / max(mean_w, 1e-9) * 100),
            "n_sales": int(n_sales)}
