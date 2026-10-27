"""M1 -- the discovery engine.

`enumerate_search_space()` is the single source of truth for how many
hypotheses exist.  It is frozen before M2 runs and is executed unchanged on
synthetic data (M2) and on real data (M5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import stats_engine as se
from config import (AL_REG_WINDOW, BASE_COST_BPS, COST_SWEEP_BPS, PAIRS,
                    REGIME_PAIRS, RESIDUAL_SPECS, RETURN_HORIZONS, RVOL_WINDOWS,
                    VOL_HORIZONS, VOL_SPILLOVER, XCORR_LAGS, Z_WINDOWS)


@dataclass(frozen=True)
class TestSpec:
    signal_id: str
    family: str            # pair_mr | residual_mr | vol_spillover | regime_vol
    description: str
    pair: str
    feature: str           # key into the feature dict
    feature_window: int
    target: str            # key into the target dict
    horizon: int
    target_kind: str       # return | direction | volatility
    traded: str            # key into the traded-series dict, or "" when not tradable
    n_legs: int
    position_mode: str     # continuous | direction | none


def enumerate_search_space() -> list[TestSpec]:
    specs: list[TestSpec] = []

    # ---- 1/2. tradable pair mean-reversion -----------------------------
    for a, b in PAIRS:
        for w in Z_WINDOWS:
            feat = f"z_{a}_{b}_{w}"
            traded = f"spread_{a}_{b}"
            for h in RETURN_HORIZONS:
                specs.append(TestSpec(
                    f"{feat}__fwdret{h}", "pair_mr",
                    f"z-score({w}d) of log({a}/{b}) vs {h}d forward spread return",
                    f"{a}/{b}", feat, w, f"fwdret_{traded}_{h}", h, "return",
                    traded, 2, "continuous"))
                specs.append(TestSpec(
                    f"{feat}__fwddir{h}", "pair_mr",
                    f"z-score({w}d) of log({a}/{b}) vs {h}d forward spread direction",
                    f"{a}/{b}", feat, w, f"fwddir_{traded}_{h}", h, "direction",
                    traded, 2, "direction"))

    # ---- 3. Avellaneda-Lee residual spread -----------------------------
    for asset, sid in RESIDUAL_SPECS:
        feat = f"al_{asset}_{sid}_{AL_REG_WINDOW}"
        traded = f"asset_{asset}"
        for h in RETURN_HORIZONS:
            specs.append(TestSpec(
                f"{feat}__fwdret{h}", "residual_mr",
                f"A-L s-score of {asset} residual vs {sid} ({AL_REG_WINDOW}d) "
                f"vs {h}d forward {asset} return",
                f"{asset}/{sid}", feat, AL_REG_WINDOW, f"fwdret_{traded}_{h}", h,
                "return", traded, 1, "continuous"))
            specs.append(TestSpec(
                f"{feat}__fwddir{h}", "residual_mr",
                f"A-L s-score of {asset} residual vs {sid} ({AL_REG_WINDOW}d) "
                f"vs {h}d forward {asset} direction",
                f"{asset}/{sid}", feat, AL_REG_WINDOW, f"fwddir_{traded}_{h}", h,
                "direction", traded, 1, "direction"))

    # ---- 4. volatility spillover TLT -> SPY ----------------------------
    # Predicts the SIZE of the move, not its direction: no tradable P&L here.
    for src, dst in VOL_SPILLOVER:
        for w in RVOL_WINDOWS:
            for lag in XCORR_LAGS:
                feat = f"rvol_{src}_{w}_lag{lag}"
                for h in VOL_HORIZONS:
                    specs.append(TestSpec(
                        f"{feat}__fwdrvol{h}", "vol_spillover",
                        f"{src} realised vol({w}d) lagged {lag}d vs {dst} "
                        f"{h}d forward realised vol",
                        f"{src}->{dst}", feat, w, f"fwdrvol_{dst}_{h}", h,
                        "volatility", "", 0, "none"))

    # ---- 5. IWM/SPY as a REGIME variable, not a spread trade -----------
    for a, b in REGIME_PAIRS:
        for w in Z_WINDOWS:
            feat = f"z_{a}_{b}_{w}"
            for h in VOL_HORIZONS:
                specs.append(TestSpec(
                    f"{feat}__fwdrvol{h}", "regime_vol",
                    f"z-score({w}d) of log({a}/{b}) vs {b} {h}d forward realised vol",
                    f"{a}/{b}", feat, w, f"fwdrvol_{b}_{h}", h, "volatility",
                    "", 0, "none"))

    ids = [s.signal_id for s in specs]
    assert len(ids) == len(set(ids)), "duplicate signal_id in search space"
    return specs


# ------------------------------------------------------------------ evaluation
def _slice(idx: pd.DatetimeIndex, start, end):
    return (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))


def evaluate_spec(spec: TestSpec, feats, tgts, traded, mask, sign=None,
                  cost_bps=BASE_COST_BPS, cost_sweep=False):
    """Evaluate one hypothesis on one date mask.

    `sign` fixes the trade direction learned on TRAIN.  When None the sign is
    FITTED here from the sign of the IC -- that fitting step is part of the
    search and the null harness performs it identically.
    """
    x = feats[spec.feature]
    y = tgts[spec.target]
    common = x.index
    xv = x.to_numpy(float)[mask]
    yv = y.to_numpy(float)[mask]

    ic = se.spearman_ic(xv, yv)
    beta, t_nw, p_nw, n_obs = se.newey_west_slope(yv, xv, lag=max(spec.horizon - 1, 0))

    res = dict(signal_id=spec.signal_id, ic=ic, beta=beta, t_stat_nw=t_nw,
               p_nw=p_nw, n_obs=n_obs, sign=np.nan, gross_sharpe=np.nan,
               net_sharpe=np.nan, turnover_annual=np.nan, max_drawdown=np.nan,
               skew=np.nan, kurtosis=np.nan, mean_net=np.nan, pnl=None)
    for c in COST_SWEEP_BPS:
        res[f"net_sharpe_{c}bps"] = np.nan

    if spec.position_mode == "none" or not spec.traded:
        return res

    if sign is None:
        sign = 1.0 if (np.isfinite(ic) and ic > 0) else -1.0
    res["sign"] = float(sign)

    r1 = traded[spec.traded].shift(-1).to_numpy(float)[mask]
    pos = se.build_position(xv, sign, spec.horizon, spec.position_mode)
    st = se.strategy_stats(pos, r1, cost_bps, spec.n_legs)
    res.update(gross_sharpe=st["gross_sharpe"], net_sharpe=st["net_sharpe"],
               turnover_annual=st["turnover_annual"], max_drawdown=st["max_drawdown"],
               skew=st["skew"], kurtosis=st["kurtosis"], mean_net=st["mean_net"],
               pnl=st["pnl"])
    if cost_sweep:
        for c in COST_SWEEP_BPS:
            res[f"net_sharpe_{c}bps"] = se.strategy_stats(
                pos, r1, c, spec.n_legs)["net_sharpe"]
    return res


def run_search(feats, tgts, traded, mask, specs, signs=None, cost_bps=BASE_COST_BPS,
               cost_sweep=False, keep_pnl=False):
    rows = []
    for sp in specs:
        sign = None if signs is None else signs.get(sp.signal_id)
        r = evaluate_spec(sp, feats, tgts, traded, mask, sign=sign,
                          cost_bps=cost_bps, cost_sweep=cost_sweep)
        if not keep_pnl:
            r.pop("pnl", None)
        rows.append(r)
    return rows


def summarise_maxima(rows):
    """The family-wise maxima that the null distribution is built from."""
    t = np.array([r["t_stat_nw"] for r in rows], float)
    ic = np.array([r["ic"] for r in rows], float)
    ns = np.array([r["net_sharpe"] for r in rows], float)
    gs = np.array([r["gross_sharpe"] for r in rows], float)
    f = lambda a: float(np.nanmax(a)) if np.isfinite(a).any() else np.nan   # noqa: E731
    return dict(
        max_abs_t_stat_nw=f(np.abs(t)),
        max_abs_ic=f(np.abs(ic)),
        max_sharpe=f(gs),
        max_net_sharpe=f(ns),
        n_tests=len(rows),
    )
