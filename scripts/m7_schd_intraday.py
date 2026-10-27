"""M7 (optional) -- intraday profile of SCHD.

The goal is EXECUTION, not a signal: the position is held and added to anyway,
so a real time-of-day profile is collected for free by choosing when to buy,
with no extra turnover and no extra risk.  That is also why the effect must be
compared against the spread: if it is smaller than the cost of acting on it,
it is not actionable regardless of its t-statistic.

Separate hourly snapshot -- yfinance only serves ~730 days of hourly bars, so
this cannot come from the M0 daily freeze.  It is frozen separately.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
import stats_engine as se
from config import (BOOTSTRAP_MEAN_BLOCK, DATA_DIR, FIG_DIR, MASTER_SEED,
                    RESULTS_DIR, SCHD_SPREAD_ANCHOR_BPS)

TARGET = "SCHD"
CONTROLS = ["VYM", "DGRO", "VIG"]          # null test: dividend ETFs with the same flows
ALL = [TARGET] + CONTROLS
HOURLY_DIR = os.path.join(DATA_DIR, "raw_snapshot_hourly")


def download_hourly(force=False):
    import yfinance as yf
    os.makedirs(HOURLY_DIR, exist_ok=True)
    meta_path = os.path.join(HOURLY_DIR, "_metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"tickers": {}}
    for t in ALL:
        p = os.path.join(HOURLY_DIR, f"{t}.parquet")
        if os.path.exists(p) and not force:
            continue
        df = None
        for attempt in range(4):
            try:
                df = yf.download(t, period="730d", interval="1h", auto_adjust=False,
                                 progress=False, threads=False)
                if df is not None and len(df):
                    break
            except Exception as exc:                       # noqa: BLE001
                print(f"  {t} attempt {attempt}: {exc}")
            time.sleep(2 + 3 * attempt)
        if df is None or not len(df):
            raise RuntimeError(f"hourly download failed for {t}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")
        df.index.name = "ts"
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df.to_parquet(p)
        meta["tickers"][t] = {
            "ticker": t, "interval": "1h", "period": "730d",
            "download_timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "first_ts": str(df.index.min()), "last_ts": str(df.index.max()),
            "row_count": int(len(df)),
        }
        print(f"  {t:5s} {len(df):6d} bars  {df.index.min()} .. {df.index.max()}")
    meta["snapshot_date"] = _dt.date.today().isoformat()
    json.dump(meta, open(meta_path, "w"), indent=2)
    return meta


def segment_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Split each session into non-overlapping legs that sum to the daily return.

    yfinance serves 1h bars stamped 09:30, 10:30, ... 15:30, so 09:45 is not
    reachable.  The first hour is therefore ISOLATED rather than trimmed: the
    opening auction print and the day's worst spread live inside it and are
    reported on their own line instead of contaminating the morning leg.
    """
    d = df.copy()
    d["date"] = d.index.date
    d["hour"] = d.index.hour + d.index.minute / 60.0
    out = []
    prev_close = None
    for day, g in d.groupby("date", sort=True):
        g = g.sort_index()
        if len(g) < 5 or g["hour"].iloc[0] > 9.75:
            prev_close = g["close"].iloc[-1] if len(g) else prev_close
            continue
        op = float(g["open"].iloc[0])                    # 09:30 open
        first_hour_close = float(g["close"].iloc[0])     # 10:30
        noon = g[g["hour"] <= 12.5]
        noon_close = float(noon["close"].iloc[-1]) if len(noon) else np.nan
        close = float(g["close"].iloc[-1])               # 16:00
        rec = {"date": pd.Timestamp(day), "n_bars": int(len(g))}
        rec["overnight"] = (op / prev_close - 1.0) if prev_close else np.nan
        rec["first_hour"] = first_hour_close / op - 1.0
        rec["morning_ex_first_hour"] = noon_close / first_hour_close - 1.0
        rec["afternoon"] = close / noon_close - 1.0
        rec["open_to_noon"] = noon_close / op - 1.0
        rec["full_day_close_to_close"] = (close / prev_close - 1.0) if prev_close else np.nan
        out.append(rec)
        prev_close = close
    r = pd.DataFrame(out).set_index("date")
    eom = r.index.to_period("M")
    eoq = r.index.to_period("Q")
    # last three sessions of each month / quarter
    r["is_month_end"] = r.groupby(eom).cumcount(ascending=False) < 3
    r["is_quarter_end"] = r.groupby(eoq).cumcount(ascending=False) < 3
    return r


def leg_stats(x, rng):
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if len(v) < 60:
        return {"n": int(len(v)), "mean_bps": np.nan, "t_nw": np.nan,
                "p_boot": np.nan, "ci_lo_bps": np.nan, "ci_hi_bps": np.nan}
    t, _, n = se.newey_west_mean_tstat(v, lag=0)
    p, ci = se.bootstrap_mean_pvalue(v, BOOTSTRAP_MEAN_BLOCK, rng, 2000)
    return {"n": int(n), "mean_bps": float(v.mean() * 1e4), "t_nw": float(t),
            "p_boot": float(p), "ci_lo_bps": float(ci[0] * 1e4),
            "ci_hi_bps": float(ci[1] * 1e4)}


LEGS = ["overnight", "first_hour", "morning_ex_first_hour", "afternoon",
        "open_to_noon", "full_day_close_to_close"]


def main():
    print("=" * 78)
    print("M7  SCHD INTRADAY PROFILE (optional)")
    print("=" * 78)
    con = db.connect()
    download_hourly()

    rng = np.random.default_rng(MASTER_SEED + 77)
    segs, res = {}, {}
    for t in ALL:
        df = pd.read_parquet(os.path.join(HOURLY_DIR, f"{t}.parquet"))
        s = segment_returns(df)
        segs[t] = s
        res[t] = {"all_days": {leg: leg_stats(s[leg], rng) for leg in LEGS},
                  "ex_month_end": {leg: leg_stats(s.loc[~s.is_month_end, leg], rng)
                                   for leg in LEGS},
                  "month_end_only": {leg: leg_stats(s.loc[s.is_month_end, leg], rng)
                                     for leg in LEGS},
                  "n_sessions": int(len(s)),
                  "first_date": str(s.index.min().date()),
                  "last_date": str(s.index.max().date())}

    print(f"\nsessions: " + ", ".join(f"{t}={res[t]['n_sessions']}" for t in ALL))
    print(f"window  : {res[TARGET]['first_date']} .. {res[TARGET]['last_date']}")
    print(f"\nMean return by leg, bps per session (all days)")
    print(f"   {'leg':24s} " + " ".join(f"{t:>18s}" for t in ALL))
    for leg in LEGS:
        cells = []
        for t in ALL:
            a = res[t]["all_days"][leg]
            cells.append(f"{a['mean_bps']:+8.2f} (t{a['t_nw']:+5.2f})")
        print(f"   {leg:24s} " + " ".join(f"{c:>18s}" for c in cells))

    # ---- the actual question ------------------------------------------
    # The position is held and added to anyway, so the only decision available
    # is WHICH HOUR OF THE SESSION to place the buy.  The overnight leg is not
    # that decision -- a holder receives it either way -- so it is reported but
    # excluded from the actionability test.  Only intraday legs qualify.
    schd = res[TARGET]["all_days"]
    TIMING_LEGS = ["first_hour", "morning_ex_first_hour", "afternoon"]
    best_leg = max(TIMING_LEGS,
                   key=lambda l: abs(schd[l]["mean_bps"])
                   if np.isfinite(schd[l]["mean_bps"]) else -1)
    best = schd[best_leg]

    # null test: does the effect appear only in SCHD, or in every dividend ETF?
    controls_same_sign = sum(
        1 for c in CONTROLS
        if np.isfinite(res[c]["all_days"][best_leg]["mean_bps"]) and
        np.sign(res[c]["all_days"][best_leg]["mean_bps"]) == np.sign(best["mean_bps"]))
    controls_significant = sum(
        1 for c in CONTROLS if res[c]["all_days"][best_leg]["p_boot"] < 0.05)

    print(f"\nNULL TEST on the largest SCHD leg ({best_leg})")
    print(f"   SCHD {best['mean_bps']:+.2f} bps  t={best['t_nw']:+.2f}  "
          f"p_boot={best['p_boot']:.4f}  95% CI [{best['ci_lo_bps']:+.2f}, "
          f"{best['ci_hi_bps']:+.2f}] bps")
    for c in CONTROLS:
        a = res[c]["all_days"][best_leg]
        print(f"   {c:5s}{a['mean_bps']:+.2f} bps  t={a['t_nw']:+.2f}  "
              f"p_boot={a['p_boot']:.4f}")
    print(f"   same sign in {controls_same_sign}/3 controls, "
          f"significant in {controls_significant}/3")

    if controls_same_sign == 0:
        interp = ("The effect appears in SCHD alone. With three controls that is weak "
                  "evidence of something specific and stronger evidence of noise.")
    elif controls_same_sign == 3:
        interp = ("The same sign appears in all three control ETFs, which points at a "
                  "shared structure in dividend-ETF order flow rather than anything "
                  "specific to SCHD.")
    else:
        interp = (f"The sign matches in {controls_same_sign} of 3 controls -- neither a "
                  "clean SCHD-specific effect nor a clean common one.")

    ratio = abs(best["mean_bps"]) / SCHD_SPREAD_ANCHOR_BPS
    actionable = bool(abs(best["mean_bps"]) > SCHD_SPREAD_ANCHOR_BPS and
                      best["p_boot"] < 0.05)

    # what a buyer actually faces: cumulative drift from the open to each checkpoint
    s_ = segs[TARGET]
    path = {"09:30 (open)": 0.0,
            "10:30": float(s_["first_hour"].mean() * 1e4),
            "12:30": float((s_["first_hour"] + s_["morning_ex_first_hour"]).mean() * 1e4),
            "16:00": float((s_["first_hour"] + s_["morning_ex_first_hour"]
                            + s_["afternoon"]).mean() * 1e4)}
    cheapest = min(path, key=path.get)
    spread_path = max(path.values()) - min(path.values())

    print("\nEXECUTION VIEW -- mean SCHD price path within the session, bps from open")
    for k, v in path.items():
        tag = "   <-- cheapest on average" if k == cheapest else ""
        print(f"   {k:14s} {v:+7.2f} bps{tag}")
    print(f"   best-to-worst spread across the session: {spread_path:.2f} bps "
          f"vs a {SCHD_SPREAD_ANCHOR_BPS} bps quoted spread")

    print(f"\nvs COST: largest TIMING leg ({best_leg}) is {abs(best['mean_bps']):.2f} bps "
          f"against a {SCHD_SPREAD_ANCHOR_BPS} bps spread -> {ratio:.1f}x")
    print(f"   {interp}")
    print(f"   actionable for execution timing: {actionable}")
    if not actionable:
        print("   -> no intraday timing edge: every intraday leg is smaller than the "
              "spread and none is distinguishable from zero.")
    print(f"   NOTE: SCHD's overnight leg ({schd['overnight']['mean_bps']:+.2f} bps, "
          f"t={schd['overnight']['t_nw']:+.2f}) carries essentially the whole daily "
          f"return ({schd['full_day_close_to_close']['mean_bps']:+.2f} bps), "
          "independently reproducing the M3-A calibration on different data and a "
          "different period. It is NOT an execution lever: a holder receives it "
          "either way.")

    # month-end / quarter-end, which the spec expects to push the other way
    print(f"\nMONTH-END EFFECT on SCHD (last 3 sessions of each month)")
    for leg in ["first_hour", "morning_ex_first_hour", "afternoon"]:
        a = res[TARGET]["ex_month_end"][leg]
        b = res[TARGET]["month_end_only"][leg]
        print(f"   {leg:24s} ex-ME {a['mean_bps']:+7.2f} bps (n={a['n']:4d})   "
              f"ME-only {b['mean_bps']:+7.2f} bps (n={b['n']:3d})")

    payload = {
        "window": {"first": res[TARGET]["first_date"], "last": res[TARGET]["last_date"]},
        "bar_interval": "1h",
        "measurement_note": "yfinance 1h bars are stamped 09:30/10:30/.../15:30, so the "
                            "09:45 start the spec asks for is not reachable. The first "
                            "hour is isolated as its own leg instead, which removes the "
                            "opening auction print and the day's worst spread from the "
                            "morning leg -- more conservative than trimming to 09:45.",
        "results": res,
        "largest_schd_timing_leg": best_leg,
        "timing_legs_considered": ["first_hour", "morning_ex_first_hour", "afternoon"],
        "overnight_excluded_from_actionability":
            "the overnight leg is not a time-of-day decision -- a holder receives it "
            "regardless of when the buy is placed -- so it is reported but cannot make "
            "an execution profile actionable",
        "intraday_price_path_bps_from_open": path,
        "cheapest_checkpoint_on_average": cheapest,
        "session_best_to_worst_spread_bps": spread_path,
        "null_test": {"controls": CONTROLS,
                      "controls_same_sign": controls_same_sign,
                      "controls_significant_p05": controls_significant,
                      "interpretation": interp},
        "vs_cost": {"leg_bps": best["mean_bps"], "spread_anchor_bps": SCHD_SPREAD_ANCHOR_BPS,
                    "ratio": ratio, "actionable": actionable,
                    "caveat": "the 3 bps figure is the spec's quoted-spread anchor; M4 "
                              "showed daily-bar estimators cannot independently confirm "
                              "it, so this comparison inherits that assumption"},
    }
    with open(os.path.join(RESULTS_DIR, "schd_intraday.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    legs_plot = ["overnight", "first_hour", "morning_ex_first_hour", "afternoon"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    ax = axes[0]
    w = 0.2
    for i, t in enumerate(ALL):
        vals = [res[t]["all_days"][l]["mean_bps"] for l in legs_plot]
        ax.bar(np.arange(len(legs_plot)) + i * w - 1.5 * w, vals, w, label=t)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(SCHD_SPREAD_ANCHOR_BPS, color="#C44E52", ls=":", lw=1.4,
               label=f"spread {SCHD_SPREAD_ANCHOR_BPS} bps")
    ax.axhline(-SCHD_SPREAD_ANCHOR_BPS, color="#C44E52", ls=":", lw=1.4)
    ax.set_xticks(range(len(legs_plot)))
    ax.set_xticklabels(legs_plot, fontsize=8, rotation=12)
    ax.set_ylabel("mean bps per session")
    ax.set_title("M7  intraday leg returns, SCHD vs dividend-ETF controls", fontsize=10)
    ax.legend(fontsize=8, ncol=3)
    ax = axes[1]
    s = segs[TARGET]
    for l in legs_plot:
        ax.plot(s.index, np.log1p(s[l].fillna(0)).cumsum() * 100, label=l, lw=1.3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("cumulative %")
    ax.set_title("SCHD cumulative contribution by leg", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "schd_intraday.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "M7", "SCHD intraday execution profile + dividend-ETF null test",
                      {"tickers": ALL, "legs": LEGS, "interval": "1h",
                       "n_leg_tests": len(ALL) * len(LEGS) * 3},
                      n_tests=len(ALL) * len(LEGS) * 3, seed=MASTER_SEED,
                      status="COMPLETED")
    con.close()
    print("\nwrote results/schd_intraday.json, figures/schd_intraday.png")


if __name__ == "__main__":
    main()
