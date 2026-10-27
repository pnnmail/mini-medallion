"""N2 -- conditional, not unconditional, bias.

M1 measured unconditional bias and found the VIX version LESS biased on three
instruments out of four. The hypothesis needs something different: bias in a
particular STATE. The variance risk premium widens in stress, so the forecast
should overstate future volatility most exactly when the next return is about to
be large and positive.

Falsifiable: if the mechanism is real the lag is concentrated in the top 10-20%
of stress days. If it is spread evenly across states, the mechanism is wrong and
this section says so.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config import TRADING_DAYS
from config_m1 import RESULTS_DIR
from config_m2 import FIG_DIR, INSTRUMENTS, MASTER_SEED_M2
from m2pair import FAST, SLOW, build_pair, dev_mask, weights_pnl
from rv_estimators import qlike

N_DECILES = 10
VRP_HORIZON = 21


def states(frame, mask, tk):
    """Descriptive conditioning variables. These are states, not trading rules,
    so full-sample quantiles are legitimate here; nothing is traded on them."""
    idx = np.flatnonzero(mask)
    vix = np.exp(frame["log_vix"].to_numpy(float))
    r = frame["ret"].to_numpy(float)
    px = np.exp(np.nancumsum(np.where(np.isfinite(r), r, 0.0)))
    peak = np.maximum.accumulate(px)
    dd = px / peak - 1.0
    days_from_high = np.zeros(len(px))
    c = 0
    for i in range(len(px)):
        c = 0 if px[i] >= peak[i] - 1e-12 else c + 1
        days_from_high[i] = c
    r5 = pd.Series(r).rolling(5, min_periods=5).sum().to_numpy()
    out = pd.DataFrame({
        "vix": vix[idx], "drawdown": dd[idx], "days_from_high": days_from_high[idx],
        "ret1": r[idx], "ret5": r5[idx]}, index=frame.index[idx])
    out["vix_decile"] = pd.qcut(out.vix, N_DECILES, labels=False, duplicates="drop")
    out["dd_decile"] = pd.qcut(out.drawdown, N_DECILES, labels=False, duplicates="drop")
    return out


def main():
    print("=" * 78)
    print("N2  CONDITIONAL BIAS BY STRESS STATE")
    print("=" * 78)
    con = db.connect()
    recs, vrp_recs, loc_rows = [], [], []

    for tk in INSTRUMENTS:
        f = build_pair(tk)
        m = dev_mask(tk, f)
        st = states(f, m, tk)
        sel = np.asarray(m)
        w_s, p_s, _ = weights_pnl(f, SLOW, m)
        w_f, p_f, _ = weights_pnl(f, FAST, m)
        tv = f["target_var"].to_numpy(float)[sel]
        pv_s = f["pred_var_" + SLOW].to_numpy(float)[sel]
        pv_f = f["pred_var_" + FAST].to_numpy(float)[sel]
        sig_s = f["sigma_" + SLOW].to_numpy(float)[sel]
        sig_f = f["sigma_" + FAST].to_numpy(float)[sel]
        real = np.sqrt(TRADING_DAYS * tv)
        fwd = f["fwd_ret"].to_numpy(float)[sel]
        d_pnl = p_s[sel] - p_f[sel]
        total_gap = float(np.nansum(d_pnl))

        for dec, g in st.groupby("vix_decile"):
            k = st.index.get_indexer(g.index)
            rec = {
                "instrument": tk, "state_var": "vix_decile", "decile": int(dec),
                "n_days": len(k),
                "mean_vix": float(g.vix.mean()),
                "bias_slow": float(np.nanmean(sig_s[k]) / np.nanmean(real[k]) - 1),
                "bias_fast": float(np.nanmean(sig_f[k]) / np.nanmean(real[k]) - 1),
                "qlike_slow": qlike(tv[k], pv_s[k]), "qlike_fast": qlike(tv[k], pv_f[k]),
                "mean_w_slow": float(np.nanmean(w_s[sel][k])),
                "mean_w_fast": float(np.nanmean(w_f[sel][k])),
                "mean_fwd_ret_ann": float(np.nanmean(fwd[k]) * TRADING_DAYS),
                "gap_contribution": float(np.nansum(d_pnl[k])),
                "gap_share": float(np.nansum(d_pnl[k]) / total_gap)
                if total_gap != 0 else np.nan,
                "day_share": float(len(k) / len(st)),
            }
            recs.append(rec)

        top2 = st.vix_decile >= N_DECILES - 2
        k2 = np.flatnonzero(top2.to_numpy())
        krest = np.flatnonzero(~top2.to_numpy())
        loc = float(np.nansum(d_pnl[k2]) / total_gap) if total_gap != 0 else np.nan
        # the share-of-gap ratio divides by a near-zero total, so it explodes.
        # the per-day mean gap inside vs outside stress is the stable version.
        g_top = float(np.nanmean(d_pnl[k2]) * TRADING_DAYS)
        g_rest = float(np.nanmean(d_pnl[krest]) * TRADING_DAYS)
        loc_rows.append({"instrument": tk, "total_gap_daily_sum": total_gap,
                         "top2_vix_decile_share_of_gap": loc,
                         "top2_share_of_days": float(top2.mean()),
                         "stress_localization_ratio": loc / float(top2.mean())
                         if np.isfinite(loc) else np.nan,
                         "gap_ann_in_top2_vix": g_top,
                         "gap_ann_outside_top2": g_rest,
                         "gap_ratio_stress_vs_calm": g_top / g_rest
                         if g_rest != 0 else np.nan})

        # variance risk premium by VIX decile
        rv_fwd = pd.Series(np.sqrt(TRADING_DAYS *
                                   f["target_var"].to_numpy(float))).rolling(
            VRP_HORIZON, min_periods=VRP_HORIZON).mean().shift(-VRP_HORIZON).to_numpy()
        rv_fwd = rv_fwd[sel]
        for dec, g in st.groupby("vix_decile"):
            k = st.index.get_indexer(g.index)
            vrp_recs.append({"instrument": tk, "decile": int(dec),
                             "mean_vix_pct": float(g.vix.mean()),
                             "mean_fwd_rv_pct": float(np.nanmean(rv_fwd[k]) * 100),
                             "vrp_pct": float(g.vix.mean() -
                                              np.nanmean(rv_fwd[k]) * 100)})

    df = pd.DataFrame(recs)
    vrp = pd.DataFrame(vrp_recs)
    loc = pd.DataFrame(loc_rows)

    print("\nCONDITIONAL BIAS by VIX decile (forecast / realised - 1)")
    for tk in INSTRUMENTS:
        s = df[df.instrument == tk]
        print(f"   {tk}")
        print(f"      {'dec':>3s} {'n':>5s} {'VIX':>6s} {'biasSlow':>9s} "
              f"{'biasFast':>9s} {'qlSlow':>8s} {'qlFast':>8s} {'wSlow':>6s} "
              f"{'wFast':>6s} {'fwdRet':>8s} {'gapShare':>9s}")
        for _, r in s.iterrows():
            print(f"      {int(r.decile):3d} {int(r.n_days):5d} {r.mean_vix:6.1f} "
                  f"{r.bias_slow:+9.1%} {r.bias_fast:+9.1%} {r.qlike_slow:8.3f} "
                  f"{r.qlike_fast:8.3f} {r.mean_w_slow:6.3f} {r.mean_w_fast:6.3f} "
                  f"{r.mean_fwd_ret_ann:+8.3f} {r.gap_share:+9.1%}")

    print("\nVARIANCE RISK PREMIUM by VIX decile (VIX minus realised vol over the "
          "next 21 sessions, pct points)")
    p = vrp.pivot(index="decile", columns="instrument", values="vrp_pct")
    print(p.round(2).to_string())

    print("\nSTRESS LOCALISATION -- share of the total gap earned in the top 2 VIX "
          "deciles")
    print(f"   {'inst':5s} {'gapShare':>9s} {'dayShare':>9s} {'ratio':>7s}")
    for _, r in loc.iterrows():
        print(f"   {r.instrument:5s} {r.top2_vix_decile_share_of_gap:+9.1%} "
              f"{r.top2_share_of_days:9.1%} "
              f"{r.stress_localization_ratio:+7.2f}")
    print("\n   STABLE VERSION -- the share-of-gap ratio divides by a near-zero total,")
    print("   so it explodes (SCHD -7.1, IWM +15.8). Mean daily gap, annualised, "
          "inside vs outside stress:")
    print(f"   {'inst':5s} {'gap in top-2 VIX':>17s} {'gap elsewhere':>15s} "
          f"{'ratio':>7s}")
    for _, r in loc.iterrows():
        print(f"   {r.instrument:5s} {r.gap_ann_in_top2_vix:+17.5f} "
              f"{r.gap_ann_outside_top2:+15.5f} "
              f"{r.gap_ratio_stress_vs_calm:+7.2f}")
    n_stress_bigger = int((loc.gap_ann_in_top2_vix > loc.gap_ann_outside_top2).sum())
    print(f"   the gap is larger in stress in {n_stress_bigger}/{len(loc)} instruments")

    mean_ratio = float(loc.stress_localization_ratio.mean())
    n_localised = int((loc.stress_localization_ratio > 2.0).sum())
    mech = bool(n_localised >= 3)
    print(f"\n   mean localisation ratio = {mean_ratio:+.2f} "
          f"(1.0 = the gap is spread evenly across states)")
    print(f"   localised (ratio > 2) in {n_localised}/{len(loc)} instruments")
    print(f"   MECHANISM {'SUPPORTED' if mech else 'NOT SUPPORTED'}: the lag is "
          f"{'concentrated in stress' if mech else 'NOT concentrated in stress'}")

    df.to_csv(os.path.join(RESULTS_DIR, "m2_n2_conditional.csv"), index=False)
    vrp.to_csv(os.path.join(RESULTS_DIR, "m2_n2_vrp.csv"), index=False)
    payload = {"by_state": df.to_dict("records"), "vrp": vrp.to_dict("records"),
               "localisation": loc.to_dict("records"),
               "mean_stress_localization_ratio": mean_ratio,
               "n_localised": n_localised,
               "mechanism_supported": mech,
               "n_instruments_gap_larger_in_stress": n_stress_bigger,
               "instability_note": "the share-of-gap localisation ratio divides by a "
                                   "total gap that N0 showed is indistinguishable from "
                                   "zero, so it is unstable by construction; the "
                                   "absolute per-day version is the one to read",
               "criterion": "the gap must be concentrated in the top 2 VIX deciles "
                            "(ratio > 2, i.e. more than twice its share of days) in at "
                            "least 3 of 4 instruments"}
    with open(os.path.join(RESULTS_DIR, "m2_n2_conditional.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    for tk in INSTRUMENTS:
        s = df[df.instrument == tk]
        ax.plot(s.decile, s.gap_share * 100, marker="o", label=tk, lw=1.3)
    ax.axhline(10, color="black", ls=":", lw=1, label="even split (10%)")
    ax.set_xlabel("VIX decile"); ax.set_ylabel("share of the total gap, %")
    ax.set_title("N2  is the gap concentrated in stress?", fontsize=10)
    ax.legend(fontsize=7)
    ax = axes[1]
    for tk in INSTRUMENTS:
        s = df[df.instrument == tk]
        ax.plot(s.decile, (s.bias_fast - s.bias_slow) * 100, marker="o", label=tk,
                lw=1.3)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("VIX decile")
    ax.set_ylabel("extra bias of the VIX version, pct points")
    ax.set_title("conditional bias difference", fontsize=10)
    ax.legend(fontsize=7)
    ax = axes[2]
    for tk in INSTRUMENTS:
        s = vrp[vrp.instrument == tk]
        ax.plot(s.decile, s.vrp_pct, marker="o", label=tk, lw=1.3)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("VIX decile"); ax.set_ylabel("VIX - forward realised vol, pts")
    ax.set_title("variance risk premium by state", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "m2_n2_conditional.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "N2", "conditional bias, QLIKE, weights and VRP by stress "
                                 "decile; stress localisation of the gap",
                      {"instruments": list(INSTRUMENTS), "n_deciles": N_DECILES,
                       "mean_localisation_ratio": mean_ratio,
                       "mechanism_supported": mech},
                      n_tests=len(df), seed=MASTER_SEED_M2,
                      status="COMPLETED" if mech else "FAILED_MECHANISM")
    con.close()
    print("\nwrote results/m2_n2_conditional.csv/.json, m2_n2_vrp.csv")


if __name__ == "__main__":
    main()
