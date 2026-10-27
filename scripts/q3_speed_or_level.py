"""N3 -- speed or level?

Two different defects produce the same symptom. Either the VIX forecast reacts
too fast, or it is systematically too high in stress. Smoothing separates them:
delay the VIX contribution and change nothing else. If the Sharpe recovers as the
smoothing window grows, the defect was speed. If it does not, smoothing cannot
help and the defect is level.

The grid k in {1, 3, 5, 10, 20} is fixed in the specification and counts as five
trials per instrument.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config import BOOTSTRAP_MEAN_BLOCK
from config_m1 import RESULTS_DIR
from config_m2 import FIG_DIR, INSTRUMENTS, MASTER_SEED_M2
from m2pair import FAST, SLOW, build_pair, dev_mask, weights_pnl
from sharpe_inference import sharpe_difference_lw
from voltarget import perf

K_GRID = [1, 3, 5, 10, 20]


def smoothed_sigma(sig: np.ndarray, k: int, index) -> np.ndarray:
    """Causal EWMA of the forecast. k = 1 leaves it untouched."""
    if k <= 1:
        return sig
    return pd.Series(sig, index=index).ewm(span=k, adjust=False,
                                           min_periods=1).mean().to_numpy()


def main():
    print("=" * 78)
    print("N3  SPEED OR LEVEL -- smoothing the VIX forecast")
    print("=" * 78)
    con = db.connect()
    rows = []

    for tk in INSTRUMENTS:
        f = build_pair(tk)
        m = dev_mask(tk, f)
        _, p_slow, _ = weights_pnl(f, SLOW, m)
        base_slow = perf(p_slow, np.asarray(m))["sharpe"]
        sig_fast = f["sigma_" + FAST].to_numpy(float)
        for k in K_GRID:
            f2 = f.copy()
            f2["sigma_smooth"] = smoothed_sigma(sig_fast, k, f.index)
            w, p, dw = weights_pnl(f2, "smooth", m)
            pf = perf(p, np.asarray(m))
            lw = sharpe_difference_lw(p_slow[np.asarray(m)], p[np.asarray(m)])
            rows.append({
                "instrument": tk, "k": k, "sharpe_smoothed": pf["sharpe"],
                "sharpe_slow": base_slow,
                "gap_slow_minus_smoothed": base_slow - pf["sharpe"],
                "p_lw_vs_slow": lw.get("p_lw", np.nan),
                "turnover_annual": float(dw[np.asarray(m)].sum() /
                                         (np.asarray(m).sum() / 252)),
                "mean_w": float(np.nanmean(w[np.asarray(m)])),
            })
    df = pd.DataFrame(rows)

    print(f"\nSharpe of the VIX forecast smoothed over k days, against the "
          f"unsmoothed HAR baseline")
    print(f"   {'inst':5s} " + " ".join(f"{'k=' + str(k):>9s}" for k in K_GRID) +
          f" {'HAR':>9s}")
    for tk in INSTRUMENTS:
        s = df[df.instrument == tk].set_index("k")
        print(f"   {tk:5s} " + " ".join(f"{s.loc[k, 'sharpe_smoothed']:9.4f}"
                                        for k in K_GRID) +
              f" {s.loc[K_GRID[0], 'sharpe_slow']:9.4f}")

    print(f"\ngap remaining (HAR minus smoothed VIX version); recovery means the gap "
          f"shrinks towards 0")
    print(f"   {'inst':5s} " + " ".join(f"{'k=' + str(k):>9s}" for k in K_GRID))
    for tk in INSTRUMENTS:
        s = df[df.instrument == tk].set_index("k")
        print(f"   {tk:5s} " + " ".join(f"{s.loc[k, 'gap_slow_minus_smoothed']:+9.4f}"
                                        for k in K_GRID))

    rec = []
    for tk in INSTRUMENTS:
        s = df[df.instrument == tk].set_index("k")
        g1, g20 = s.loc[1, "gap_slow_minus_smoothed"], s.loc[20, "gap_slow_minus_smoothed"]
        best_k = int(s.sharpe_smoothed.idxmax())
        closes = bool(abs(g20) < abs(g1) * 0.5)
        rec.append({"instrument": tk, "gap_k1": g1, "gap_k20": g20,
                    "gap_closed_by_half": closes, "best_k": best_k,
                    "best_sharpe": float(s.sharpe_smoothed.max()),
                    "beats_slow_at_best_k": bool(s.sharpe_smoothed.max() > s.loc[1, "sharpe_slow"])})
    rd = pd.DataFrame(rec)
    n_close = int(rd.gap_closed_by_half.sum())
    n_beat = int(rd.beats_slow_at_best_k.sum())
    verdict = "SPEED" if n_close >= 3 else "LEVEL_OR_NEITHER"
    print(f"\n   {'inst':5s} {'gap k=1':>9s} {'gap k=20':>9s} {'halved':>7s} "
          f"{'best k':>7s} {'best Sharpe':>12s} {'beats HAR':>10s}")
    for _, r in rd.iterrows():
        print(f"   {r.instrument:5s} {r.gap_k1:+9.4f} {r.gap_k20:+9.4f} "
              f"{'yes' if r.gap_closed_by_half else 'no':>7s} {r.best_k:7d} "
              f"{r.best_sharpe:12.4f} "
              f"{'yes' if r.beats_slow_at_best_k else 'no':>10s}")
    print(f"\n   gap halved by k=20 in {n_close}/{len(rd)} instruments; the smoothed "
          f"version beats plain HAR at its best k in {n_beat}/{len(rd)}")
    print(f"   -> defect looks like {verdict}")
    print(f"   NOTE: 'best k' is chosen in-sample over a 5-point grid on a gap that "
          f"N0 could not")
    print(f"   distinguish from zero. It is reported, not recommended.")

    df.to_csv(os.path.join(RESULTS_DIR, "m2_n3_smoothing.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "m2_n3_smoothing.json"), "w") as fh:
        json.dump({"k_grid": K_GRID, "rows": df.to_dict("records"),
                   "summary": rd.to_dict("records"),
                   "n_gap_halved": n_close, "n_beats_har_at_best_k": n_beat,
                   "verdict": verdict,
                   "n_trials": len(df),
                   "caveat": "best k is an in-sample pick over a 5-point grid on a gap "
                             "N0 classified GAP_UNDERPOWERED"}, fh, indent=2,
                  default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for tk in INSTRUMENTS:
        s = df[df.instrument == tk].set_index("k")
        ax.plot(K_GRID, [s.loc[k, "gap_slow_minus_smoothed"] for k in K_GRID],
                marker="o", label=tk, lw=1.4)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("EWMA smoothing span k (days)")
    ax.set_ylabel("Sharpe gap: HAR minus smoothed HAR+VIX")
    ax.set_title("N3  does delaying the VIX contribution close the gap?", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "m2_n3_smoothing.png"), dpi=135)
    plt.close(fig)

    db.log_experiment(con, "N3", "smoothing grid on the VIX forecast: speed or level",
                      {"k_grid": K_GRID, "verdict": verdict,
                       "n_trials": len(df)},
                      n_tests=len(df), seed=MASTER_SEED_M2, status="COMPLETED")
    con.close()
    print("\nwrote results/m2_n3_smoothing.csv/.json")


if __name__ == "__main__":
    main()
