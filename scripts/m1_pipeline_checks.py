"""M1 -- freeze the search space and run the BLOCKING look-ahead test.

Results are deliberately not inspected here: this milestone only proves the
machinery is causal and enumerates how many hypotheses exist.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import db
from config import RESULTS_DIR, SPLITS
from discovery import enumerate_search_space
from features import build_features, half_life, log_ratio, log_returns
from panel import research_panel, split_mask
from targets import build_targets, forward_realized_vol, forward_return

SHIFTS = [1, 5, 20]


def lookahead_test(prices, macro, con):
    """Shift every input by N days; every feature must shift by exactly N.

    A feature that peeks at the future does not commute with the shift, so the
    comparison fails.  Any mismatch stops the pipeline.
    """
    base = build_features(prices, macro, None)
    rows, all_ok = [], True
    for n in SHIFTS:
        sp = prices.shift(n)
        sm = macro.shift(n)
        shifted = build_features(sp, sm, None)
        for name, s_base in base.items():
            expect = s_base.shift(n)
            got = shifted[name]
            m = expect.notna() & got.notna()
            n_cmp = int(m.sum())
            if n_cmp < 100:
                ok, note = False, f"too few comparable points ({n_cmp})"
            else:
                diff = (expect[m] - got[m]).abs()
                scale = expect[m].abs().median()
                tol = max(1e-8, 1e-6 * float(scale if np.isfinite(scale) else 1.0))
                worst = float(diff.max())
                ok = worst <= tol
                note = f"max_abs_diff={worst:.3e} tol={tol:.3e} n={n_cmp}"
            all_ok &= ok
            rows.append((name, n, name, n if ok else -1, ok, note))
    con.execute("DELETE FROM lookahead_tests")
    con.executemany("INSERT INTO lookahead_tests VALUES (?,?,?,?,?,?)", rows)
    return all_ok, rows


def target_direction_test():
    """Targets must look strictly FORWARD: verify against a hand-built sum."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(size=400), index=pd.bdate_range("2010-01-01", periods=400))
    out = {}
    for h in [1, 3, 5, 10]:
        fr = forward_return(r, h)
        manual = pd.Series(
            [r.values[i + 1:i + 1 + h].sum() if i + h < len(r) else np.nan
             for i in range(len(r))], index=r.index)
        m = fr.notna() & manual.notna()
        out[f"forward_return_{h}"] = {
            "n": int(m.sum()),
            "max_abs_diff": float((fr[m] - manual[m]).abs().max()),
            "passed": bool((fr[m] - manual[m]).abs().max() < 1e-10),
        }
    for h in [5, 10]:
        fv = forward_realized_vol(r, h)
        manual = pd.Series(
            [np.sqrt(252 * np.mean(r.values[i + 1:i + 1 + h] ** 2)) if i + h < len(r) else np.nan
             for i in range(len(r))], index=r.index)
        m = fv.notna() & manual.notna()
        out[f"forward_rvol_{h}"] = {
            "n": int(m.sum()),
            "max_abs_diff": float((fv[m] - manual[m]).abs().max()),
            "passed": bool((fv[m] - manual[m]).abs().max() < 1e-10),
        }
    return out


def main():
    print("=" * 78)
    print("M1  PIPELINE + FROZEN SEARCH SPACE")
    print("=" * 78)
    con = db.connect()
    prices, macro = research_panel()
    print(f"research panel: {prices.shape[1]} tickers, {len(prices)} rows, "
          f"{prices.index.min().date()} .. {prices.index.max().date()}")

    specs = enumerate_search_space()
    fam = pd.Series([s.family for s in specs]).value_counts().to_dict()
    print(f"search space: {len(specs)} hypotheses  {fam}")

    feats = build_features(prices, macro, None)
    tgts, traded = build_targets(prices)
    print(f"features built: {len(feats)}   targets built: {len(tgts)}   traded: {len(traded)}")

    print("\n-- look-ahead test (N = 1, 5, 20) --")
    ok, rows = lookahead_test(prices, macro, con)
    bad = [r for r in rows if not r[4]]
    print(f"   {len(rows)} feature x shift comparisons, {len(bad)} failures")
    for b in bad[:10]:
        print("   FAIL", b[0], "shift", b[1], b[5])
    if not ok:
        db.log_experiment(con, "M1", "look-ahead test", {"shifts": SHIFTS},
                          len(specs), None, "FAILED_LOOKAHEAD")
        con.close()
        raise SystemExit("LOOK-AHEAD TEST FAILED -- pipeline halted")

    print("\n-- target direction test --")
    td = target_direction_test()
    for k, v in td.items():
        print(f"   {k:22s} n={v['n']:4d} max_diff={v['max_abs_diff']:.2e} "
              f"{'PASS' if v['passed'] else 'FAIL'}")
    if not all(v["passed"] for v in td.values()):
        con.close()
        raise SystemExit("TARGET DIRECTION TEST FAILED -- pipeline halted")

    # half-life of every underlying spread, on TRAIN only
    tr = split_mask(prices.index, "TRAIN")
    hl = {}
    for a, b in [("GDX", "GLD"), ("XOP", "XLE"), ("IWM", "SPY")]:
        v, note = half_life(log_ratio(prices[a], prices[b])[tr])
        hl[f"{a}/{b}"] = {"half_life_days": v, "note": note}
    from config import AL_REG_WINDOW
    from features import avellaneda_lee_score
    al = avellaneda_lee_score(log_returns(prices["XLF"]), macro.diff(), AL_REG_WINDOW)
    hl["XLF/T10Y2Y (median of rolling AL fits)"] = {
        "half_life_days": float(np.nanmedian(al["half_life_days"][tr])),
        "note": "rolling window fits; NaN where AR(1) b outside (0,1)",
        "pct_windows_mean_reverting": float(np.isfinite(al["ar_b"][tr]).mean()),
    }

    payload = {
        "spec_version": __import__("config").SPEC_VERSION,
        "n_hypotheses": len(specs),
        "by_family": fam,
        "n_features": len(feats),
        "n_targets": len(tgts),
        "panel": {"tickers": list(prices.columns), "rows": int(len(prices)),
                  "first_date": str(prices.index.min().date()),
                  "last_date": str(prices.index.max().date())},
        "splits": {k: list(v) for k, v in SPLITS.items()},
        "lookahead_test": {"shifts": SHIFTS, "comparisons": len(rows),
                           "failures": len(bad), "passed": bool(ok)},
        "target_direction_test": td,
        "half_life_train": hl,
        "signals": [s.__dict__ if hasattr(s, "__dict__") else
                    {f: getattr(s, f) for f in s.__dataclass_fields__} for s in specs],
    }
    out = os.path.join(RESULTS_DIR, "search_space.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    db.log_experiment(con, "M1", "search space frozen; look-ahead + target tests passed",
                      {"shifts": SHIFTS, "families": fam}, len(specs), None, "PASSED")
    con.close()
    print("\nhalf-life on TRAIN:")
    for k, v in hl.items():
        print(f"   {k:38s} {v['half_life_days']}  ({v['note']})")
    print(f"\nwrote {out}")
    print("LOOK-AHEAD TEST: PASSED")


if __name__ == "__main__":
    main()
