"""M2 -- what does OUR OWN search find on data that contains no signal?

Until that number is known, no statistic computed on real data is
interpretable.  The synthetic data is produced by a stationary block bootstrap
(Politis-Romano) applied INDEPENDENTLY to each instrument, which keeps local
dependence, volatility clustering and fat tails inside each series while
destroying cross-asset synchronisation and lead-lag structure -- exactly the
structure every feature in the search space claims to exploit.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

import stats_engine as se
from config import BOOTSTRAP_MEAN_BLOCK, MASTER_SEED
from discovery import run_search, summarise_maxima
from features import build_features, log_returns
from targets import build_targets


def derive_seed(replication_id: int, name: str, master_seed: int = MASTER_SEED) -> int:
    """Deterministic per-replication, per-series seed. Logged, hence checkable."""
    h = zlib.crc32(name.encode("utf-8"))
    return int((master_seed + replication_id * 1_000_003 + h) % (2 ** 31 - 1))


def synthetic_panel(prices: pd.DataFrame, macro: pd.Series, replication_id: int,
                    mean_block: int = BOOTSTRAP_MEAN_BLOCK, master_seed: int = MASTER_SEED):
    """Block-bootstrap every series independently and rebuild levels."""
    idx = prices.index
    out = {}
    seeds = {}
    for t in prices.columns:
        r = log_returns(prices[t]).to_numpy(float)[1:]
        seed = derive_seed(replication_id, t, master_seed)
        seeds[t] = seed
        rng = np.random.default_rng(seed)
        bi = se.stationary_bootstrap_index(len(r), mean_block, rng)
        rb = r[bi]
        p0 = float(prices[t].iloc[0])
        out[t] = pd.Series(p0 * np.exp(np.concatenate(([0.0], np.cumsum(rb)))), index=idx)
    syn = pd.DataFrame(out, index=idx)

    dm = macro.diff().to_numpy(float)[1:]
    dm = np.where(np.isfinite(dm), dm, 0.0)
    seed = derive_seed(replication_id, "MACRO_T10Y2Y", master_seed)
    seeds["MACRO_T10Y2Y"] = seed
    rng = np.random.default_rng(seed)
    bi = se.stationary_bootstrap_index(len(dm), mean_block, rng)
    m0 = float(macro.dropna().iloc[0])
    syn_macro = pd.Series(m0 + np.concatenate(([0.0], np.cumsum(dm[bi]))), index=idx)
    return syn, syn_macro, seeds


def run_replication(prices, macro, specs, train_mask, val_mask, replication_id,
                    mean_block=BOOTSTRAP_MEAN_BLOCK, master_seed=MASTER_SEED):
    """One null replication: the FULL search space, exactly as M5 will run it."""
    syn, syn_macro, seeds = synthetic_panel(prices, macro, replication_id,
                                            mean_block, master_seed)
    feats = build_features(syn, syn_macro, None)
    tgts, traded = build_targets(syn)

    rows = run_search(feats, tgts, traded, train_mask, specs)
    mx = summarise_maxima(rows)

    # second stage: the train-selected best is carried to VALIDATION with the
    # sign frozen on TRAIN -- the null for the whole selection cascade
    t_abs = np.array([abs(r["t_stat_nw"]) if np.isfinite(r["t_stat_nw"]) else -np.inf
                      for r in rows])
    best_val_t = np.nan
    if np.isfinite(t_abs).any() and t_abs.max() > -np.inf:
        k = int(np.argmax(t_abs))
        best = specs[k]
        signs = {best.signal_id: rows[k]["sign"] if np.isfinite(rows[k]["sign"]) else 1.0}
        vrows = run_search(feats, tgts, traded, val_mask, [best], signs=signs)
        best_val_t = abs(vrows[0]["t_stat_nw"])

    mx["max_abs_t_val_selected"] = float(best_val_t) if np.isfinite(best_val_t) else np.nan
    mx["replication_id"] = replication_id
    mx["seed"] = seeds[prices.columns[0]]
    mx["seeds"] = seeds
    return mx
