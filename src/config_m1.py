"""MILESTONE 1 -- FROZEN CONFIGURATION.

Separate file on purpose: `config.py` is inside the M0 estimator fingerprint, so
touching it would invalidate the M0 null thresholds that M5 still checks against.
M1 gets its own constants and its own fingerprint.

Everything here is fixed BEFORE any M1 result is looked at.
"""
from __future__ import annotations

import os

from config import (BOOTSTRAP_MEAN_BLOCK, COST_SWEEP_BPS, DATA_DIR, FIG_DIR,
                    LOG_DIR, MASTER_SEED, RESULTS_DIR, ROOT, SPLITS, TRADING_DAYS)

SPEC_VERSION_M1 = "M1.0.0"
MASTER_SEED_M1 = MASTER_SEED            # 20260829, same ledger

M1_SNAPSHOT_DIR = os.path.join(DATA_DIR, "raw_snapshot_m1")

# ---------------------------------------------------------------- removed for good
# Deleted in M1 by explicit decision, recorded in experiment_log.
# Reason is structural, not statistical: operating leverage guarantees correlation,
# not mean reversion. Re-testing with different windows is FORBIDDEN -- the
# parameters were never the problem.
REMOVED_FEATURES = {
    "pair_GDX_GLD": "half-life 1135d vs a 60d admissibility threshold; operating "
                    "leverage produces correlation, not mean reversion",
    "pair_XOP_XLE": "half-life 449d vs a 60d admissibility threshold; same reason",
    "corwin_schultz_as_level_estimator":
        "45-65% negative daily estimates on liquid ETFs; cross-ticker correlation "
        "+0.945 with volatility vs +0.118 with log ADV. Built for quote-less "
        "illiquid names; an ETF with billions of ADV has nothing for it to estimate",
}

# ---------------------------------------------------------------- universe
VOL_TARGETS = ["SPY", "SCHD", "TLT", "IWM"]        # instruments we may size
VOL_SOURCES = ["SPY", "TLT", "GLD", "XLE", "IWM"]  # cross-asset volatility sources
CALIB_INSTRUMENTS = ["SPY", "QQQ", "IWM", "TLT", "SHY", "GLD", "XLE", "XLF", "SCHD"]

VIX_TICKER = "^VIX"
# SPY trades from 1993; this extended TRAIN is declared here, before any result.
SPY_EXTENDED_TRAIN_START = "1993-01-29"
M1_DOWNLOAD_START = "1990-01-01"
M1_DOWNLOAD_END = "2026-08-29"

# ---------------------------------------------------------------- RV estimators
RV_ESTIMATORS = ["cc", "parkinson", "gk", "gk_overnight"]
RV_FLOOR = 1e-8                     # (1 bp daily move)^2, guards log(0)

# ---------------------------------------------------------------- N2 baseline
HAR_LAGS = (1, 5, 22)               # daily / weekly / monthly, Corsi (2009)
HAR_MIN_TRAIN_OBS = 750             # ~3 years before the first out-of-sample forecast
FORECAST_HORIZONS = [1, 5, 10]
EWMA_LAMBDA = 0.94
MA_WINDOW = 20

# ---------------------------------------------------------------- N3 candidates
VIX_Z_WINDOWS = [60, 120, 252]
RV_LOOKBACK = 20                    # window for candidate-side RV aggregations
SEMIVAR_WINDOW = 20

# ---------------------------------------------------------------- N3 inference
RW_REPLICATIONS = 1000              # Romano-Wolf / null-harness bootstrap draws
RW_ALPHA = 0.05
BH_ALPHA_DIAGNOSTIC = 0.10          # kept as a calibration diagnostic, NOT a criterion
NULL_PERCENTILE = 95

# ---------------------------------------------------------------- N4 decision layer
VOL_TARGET_ANNUAL = 0.10            # 10% annualised target
VOL_TARGET_CAPS = [1.0, 1.5]        # de-risking overlay / limited leverage
REBALANCE_BAND = 0.20               # +-20% of the target weight
SIZING_HORIZON = 10                 # HAR forecast horizon used for sizing
MIN_SIGMA = 0.02                    # floor on forecast vol, prevents absurd weights

# ---------------------------------------------------------------- N5 crisis analysis
CRISIS_TOP_N = 5
CRISIS_MIN_DEPTH = 0.10             # a drawdown episode must reach -10% to count
CRISIS_BOOTSTRAP_REPS = 2000
SHORT_TERM_DAYS = 365               # holding period below which a gain is short-term

# ---------------------------------------------------------------- calibration gate
CALIB_VOL_PERSISTENCE_MIN_T = 5.0   # |t| > 5 on EVERY instrument, individually

M1_SPLITS = dict(SPLITS)            # TRAIN 2006-2015, VAL 2016-2020, TEST 2021+
