"""
MINI-MEDALLION / MILESTONE 0 -- FROZEN CONFIGURATION.

Everything in this file is fixed BEFORE any result on real data is looked at.
The search space defined here is executed:
  * first on synthetic (block-bootstrapped) data in M2,
  * only afterwards on real data in M5.

Any change to this file after M2 is a NEW EXPERIMENT and must be recorded in
`experiment_log` with a bumped SPEC_VERSION.
"""
from __future__ import annotations

import os

SPEC_VERSION = "M0.1.0"
MASTER_SEED = 20260829
SNAPSHOT_DATE = "2026-08-29"

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_SNAPSHOT_DIR = os.path.join(DATA_DIR, "raw_snapshot")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
LOG_DIR = os.path.join(ROOT, "logs")
DB_PATH = os.path.join(ROOT, "mini_medallion.duckdb")

# ---------------------------------------------------------------- M0 universe
EQUITY_TICKERS = [
    "GLD", "GDX", "XLE", "XOP", "XLF",
    "SPY", "QQQ", "IWM", "TLT", "SHY", "SCHD",
]
FUTURES_TICKERS = ["CL=F"]          # continuous front-month; roll gaps -> diagnostics only
ALL_TICKERS = EQUITY_TICKERS + FUTURES_TICKERS

MACRO_SERIES = ["T10Y2Y"]           # FRED, no API key required

DOWNLOAD_START = "2000-01-01"
DOWNLOAD_END = "2026-08-29"

# Known corporate action that must be visible in the integrity report.
KNOWN_SPLITS = {"SCHD": [("2024-10-10", 3.0)]}

# ---------------------------------------------------------------- M5 splits
# Chronological only. Random shuffle is forbidden.
SPLITS = {
    "TRAIN": ("2006-01-01", "2015-12-31"),
    "VALIDATION": ("2016-01-01", "2020-12-31"),
    "TEST": ("2021-01-01", "2099-12-31"),
}

# ---------------------------------------------------------------- M1 search space
Z_WINDOWS = [60, 120, 252]           # fixed list; never tuned
RVOL_WINDOWS = [5, 10, 20]
XCORR_LAGS = [0, 1, 2, 3, 4, 5]
AL_REG_WINDOW = 252                  # Avellaneda-Lee rolling regression window
STD_WINDOW = 252                     # causal standardisation window for non-z features

RETURN_HORIZONS = [1, 3, 5, 10]
VOL_HORIZONS = [5, 10]

# Pairs traded as a spread: (long_leg, short_leg)
PAIRS = [
    ("GDX", "GLD"),
    ("XOP", "XLE"),
]

# Regime pair -- explicitly NOT mean reversion, volatility targets only.
REGIME_PAIRS = [("IWM", "SPY")]

# Volatility spillover: (source_ticker, destination_ticker)
VOL_SPILLOVER = [("TLT", "SPY")]

# Avellaneda-Lee residual spread: (asset, macro_series)
RESIDUAL_SPECS = [("XLF", "T10Y2Y")]

# Position construction
POSITION_CLIP = 2.0                  # z-score clip before scaling to [-1, 1]

# Declared IN ADVANCE (M1), never re-tuned after seeing results.
# Applied as a REPORTED diagnostic / status downgrade only -- never as a filter
# that would differ between the null harness and the real run.
MAX_ADMISSIBLE_HALF_LIFE = 60        # trading days
HORIZON_HALF_LIFE_MULTIPLE = 2.0     # H <= 2 * half_life

# ---------------------------------------------------------------- M2 null harness
BOOTSTRAP_MEAN_BLOCK = 20            # stationary block bootstrap, Politis-Romano
NULL_REPLICATIONS_MIN = 500
NULL_REPLICATIONS_TARGET = 1000
NULL_TIME_BUDGET_SECONDS = 30 * 60
NULL_PERCENTILES = [90, 95, 99]
PRIMARY_NULL_PERCENTILE = 95         # 99 reported only when N >= 1000

# ---------------------------------------------------------------- M4 costs
COST_SWEEP_BPS = [2, 5, 10, 20]      # round-trip
BASE_COST_BPS = 5
SCHD_SPREAD_ANCHOR_BPS = 3.0         # sanity anchor for the Corwin-Schultz estimator
TRADING_DAYS = 252

# ---------------------------------------------------------------- M5 statistics
NEWEY_WEST_MIN_LAG = 0
BH_FDR_ALPHA = 0.10
BOOTSTRAP_CI_REPLICATIONS = 2000
