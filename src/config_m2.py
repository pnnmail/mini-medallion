"""MILESTONE 2 -- FROZEN CONFIGURATION.

Separate file again: config.py sits inside the M0 estimator fingerprint and
config_m1.py carries M1's declarations. Nothing here changes either.

Every sizing constant is INHERITED from M1 unchanged. M2 is not allowed to
re-tune the decision layer -- it is asking a question about that layer, and a
layer that moves while being questioned answers nothing.
"""
from __future__ import annotations

from config import BOOTSTRAP_MEAN_BLOCK, COST_SWEEP_BPS, FIG_DIR, RESULTS_DIR, ROOT
from config_m1 import (M1_SPLITS, MASTER_SEED_M1, MIN_SIGMA, REBALANCE_BAND,
                       SIZING_HORIZON, TRADING_DAYS, VOL_TARGET_ANNUAL,
                       VOL_TARGET_CAPS, VOL_TARGETS)

SPEC_VERSION_M2 = "M2.0.0"
MASTER_SEED_M2 = MASTER_SEED_M1          # 20260829, one ledger for the whole project

# ---------------------------------------------------------------- inherited, frozen
CAP = 1.0                 # the de-risking overlay; cap 1.5 is reported, never ranked
SIGMA_TARGET = VOL_TARGET_ANNUAL
BAND = REBALANCE_BAND
HORIZON = SIZING_HORIZON
INSTRUMENTS = list(VOL_TARGETS)
SPLITS_M2 = dict(M1_SPLITS)

# ---------------------------------------------------------------- P0 universe
# ALL M1 candidates, not the 30 that survived. Restricting to survivors would
# import a selection made under a different criterion and turn the selector race
# into a re-test of a choice already made.
USE_ALL_CANDIDATES = True
BASELINE_NAME = "HAR"

# ---------------------------------------------------------------- P3 selectors
SELECTORS = ["QLIKE", "SHARPE_IS", "TIMING_COV"]
REFERENCE_LINES = ["random_choice", "oracle_validation", "baseline_HAR"]

# ---------------------------------------------------------------- P2 null
NULL_REPLICATIONS_M2 = 1000
NULL_MEAN_BLOCK = BOOTSTRAP_MEAN_BLOCK   # 20 trading days
NULL_PERCENTILE_M2 = 95

# ---------------------------------------------------------------- tolerances
DECOMP_TOLERANCE = 1e-12   # the P1 identity must close to machine precision
