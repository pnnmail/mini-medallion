"""DuckDB storage. Every result of every milestone must be reachable by SQL."""
from __future__ import annotations

import datetime as _dt
import json
import os
import uuid

import duckdb

from config import DB_PATH, RESULTS_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment_log (
    experiment_id   VARCHAR,
    timestamp       TIMESTAMP,
    git_commit      VARCHAR,
    spec_version    VARCHAR,
    milestone       VARCHAR,
    description     VARCHAR,
    parameters      VARCHAR,
    n_tests         BIGINT,
    seed            BIGINT,
    status          VARCHAR
);

CREATE TABLE IF NOT EXISTS test_access_log (
    access_id       VARCHAR,
    timestamp       TIMESTAMP,
    split_name      VARCHAR,
    accessed_by     VARCHAR,
    experiment_id   VARCHAR,
    n_rows_read     BIGINT,
    note            VARCHAR
);

CREATE TABLE IF NOT EXISTS prices (
    ticker          VARCHAR,
    date            DATE,
    open            DOUBLE,
    high            DOUBLE,
    low             DOUBLE,
    close           DOUBLE,
    adj_close       DOUBLE,
    volume          DOUBLE
);

CREATE TABLE IF NOT EXISTS macro (
    series_id       VARCHAR,
    date            DATE,
    value           DOUBLE
);

CREATE TABLE IF NOT EXISTS null_distribution (
    replication_id  BIGINT,
    seed            BIGINT,
    n_tests         BIGINT,
    max_abs_t_stat_nw   DOUBLE,
    max_abs_ic          DOUBLE,
    max_sharpe          DOUBLE,
    max_net_sharpe      DOUBLE,
    max_abs_t_val_selected DOUBLE,
    elapsed_seconds     DOUBLE
);

CREATE TABLE IF NOT EXISTS calibration_results (
    test_name       VARCHAR,
    ticker          VARCHAR,
    detail          VARCHAR,
    metric          VARCHAR,
    value           DOUBLE,
    expected_sign   VARCHAR,
    actual_sign     VARCHAR,
    passed          BOOLEAN
);

CREATE TABLE IF NOT EXISTS cost_estimates (
    ticker          VARCHAR,
    month           DATE,
    cs_spread_bps   DOUBLE,
    n_days          BIGINT
);

CREATE TABLE IF NOT EXISTS signal_results (
    signal_id       VARCHAR,
    description     VARCHAR,
    pair            VARCHAR,
    feature         VARCHAR,
    feature_window  BIGINT,
    target          VARCHAR,
    horizon         BIGINT,
    n_obs           BIGINT,
    train_ic        DOUBLE,
    val_ic          DOUBLE,
    test_ic         DOUBLE,
    gross_sharpe    DOUBLE,
    net_sharpe_2bps DOUBLE,
    net_sharpe_5bps DOUBLE,
    net_sharpe_10bps DOUBLE,
    net_sharpe_20bps DOUBLE,
    turnover_annual DOUBLE,
    half_life       DOUBLE,
    half_life_note  VARCHAR,
    t_stat_nw       DOUBLE,
    p_bootstrap     DOUBLE,
    p_bh_adjusted   DOUBLE,
    deflated_sharpe DOUBLE,
    null_t_threshold_95 DOUBLE,
    beats_null      BOOLEAN,
    max_drawdown    DOUBLE,
    skew            DOUBLE,
    kurtosis        DOUBLE,
    status          VARCHAR
);

CREATE TABLE IF NOT EXISTS lookahead_tests (
    series_name VARCHAR, shift_n BIGINT, feature VARCHAR,
    observed_shift BIGINT, passed BOOLEAN, note VARCHAR
);

CREATE TABLE IF NOT EXISTS xcorr_diagnostics (
    source VARCHAR, dest VARCHAR, rvol_window BIGINT, lag_days BIGINT,
    corr DOUBLE, split_name VARCHAR
);

CREATE TABLE IF NOT EXISTS rv_estimator_comparison (
    ticker VARCHAR, estimator VARCHAR, horizon BIGINT, split_name VARCHAR,
    qlike DOUBLE, mse DOUBLE, r2_oos_log DOUBLE, n_obs BIGINT
);

CREATE TABLE IF NOT EXISTS har_baseline (
    ticker VARCHAR, estimator VARCHAR, horizon BIGINT, split_name VARCHAR,
    model VARCHAR, qlike DOUBLE, mse DOUBLE, r2_oos_log DOUBLE,
    beats_naive_qlike BOOLEAN, n_obs BIGINT
);

CREATE TABLE IF NOT EXISTS incremental_results (
    model_id VARCHAR, target_instrument VARCHAR, horizon BIGINT, family VARCHAR,
    candidate VARCHAR, split_name VARCHAR,
    beta_incremental DOUBLE, t_incremental DOUBLE, p_analytic DOUBLE,
    t_null_threshold_rw DOUBLE, p_romano_wolf DOUBLE, p_bh_adjusted DOUBLE,
    rw_rejected BOOLEAN, bh_rejected BOOLEAN,
    baseline_r2_oos DOUBLE, model_r2_oos DOUBLE, delta_r2 DOUBLE,
    qlike_baseline DOUBLE, qlike_model DOUBLE, n_obs BIGINT, status VARCHAR
);

CREATE TABLE IF NOT EXISTS crisis_episodes (
    instrument VARCHAR, episode_id BIGINT, peak_date DATE, trough_date DATE,
    recovery_date DATE, depth DOUBLE, length_days BIGINT
);

CREATE TABLE IF NOT EXISTS vol_results (
    model_id VARCHAR, target_instrument VARCHAR, horizon BIGINT, estimator_rv VARCHAR,
    baseline_r2_oos DOUBLE, model_r2_oos DOUBLE, delta_r2 DOUBLE,
    qlike_baseline DOUBLE, qlike_model DOUBLE,
    beta_incremental DOUBLE, t_incremental DOUBLE, t_null_threshold_rw DOUBLE,
    beats_baseline BOOLEAN,
    strategy_id VARCHAR, cap DOUBLE, band DOUBLE, turnover_annual DOUBLE,
    n_rebalances_yr DOUBLE,
    sharpe_bh DOUBLE, sharpe_vt DOUBLE, sharpe_vt_matched_vol DOUBLE,
    sharpe_vt_matched_exposure DOUBLE,
    maxdd_bh DOUBLE, maxdd_vt DOUBLE,
    sharpe_ex_2008 DOUBLE, sharpe_ex_2020 DOUBLE, crisis_contribution_top5 DOUBLE,
    net_sharpe_2bps DOUBLE, net_sharpe_5bps DOUBLE, net_sharpe_10bps DOUBLE,
    net_sharpe_20bps DOUBLE,
    short_term_gain_turnover DOUBLE, status VARCHAR
);

CREATE TABLE IF NOT EXISTS m2_results (
    model_id VARCHAR, instrument VARCHAR, candidate VARCHAR, family VARCHAR,
    split_name VARCHAR, n_obs BIGINT,
    mean_w DOUBLE, cov_w_r DOUBLE, exposure_term DOUBLE, timing_term DOUBLE,
    delta_mean_total DOUBLE, residual_check DOUBLE,
    qlike DOUBLE, sharpe_is DOUBLE, timing_cov DOUBLE, sharpe_oos DOUBLE,
    corr_delta_fwd_ret DOUBLE, timing_null_p95 DOUBLE, timing_beats_null BOOLEAN,
    selector_qlike_rank BIGINT, selector_sharpe_rank BIGINT,
    selector_timing_rank BIGINT, status VARCHAR
);

CREATE TABLE IF NOT EXISTS m2_selector_race (
    selector VARCHAR, instrument VARCHAR, split_name VARCHAR,
    picked_model VARCHAR, picked_sharpe_oos DOUBLE,
    random_choice_sharpe DOUBLE, oracle_sharpe DOUBLE, baseline_sharpe DOUBLE,
    beats_random BOOLEAN, beats_baseline BOOLEAN, oracle_gap DOUBLE
);

CREATE TABLE IF NOT EXISTS decision_loss_results (
    test_id VARCHAR, instrument VARCHAR, forecast_version VARCHAR,
    rho_strategies DOUBLE, sharpe_diff DOUBLE, se_diff_ledoit_wolf DOUBLE,
    p_lw DOUBLE, p_bootstrap_paired DOUBLE, gap_status VARCHAR,
    delta_E_w_times_E_r DOUBLE, delta_cov_w_r DOUBLE, delta_numerator DOUBLE,
    delta_denominator DOUBLE,
    b_predictive DOUBLE, b_stambaugh_adjusted DOUBLE, t_hac DOUBLE,
    bias_by_stress_decile VARCHAR, qlike_by_stress_decile VARCHAR,
    vrp_by_stress_decile VARCHAR, stress_localization_ratio DOUBLE,
    k_smoothing BIGINT, sharpe_by_k VARCHAR,
    reversal_overlay_sharpe DOUBLE, reversal_match_residual DOUBLE,
    crisis_contribution_top5 DOUBLE, sharpe_diff_ex_2008 DOUBLE,
    sharpe_diff_ex_2020 DOUBLE,
    lambda_blend DOUBLE, cap DOUBLE, n_trials_total BIGINT,
    t_null_threshold_rw DOUBLE, beats_null BOOLEAN,
    net_sharpe_2bps DOUBLE, net_sharpe_5bps DOUBLE, net_sharpe_10bps DOUBLE,
    net_sharpe_20bps DOUBLE, status VARCHAR
);
"""


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(DB_PATH, read_only=read_only)
    if not read_only:
        con.execute(_SCHEMA)
    return con


def _now():
    return _dt.datetime.now().replace(microsecond=0)


def log_experiment(con, milestone, description, parameters, n_tests, seed, status):
    """Every experiment is logged -- including failures and post-hoc iterations."""
    eid = f"{milestone}-{uuid.uuid4().hex[:8]}"
    con.execute(
        "INSERT INTO experiment_log (experiment_id, timestamp, git_commit, "
        "spec_version, milestone, description, parameters, n_tests, seed, status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        [eid, _now(), "no-git-repo", __import__("config").SPEC_VERSION, milestone,
         description, json.dumps(parameters, default=str), int(n_tests),
         int(seed) if seed is not None else None, status],
    )
    return eid


def log_test_access(con, split_name, accessed_by, experiment_id, n_rows, note):
    con.execute(
        "INSERT INTO test_access_log VALUES (?,?,?,?,?,?,?)",
        [uuid.uuid4().hex[:12], _now(), split_name, accessed_by,
         experiment_id, int(n_rows), note],
    )


def dump_table_csv(con, table, filename=None):
    filename = filename or os.path.join(RESULTS_DIR, f"{table}.csv")
    df = con.execute(f"SELECT * FROM {table}").df()
    df.to_csv(filename, index=False)
    return filename
