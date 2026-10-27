# MINI-MEDALLION — MILESTONE 2
## Why a more accurate forecast leads to a worse decision

**Status:** RESEARCH ONLY. No broker connection, no orders.
The specification was frozen **before** any P1–P4 run and is not changed afterwards.

---

## Why this question

M1 left a contradiction that nothing tested so far explains.
`HAR + log(VIX)` beats `HAR` on QLIKE on all four instruments, yet the Sharpe ratio
of the corresponding sizing strategy is worse — also on all four.

Measured before this spec was written (TRAIN+VALIDATION, cap 1.0):

| | forecast bias | turnover | mean exposure | QLIKE | Sharpe |
|---|---|---|---|---|---|
| SPY  | +2.9% / +1.1% | 2.27 / 2.08 | 0.653 / 0.666 | −8.2026 / −8.1682 | **0.534 / 0.575** |
| SCHD | −4.3% / −5.1% | 3.58 / 3.93 | 0.775 / 0.761 | −8.4689 / −8.4297 | **0.615 / 0.635** |
| TLT  | −1.1% / +0.1% | 0.82 / 0.96 | 0.764 / 0.767 | −8.5795 / −8.5741 | **0.535 / 0.549** |
| IWM  | +2.3% / +4.8% | 2.03 / 2.25 | 0.524 / 0.516 | −7.7391 / −7.7087 | **0.400 / 0.408** |

(first number — HAR+log(VIX), second — HAR)

For SCHD, TLT and IWM the VIX version is **less biased and trades less**, and still
loses. Four obvious explanations were ruled out by measurement, not by argument.

**The central question of M2:** which property of a volatility forecast determines
whether it improves the position-sizing decision — and can it be measured in advance,
on TRAIN, without looking at the strategy's P&L?

---

## Stage order — do not change

```
P0 FREEZE          specification, model universe, gates
   ↓
P1 DECOMPOSITION   exact algebraic decomposition: exposure vs. timing
   ↓
P2 MECHANISM       where the timing term comes from; null calibration
   ↓
P3 SELECTOR RACE   three pre-registered selectors, TRAIN → VALIDATION
   ↓
P4 TEST + REPORT   the single access to TEST
```

---

## P0 — FREEZE

No new data is needed: the M0 (`data/raw_snapshot/`) and M1
(`data/raw_snapshot_m1/`) snapshots are sufficient and stay immutable. No new
downloads.

**Model universe.** For each instrument in `VOL_TARGETS`
(SPY, SCHD, TLT, IWM) at the 10-day sizing horizon:
the `HAR` baseline plus `HAR + X` for **every** M1 candidate (21–22 of them).
88 (instrument, model) pairs in total.

**All** candidates are used, not the 30 M1 survivors. Restricting to survivors would
import into M2 a selection made by a different criterion and turn the selector race
into a check of a choice already made.

---

## P1 — DECOMPOSITION (analytical core)

Strategy P&L: `p_t = w_t · r_{t+1}`, where `w_t = min(σ*/σ̂_t, cap)`.

```
mean(p) = mean(w)·mean(r) + cov(w, r)
```

This is an identity, not a fit. For two models A and B:

```
Δmean = [mean(w_A) − mean(w_B)]·mean(r)   +   [cov(w_A,r) − cov(w_B,r)]
        ──────── exposure term ─────────      ───────── timing term ─────────
```

**H1 (pre-registered):** the VIX version's loss sits in the timing term, not in
exposure.

Sharpe does not decompose exactly (the denominator is not additive), so a two-factor
Sharpe attribution is reported as well, with the **interaction residual shown
explicitly**. An attribution that hides its residual is fitting to the explanation.

---

## P2 — MECHANISM

If H1 holds, the cause has to be named and tested separately.

Decompose `σ̂^{HAR+X}_t = σ̂^{HAR}_t + δ_t` and measure:

1. `corr(δ_t, r_{t+1})` — the own timing of the increment the candidate adds;
2. `corr(Δσ̂_t, r_{t+1})` for each model;
3. the same, conditional on regimes: drawdown, VIX spike, calm market.

**H2 (pre-registered):** `corr(δ_t, r_{t+1}) < 0` for VIX — VIX reacts to
sell-offs, sell-offs are followed by higher subsequent returns, so the increment
cuts the position near a local bottom.

**Null calibration is mandatory.** The timing term is a sample covariance on
overlapping, autocorrelated series; its null distribution is built with a
stationary block bootstrap (mean block 20), the same machinery as in M0/M1.
No value is reported without comparison against the threshold.

---

## P3 — SELECTOR RACE (the milestone's product)

The point: can we tell **on TRAIN**, without looking at out-of-sample P&L, which
forecast will produce the better decision?

Three selectors, frozen here:

| selector | what it computes on TRAIN |
|---|---|
| `QLIKE` | forecast accuracy (the M1 criterion) |
| `SHARPE_IS` | in-sample Sharpe of the strategy implied by the forecast |
| `TIMING_COV` | `cov(w_t, r_{t+1})` — the timing term only |

Procedure: within each instrument, rank all models by each selector on TRAIN →
take the top 1 → measure Sharpe on VALIDATION. The selector whose choice gives the
best out-of-sample Sharpe wins.

**Reference lines without which the numbers cannot be read:**
- random model choice (mean over all models on VALIDATION);
- best possible choice on VALIDATION (oracle, unattainable);
- the plain `HAR` baseline with no additions.

**GATE P3:** if all three selectors give the same ordering, or none beats random
choice, the milestone's premise is wrong and the report says so plainly.

---

## P4 — TEST AND REPORT

TEST is opened **exactly once**, at the very end, for everything at once.
`test_access_log` is mandatory.

The split is unchanged: TRAIN 2006–2015 (SPY from 1993), VALIDATION 2016–2020,
TEST 2021–present.

Table `m2_results` in DuckDB + CSV:

```
model_id, instrument, candidate, split_name,
mean_w, cov_w_r, exposure_term, timing_term, delta_mean_total, residual_check,
qlike, sharpe_is, timing_cov, sharpe_oos,
corr_delta_fwd_ret, timing_null_p95, timing_beats_null,
selector_qlike_rank, selector_sharpe_rank, selector_timing_rank, status
```

`status ∈ {PASSED, FAILED_NULL, FAILED_OOS, FAILED_PREMISE, NEUTRAL}`

---

## HARD RULES

1. No real money, no broker API, no orders.
2. No ML and no hyperparameter tuning. σ*, cap, band and sizing horizon are
   inherited from M1 unchanged.
3. No additional data is downloaded. Anything removed in M1 does not come back.
4. Every experiment is logged, including iterations after seeing results.
5. The P1 decomposition is an identity; any residual is printed, not absorbed.
6. No timing value is reported without null calibration.
7. TEST is opened once.
8. All computations are deterministic Python.

---

## COMPLETION GATE

- [ ] model universe built from all candidates, not from M1 survivors
- [ ] P1 decomposition closes numerically (residual at machine precision)
- [ ] H1 tested: exposure and timing separated and reported
- [ ] H2 tested, `corr(δ, r)` null-calibrated
- [ ] three selectors implemented, ranking on TRAIN
- [ ] reference lines built: random choice, oracle, plain HAR
- [ ] TEST opened exactly once
- [ ] final table contains all 88 pairs, not only survivors
- [ ] verdict produced automatically

---

**FINAL INSTRUCTION**

M0 asked "is there anything here at all". M1 asked "does it change the decision".
M2 asks: **can a forecast that will improve the decision be told apart in advance
from one that will make it worse.**

The most likely outcome: accuracy and decision usefulness diverge, but none of the
selectors reliably beats random choice out of sample — i.e. they cannot be told
apart in advance. That outcome must be reported plainly.

`CANNOT BE SELECTED EX ANTE` is a successful completion of Milestone 2.
Do not move on to the next milestone without explicit permission.
