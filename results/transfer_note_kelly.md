# Transfer note — does the M2 identity apply to the fractional-Kelly stage?

**Revised 2026-08-30.** The first version of this note concluded that
`Cov(w_t, r_{t+1}) ≡ 0` there and the M2 failure mode "cannot" occur. That was
wrong: the identity requires literally constant `w`, and `w` is not constant.
Corrected below, with both channels verified in the code.

**Where it lives:** the separate `portfolio-risk` project —
`analysis/factor_risk.py::kelly_sizing`, `fraction = 0.25`, wired through
`cli.py --kelly-fraction`. Its own docstring identifies the input as
"EXPECTED return from the theses, NOT the historical mean", which is what makes
it the thesis-bot.

## What is and is not there

```
E[w_t · r_{t+1}] = E[w_t]·E[r_{t+1}] + Cov(w_t, r_{t+1})
```

`kelly_sizing` itself is a one-shot computation: `f = 0.25 · Σ⁻¹μ`, evaluated
once from hand-supplied thesis expectations. There is no rebalance loop.

**The scheduled timing channel is therefore absent. Two unmeasured ones are not.**

### 1. Realised weights drift — confirmed in code

`PortfolioState.weights()` returns `market_values() / market_values().sum()`, and
`market_values()` is computed from live prices loaded on every run
(`portfolio_state.py:59-64`). Positions are share counts in a YAML file. So
between manual edits the realised weights **drift with returns**, and drifting
weights are not constant.

They also covary with returns mechanically and with a known sign: winners grow in
share. That is a momentum tilt obtained by inaction rather than by decision, and
it sits in exactly the covariance term M2 measured. Its magnitude is unmeasured
and measurable — the weight path is reconstructible from the position file plus
price history.

### 2. Manual μ revision is an update, and a worse one than a scheduled update

Irregular is not absent. The trigger for a hand revision is that the analyst
noticed something, and one usually notices after a move. Revising μ downward
after a drawdown is precisely the M2 failure mode — the forecast is improved by
information that arrives late, the weight is cut, and the cut lands before the
rebound. Discretionary and unlogged is the worst version of it, not the mildest:
a scheduled rule at least has a fixed relationship to price history.

**This channel is currently unauditable, and permanently so.** `theses/schema.py`
contains no `updated_at`, no revision history and no change log — grep finds
nothing. Revision events leave no trace, so the covariance term attributable to
them cannot be estimated at all, now or later.

## The one action with immediate payoff

**Log μ-revision events**: timestamp, which ticker, old and new μ, and what
triggered the revision. Three fields in `theses/schema.py` and an append-only
file.

Without that log the covariance term there is unmeasurable *permanently*, not
"until a refresh loop appears". With it, the M2 machinery applies directly:
`src/decomposition.py` (`decompose_pair`, `sharpe_attribution`,
`circular_shift_null`) needs only a weight path and a return path.

## The power lesson transfers regardless, and it is the more useful half

- M2 measured **82–331 distinct weight levels** from automated band rebalancing
  over 6–25 years — not the ~5000 daily observations the sample appears to hold.
  That is the effective sample for anything driven by weight variation, and it
  was not enough to resolve a 0.008–0.042 Sharpe gap.
- A hand-updated thesis portfolio produces revision events in the **tens** over
  its whole life.

So a μ-refresh policy there should be justified structurally — by what the thesis
says and what the correlation haircut implies — and not by backtesting which
refresh rule produced better realised returns. There will never be enough
distinct weight levels to tell, and the drift channel will contaminate the
comparison with a momentum tilt nobody chose.
