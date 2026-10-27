# MINI-MEDALLION — what the programme established

Self-contained. Written for a reader who has seen none of the milestones.

**Status: CONCLUDED.** Research only; no broker was connected, no order was placed.

---

## The one-sentence result

The programme did not find a tradeable edge, and it built an apparatus that shows
reliably that there is not one to find **in this data** — then measured precisely
what that apparatus could and could not have seen.

That last part is the durable half. The failure to find an edge is a statement
about power, not about markets, and the programme is able to say which.

---

## What was actually asked, and answered

| | question | answer |
|---|---|---|
| M0 | is there anything here? | `NO ROBUST EDGE` — 98 hypotheses, 0 passed |
| M1 | does something already known change a decision? | `NO USEFUL IMPROVEMENT` — survives in 2 of 16, both on one instrument |
| M2 | why did accuracy and usefulness diverge? | `GAP_UNDERPOWERED` — the gap is too small to explain |
| M4 | what could the apparatus have seen? | `CONCLUDE` — see below |

---

## The decisive number

Running the identical search on **synthetic data containing no signal by
construction** gives the threshold a real result must clear. Measured over 1000
block-bootstrap replications:

| hypotheses searched | 95% null threshold on max\|t\| |
|---|---|
| 1 | 2.90 |
| 10 | 4.41 |
| 98 | 5.91 |
| 1268 (everything the programme ever ran) | 7.00 |

Converted into the smallest effect detectable with 84% power, on the return data
this programme had (n = 2399):

| | minimum detectable IC |
|---|---|
| K = 1 (a single pre-registered hypothesis) | **0.0796** |
| K = 1268 | 0.1632 |

Now compare with every real return effect the programme or the literature has:

| effect | IC |
|---|---|
| short-term reversal (measured here, correct sign in 36 of 36 cells) | 0.056 |
| overnight premium on SPY | 0.036 |
| typical published return predictor | 0.02 – 0.05 |
| 12−1 momentum | 0.024 |

**Every one of them sits below the detection floor at K = 1.** Not below the floor
for a wide search — below the floor for a *single pre-registered hypothesis*. On
this data, a return effect of realistic size is invisible no matter how narrow the
question. That is why the programme concluded rather than continued.

---

## Where the apparatus did have power, and what it found there

Predicting **volatility** is a different regime: baseline R² is about 0.49, against
about 0.005 for returns. Detection floor there is 0.051 (K = 1) to 0.105 (K = 1268).

Two effects cleared it, and both are textbook:

- **volatility persistence**, IC 0.160 — known, and untradeable on its own;
- **implied volatility adding information over historical**, IC 0.190, t = 14.13 —
  one of the most replicated results in the volatility literature.

The second was checked hard, because a t of 14 deserves suspicion. Four artefact
explanations were tested and rejected: measurement overlap (a *better*-measured
target makes the effect **stronger**, not weaker — the signature of a real
relationship measured badly), an alignment leak (a full day of genuine look-ahead
adds only 26%, so a 15-minute settlement offset cannot manufacture it), horizon
overlap (the effect peaks at 5 days and decays toward 30), and a weak baseline
(96% of the statistic survives strengthening it).

**Neither is a finding of this programme.** They are independent re-derivations of
known results, with the right magnitudes. That makes them a **calibration success**:
evidence the machinery detects real effects when they are present.

---

## The one structural result

```
E[w · r] = E[w] · E[r] + Cov(w, r)
```

For any rule that sizes a position from a forecast, the realised return splits
exactly into an **exposure** term (how much was held on average) and a **timing**
term (whether more was held at the right moments). The identity closes to 1.4e-20;
it is algebra, not a model.

Measured on this data: **77% of the difference between two sizing rules ran through
the timing term.**

The general form: **a forecast feeding a sizing rule is selected on accuracy but
acts through covariance.** Those are different quantities, and the programme
measured a case where they point in opposite directions — a volatility forecast
that was overwhelmingly more accurate (t = 14.13) produced a *worse* sizing
decision in **8 of 8** comparisons on all data available before the held-out set
was opened.

This result needs no multiple-testing correction, and that is not a technicality:
it was never selected from a pool of candidates, so it carries no selection bias by
construction. That is precisely why it is the one thing that survived.

---

## What the multiplicity arithmetic cost

The penalty for searching grows as √(2 ln K) — logarithmically. Going from 98
hypotheses to all 1268 raises the bar by 10–25% depending on the family. That was
computed and **paid**: thresholds throughout are reported at the cumulative K, not
per milestone.

The point worth carrying: **the cost of the search space exceeded the strength of
every known real effect in the returns regime.** Not "made it hard" — exceeded it,
at K = 1.

---

## Questions that can no longer be asked

Held-out data was opened four times across the programme; each access is recorded
with its full content in `test_exposure_ledger`. One of those accesses was **not
logged when it happened** and was disclosed retroactively.

| id | question now spent | on |
|---|---|---|
| BQ-1 | HAR vs HAR+log(VIX) compared by Sharpe | SPY, SCHD, TLT, IWM, 2021+ |
| BQ-2 | model selection among HAR-plus-one-candidate variants by out-of-sample Sharpe | same panel, 2021+ |
| BQ-3 | whether any M0 feature family predicts | M0 universe, 2021+ |
| BQ-4 | whether volatility targeting beats buy-and-hold | same panel, 2021+ |

Each remains askable on data that did not exist at the time of the access, or on a
different panel. Full table: `burned_questions`.

---

## What no amount of further work fixes

Four levers were assessed against effective sample size:

| lever | cost | effect on the detection floor |
|---|---|---|
| shrink the search space, 1268 → 1 | free | ×2.05 — still 0.0796, above every real effect |
| longer history (SPY from 1993) | free, already used | ×1.28 — 0.062, still above 0.05 |
| intraday data | free for 730 days | unclear for returns; **nothing at all** for sizing decisions |
| wide cross-section (500 names) | paid point-in-time data | ×15.8 — a genuine lever |

The last one works for the returns regime. It does not work for anything this
programme actually asked, because **no lever changes the number of crises**. The
sample contains about five. Every regime-dependent question — which is every
decision-level question here — stays at that n permanently.

Intraday data deserves its own line: for position sizing it gives nothing, because
the number of rebalancing episodes is set by the tolerance band, not by data
frequency. Measured: **82–331 distinct weight levels** over 6–25 years, not the
~5000 daily observations the sample appears to contain.

---

## What to do with this

**Do not** re-run a narrower version of the same search on the same panel. The
detection floor at K = 1 rules it out, and that is a computed number, not a
judgement.

**Do** treat the covariance identity as reusable: it applies to any forecast that
feeds a sizing rule, in any system. One transfer has been made already —
the separate `portfolio-risk` project now logs μ-revision events, because without them the
timing term there is unmeasurable permanently rather than temporarily
(`theses/MU_REVISION_LOG.md`).

**Do** reuse the null harness. Its central lesson generalises: a p-value from a
filter sweep means nothing until the sweep's own null distribution is known. The
same search that returned max |t| = 5.91 on pure noise would have produced
p ≈ 3×10⁻⁹ if read conventionally.

---

## Reproducing any of it

Everything is deterministic (`MASTER_SEED = 20260829`), stored in DuckDB
(`mini_medallion.duckdb`) and reachable by SQL. Re-running the null harness
reproduces its thresholds exactly, which has been verified.

```sql
SELECT status, COUNT(*) FROM signal_results GROUP BY status;   -- M0
SELECT * FROM vol_results;                                     -- M1
SELECT * FROM decision_loss_results;                           -- M2
SELECT * FROM experiment_log ORDER BY timestamp;               -- every trial
SELECT * FROM test_access_log;  SELECT * FROM test_exposure_ledger;
SELECT * FROM burned_questions;
```

Per-milestone detail is in `README.md`. Stage artefacts are in `results/`.
Sign invariants that gate the pre-registration process: `tests/test_sign_invariants.py`.
