# Model trained on real DDL data (2010–2018)

**Data:** 80.9M district-court case records (DDL judicial data, CC BY-NC-SA 4.0; see `../../DATA_LICENSE.md`),
632 districts, filings 2010-01 to 2018-12. No NJDG data, so this is the DDL-only mode: 12-month horizon, forecast
origin Dec 2018 → Dec 2019.

**Model:** LightGBM quantile models (q10/q50/q90) for 12-month backlog growth and clearance ratio (disposals ÷
filings), predicting the change over a trend-continuation anchor. Validated on two unseen years (2016 and 2017
origins) against two benchmarks: trend continuation, and a 7-variable linear quantile regression. The production
model is trained on 2013–2015 origins and calibrated on 2017, the most recent year whose outcomes are known.
Saved in `../../models/ddl_2010_2018/`.

## Graded against what actually happened in 2019
`forecasts.csv` (LightGBM, linear and trend forecasts) was committed on 5 Oct 2026, before any 2019 outcome was
looked at. It was then scored against official state-wise figures: Lok Sabha Unstarred Question 1838 (16 Dec 2022),
Annexures III and VIII, from the Supreme Court of India / NJDG (`../../grade_state_2019.py`,
`grading_2019_state/`). The 27 graded states hold 99.8% of India's district-court cases.

| 2019, state level | LightGBM model | Linear model | Trend continuation | Model's error reduction vs trend (95% range) |
|---|---|---|---|---|
| Backlog growth, mean abs. error | 0.077 | 0.077 | 0.109 | **30%** (11% to 53%) |
| Clearance ratio, mean abs. error | 0.088 | 0.099 | 0.116 | **24%** (13% to 36%) |

- National backlog growth in 2019: actual +7.4%; model +4.6%; linear +3.6%; trend continuation +1.7%.
- Without Tripura and Uttarakhand, whose official backlogs fall 75% and 17% in one year (likely data clean-ups),
  the reduction is 45% for growth and 32% for clearance.
- The model ranks states only moderately well (Spearman 0.44 for growth, 0.50 for clearance; trend continuation is
  similar). It wins mainly by getting the level right, which trend continuation underestimates.
- Caveat: official counts include cases filed before 2010, which the model never sees. All three forecasters share
  this gap. District-level grading needs NJDG district history (`../../grade_2019.py`).

## Validation (2016 + 2017 pooled, 15k district-months)
Improvement = lower pinball loss than the benchmark. The 95% ranges come from resampling whole High Courts
(29 clusters), because districts in one High Court share shocks.

| Target | Quantile | vs trend baseline (95% range) | vs linear model (95% range) | Coverage (target) |
|---|---|---|---|---|
| Backlog growth | q10 | 39% (32% to 46%) | +12% (+0% to +23%) | 0.11 (0.10) |
| Backlog growth | q50 | 36% (29% to 43%) | +2% (-1% to +6%) | 0.60 (0.50) |
| Backlog growth | q90 | 20% (5% to 31%) | -6% (-12% to -2%) | 0.94 (0.90) |
| Clearance ratio | q10 | 19% (2% to 31%) | +3% (-7% to +13%) | 0.06 (0.10) |
| Clearance ratio | q50 | 20% (13% to 26%) | +4% (+1% to +7%) | 0.40 (0.50) |
| Clearance ratio | q90 | 25% (17% to 32%) | +2% (-3% to +6%) | 0.88 (0.90) |

- The improvement over trend continuation holds in both validation years separately (`validation_metrics.csv`).
- **LightGBM is not what earns it.** The linear quantile regression does about as well: LightGBM is better for
  best-case growth and median clearance, worse for worst-case growth, and level elsewhere. The gain comes from the
  features and the anchor-plus-change setup.
- The median clearance forecast runs low (coverage 0.40 vs 0.50), and the median growth forecast runs high
  (0.60). The q10–q90 band covers 83% of outcomes (target 80%).

## Forecast for 2019 (631 districts; one onboarded to eCourts in 2018 is excluded; 2 inactive ones have no clearance forecast)
- **65% of districts:** median clearance ratio below 1 (more filings than disposals). The figure depends on the
  forecaster: 51% for trend continuation and 74% for the linear model. It counts only disposals of cases filed
  since 2010 (older cases are not in the data), so all three are biased towards "falling behind".
- 409 districts flagged (median clearance < 1 and worst-case growth of the post-2010 backlog > 5%).
- At data collection (Feb 2019 for most districts), a pending case's next hearing was scheduled a median of
  ~50 days after its last one (`hearing_at_scrape.csv`). This describes one moment, not a trend, and is not used
  by the model.
- **What the worst-case model relies on** (TreeSHAP, flagged districts; associations, not causes): recent
  backlog trend and inflow/disposal balance 61%, peer districts in the same High Court 12%, the trend anchor 9%,
  case mix and age 8%, disposal momentum 4%, filing spikes 3%, long-run filing trend 3%.

## Judge effect: not identified, so no "judges needed" numbers
- Fixed-effects OLS gives 0.15 (a 10% increase in judges goes with 1.5% more disposals). OLS is biased: judges
  get posted where backlogs are high, and DDL's judge records thin out further back in time. DDL records 2.8k
  judges in Dec 2010 and 16.3k in Dec 2018, while actual strength was roughly flat.
- An instrument based on each district's mix of judge grades × state-wide recruitment by grade is too weak
  (first-stage F = 6.2; F ≥ 10 is required). Its estimate is -0.00 ± 0.52, which tells us nothing.
- So `policy_levers.csv` is not produced. NJDG aggregates alone will not fix this. It needs a cleaner source of
  variation in judge numbers (for example, dated recruitment and joining records from High Courts).

## What changed after the October 2026 review
- Hearing-stage and hearing-gap features were removed. DDL keeps only each pending case's latest hearing, so
  values for past months depend on later events (4.8k observations in 2013 vs 6.4M in 2018).
- Age and workload features are now measured among cases under 3 years old, which are fully observed at every
  origin. The ">5 years old" share and the age-skew feature were dropped (always zero before 2015 by construction).
- Judge-count features were dropped in this mode (see above), and the eCourts onboarding check uses only data up
  to each origin.
- Validation now covers two years, with error bars and a linear benchmark. The production model is calibrated on
  2017 instead of 2015.
- Effect on accuracy, same 2017 validation year: essentially none (median growth 31% vs 32% before;
  median clearance 15% vs 15%).

## Limitations (read before quoting numbers)
- **Pre-2010 cases are missing**, so "pending" counts only cases filed since 2010 (23.8M at Dec 2018). Backlog
  growth is overstated and the clearance ratio understated. Training starts in 2013 to limit this.
- Case-type mapping leaves 9% of cases (ambiguous local codes such as `chi`, `mjc`) unclassified and treated as civil.
- Forecasts are for 2019 (the data ends in 2018). They are graded at state level only; district-level grading
  needs NJDG district history.
