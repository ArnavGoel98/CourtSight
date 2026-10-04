# Model trained on real DDL data (2010–2018)

**Data:** 80.9M district-court case records (DDL judicial data, CC BY-NC-SA 4.0), 632 districts, filings 2010-01 to 2018-12.
No NJDG data, so this is the DDL-only mode: 12-month horizon, forecast origin Dec 2018 → Dec 2019.

**Model:** LightGBM quantile models (q10/q50/q90) for 12-month backlog growth and clearance ratio
(disposals ÷ filings). Trained on 2013 origins, calibrated on 2015, validated on all of 2017 (7.5k district-months, out of sample).
Saved in `../../models/ddl_2010_2018/` (`manifest.json` + LightGBM text files; reload with `load_model_bundle`).

## Validation (2017, unseen)
| Target | Quantile | Pinball loss | Drift baseline | Improvement | Coverage (target) |
|---|---|---|---|---|---|
| Backlog growth | q10 | 0.022 | 0.033 | 34% | 0.13 (0.10) |
| Backlog growth | q50 | 0.043 | 0.063 | 31% | 0.61 (0.50) |
| Backlog growth | q90 | 0.021 | 0.028 | 25% | 0.93 (0.90) |
| Clearance ratio | q10 | 0.030 | 0.036 | 16% | 0.05 (0.10) |
| Clearance ratio | q50 | 0.066 | 0.077 | 15% | 0.35 (0.50) |
| Clearance ratio | q90 | 0.035 | 0.045 | 22% | 0.82 (0.90) |

## Forecast for 2019 (631 districts; 1 excluded as onboarded to eCourts in 2018)
- 71% of districts: median forecast clearance ratio below 1 (more cases filed than disposed).
- 70% of districts: worst-case (q90) backlog growth above 10% in 12 months.
- 446 districts flagged (q50 clearance < 1 and q90 growth > 5%).
- Median gap between hearings for pending cases: ~86 days.
- Main drivers of the worst-case forecasts (TreeSHAP): recent backlog momentum and inflow/disposal balance (51%),
  judicial capacity incl. workload per judge (16%), same-High-Court peer districts (12%), procedural stage/case age (11%);
  short-term filing spikes explain ~1%.

## Limitations (read before quoting numbers)
- **Pre-2010 cases are missing** from DDL, so "pending" counts only cases filed since 2010 (23.8M at Dec 2018);
  early-year backlog growth is inflated by this. Training starts in 2013 to limit it. NJDG data would fix it.
- **Policy levers are not reliable in this mode.** The estimated judge elasticity (0.15) is distorted by the same
  truncation, so "judges required" numbers (`policy_levers.csv`) should not be quoted.
- Clearance-ratio q50 is biased low (35% coverage vs 50%): 2017 cleared faster than calibration years.
- Case-type mapping left 45% of labels unclassified (treated as civil); affects only workload weights.
- Forecasts are for 2019 (the data ends in 2018), not the present.
