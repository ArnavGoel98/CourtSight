# Locked forecast: pending cases on 31 December 2026

**Status: open.** The outcome does not exist yet. This file, `forecast_states_2026.csv` and `forecast_meta.json` were
committed on 8 October 2026 using official data through 31.12.2025. The git history is the timestamp: anything
changed after this commit does not count.

Made in October 2026, so part of 2026 has already happened. The forecast uses only official year-end figures through
31.12.2025; NJDG's live dashboard was not consulted. Treat it as a one-year-ahead forecast from the latest official
data, not as a forecast made on 1 January.

## The forecast
India's District & Subordinate Courts will hold **4.93 crore** pending cases on 31.12.2026
(range 4.78–5.07 crore), up from 4.77 crore on 31.12.2025.
State forecasts with q10/q50/q90 are in `forecast_states_2026.csv` (34 units).

## How the method was chosen (rule written into `forecast_2026.py` before the backtest ran)
Five simple methods were backtested one year ahead on every year outside the COVID disruption (targets 2017–2019
and 2023–2025). The rule picks the lowest caseload-weighted mean absolute error of state log growth.

| Method | Mean error per state | Weighted by caseload | National error |
|---|---|---|---|
| blend | 0.084 | 0.051 | 0.021 |
| own_last | 0.108 | 0.053 | 0.018 |
| national | 0.083 | 0.054 | 0.019 |
| own_mean3 | 0.095 | 0.054 | 0.027 |
| zero | 0.091 | 0.068 | 0.040 |

Chosen: **blend** (half the state's own 3-year trend, half the latest national trend).
State ranges use the q10/q90 of this method's backtest errors; the national range spans its backtest national errors.

## How it will be scored (fixed now)
- **Data:** the first Ministry of Law & Justice answer in Parliament that reports state-wise pending cases in
  District & Subordinate Courts on 31.12.2026 (source: NJDG), with combined units summed as in `official_series.py`.
- **Primary score:** caseload-weighted mean absolute error of state log growth, 2025→2026.
- **It passes if:** that error is below the no-change forecast's, and the national total falls inside
  4.78–5.07 crore.
- Also reported, whatever they show: state-level q10–q90 coverage (target 80%), national error, and a comparison
  with an oracle that knows the true national growth rate.
- Run: `python3 grade_2026.py --actual actual_2026.csv`.
