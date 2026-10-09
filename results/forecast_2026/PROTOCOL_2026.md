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

## Addendum (8 October 2026, after locking; the forecast above is unchanged)
A review after locking found these weaknesses. They are recorded here instead of fixing the forecast, because
changing a locked forecast would defeat its purpose. They will be reported alongside the score.
- **State ranges are not equally reliable by size.** In the backtest, the q10–q90 range held 79% of outcomes
  overall (target 80%), but 69% for the smallest third of states and union territories and 90% for the largest
  third. Expect small units to fall outside their range more often than 1 in 5.
- **The national range is narrow.** It spans the minimum to maximum of only six backtest errors, so it is not a
  calibrated probability interval.
- **The "2022" figure is mid-December, not year-end** (see `official_series.py`). This slightly distorts the
  backtest targets 2022 and 2023.
- **The chosen method wins narrowly.** Its caseload-weighted error (0.051) is close to the next two methods'
  (0.053 and 0.054). Picking any of them would have been reasonable.

## Integrity
SHA-256 of the files as locked in commit 84732fc (8 October 2026; the same files were first committed as 391dca6,
whose hash changed when the repository history was rewritten to move it to this repository; the file contents and
these digests did not change):
- `forecast_states_2026.csv`: `adf62cd20085e545730b10e62a7ffa66dae5e3f90207d197c5207f2aa5c96d2d`
- `forecast_meta.json`: `1715bda1eed20622d7c43f89ea813ccf8b4d294f69a4380c40d39578174f4c13`

`tests/test_locked_forecast.py` fails if either file changes.
