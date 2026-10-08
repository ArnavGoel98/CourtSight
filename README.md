# CourtSight

**Forecasting the backlog of India's district courts, and checking the forecasts against what actually happened.**

India's district and subordinate courts held **4.77 crore (47.7 million) pending cases** on 31 December 2025. CourtSight
models where that backlog comes from and where it is heading, using 80.9 million individual court records and the
official figures the Ministry of Law & Justice reports to Parliament. Every claim below is reproducible from this repo,
and the weaker results are reported alongside the strong ones.

## Three results

### 1. COVID left India's district courts about 48 lakh cases behind
![COVID excess backlog](results/ddl_2010_2018/charts/covid_excess.png)

At the end of 2021, the courts held **48 lakh (4.8 million) more pending cases** than each state's pre-pandemic trend
implies (range 32–70 lakh). That is about **one in every eight pending cases**. The trend is corrected by its own
error in pre-COVID backtests, and the range spans that observed error rather than a model of it. By 2024 the gap can
no longer be told apart from the trend. Code: [`covid_excess.py`](covid_excess.py); results:
[`results/covid_excess/`](results/covid_excess/).

### 2. A forecast for 31 December 2026, locked before the outcome exists
**4.93 crore pending cases** nationally (range 4.78–5.07 crore), with a forecast for every state. Five methods were
backtested on every non-COVID year. The selection rule was written into the code before the backtest ran, and the
scoring rule is committed in [`PROTOCOL_2026.md`](results/forecast_2026/PROTOCOL_2026.md). When the Ministry publishes
the 31.12.2026 figures, [`grade_2026.py`](grade_2026.py) scores the forecast in one command. Until then the result is
open.

### 3. A district-level model on 80.9 million cases, graded honestly
- **Model:** quantile models (best case, median, worst case) of each district's 12-month backlog growth and clearance
  ratio (disposals ÷ filings). They are built from every case's filing and decision dates in Development Data Lab's
  eCourts data: 632 districts, 2010–2018.
- **Tested on years it never saw (2016 and 2017):** the median forecasts beat the strongest simple forecast by
  **12–25%**. The range comes from resampling whole High Courts.
- **Graded against official 2019 figures** (forecasts saved before the outcome was looked at): at state level the model
  **matches simple forecasts built from the official series but does not beat them**.
- **The model first looked 30% better.** That was against a trend benchmark that flattered it; it was caught and
  corrected, and both comparisons are published ([`RESULTS.md`](results/ddl_2010_2018/RESULTS.md)).

![2019 graded](results/ddl_2010_2018/charts/graded_2019.png)

## What the project does not claim
- **No "judges needed" numbers.** The data cannot separate the effect of adding judges from where judges are posted.
  A recruitment-based instrument was tried and is too weak (first-stage F = 6.2), so those numbers are withheld.
- **A simple model nearly matches the gradient-boosted one.** A 7-variable linear model ties it on backlog growth.
  The gain comes from how the problem is set up, not from the algorithm.
- **The DDL data starts in 2010.** Older cases are invisible to the district model, which biases its clearance ratios
  downward. This is stated wherever it matters.

## Repository map
| Path | What it is |
|---|---|
| `official_series.py` | Official state-wise pending cases 2014–2025, transcribed from Parliament answers (checked against the printed totals) |
| `covid_excess.py` | COVID excess-backlog estimate with backtest-calibrated ranges |
| `forecast_2026.py`, `grade_2026.py`, `results/forecast_2026/` | The locked 2026 forecast, its protocol and its grader |
| `pendency_forecast.py` | District pipeline: stock-flow reconstruction, features, quantile models, validation, attribution, elasticities |
| `compress_ddl.py`, `ddl_compact/` | Shrinks the 5 GB DDL download to aggregated counts on your own computer |
| `grade_state_2019.py`, `naive_benchmarks.py` | 2019 grading and the strict benchmarks |
| `results/ddl_2010_2018/` | Model outputs, metrics, charts and `RESULTS.md` |
| `tests/` | Identity tests for the grading arithmetic |

## Run it
```
pip install -r requirements.txt
python3 covid_excess.py                     # COVID estimate (official data is built in)
python3 forecast_2026.py                    # rebuilds the 2026 forecast and its backtest
python3 pendency_forecast.py --ddl-compact ddl_compact --ddl-only --out outputs/   # district model (~5 min)
python3 make_charts.py                      # charts
python3 tests/test_grade_2019.py            # tests
```

## Data and licences
- **Case records:** Development Data Lab, Indian judicial data (eCourts, 2010–2018), CC BY-NC-SA 4.0. Derived files
  here carry the same licence; see [`DATA_LICENSE.md`](DATA_LICENSE.md).
- **Official pendency:** Lok Sabha Unstarred Questions 1838 (16.12.2022) and 2362 (13.02.2026), Ministry of Law &
  Justice; figures from NJDG and the Supreme Court of India.
