# Data sources and licences

## DDL judicial data (eCourts, 2010–2018)
Source: Development Data Lab, Indian judicial data scraped from eCourts (case, judge and key files), released
under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 licence (CC BY-NC-SA 4.0; see the
`license.txt` shipped with the download).

Derived from it and released under the **same CC BY-NC-SA 4.0 licence** (attribution to Development Data Lab,
non-commercial use only, share-alike):
- `ddl_compact/` (aggregated filing x decision month counts, hearing snapshots, judge postings, district key, and
  `detail/`: case-type counts and courtroom-month filings and disposals)
- `results/time_to_decision/`, `results/judge_transfer/` and the pages in `docs/` built from them
- `models/ddl_2010_2018/` and `results/ddl_2010_2018/` (models, forecasts and metrics trained on that data)

## NJDG district data via Dataful (Factly)
Datasets 21265 / 21282 (and related) are paid downloads. They are **not** redistributed here: `dataful/` is
git-ignored, `prepare_dataful.py` and `grade_2019.py` run locally, and only aggregate scores
(`results/*/grading_2019/grading_summary.csv`) and district-name crosswalks are published.

The code is released under the MIT licence (`LICENSE`); it is separate from the data and the data licences above
do not apply to it.
