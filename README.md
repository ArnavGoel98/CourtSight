# District court pendency forecasting (DDL eCourts × NJDG)

Direct multi-horizon (12/24/36-month) quantile forecasts (q10/q50/q90) of district-level backlog growth and
clearance ratio, with TreeSHAP driver decomposition and fixed-effects disposal elasticities that are
converted into bench, hearing-cadence and surge-capacity levers.

```
pip install -r requirements.txt
python pendency_forecast.py --synthetic --out outputs/            # end-to-end on DDL/NJDG-schema synthetic data
python pendency_forecast.py \
  --ddl-csv-glob 'ddl/cases/cases_*.csv' --ddl-parquet data/ddl_parquet \
  --ddl-judges ddl/judges_clean.csv --njdg data/njdg_monthly.csv \
  --edges data/district_edges.csv --out outputs/
```

Inputs
- DDL judicial data: per-year case CSVs + judges file; column map in `DDL_CASE_COLS` / `DDL_JUDGE_COLS`
  (check it against the release README before running on real data).
- NJDG: one row per district-month; required columns in `NJDG_REQUIRED`, optional columns in `NJDG_OPTIONAL`.
  NJDG publishes snapshots, not history, so this file has to be built by archiving snapshots month by month.
- Edges: undirected district adjacency (`src_state,src_dist,dst_state,dst_dist`), e.g. Queen contiguity on SHRUG pc11 polygons.

Outputs: `panel_features.parquet`, `validation_metrics.csv`, `forecasts.csv`, `shap_q90_flagged.csv`,
`drivers_flagged.csv`, `elasticities.csv`, `policy_levers.csv`.
