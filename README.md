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

## Using the real DDL data when it's too big to upload

The ~5 GB DDL download is shrunk on your own computer into a small package of aggregated counts (tens of MB):

```
# 1. Get the code (or download the three files compress_ddl.py, taxonomy.py from GitHub)
git clone https://github.com/Arnavthemighty/areudumb && cd areudumb
git checkout claude/hopeful-wright-qibytw
pip install pandas pyarrow

# 2. Check what it finds, then shrink (Windows: use "C:\path\to\folder")
python court_pendency/compress_ddl.py --src "/path/to/unzipped/ddl" --list-only
python court_pendency/compress_ddl.py --src "/path/to/unzipped/ddl" --out court_pendency/ddl_compact

# 3. Send it: commit court_pendency/ddl_compact/ and push, or upload the folder's files on GitHub
#    (Add file -> Upload files); every file is kept under 24 MB.
```

Then train with `python pendency_forecast.py --ddl-compact ddl_compact --njdg <njdg_monthly.csv> --out outputs/`.
Only aggregated counts leave your computer, no case-level rows or names.

Inputs
- DDL judicial data: per-year case CSVs + judges file; column map in `DDL_CASE_COLS` / `DDL_JUDGE_COLS`
  (check it against the release README before running on real data).
- NJDG: one row per district-month; required columns in `NJDG_REQUIRED`, optional columns in `NJDG_OPTIONAL`.
  NJDG publishes snapshots, not history, so this file has to be built by archiving snapshots month by month.
- Edges: undirected district adjacency (`src_state,src_dist,dst_state,dst_dist`), e.g. Queen contiguity on SHRUG pc11 polygons.

Outputs: `panel_features.parquet`, `validation_metrics.csv`, `forecasts.csv`, `shap_q90_flagged.csv`,
`drivers_flagged.csv`, `elasticities.csv`, `policy_levers.csv`.
