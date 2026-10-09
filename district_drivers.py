#!/usr/bin/env python3
"""What do slow districts have in common? Correlates of time to decision across district courts, DDL 2010-2018.

This is descriptive. A district with more courts may also be richer, more urban, or better run; nothing here says
adding courts would make a district faster. The judge-handover study (judge_transfer.py) shows how hard it is to get
causal answers from this data.

Outcome per district: share of civil (and, separately, criminal) cases still pending 3 years after filing, from
results/time_to_decision/by_district_civil_criminal.csv (Kaplan-Meier, censored at collection).
Features per district, from ddl_compact:
  log_filings_per_court  log of cases filed per year per active courtroom, 2014-2017 (court_month.parquet)
  log_filings         size: log of cases filed per year
  criminal_share      share of filings that are criminal (cube)
  judge_tenure_months median months a judge stays in one courtroom (judges.csv.gz)
  disposal_ratio      cases decided / cases filed per year, 2014-2017 (throughput: nearly the outcome itself)
Models: OLS of each outcome on the standardised structural features (all but disposal_ratio), and on all five,, with a 95% interval from 2,000 bootstrap resamples of
districts. Also: Spearman correlation of each feature with the outcome alone.

Output: results/district_drivers/{districts.csv, coefficients.csv, summary.json}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE: Final = Path(__file__).resolve().parent
DDL: Final = HERE / "ddl_compact"
OUT: Final = HERE / "results" / "district_drivers"
FEATURES: Final = ["log_filings_per_court", "log_filings", "criminal_share", "judge_tenure_months", "disposal_ratio"]
STRUCTURAL: Final = FEATURES[:4]  # disposal_ratio is close to a restatement of the outcome, so it gets its own model
MIN_CASES: Final = 2000


def features() -> pd.DataFrame:
    cm = pd.read_parquet(DDL / "detail" / "court_month.parquet")
    cm = cm.loc[(cm["ym"] >= 201401) & (cm["ym"] <= 201712)]
    active = cm.loc[(cm["filed"] + cm["disposed"]) > 0]
    courts = active.groupby(["district_id", "ym"])["court_no"].nunique().groupby("district_id").mean()
    flows = cm.groupby("district_id")[["filed", "disposed"]].sum() / 4
    cube = pd.concat(pd.read_parquet(p) for p in sorted((DDL / "cube").glob("*.parquet")))
    crim = cube.groupby(["district_id", "crim"])["n"].sum().unstack(fill_value=0)

    j = pd.read_csv(DDL / "judges.csv.gz")
    for c in ("start_date", "end_date"):
        j[c] = pd.to_datetime(j[c], format="%d-%m-%Y", errors="coerce")
    j = j.dropna(subset=["start_date", "end_date"])
    j = j.loc[(j["end_date"] >= j["start_date"]) & (j["start_date"].dt.year >= 2005)]
    j["months"] = (j["end_date"] - j["start_date"]).dt.days / 30.44
    j["district_id"] = j["state_code"] * 1000 + j["dist_code"]
    tenure = j.groupby("district_id")["months"].median()

    f = pd.DataFrame({"courts": courts, "filed_per_year": flows["filed"]})
    f["log_filings_per_court"] = np.log(f["filed_per_year"] / f["courts"])
    f["log_filings"] = np.log(f["filed_per_year"])
    f["criminal_share"] = crim[1] / crim.sum(axis=1)
    f["judge_tenure_months"] = tenure
    f["disposal_ratio"] = flows["disposed"] / flows["filed"]
    return f


def ols_boot(X: np.ndarray, y: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    A = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(seed)
    bs = np.empty((reps, A.shape[1]))
    for b in range(reps):
        i = rng.integers(0, len(y), len(y))
        bs[b] = np.linalg.lstsq(A[i], y[i], rcond=None)[0]
    return beta[1:], np.percentile(bs[:, 1:], [2.5, 97.5], axis=0), float(r2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f = features()
    ttd = pd.read_csv(HERE / "results" / "time_to_decision" / "by_district_civil_criminal.csv")
    ttd = ttd.loc[ttd["cases"] >= MIN_CASES].pivot(index="district_id", columns="case_type",
                                                    values=["pending_after_3y", "median_months"])
    ttd.columns = [f"{a}_{b}" for a, b in ttd.columns]
    names = pd.read_csv(HERE / "results" / "time_to_decision" / "by_district_case_type.csv")
    names = names.dropna(subset=["district_name"]).groupby("district_id")[["state_name", "district_name"]].first()
    d = f.join(ttd, how="inner").join(names, how="left")
    d = d.loc[np.isfinite(d[FEATURES]).all(axis=1) & d["courts"].ge(2)]
    d.reset_index().round(4).to_csv(OUT / "districts.csv", index=False)

    rows, summ = [], {}
    for kind in ("civil", "criminal"):
        y = d[f"pending_after_3y_{kind}"]
        ok = y.notna()
        for model, feats in (("structural", STRUCTURAL), ("with_throughput", FEATURES)):
            X = d.loc[ok, feats]
            Z = ((X - X.mean()) / X.std()).to_numpy()
            beta, ci, r2 = ols_boot(Z, y[ok].to_numpy())
            summ.setdefault(kind, {})[f"r2_{model}"] = round(r2, 3)
            for k, name in enumerate(feats):
                rows.append({"outcome": kind, "model": model, "feature": name, "coef_per_sd": beta[k],
                             "lo": ci[0, k], "hi": ci[1, k], "spearman": spearmanr(X[name], y[ok]).statistic})
        summ[kind] |= {"districts": int(ok.sum()),
                      "mean_pending_3y": round(float(y[ok].mean()), 4),
                      "p10_p90_pending_3y": [round(float(y[ok].quantile(q)), 4) for q in (0.1, 0.9)]}
    co = pd.DataFrame(rows).round(4)
    co.to_csv(OUT / "coefficients.csv", index=False)
    summ["feature_means"] = {k: round(float(d[k].mean()), 3) for k in FEATURES}
    summ["feature_sds"] = {k: round(float(d[k].std()), 3) for k in FEATURES}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(co.to_string(index=False))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
