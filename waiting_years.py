#!/usr/bin/env python3
"""Years of waiting: how much time cases filed 2010-2018 had already spent undecided when the data was collected.

Input: ddl_compact/cube (district x filing month x decision month x civil/criminal, from `compress_ddl.py`).
For each state, "now" is its collection month (same rule as time_to_decision.py). A case filed by then and not
decided by then is pending; its age is the months since filing. Summing ages gives case-years of waiting still
running; adding the time decided cases spent before their decision gives all waiting the 2010-2018 filings caused.
A case-year is one case undecided for one year; a case usually involves two or more people, so people-years are larger.
Cases filed before 2010 are not in the data, so the real stock is older than this.

Output (results/waiting_years/):
  national.json       totals, mean and median age of pending cases, share older than 1/3/5 years
  age_profile.csv     pending cases by age in years (0-9), civil and criminal
  districts.csv       per district: pending cases, case-years, mean age, share over 3 years
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from time_to_decision import collection_month, months

HERE: Final = Path(__file__).resolve().parent
OUT: Final = HERE / "results" / "waiting_years"


def load() -> pd.DataFrame:
    c = pd.concat(pd.read_parquet(p) for p in sorted((HERE / "ddl_compact" / "cube").glob("*.parquet")))
    cut = collection_month(c)
    c = c.assign(fm=months(c["f"]), cut=(c["district_id"] // 1000).map(cut))
    c = c.loc[c["cut"].notna() & (c["fm"] <= c["cut"])].copy()
    dm = np.where(c["d"] > 0, months(c["d"]), np.nan)
    c["pending"] = ~(np.isfinite(dm) & (dm <= c["cut"].to_numpy()))
    c["age"] = np.where(c["pending"], c["cut"] - c["fm"], dm - c["fm"]).clip(0)
    return c


def weighted_median(x: pd.Series, w: pd.Series) -> float:
    o = np.argsort(x.to_numpy())
    cw = np.cumsum(w.to_numpy()[o])
    return float(x.to_numpy()[o][np.searchsorted(cw, cw[-1] / 2)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = load()
    p = c.loc[c["pending"]]
    n, yrs = int(p["n"].sum()), float((p["n"] * p["age"]).sum() / 12)
    nat = {
        "cases_filed": int(c["n"].sum()),
        "pending_at_collection": n,
        "case_years_pending": round(yrs),
        "case_years_all": round(float((c["n"] * c["age"]).sum() / 12)),
        "mean_age_years": round(yrs / n, 2),
        "median_age_years": round(weighted_median(p["age"], p["n"]) / 12, 2),
        **{f"share_over_{k}y": round(float(p.loc[p["age"] >= 12 * k, "n"].sum() / n), 4) for k in (1, 3, 5)},
        "civil_share_of_case_years": round(float((p["n"] * p["age"])[p["crim"] == 0].sum() / (p["n"] * p["age"]).sum()), 4),
        "states": int(c["cut"].groupby(c["district_id"] // 1000).first().size),
        "collection_months": sorted({f"{int(m) // 12}-{int(m) % 12 + 1:02d}" for m in c["cut"].unique()}),
    }
    (OUT / "national.json").write_text(json.dumps(nat, indent=2) + "\n")

    prof = (p.assign(age_y=(p["age"] // 12).clip(upper=9).astype(int), kind=np.where(p["crim"] == 1, "criminal", "civil"))
            .groupby(["age_y", "kind"])["n"].sum().unstack(fill_value=0).reset_index())
    prof.to_csv(OUT / "age_profile.csv", index=False)

    w = p.assign(cy=p["n"] * p["age"] / 12, old=p["n"] * (p["age"] >= 36))
    d = w.groupby("district_id").agg(pending=("n", "sum"), case_years=("cy", "sum"), over3=("old", "sum"))
    d = d.join(c.groupby("district_id")["n"].sum().rename("filed"))
    d["mean_age_years"] = d["case_years"] / d["pending"]
    d["share_over_3y"] = d["over3"] / d["pending"]
    d["case_years"] = d["case_years"].round().astype(int)
    d = d.drop(columns="over3").reset_index().sort_values("case_years", ascending=False)
    d.round(4).to_csv(OUT / "districts.csv", index=False)
    print(json.dumps(nat, indent=2))


if __name__ == "__main__":
    main()
