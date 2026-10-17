#!/usr/bin/env python3
"""Audit: the checks behind the caveats on the site, each reproducible from ddl_compact/.

1. Digitisation (left truncation). Recorded filings rise from 4.3 million (2010) to 13.7 million (2018), far faster
   than real filings grew. Evidence that early records are the slow survivors: among cases filed in the same year,
   those filed before their district's records become complete were decided much less often within 12 months.
2. Effect of the correction (delayed entry from the completeness month) on national figures and on every
   district x case type, and how sensitive it is to the completeness threshold (30% / 50% / 70%) and to the
   collection-month rule (10% / 40% instead of 20%).
3. Rank uncertainty. Each district's share pending after 3 years has a sampling interval (Greenwood); ranks are
   simulated from it, giving a 90% range for each district's rank within its case type.
4. Units that are not districts. Some DDL "districts" are court complexes (e.g. Mumbai City Civil Court).

Outputs (results/audit/): summary.json, cohort_speed.csv, correction_by_type.csv, rank_intervals.csv
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

import time_to_decision as T

HERE: Final = Path(__file__).resolve().parent
OUT: Final = HERE / "results" / "audit"
COLS: Final = ["median_months", "pending_after_1y", "pending_after_3y", "pending_after_5y"]
NOT_DISTRICT: Final = re.compile(r"court|tribunal|cmm|claims|family|city civil|small causes", re.I)


def cohort_speed(c: pd.DataFrame, entry: pd.Series) -> pd.DataFrame:
    fm, dm = T.months(c["f"]), np.where(c["d"] > 0, T.months(c["d"]), np.nan)
    complete = fm >= c["district_id"].map(entry).fillna(0)
    dec12 = np.isfinite(dm) & (dm - fm <= 12)
    d = pd.DataFrame({"year": c["f"] // 100, "complete": complete, "n": c["n"], "d12": c["n"] * dec12})
    g = d.groupby(["year", "complete"])[["n", "d12"]].sum()
    g["decided_within_12m"] = g["d12"] / g["n"]
    out = g["decided_within_12m"].unstack().rename(columns={True: "after_records_complete", False: "before"})
    out["cases_filed"] = d.groupby("year")["n"].sum()
    return out.reset_index()


def rank_intervals(t: pd.DataFrame, draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for ct, g in t.dropna(subset=["pending_after_3y"]).groupby("case_type"):
        se = ((g["pending_after_3y_hi"] - g["pending_after_3y_lo"]) / 3.92).fillna(0).to_numpy()
        sim = g["pending_after_3y"].to_numpy() + rng.standard_normal((draws, len(g))) * se
        ranks = sim.argsort(axis=1).argsort(axis=1) + 1
        lo, hi = np.percentile(ranks, [5, 95], axis=0)
        point = g["pending_after_3y"].rank(method="first").to_numpy()
        rows.append(pd.DataFrame({"district_id": g["district_id"].to_numpy(), "case_type": ct, "rank": point.astype(int),
                                  "rank_lo": lo.round().astype(int), "rank_hi": hi.round().astype(int), "of": len(g)}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c, _ = T.load(HERE / "ddl_compact")
    entry = T.digitisation_month(c)
    cohort_speed(c, entry).round(4).to_csv(OUT / "cohort_speed.csv", index=False)

    nat = {"uncorrected": T.km_table(c.assign(district_id=0), 200)}
    for share in (0.3, 0.5, 0.7):
        e = T.digitisation_month(c, share)
        nat[f"corrected_{int(share * 100)}"] = T.km_table(c.assign(entry_m=c["district_id"].map(e), district_id=0), 200)
    e50 = T.digitisation_month(c)
    for frac in (0.1, 0.4):   # the collection-month rule (20% of 2018 monthly decisions) moved both ways
        T.COLLECTION_FRAC = frac
        nat[f"collection_{int(frac * 100)}"] = T.km_table(c.assign(entry_m=c["district_id"].map(e50), district_id=0), 200)
    T.COLLECTION_FRAC = 0.2
    by_type = pd.concat([v.set_index("case_type")[COLS].add_suffix(f"__{k}") for k, v in nat.items()], axis=1)
    by_type.round(4).reset_index().to_csv(OUT / "correction_by_type.csv", index=False)

    naive = T.km_table(c, 200)
    corr = pd.read_csv(HERE / "results" / "time_to_decision" / "by_district_case_type.csv")
    m = naive.merge(corr, on=["district_id", "case_type"], suffixes=("_naive", ""))
    shift = (m["pending_after_3y"] - m["pending_after_3y_naive"]).dropna()
    civ = m.loc[m["case_type"] == "civ_suit"].dropna(subset=["pending_after_3y", "pending_after_3y_naive"])
    rshift = (civ["pending_after_3y"].rank() - civ["pending_after_3y_naive"].rank()).abs()

    ri = rank_intervals(corr)
    ri.to_csv(OUT / "rank_intervals.csv", index=False)
    civ_ri = ri.loc[ri["case_type"] == "civ_suit"]
    width = civ_ri["rank_hi"] - civ_ri["rank_lo"]

    names = corr.dropna(subset=["district_name"]).drop_duplicates("district_id")
    courts = names.loc[names["district_name"].str.contains(NOT_DISTRICT), "district_name"].tolist()

    yearly = c.groupby(c["f"] // 100)["n"].sum()
    summary = {
        "recorded_filings_2010": int(yearly.loc[2010]), "recorded_filings_2018": int(yearly.loc[2018]),
        "share_of_cases_filed_before_records_complete": round(float(
            c.loc[T.months(c["f"]) < c["district_id"].map(entry).fillna(0), "n"].sum() / c["n"].sum()), 4),
        "districts_complete_from": {int(k): int(v) for k, v in (entry // 12).value_counts().sort_index().items()},
        "districts_never_complete": int(entry.isna().sum()),
        "rows_compared": int(len(shift)),
        "rows_shift_over_5pp": int((shift.abs() > 0.05).sum()), "rows_shift_over_10pp": int((shift.abs() > 0.10).sum()),
        "median_shift_pp": round(float(shift.median() * 100), 2),
        "civil_suit_rank_spearman_naive_vs_corrected": round(float(civ[["pending_after_3y", "pending_after_3y_naive"]]
                                                                    .corr("spearman").iloc[0, 1]), 3),
        "civil_suit_median_rank_shift": int(rshift.median()), "civil_suit_max_rank_shift": int(rshift.max()),
        "civil_suit_districts_ranked": int(len(civ_ri)),
        "civil_suit_median_rank_interval_width": int(width.median()),
        "civil_suit_share_rank_interval_over_50": round(float((width > 50).mean()), 3),
        "units_that_are_court_complexes": courts,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(by_type.loc[:, by_type.columns.str.startswith("pending_after_3y")].round(3).to_string())


if __name__ == "__main__":
    main()
