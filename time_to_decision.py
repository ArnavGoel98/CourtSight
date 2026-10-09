#!/usr/bin/env python3
"""How long does a case take? Kaplan-Meier time to decision for every district and case type, DDL 2010-2018.

Input: ddl_compact/detail/cube_cat (district x filing month x decision month x case type; from
`compress_ddl.py --detail`), or, without it, ddl_compact/cube (civil vs criminal only).
Each state was scraped at a different time, so a case still undecided is censored at its state's collection month:
the last month whose decisions reach 20% of that state's 2018 monthly average. Decision dates after it (typos such
as year 9201) are treated as undecided.
Output per district x case type: cases, median months to decision (blank if over half are still pending when
the data ends), 90th percentile, and the share still pending after 1, 3 and 5 years, with a 95% interval
(Greenwood) on the 3-year figure. Rows with fewer than --min-cases cases are dropped.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
H: Final = 120  # months of follow-up kept (10 years; the data spans at most ~9.5)


def months(ym: pd.Series) -> pd.Series:
    return (ym // 100) * 12 + ym % 100 - 1


def load(compact: Path) -> tuple[pd.DataFrame, str]:
    det = sorted((compact / "detail" / "cube_cat").glob("*.parquet"))
    if det:
        return pd.concat(pd.read_parquet(p) for p in det).rename(columns={"category": "case_type"}), "case_type"
    cube = pd.concat(pd.read_parquet(p) for p in sorted((compact / "cube").glob("*.parquet")))
    return cube.assign(case_type=np.where(cube["crim"] == 1, "criminal", "civil")).drop(columns="crim"), "civil_criminal"


def collection_month(c: pd.DataFrame) -> pd.Series:
    """Per state: last month with decisions >= 20% of the state's 2018 monthly mean."""
    dec = c.loc[(c["d"] >= 201801) & (c["d"] <= 202112)]
    per = dec.groupby([dec["district_id"] // 1000, "d"])["n"].sum().reset_index()
    per.columns = ["state", "d", "n"]
    base = per.loc[per["d"] // 100 == 2018].groupby("state")["n"].mean()
    per = per.join(base.rename("base"), on="state")
    return months(per.loc[per["n"] >= 0.2 * per["base"]].groupby("state")["d"].max())


def km_table(c: pd.DataFrame, min_cases: int) -> pd.DataFrame:
    cut = collection_month(c)
    c = c.assign(fm=months(c["f"]), cut=(c["district_id"] // 1000).map(cut))
    c = c.loc[c["cut"].notna() & (c["fm"] <= c["cut"])]
    dm = np.where(c["d"] > 0, months(c["d"]), np.nan)
    event = np.isfinite(dm) & (dm <= c["cut"].to_numpy())
    dur = np.where(event, dm - c["fm"], c["cut"] - c["fm"]).clip(0, H).astype(np.int64)
    g = pd.DataFrame({"district_id": c["district_id"].to_numpy(), "case_type": c["case_type"].to_numpy(),
                      "dur": dur, "event": event.astype(np.int64), "n": c["n"].to_numpy()})
    agg = g.groupby(["district_id", "case_type", "dur", "event"])["n"].sum().unstack("event", fill_value=0)
    agg = agg.reindex(columns=[0, 1], fill_value=0).rename(columns={0: "cens", 1: "dec"}).reset_index()
    rows = []
    for (dist, ct), grp in agg.groupby(["district_id", "case_type"], sort=False):
        n_total = int(grp["cens"].sum() + grp["dec"].sum())
        if n_total < min_cases:
            continue
        dec = np.bincount(grp["dur"], weights=grp["dec"], minlength=H + 1)
        cen = np.bincount(grp["dur"], weights=grp["cens"], minlength=H + 1)
        at_risk = n_total - np.concatenate([[0], np.cumsum(dec + cen)[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(at_risk > 0, dec / at_risk, 0.0)
            gw = np.where((at_risk > 0) & (at_risk > dec), dec / (at_risk * (at_risk - dec)), 0.0)
        s = np.cumprod(1 - h)  # S(m): share still pending after m months
        var = s ** 2 * np.cumsum(gw)
        reach = int(np.max(np.nonzero(at_risk > 0)[0])) if (at_risk > 0).any() else 0

        def at(m: int) -> float:
            return float(s[m]) if m <= reach else np.nan

        def q(p: float) -> float:
            idx = np.nonzero(s <= 1 - p)[0]
            return float(idx[0]) if len(idx) else np.nan

        se36 = float(np.sqrt(var[36])) if 36 <= reach else np.nan
        rows.append({"district_id": int(dist), "case_type": ct, "cases": n_total,
                     "median_months": q(0.5), "p90_months": q(0.9),
                     "pending_after_1y": at(12), "pending_after_3y": at(36), "pending_after_5y": at(60),
                     "pending_after_3y_lo": max(at(36) - 1.96 * se36, 0.0) if 36 <= reach else np.nan,
                     "pending_after_3y_hi": min(at(36) + 1.96 * se36, 1.0) if 36 <= reach else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compact", type=Path, default=HERE / "ddl_compact")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "time_to_decision")
    ap.add_argument("--min-cases", type=int, default=200)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    c, level = load(args.compact)
    t = km_table(c, args.min_cases)
    key = args.compact / "district_key.csv"
    if key.exists():
        k = pd.read_csv(key)
        if "year" in k:
            k = k.sort_values("year").drop_duplicates(["state_code", "dist_code"], keep="last")
        k["district_id"] = k["state_code"] * 1000 + k["dist_code"]
        t = t.merge(k[["district_id", "state_name", "district_name"]], on="district_id", how="left")
    t.to_csv(args.out / f"by_district_{level}.csv", index=False)
    nat = km_table(c.assign(district_id=0), args.min_cases).drop(columns="district_id")
    nat.to_csv(args.out / f"national_{level}.csv", index=False)
    pd.set_option("display.width", 200)
    print(nat.round(3).to_string(index=False))
    print(f"{len(t)} district x case-type rows; districts: {t['district_id'].nunique()}")
    print(t[["median_months", "pending_after_3y"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
