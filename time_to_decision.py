#!/usr/bin/env python3
"""How long does a case take? Kaplan-Meier time to decision for every district and case type, DDL 2010-2018.

Input: ddl_compact/detail/cube_cat (district x filing month x decision month x case type; from
`compress_ddl.py --detail`), or, without it, ddl_compact/cube (civil vs criminal only).
Each state was scraped at a different time, so a case still undecided is censored at its state's collection month:
the last month whose decisions reach 20% of that state's 2018 monthly average. Decision dates after it (typos such
as year 9201) are treated as undecided.
Left truncation: eCourts reached districts at different times, and cases already decided by then were mostly never
entered, so early filings in the data are the slow survivors (of 2010 filings, 16% were decided within a year among
those filed before their district's records become complete, against 39% after). Each district's completeness month
is estimated (digitisation_month) and earlier cases enter the risk set only from then (delayed entry). Use
--no-truncation-fix for the uncorrected estimate.
Output per district x case type: cases, median months to decision (blank if over half are still pending when
the data ends), 90th percentile, and the share still pending after 1, 3 and 5 years, with a 95% interval
(Greenwood) on the 3-year figure. The whole curve (share still pending every 3 months, 0-120) goes to
curves_*.csv for the charts. Rows with fewer than --min-cases cases are dropped.
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


def load(compact: Path, level: str = "auto") -> tuple[pd.DataFrame, str]:
    det = sorted((compact / "detail" / "cube_cat").glob("*.parquet"))
    if det and level != "civil_criminal":
        return pd.concat(pd.read_parquet(p) for p in det).rename(columns={"category": "case_type"}), "case_type"
    cube = pd.concat(pd.read_parquet(p) for p in sorted((compact / "cube").glob("*.parquet")))
    return cube.assign(case_type=np.where(cube["crim"] == 1, "criminal", "civil")).drop(columns="crim"), "civil_criminal"


COLLECTION_FRAC = 0.2  # see collection_month; data_audit.py checks 0.1 and 0.4


def collection_month(c: pd.DataFrame) -> pd.Series:
    """Per state: last month with decisions >= 20% of the state's 2018 monthly mean."""
    dec = c.loc[(c["d"] >= 201801) & (c["d"] <= 202112)]
    per = dec.groupby([dec["district_id"] // 1000, "d"])["n"].sum().reset_index()
    per.columns = ["state", "d", "n"]
    base = per.loc[per["d"] // 100 == 2018].groupby("state")["n"].mean()
    per = per.join(base.rename("base"), on="state")
    return months(per.loc[per["n"] >= COLLECTION_FRAC * per["base"]].groupby("state")["d"].max())


def digitisation_month(c: pd.DataFrame, share: float = 0.5) -> pd.Series:
    """Per district: the month from which its case records look complete.

    eCourts reached districts at different times, and when a district went online its already-decided cases were
    mostly never entered: recorded 2010 filings are under half the 2016-17 level in 85% of districts. So for each
    district this is the first month from which every 6-month window up to Dec 2017 holds at least `share` of the
    district's 2016-17 average filings. Cases filed earlier were seen only if still pending then (left truncation).
    """
    f = c.groupby(["district_id", "f"])["n"].sum().reset_index()
    f["m"] = months(f["f"])
    lo, hi = months(pd.Series([201001])).iloc[0], months(pd.Series([201712])).iloc[0]
    f = f.loc[(f["m"] >= lo) & (f["m"] <= hi)]
    wide = f.pivot_table(index="district_id", columns="m", values="n", aggfunc="sum").reindex(
        columns=range(lo, hi + 1)).fillna(0)
    base = wide.loc[:, months(pd.Series([201601])).iloc[0]:].mean(axis=1)
    win = wide.T.rolling(6).sum().shift(-5).T.loc[:, :hi - 5]      # filings in [m, m+5]
    ok = win.ge(6 * share * base, axis=0)
    good_after = ok.loc[:, ::-1].cummin(axis=1).loc[:, ::-1]        # True from m onwards only if every later window is ok
    first = good_after.idxmax(axis=1).where(good_after.any(axis=1))
    return first.rename("digitised_m")


def km_table(c: pd.DataFrame, min_cases: int, entry: pd.Series | None = None) -> pd.DataFrame:
    """Kaplan-Meier by district x case type. With `entry` (per district, the month records become complete, see
    digitisation_month), cases filed earlier enter the risk set only at that month (delayed entry), and those
    decided before it are dropped: the standard correction for left truncation."""
    cut = collection_month(c)
    c = c.assign(fm=months(c["f"]), cut=(c["district_id"] // 1000).map(cut))
    c = c.loc[c["cut"].notna() & (c["fm"] <= c["cut"])]
    dm = np.where(c["d"] > 0, months(c["d"]), np.nan)
    event = np.isfinite(dm) & (dm <= c["cut"].to_numpy())
    dur = np.where(event, dm - c["fm"], c["cut"] - c["fm"]).clip(0, H).astype(np.int64)
    ent = np.zeros(len(c), dtype=np.int64)
    if entry is not None or "entry_m" in c:   # entry_m: per-row entry month, for pooled (e.g. national) tables
        e = (c["entry_m"] if "entry_m" in c else c["district_id"].map(entry)).to_numpy(dtype=float)
        ent = np.nan_to_num(e - c["fm"].to_numpy(), nan=0.0).clip(0, H).astype(np.int64)
    keep = dur >= ent
    g = pd.DataFrame({"district_id": c["district_id"].to_numpy(), "case_type": c["case_type"].to_numpy(),
                      "dur": dur, "event": event.astype(np.int64), "ent": ent, "n": c["n"].to_numpy()})[keep]
    g["dec"] = g["n"] * g["event"]
    g["cens"] = g["n"] - g["dec"]
    agg = g.groupby(["district_id", "case_type", "dur", "ent"])[["dec", "cens"]].sum().reset_index()
    rows = []
    for (dist, ct), grp in agg.groupby(["district_id", "case_type"], sort=False):
        n_total = int(grp["cens"].sum() + grp["dec"].sum())
        if n_total < min_cases:
            continue
        dec = np.bincount(grp["dur"], weights=grp["dec"], minlength=H + 1)
        cen = np.bincount(grp["dur"], weights=grp["cens"], minlength=H + 1)
        enter = np.bincount(grp["ent"], weights=grp["dec"] + grp["cens"], minlength=H + 1)
        at_risk = np.cumsum(enter) - np.concatenate([[0], np.cumsum(dec + cen)[:-1]])
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
                     "pending_after_3y_hi": min(at(36) + 1.96 * se36, 1.0) if 36 <= reach else np.nan,
                     "curve_q": " ".join("" if m > reach else f"{s[m]:.3f}" for m in range(0, H + 1, 3))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compact", type=Path, default=HERE / "ddl_compact")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "time_to_decision")
    ap.add_argument("--min-cases", type=int, default=200)
    ap.add_argument("--level", choices=["auto", "civil_criminal"], default="auto",
                    help="civil_criminal: two groups instead of the ten case types")
    ap.add_argument("--no-truncation-fix", action="store_true",
                    help="ignore digitisation dates (the uncorrected estimate, for comparison only)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    c, level = load(args.compact, args.level)
    entry = None if args.no_truncation_fix else digitisation_month(c)
    if entry is not None:
        y = (entry // 12).rename("first_complete_year")
        pd.DataFrame({"digitised_m": entry, "first_complete_year": y}).reset_index().to_csv(
            args.out / "digitisation.csv", index=False)
    t = km_table(c, args.min_cases, entry)
    key = args.compact / "district_key.csv"
    if key.exists():
        k = pd.read_csv(key)
        if "year" in k:
            k = k.sort_values("year").drop_duplicates(["state_code", "dist_code"], keep="last")
        k["district_id"] = k["state_code"] * 1000 + k["dist_code"]
        t = t.merge(k[["district_id", "state_name", "district_name"]], on="district_id", how="left")
    curve_cols = ["district_id", "case_type", "curve_q"]
    t[curve_cols].to_csv(args.out / f"curves_by_district_{level}.csv.gz", index=False)
    t = t.drop(columns="curve_q")
    t.to_csv(args.out / f"by_district_{level}.csv", index=False)
    pooled = c.assign(district_id=0) if entry is None else c.assign(entry_m=c["district_id"].map(entry), district_id=0)
    nat = km_table(pooled, args.min_cases).drop(columns="district_id")
    nat[["case_type", "curve_q"]].to_csv(args.out / f"curves_national_{level}.csv", index=False)
    nat = nat.drop(columns="curve_q")
    nat.to_csv(args.out / f"national_{level}.csv", index=False)
    pd.set_option("display.width", 200)
    print(nat.round(3).to_string(index=False))
    print(f"{len(t)} district x case-type rows; districts: {t['district_id'].nunique()}")
    print(t[["median_months", "pending_after_3y"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
