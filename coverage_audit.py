#!/usr/bin/env python3
"""Audit: how much of the court system the DDL records cover, and whether the 2010-2017 "speed-up" is real.

1. Coverage. The official figures (official_series.py) give each state's filings by the stock-flow identity,
   filed(t) = pending(t) - pending(t-1) + disposed(t), for 2015-2021. DDL's recorded filings for 2015-2018 are
   compared with them, state by state. The identity also counts transfers, restorations and clean-ups, and small
   states' figures are noisy, so a single state-year is only indicative; four-year totals are steadier.
2. Speed-up. Among records filed after their district's records became complete, the share decided within 12 months
   rises from 2010 to 2017 filings. It is split into case mix (holding the case-type mix fixed) and, separately,
   into states whose coverage was stable in 2015-2018 and states whose coverage grew. If the rise were real court
   behaviour it would show in both groups; if it were records still filling in, it would sit in the growing group.
3. Are "pending" records live? A case that was decided but never marked decided would sit in the data as pending
   and make a court look slower. For cases pending at collection, the date each was last listed for a hearing is
   compared with the state's collection month: a live case is listed every few months.
4. Age of the pending stock against NJDG. The National Judicial Data Grid's age bands for pending district-court
   cases on 24 May 2020 (as reported in the press) are compared with the ages of cases pending at collection here.
   The records hold only cases filed 2010-2018, so cases under a year old are missing wherever a state was
   collected after 2018; the comparison uses cases aged 1-10 years in states collected by December 2019, for which
   those ages are fully covered.
5. Sensitivity of the national figures to the remaining gaps: (a) counting every record only from January 2014,
   when coverage had largely settled, and (b) weighting each state by its official filings instead of its recorded
   ones (each record counts 1 / coverage), so that thinly covered states are not under-represented.

Outputs (results/audit/): coverage_by_state.csv, speedup_by_year.csv, pending_activity.csv, national_sensitivity.csv,
age_check.json, coverage.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import official_series as O
import time_to_decision as T

HERE: Final = Path(__file__).resolve().parent
OUT: Final = HERE / "results" / "audit"
YEARS: Final = [2015, 2016, 2017, 2018]
MIN_OFFICIAL: Final = 200_000   # four-year official filings below this are too noisy to classify a state
COLS: Final = ["median_months", "pending_after_1y", "pending_after_3y", "pending_after_5y"]
# DDL state names -> the units official_series uses (states that split are combined there)
TO_OFFICIAL: Final = {"Andhra Pradesh": "Andhra Pradesh + Telangana", "Telangana": "Andhra Pradesh + Telangana",
                      "Orissa": "Odisha", "Jammu and Kashmir": "Jammu & Kashmir + Ladakh",
                      "Diu and Daman": "Dadra & Nagar Haveli + Daman & Diu",
                      "DNH at Silvasa": "Dadra & Nagar Haveli + Daman & Diu"}


def state_units() -> pd.Series:
    """DDL state code -> official unit name."""
    k = pd.read_csv(HERE / "ddl_compact" / "district_key.csv")
    return k.groupby("state_code")["state_name"].first().replace(TO_OFFICIAL)


def coverage(c: pd.DataFrame, units: pd.Series) -> pd.DataFrame:
    unit = (c["district_id"] // 1000).map(units)
    ddl = c.groupby([unit, c["f"] // 100])["n"].sum().unstack().reindex(columns=YEARS)
    off = O.instituted().reindex(columns=YEARS).loc[ddl.index]
    both = off.notna() & ddl.notna()
    out = pd.DataFrame({f"coverage_{y}": ddl[y] / off[y] for y in YEARS})
    out["recorded_2015_18"] = ddl.where(both).sum(axis=1)
    out["official_2015_18"] = off.where(both).sum(axis=1)
    out["coverage_2015_18"] = out["recorded_2015_18"] / out["official_2015_18"]
    out["change_2015_18"] = (out["coverage_2018"] - out["coverage_2015"]).round(2)
    big = out["official_2015_18"] >= MIN_OFFICIAL
    stable = (out["change_2015_18"].abs() <= 0.1) & (out["coverage_2015_18"] >= 0.5)
    out["group"] = np.select([~big, stable, out["change_2015_18"] > 0.1], ["too small to classify", "stable", "growing"],
                             "other")
    out.index.name = "state"
    return out


def speedup(c: pd.DataFrame, entry: pd.Series, groups: pd.Series, units: pd.Series) -> pd.DataFrame:
    fm = T.months(c["f"])
    dm = np.where(c["d"] > 0, T.months(c["d"]), np.nan)
    keep = (fm >= c["district_id"].map(entry).fillna(np.inf)) & (c["f"] // 100 <= 2017)
    d = pd.DataFrame({"year": c["f"] // 100, "case_type": c["case_type"], "n": c["n"],
                      "d12": c["n"] * (np.isfinite(dm) & (dm - fm <= 12)),
                      "group": (c["district_id"] // 1000).map(units).map(groups)})[keep.to_numpy()]
    g = d.groupby(["year", "case_type"])[["n", "d12"]].sum()
    rate = (g["d12"] / g["n"]).unstack()
    mix = g["n"].unstack()
    mix = (mix.div(mix.sum(axis=1), axis=0)).mean()          # the average 2010-2017 case-type mix
    out = pd.DataFrame({"all_complete_records": g.groupby("year")["d12"].sum() / g.groupby("year")["n"].sum(),
                        "fixed_case_mix": rate.mul(mix, axis=1).sum(axis=1)})
    for grp in ("stable", "growing"):
        s = d.loc[d["group"] == grp].groupby("year")[["n", "d12"]].sum()
        out[f"{grp}_coverage_states"] = s["d12"] / s["n"]
    for ct in ("civ_suit", "crim_magisterial", "civ_misc"):
        out[ct] = rate[ct]
    out.index.name = "filing_year"
    return out


def pending_activity(c: pd.DataFrame) -> pd.DataFrame:
    """By state: share of cases pending at collection last listed more than 12 / 24 months before it."""
    cut = T.collection_month(c)
    s = pd.read_parquet(HERE / "ddl_compact" / "snapshot.parquet")
    sc = s["district_id"] // 1000
    age = sc.map(cut) - T.months(s["t_ym"])
    names = pd.read_csv(HERE / "ddl_compact" / "district_key.csv").groupby("state_code")["state_name"].first()
    d = pd.DataFrame({"state": sc.map(names), "n": s["n"], "o12": s["n"] * (age > 12), "o24": s["n"] * (age > 24)})
    g = d.groupby("state")[["n", "o12", "o24"]].sum()
    out = pd.DataFrame({"pending_with_listing": g["n"], "not_listed_in_12m": g["o12"] / g["n"],
                        "not_listed_in_24m": g["o24"] / g["n"]})
    out.loc["All states"] = [g["n"].sum(), g["o12"].sum() / g["n"].sum(), g["o24"].sum() / g["n"].sum()]
    return out


# NJDG, district and taluka courts, 24 May 2020: share of pending cases by age, as printed (sources/README.md)
NJDG_2020: Final = {"under_1y": 32.0, "1_3y": 28.0, "3_5y": 15.0, "5_10y": 15.28, "10_20y": 7.1, "20_30y": 1.28,
                    "over_30y": 0.26}


def age_check() -> dict[str, object]:
    import waiting_years as W
    c = W.load()
    cut = c.groupby(c["district_id"] // 1000)["cut"].first()
    ok = cut.index[cut <= T.months(pd.Series([201912])).iloc[0]]
    p = c.loc[c["pending"] & (c["district_id"] // 1000).isin(ok)]
    yrs = p["age"] // 12
    bands = {"1_3y": (1, 2), "3_5y": (3, 4), "5_10y": (5, 9)}
    tot = p.loc[yrs.between(1, 9), "n"].sum()
    ours = {k: round(float(p.loc[yrs.between(a, b), "n"].sum() / tot), 3) for k, (a, b) in bands.items()}
    nj = sum(NJDG_2020[k] for k in bands)
    njdg = {k: round(NJDG_2020[k] / nj, 3) for k in bands}
    return {"states": int(len(ok)), "pending_cases_aged_1_10y": int(tot), "records_share": ours, "njdg_share": njdg,
            "max_abs_gap_points": round(max(abs(ours[k] - njdg[k]) for k in bands) * 100, 1),
            "njdg_source": "NJDG figures for 24 May 2020 as reported by LiveLaw (shares of cases aged 1-10 years)"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c, _ = T.load(HERE / "ddl_compact")
    entry = T.digitisation_month(c)
    units = state_units()

    cov = coverage(c, units)
    cov.round(4).reset_index().to_csv(OUT / "coverage_by_state.csv", index=False)
    sp = speedup(c, entry, cov["group"], units)
    sp.round(4).reset_index().to_csv(OUT / "speedup_by_year.csv", index=False)

    ages = age_check()
    (OUT / "age_check.json").write_text(json.dumps(ages, indent=2) + "\n")
    act = pending_activity(c)
    act.round(4).reset_index().to_csv(OUT / "pending_activity.csv", index=False)

    unit = (c["district_id"] // 1000).map(units)
    weight = 1 / unit.map(cov["coverage_2015_18"].clip(upper=1.0)).fillna(1.0)
    e = c["district_id"].map(entry)
    m2014 = float(T.months(pd.Series([201401])).iloc[0])
    runs = {"as_published": c.assign(entry_m=e),
            "records_from_2014": c.assign(entry_m=np.maximum(e.fillna(np.inf), m2014)),
            "weighted_to_official_filings": c.assign(entry_m=e, n=c["n"] * weight)}
    nat = pd.concat({k: T.km_table(v.assign(district_id=0), 200).set_index("case_type")[COLS]
                     for k, v in runs.items()}, axis=1)
    nat.columns = [f"{b}__{a}" for a, b in nat.columns]
    nat.round(4).reset_index().to_csv(OUT / "national_sensitivity.csv", index=False)

    # are thinly covered states faster or slower? state-level civil-suit results against coverage
    st = T.km_table(c.assign(entry_m=e, district_id=(c["district_id"] // 1000) * 1000), 2000)
    st = st.loc[st["case_type"] == "civ_suit"].assign(unit=lambda x: (x["district_id"] // 1000).map(units))
    st = st.groupby("unit")["pending_after_3y"].mean().to_frame().join(cov["coverage_2015_18"], how="inner")
    st = st.loc[cov.loc[st.index, "official_2015_18"] >= MIN_OFFICIAL]
    rho = spearmanr(st["coverage_2015_18"], st["pending_after_3y"]).statistic

    nat_cov = cov[["recorded_2015_18", "official_2015_18"]].sum()
    nat_cov["official_2015_18"] += O.instituted().loc[~O.instituted().index.isin(cov.index), YEARS].sum().sum()
    off = O.instituted()
    big = cov.loc[cov["group"] != "too small to classify"]
    summary = {
        "coverage_2015_18": round(float(nat_cov["recorded_2015_18"] / nat_cov["official_2015_18"]), 3),
        # each year over the units with an official figure that year (Kerala has none for 2016)
        "coverage_by_year": {y: round(float(c.loc[unit.isin(off.index[off[y].notna()]) & (c["f"] // 100 == y), "n"]
                                            .sum() / off[y].sum()), 3) for y in YEARS},
        "official_units_with_no_records": sorted(set(off.index) - set(cov.index)),
        "lowest_coverage": {k: round(float(v), 2) for k, v in big["coverage_2015_18"].nsmallest(4).items()},
        "highest_coverage": {k: round(float(v), 2) for k, v in big["coverage_2015_18"].nlargest(4).items()},
        "states_stable": sorted(cov.index[cov["group"] == "stable"]),
        "states_growing": sorted(cov.index[cov["group"] == "growing"]),
        "decided_within_12m": {k: {str(y): round(float(sp.loc[y, k]), 3) for y in (2010, 2014, 2017)}
                               for k in ("all_complete_records", "fixed_case_mix", "stable_coverage_states",
                                         "growing_coverage_states")},
        "pending_not_listed_in_12m": round(float(act.loc["All states", "not_listed_in_12m"]), 3),
        "pending_not_listed_in_24m": round(float(act.loc["All states", "not_listed_in_24m"]), 3),
        "pending_not_listed_in_12m_worst": {k: round(float(v), 2) for k, v in
                                            act.drop("All states")["not_listed_in_12m"].nlargest(3).items()},
        "pending_not_listed_in_12m_bihar": round(float(act.loc["Bihar", "not_listed_in_12m"]), 3),
        "age_check_max_gap_points": ages["max_abs_gap_points"],
        "spearman_state_coverage_vs_civil_suit_pending_3y": round(float(rho), 3),
        "states_in_spearman": int(len(st)),
    }
    (OUT / "coverage.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(cov.round(2).sort_values("coverage_2015_18").to_string())
    print(sp.round(3).to_string())
    print(nat.round(3).to_string())


if __name__ == "__main__":
    main()
