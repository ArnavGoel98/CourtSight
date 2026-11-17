"""A second, independent calculation of the headline figures, checked against the published results.

Nothing here imports the analysis code. Every rule (where each district's records end, when a district's records become
complete, the Kaplan-Meier estimate with delayed entry, the 20-cases-still-followed stop) is written again from its
description, with plain loops instead of the vectorised code in time_to_decision.py, waiting_years.py, hearings.py and
coverage_audit.py. If the two disagree, one of them is wrong.

Checked: every national and district time-to-decision figure, the decided-only medians, years of waiting, the hearing
stages and gaps, and the national coverage of official filings.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DDL, RES = ROOT / "ddl_compact", ROOT / "results"
H, MIN_AT_RISK, MIN_CASES = 120, 20, 200


def month(ym: np.ndarray) -> np.ndarray:
    """201503 -> months since year 0 (2015 * 12 + 2)."""
    y, m = np.divmod(ym, 100)
    return y * 12 + m - 1


def read(folder: str, cols: list[str]) -> dict[str, np.ndarray]:
    parts = [pq.read_table(p, columns=cols) for p in sorted((DDL / folder).glob("*.parquet"))]
    return {c: np.concatenate([t[c].to_numpy(zero_copy_only=False) for t in parts]) for c in cols}


def csv_rows(path: Path) -> list[dict[str, str]]:
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(x: str) -> float:
    return float(x) if x not in ("", "nan") else math.nan


def close(a: float, b: float, tol: float = 5e-4) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol


# ---- rules, written again ------------------------------------------------------------------------------------------

def collection_months(dist: np.ndarray, d: np.ndarray, n: np.ndarray) -> dict[int, int]:
    """Per district: where its records end. The last month (2018-2021) whose decisions reach half the district's
    average 2018 month, and no later than the last month in which its state's decisions reach 20% of the state's
    average 2018 month. A district with no decisions in 2018 takes its state's month."""
    by_state: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    by_dist: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    ok = (d >= 201801) & (d <= 202112)
    for x, dd, nn in zip(dist[ok].tolist(), d[ok].tolist(), n[ok].tolist()):
        by_state[x // 1000][dd] += nn
        by_dist[x][dd] += nn
    as_month = lambda ym: (ym // 100) * 12 + ym % 100 - 1
    state_end = {}
    for s, by in by_state.items():
        m18 = [v for k, v in by.items() if k // 100 == 2018]
        if m18:
            state_end[s] = as_month(max(k for k, v in by.items() if v >= 0.2 * sum(m18) / len(m18)))
    out = {}
    for x in np.unique(dist).tolist():
        if x // 1000 not in state_end:
            continue
        by = by_dist.get(x, {})
        base = sum(v for k, v in by.items() if k // 100 == 2018) / 12
        own = [as_month(k) for k, v in by.items() if base > 0 and v >= 0.5 * base]
        out[x] = min([state_end[x // 1000]] + ([max(own)] if own else []))
    return out


def complete_from(dist: np.ndarray, f: np.ndarray, d: np.ndarray, n: np.ndarray) -> dict[int, float]:
    """Per district: the first month from which every six-month window up to December 2017 holds at least half the
    district's 2016-17 filing rate and (from July 2010) records decisions of at least a fifth of its filings."""
    lo = 2010 * 12
    ids = {x: i for i, x in enumerate(np.unique(dist).tolist())}
    row = np.array([ids[x] for x in dist.tolist()])
    fil, dec = np.zeros((len(ids), 96)), np.zeros((len(ids), 96))
    fm, dm = month(f) - lo, month(np.where(d > 0, d, 100)) - lo
    okf = (fm >= 0) & (fm < 96)
    np.add.at(fil, (row[okf], fm[okf]), n[okf])
    okd = (d > 0) & (dm >= 0) & (dm < 96)
    np.add.at(dec, (row[okd], dm[okd]), n[okd])
    out = {}
    for x, i in ids.items():
        if fil[i].sum() == 0:
            continue
        base = fil[i, 72:].mean()
        first = math.nan
        for m in range(90, -1, -1):                      # walk back from the last window while every window passes
            fw, dw = fil[i, m:m + 6].sum(), dec[i, m:m + 6].sum()
            if fw >= 3 * base and (dw >= 0.2 * fw or m < 6):
                first = lo + m
            else:
                break
        out[x] = first
    return out


def kaplan_meier(ent: np.ndarray, dur: np.ndarray, event: np.ndarray, n: np.ndarray) -> dict[str, float]:
    """Share still pending by months since filing. A case is followed from `ent` to `dur` months after filing."""
    at, ev = np.zeros((H + 1, H + 1)), np.zeros(H + 1)
    np.add.at(at, (ent, dur), n)
    np.add.at(ev, dur[event], n[event])
    s, surv, reach = 1.0, [], H
    for m in range(H + 1):
        risk = at[:m + 1, m:].sum()                      # entered by month m and not yet decided or lost to follow-up
        if risk < MIN_AT_RISK and reach == H:
            reach = m - 1
        if risk > 0:
            s *= 1 - ev[m] / risk
        surv.append(s)
    seen = surv[:reach + 1]
    first = lambda level: next((float(m) for m, v in enumerate(seen) if v <= level), math.nan)
    val = lambda m: surv[m] if m <= reach else math.nan
    return {"cases": float(n.sum()), "median_months": first(0.5), "p90_months": first(0.1), "followed_months": reach,
            "pending_after_1y": val(12), "pending_after_3y": val(36), "pending_after_5y": val(60)}


# ---- checks --------------------------------------------------------------------------------------------------------

def check_time_to_decision() -> list[str]:
    bad = []
    c = read("detail/cube_cat", ["district_id", "f", "d", "category", "n"])
    dist, f, d, n, cat = c["district_id"], c["f"], c["d"], c["n"].astype(np.int64), c["category"]
    cut_of, entry_of = collection_months(dist, d, n), complete_from(dist, f, d, n)
    cut = np.array([cut_of.get(x, -1) for x in dist.tolist()])
    entry = np.array([entry_of.get(x, math.nan) for x in dist.tolist()])
    fm = month(f)
    dm = month(np.where(d > 0, d, 100))
    decided = (d > 0) & (dm <= cut)
    dur = np.clip(np.where(decided, dm - fm, cut - fm), 0, H)
    in_data = (cut >= 0) & (fm <= cut)

    # Kaplan-Meier with delayed entry: national, and every district x case type
    usable = in_data & ~np.isnan(entry)
    ent = np.clip(np.nan_to_num(entry) - fm, 0, H).astype(np.int64)
    usable &= dur >= ent
    keys = ("median_months", "p90_months", "followed_months", "pending_after_1y", "pending_after_3y", "pending_after_5y")
    types = sorted(set(cat.tolist()))
    is_type = {t: cat == t for t in types}
    for t in types:
        k = usable & is_type[t]
        mine = kaplan_meier(ent[k], dur[k], decided[k], n[k])
        pub = next(r for r in csv_rows(RES / "time_to_decision" / "national_case_type.csv") if r["case_type"] == t)
        bad += [f"national {t} {x}: {mine[x]} vs published {pub[x]}" for x in ("cases", *keys)
                if not close(mine[x], num(pub[x]))]
    pub = {(int(r["district_id"]), r["case_type"]): r for r in csv_rows(RES / "time_to_decision" / "by_district_case_type.csv")}
    order = np.lexsort((cat, dist))
    order = order[usable[order]]
    cuts = np.flatnonzero((dist[order][1:] != dist[order][:-1]) | (cat[order][1:] != cat[order][:-1])) + 1
    seen = set()
    for idx in np.split(order, cuts):
        key = (int(dist[idx[0]]), str(cat[idx[0]]))
        if n[idx].sum() < MIN_CASES:
            if key in pub:
                bad.append(f"{key} is published but has under {MIN_CASES} cases")
            continue
        seen.add(key)
        if key not in pub:
            bad.append(f"{key} is missing from the published table")
            continue
        mine = kaplan_meier(ent[idx], dur[idx], decided[idx], n[idx])
        bad += [f"{key} {x}: {mine[x]} vs published {pub[key][x]}" for x in ("cases", *keys)
                if not close(mine[x], num(pub[key][x]))]
    bad += [f"{k} is published but was not computed" for k in set(pub) - seen]
    print(f"time to decision: {len(types)} national and {len(seen)} district figures recomputed")

    # the usual statistic (decided cases only) against Kaplan-Meier on exactly the same cases
    same = in_data & (fm >= np.nan_to_num(entry, nan=np.inf))
    for r in csv_rows(RES / "time_to_decision" / "national_decided_only_case_type.csv"):
        k = same & is_type[r["case_type"]]
        kd = k & decided
        waits = np.repeat(np.clip(dm[kd] - fm[kd], 0, None), n[kd])
        waits.sort()
        naive = float(waits[(len(waits) - 1) // 2]) if len(waits) % 2 else float(waits[len(waits) // 2 - 1])
        km = kaplan_meier(np.zeros(k.sum(), dtype=np.int64), dur[k], decided[k], n[k])["median_months"]
        if not close(naive, num(r["median_months_decided_only"])) or int(r["decided_cases"]) != int(n[kd].sum()):
            bad.append(f"decided-only {r['case_type']}: {naive} ({int(n[kd].sum())} cases) vs published "
                       f"{r['median_months_decided_only']} ({r['decided_cases']})")
        if not close(km, num(r["median_months_kaplan_meier_same_cases"])):
            bad.append(f"same-cases Kaplan-Meier {r['case_type']}: {km} vs {r['median_months_kaplan_meier_same_cases']}")
    return bad


def check_waiting_years() -> list[str]:
    c = read("cube", ["district_id", "f", "d", "n"])
    dist, f, d, n = c["district_id"], c["f"], c["d"], c["n"].astype(np.int64)
    cut_of = collection_months(dist, d, n)
    cut = np.array([cut_of.get(x, -1) for x in dist.tolist()])
    fm, dm = month(f), month(np.where(d > 0, d, 100))
    keep = (cut >= 0) & (fm <= cut)
    pending = keep & ~((d > 0) & (dm <= cut))
    age = cut - fm
    waited = np.where(pending, age, np.clip(dm - fm, 0, None))
    ages = np.repeat(age[pending], n[pending])
    mine = {"cases_filed": int(n[keep].sum()), "pending_at_collection": int(n[pending].sum()),
            "case_years_pending": round(float((n[pending] * age[pending]).sum() / 12)),
            "case_years_all": round(float((n[keep] * waited[keep]).sum() / 12)),
            "share_over_3y": round(float((ages >= 36).mean()), 4), "share_over_5y": round(float((ages >= 60).mean()), 4)}
    pub = json.loads((RES / "waiting_years" / "national.json").read_text())
    print(f"years of waiting: {len(mine)} figures recomputed")
    return [f"waiting years {k}: {v} vs published {pub[k]}" for k, v in mine.items() if not close(v, pub[k], 1e-4)]


def check_hearings() -> list[str]:
    s = read("", ["district_id", "stage", "n", "n_gap", "sum_log_gap"])
    group = {"service": "service", "charge": "charge", "evidence": "evidence", "interlocutory": "hearing",
             "arguments": "final", "judgment": "final", "unknown": "unclear"}
    pub = json.loads((RES / "hearings" / "national.json").read_text())
    names = {int(r["state_code"]): r["state_name"] for r in csv_rows(DDL / "district_key.csv")}
    stale = {k for k, v in names.items() if v in pub["stale_record_states_left_out"]}
    total, by, logs, gaps = 0, defaultdict(int), 0.0, 0
    for dist, stage, n, n_gap, sl in zip(s["district_id"].tolist(), s["stage"].tolist(), s["n"].tolist(),
                                         s["n_gap"].tolist(), s["sum_log_gap"].tolist()):
        total += n
        by[group[stage]] += n
        if dist // 1000 not in stale:
            logs, gaps = logs + sl, gaps + n_gap
    bad = [f"hearing stage {g}: {by[g] / total:.4f} vs published {v}" for g, v in pub["share_by_group"].items()
           if not close(by[g] / total, v)]
    if total != pub["pending_listed"]:
        bad.append(f"pending cases with a listing: {total} vs published {pub['pending_listed']}")
    gap = math.exp(logs / gaps)
    if not close(gap, pub["gap_days_national_without_stale_states"], 0.05):
        bad.append(f"typical gap: {gap:.1f} days vs published {pub['gap_days_national_without_stale_states']}")
    print("hearings: stage shares and the typical gap recomputed")
    return bad


def check_coverage() -> list[str]:
    """Records / official filings, 2015-2018, where official filings = pending now - pending a year before +
    disposals in the year, straight from the printed official tables."""
    sys.path.insert(0, str(ROOT))
    import official_series as official            # the transcribed tables only; no analysis code
    pend, disp = official.pending(), official.disposed()
    merged = {"Andhra Pradesh": "Andhra Pradesh + Telangana", "Telangana": "Andhra Pradesh + Telangana",
              "Orissa": "Odisha", "Jammu and Kashmir": "Jammu & Kashmir + Ladakh",
              "Diu and Daman": "Dadra & Nagar Haveli + Daman & Diu", "DNH at Silvasa": "Dadra & Nagar Haveli + Daman & Diu"}
    names = {int(r["state_code"]): merged.get(r["state_name"], r["state_name"]) for r in csv_rows(DDL / "district_key.csv")}
    c = read("cube", ["district_id", "f", "n"])
    recorded: dict[tuple[str, int], int] = defaultdict(int)
    for dist, f, n in zip(c["district_id"].tolist(), c["f"].tolist(), c["n"].tolist()):
        if dist // 1000 in names:                 # 927 cases carry a state code the key does not name
            recorded[(names[dist // 1000], f // 100)] += n
    rec = off = 0.0
    for unit in pend.index:
        for y in (2015, 2016, 2017, 2018):
            implied = pend.loc[unit, y] - pend.loc[unit, y - 1] + disp.loc[unit, y]
            if not math.isnan(implied):
                off += implied
                rec += recorded.get((unit, y), 0)
    pub = json.loads((RES / "audit" / "coverage.json").read_text())["coverage_2015_18"]
    print("coverage: records against official filings recomputed")
    return [] if close(rec / off, pub, 2e-3) else [f"coverage 2015-18: {rec / off:.3f} vs published {pub}"]


if __name__ == "__main__":
    problems = check_time_to_decision() + check_waiting_years() + check_hearings() + check_coverage()
    for p in problems[:40]:
        print("MISMATCH", p)
    if problems:
        sys.exit(f"{len(problems)} figures disagree with the independent calculation")
    print("independent recalculation agrees with the published results")
