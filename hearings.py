#!/usr/bin/env python3
"""Where pending cases are stuck, and how far apart their hearings are scheduled.

For every case still pending when its state's data was collected, the DDL records give the date it was last listed
for a hearing, the date of its next listing, and the purpose of that listing (free text typed by court staff, e.g.
"summons", "evidence", "argument"). compress_ddl.py grouped the purposes into stages (taxonomy.STAGE_PATTERNS) and kept
counts and the sum of log(next - last, in days, clipped to 1..730) per district x month x stage
(ddl_compact/snapshot.parquet), plus every distinct purpose label with its count (audit_purpose_labels.csv).

1. Stages. The labels are reported in six groups named for what they mostly contain, not for the regex names.
   The label audit shows what each group holds and what is misfiled (for example police "final form" reports,
   which come before a trial, fall in "arguments, orders, judgment").
2. Hearing gap: the typical (geometric mean) number of days between a pending case's last listing and its next one,
   by stage and by district, and a stage-adjusted version that gives every district the national stage mix. Only
   cases pending at collection are covered, so these are the cases still in the system, not all cases.
3. Do districts with longer gaps take longer? Spearman correlation with each district's share of cases pending
   after 3 years (time_to_decision.py), the same by fifths of districts, and an OLS with the structural features of
   district_drivers.py (caseload per courtroom, size, criminal share, judge tenure, records-complete year) to check
   that the gap is not a stand-in for caseload. States whose hearing records look stale (West Bengal, Odisha:
   see coverage_audit.py) are left out, since a stale record can show a long gap that never happened; the
   correlation with them included, and within states only, is reported alongside. Correlation, not cause: congested
   courts may space hearings out.

Outputs (results/hearings/): national.json, by_stage.csv, districts.csv, labels_top.csv, gap_fifths.csv,
coefficients.csv
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from district_drivers import STRUCTURAL, ols_boot

HERE: Final = Path(__file__).resolve().parent
DDL: Final = HERE / "ddl_compact"
OUT: Final = HERE / "results" / "hearings"
MIN_LISTED: Final = 1000      # districts with fewer pending cases that have a next date are left out of comparisons
STALE: Final = 0.2            # states where this share of pending cases had no hearing listed in the year before
                              # collection (coverage_audit.py) have stale hearing records; left out of comparisons
# reporting groups: (key, name, snapshot stages, audit-file stage names)
GROUPS: Final = [
    ("service", "Getting the parties to court (summons, notices, appearance, warrants)", ["service"], ["service"]),
    ("charge", "Framing of charges or issues", ["charge"], ["charge"]),
    ("evidence", "Evidence and witnesses", ["evidence"], ["evidence"]),
    ("hearing", "Applications and general hearings", ["interlocutory"], ["interlocutory"]),
    ("final", "Arguments, orders and judgment", ["arguments", "judgment"], ["arguments", "judgment"]),
    ("unclear", "Unclear, blank or not heard", ["unknown"], ["other/unknown"]),
]
POLICE_REPORT: Final = re.compile(r"final form|charge ?sheet|awa\w* .*final", re.I)


def gm(sum_log: pd.Series | float, n: pd.Series | float) -> pd.Series | float:
    """Geometric mean in days from the sum of log(days) and the count."""
    return np.exp(sum_log / n)


def stage_table(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, name, stages, _ in GROUPS:
        x = s.loc[s["stage"].isin(stages), ["n", "n_gap", "sum_log_gap"]].sum()
        rows.append({"group": key, "name": name, "pending_listed": int(x["n"]),
                     "share": x["n"] / s["n"].sum(),
                     "gap_days": gm(x["sum_log_gap"], x["n_gap"]) if x["n_gap"] else np.nan})
    return pd.DataFrame(rows)


def label_audit() -> tuple[pd.DataFrame, dict[str, float]]:
    a = pd.read_csv(DDL / "audit_purpose_labels.csv")
    to_group = {st: key for key, _, _, sts in GROUPS for st in sts}
    a["group"] = a["assigned"].map(to_group)
    top = (a.sort_values("n", ascending=False).groupby("group").head(8)
           .assign(share_of_group=lambda x: x["n"] / x["group"].map(a.groupby("group")["n"].sum())))
    total = a["n"].sum()
    police = a["label"].fillna("").str.contains(POLICE_REPORT) & (a["group"] == "final")
    checks = {"police_reports_filed_as_final": round(float(a.loc[police, "n"].sum() / total), 4),
              "generic_order_label_in_final": round(float(a.loc[(a["group"] == "final") & a["label"].fillna("")
                                                             .str.fullmatch(r"orders?"), "n"].sum() / total), 4),
              "blank_label": round(float(a.loc[a["label"].isna(), "n"].sum() / total), 4),
              "no_sitting_notified": round(float(a.loc[a["label"].fillna("").str.contains("no sitting"), "n"].sum()
                                                 / total), 4)}
    return top[["group", "label", "n", "share_of_group"]], checks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(DDL / "snapshot.parquet")
    by_stage = stage_table(s)
    by_stage.round(4).to_csv(OUT / "by_stage.csv", index=False)
    labels, checks = label_audit()
    labels.round(4).to_csv(OUT / "labels_top.csv", index=False)

    # districts
    d = s.groupby("district_id")[["n", "n_gap", "sum_log_gap"]].sum()
    d["gap_days"] = gm(d["sum_log_gap"], d["n_gap"])
    to_group = {st: key for key, _, stages, _ in GROUPS for st in stages}
    shares = (s.assign(group=s["stage"].map(to_group)).pivot_table(index="district_id", columns="group", values="n",
                                                                   aggfunc="sum").fillna(0))
    shares = shares.div(shares.sum(axis=1), axis=0).add_prefix("share_")
    d = d.join(shares).rename(columns={"n": "pending_listed", "n_gap": "with_next_date"})
    names = pd.read_csv(HERE / "results" / "time_to_decision" / "by_district_case_type.csv")
    names = names.dropna(subset=["district_name"]).groupby("district_id")[["state_name", "district_name"]].first()
    key = pd.read_csv(DDL / "district_key.csv").sort_values("year")
    key = key.assign(district_id=key["state_code"] * 1000 + key["dist_code"]).groupby("district_id")[
        ["state_name", "district_name"]].last()
    d = d.join(names.combine_first(key), how="left")
    d = d.drop(columns="sum_log_gap").reset_index()

    # the same gap with every district given the national stage mix, so a district is not "slow to list" merely
    # because more of its cases sit at stages that are listed less often
    cell = s.assign(group=s["stage"].map(to_group)).groupby(["district_id", "group"])[["n_gap", "sum_log_gap"]].sum()
    lg = (cell["sum_log_gap"] / cell["n_gap"]).unstack()
    wts = by_stage.set_index("group")["share"]
    nat_lg = np.log(by_stage.set_index("group")["gap_days"])
    adj = np.exp((lg.fillna(nat_lg) * wts).sum(axis=1) / wts.sum())    # a stage a district never lists: national gap
    d["gap_days_stage_adjusted"] = d["district_id"].map(adj)
    d.round(4).to_csv(OUT / "districts.csv", index=False)

    # gap and delay
    ttd = pd.read_csv(HERE / "results" / "time_to_decision" / "by_district_civil_criminal.csv")
    ttd = ttd.loc[ttd["cases"] >= 2000].pivot(index="district_id", columns="case_type", values="pending_after_3y")
    act = pd.read_csv(HERE / "results" / "audit" / "pending_activity.csv").set_index("state")["not_listed_in_12m"]
    stale_states = sorted(act.index[(act >= STALE) & (act.index != "All states")])
    every = d.set_index("district_id").join(ttd, how="inner")
    every = every.loc[every["with_next_date"] >= MIN_LISTED]
    rho_all = {k: round(float(spearmanr(*every[["gap_days", k]].dropna().to_numpy().T).statistic), 3)
               for k in ("civil", "criminal")}
    x = every.loc[~every["state_name"].isin(stale_states)].copy()
    within = {}
    for k in ("civil", "criminal"):   # ranks within each state: compares districts that share a record system
        r = x[["state_name", "gap_days", k]].dropna()
        g_r, o_r = r.groupby("state_name")["gap_days"].rank(pct=True), r.groupby("state_name")[k].rank(pct=True)
        within[k] = round(float(spearmanr(g_r, o_r).statistic), 3)
    x["fifth"] = pd.qcut(x["gap_days"], 5, labels=False) + 1
    fifths = x.groupby("fifth").agg(districts=("gap_days", "size"), gap_days=("gap_days", "median"),
                                    pending_3y_civil=("civil", "median"), pending_3y_criminal=("criminal", "median"))
    fifths.round(4).reset_index().to_csv(OUT / "gap_fifths.csv", index=False)
    rho = {k: round(float(spearmanr(*x[["gap_days", k]].dropna().to_numpy().T).statistic), 3) for k in ("civil", "criminal")}
    rho_adj = {k: round(float(spearmanr(*x[["gap_days_stage_adjusted", k]].dropna().to_numpy().T).statistic), 3)
               for k in ("civil", "criminal")}
    rho_service = {k: round(float(spearmanr(*x[["share_service", k]].dropna().to_numpy().T).statistic), 3)
                   for k in ("civil", "criminal")}

    # is the gap just caseload? OLS with district_drivers' structural features
    f = pd.read_csv(HERE / "results" / "district_drivers" / "districts.csv").set_index("district_id")[STRUCTURAL]
    z = x.join(f, how="inner").assign(log_gap_days=lambda v: np.log(v["gap_days"]))
    feats = ["log_gap_days", *STRUCTURAL]
    rows, r2 = [], {}
    for kind in ("civil", "criminal"):
        ok = z[[kind, *feats]].dropna()
        X = ok[feats]
        beta, ci, r2[kind] = ols_boot(((X - X.mean()) / X.std()).to_numpy(), ok[kind].to_numpy())
        rows += [{"outcome": kind, "feature": n, "coef_per_sd": beta[i], "lo": ci[0, i], "hi": ci[1, i]}
                 for i, n in enumerate(feats)]
    co = pd.DataFrame(rows)
    co.round(4).to_csv(OUT / "coefficients.csv", index=False)
    load_rho = round(float(spearmanr(*z[["gap_days", "log_filings_per_court"]].dropna().to_numpy().T).statistic), 3)

    st_names = pd.read_csv(DDL / "district_key.csv").groupby("state_code")["state_name"].first()
    stale_codes = [c for c, n in st_names.items() if n in stale_states]
    tot = s[["n", "n_gap", "sum_log_gap"]].sum()
    g = by_stage.set_index("group")
    summary = {
        "pending_listed": int(tot["n"]), "share_with_next_date": round(float(tot["n_gap"] / tot["n"]), 4),
        "gap_days_national": round(float(gm(tot["sum_log_gap"], tot["n_gap"])), 1),
        "gap_days_district_p10_p50_p90": [round(float(v), 1) for v in d.loc[(d["with_next_date"] >= MIN_LISTED)
                                          & ~d["state_name"].isin(stale_states), "gap_days"].quantile([.1, .5, .9])],
        "share_by_group": {k: round(float(g.loc[k, "share"]), 4) for k in g.index},
        "gap_days_by_group": {k: round(float(g.loc[k, "gap_days"]), 1) for k in g.index},
        "stale_record_states_left_out": stale_states,
        "gap_days_national_without_stale_states": round(float(gm(
            s.loc[~(s["district_id"] // 1000).isin(stale_codes), "sum_log_gap"].sum(),
            s.loc[~(s["district_id"] // 1000).isin(stale_codes), "n_gap"].sum())), 1),
        "districts_compared": int(len(x)),
        "spearman_gap_vs_pending_3y": rho,
        "spearman_gap_vs_pending_3y_all_states": rho_all,
        "spearman_gap_vs_pending_3y_within_states": within, "spearman_service_share_vs_pending_3y": rho_service,
        "spearman_stage_adjusted_gap_vs_pending_3y": rho_adj,
        "spearman_gap_vs_caseload_per_court": load_rho,
        "r2_with_gap": {k: round(v, 3) for k, v in r2.items()},
        "label_checks": checks,
    }
    (OUT / "national.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(by_stage.round(3).to_string(index=False))
    print(fifths.round(3).to_string())
    print(co.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
