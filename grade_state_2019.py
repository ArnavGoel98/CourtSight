#!/usr/bin/env python3
"""Grade the committed Dec-2018-origin forecasts against official 2019 outcomes, state by state.

Official source (public, free): Lok Sabha Unstarred Question 1838, answered 16.12.2022, Ministry of Law & Justice,
Annexure-III (cases pending in District & Subordinate Courts at year end, 2014-2022) and Annexure-VIII (cases
disposed in District & Subordinate Courts per year, 2014-2022); figures from the Supreme Court of India / NJDG.
https://sansad.in/getFile/loksabhaquestions/annex/1710/AU1838.pdf?source=pqals

Actuals per state:  growth_2019 = log(P_2019 / P_2018);  filings_2019 = P_2019 - P_2018 + D_2019 (stock-flow
identity);  clearance_2019 = D_2019 / filings_2019.
Forecasts per state are built from the district forecasts: P_hat = sum_d P_d,2018 * exp(g_hat_d) and a
filing-weighted clearance ratio.

Definitions differ: official counts include cases filed before 2010, which DDL (and so the model) never sees, and
that older stock mostly shrinks. Every forecaster here (model, linear, trend) shares this gap, so the fair tests are
(1) which forecaster is closer, (2) whether it ranks states correctly (Spearman), and (3) error after removing the
common level bias. Andhra Pradesh and Telangana are combined (one row in the official 2018 data).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
SOURCE: Final = "Lok Sabha USQ 1838 (16.12.2022), Annexures III and VIII; Supreme Court of India / NJDG"
# eCourts state code -> official row name (AP + Telangana merged)
STATE_OF_CODE: Final = {
    13: "Uttar Pradesh", 1: "Maharashtra", 8: "Bihar", 3: "Karnataka", 9: "Rajasthan", 23: "Madhya Pradesh",
    4: "Kerala", 16: "West Bengal", 17: "Gujarat", 10: "Tamil Nadu", 11: "Odisha", 14: "Haryana", 26: "Delhi",
    22: "Punjab", 2: "Andhra Pradesh + Telangana", 29: "Andhra Pradesh + Telangana", 7: "Jharkhand", 6: "Assam",
    18: "Chhattisgarh", 5: "Himachal Pradesh", 15: "Uttarakhand", 12: "Jammu & Kashmir", 30: "Goa",
    27: "Chandigarh", 20: "Tripura", 25: "Manipur", 21: "Meghalaya", 19: "Mizoram", 24: "Sikkim",
}
# (pending end-2018, pending end-2019, disposed during 2019) from the annexures
OFFICIAL: Final = {
    "Uttar Pradesh": (6987417, 7807863, 3426942),
    "Andhra Pradesh + Telangana": (1068400, 567096 + 580193, 364947 + 331963),
    "Maharashtra": (3531425, 3821487, 1877895),
    "Goa": (42783, 49049, 32634),
    "West Bengal": (1950492, 2048697, 683238),
    "Chhattisgarh": (267429, 285025, 214399),
    "Delhi": (834813, 882366, 814555),
    "Gujarat": (1447459, 1595813, 1142383),
    "Assam": (291960, 301427, 254823),
    "Meghalaya": (13584, 13673, 7890),
    "Manipur": (6216, 6516, 3717),
    "Tripura": (58261, 27491, 90786),
    "Mizoram": (6154, 6589, 15107),
    "Himachal Pradesh": (256640, 293706, 483869),
    "Jammu & Kashmir": (163520, 172769, 81520),
    "Jharkhand": (330607, 365642, 187370),
    "Karnataka": (1494608, 1531008, 1272673),
    "Kerala": (1652509, 1614277, 1005350),
    "Madhya Pradesh": (1354602, 1455435, 1207541),
    "Tamil Nadu": (1084286, 1137684, 849240),
    "Odisha": (1319031, 1433522, 296535),
    "Bihar": (2502204, 2714344, 405347),
    "Punjab": (602014, 642327, 670175),
    "Haryana": (728097, 853375, 614384),
    "Chandigarh": (56357, 62955, 146256),
    "Rajasthan": (1732308, 1769823, 1508232),
    "Sikkim": (1208, 1142, 1906),
    "Uttarakhand": (232338, 195281, 341452),
}
# official pending at end of 2017 and disposed during 2018 (same annexures): for naive baselines built on the
# official series itself, which is the benchmark a skeptic would use
OFFICIAL_2017_D2018: Final = {
    "Uttar Pradesh": (6390684, 3282885), "Andhra Pradesh + Telangana": (1040864, 741390),
    "Maharashtra": (3340050, 2196271), "Goa": (39249, 36235), "West Bengal": (2141254, 1016319),
    "Chhattisgarh": (277338, 229548), "Delhi": (747704, 808156), "Gujarat": (1555203, 1418688),
    "Assam": (276520, 311150), "Meghalaya": (14775, 8517), "Manipur": (6799, 4379), "Tripura": (107089, 139931),
    "Mizoram": (5148, 12563), "Himachal Pradesh": (234639, 343667), "Jammu & Kashmir": (161674, 146194),
    "Jharkhand": (338680, 194200), "Karnataka": (1432952, 1120397), "Kerala": (1623212, 961840),
    "Madhya Pradesh": (1332566, 1386280), "Tamil Nadu": (1065878, 906184), "Odisha": (1178882, 255005),
    "Bihar": (2223744, 361063), "Punjab": (572802, 712529), "Haryana": (643394, 628939),
    "Chandigarh": (41695, 139172), "Rajasthan": (1635389, 1468290), "Sikkim": (1405, 2440),
    "Uttarakhand": (210018, 288999),
}
FORECASTERS: Final = {"model": "", "linear": "linear_", "drift": "drift_"}
# naive forecasts from the official series: each state repeats its own 2018; every state gets the 2018 median
BASELINES: Final = ("official_last_year", "official_common")


def official_frame() -> pd.DataFrame:
    o = pd.DataFrame.from_dict(OFFICIAL, orient="index", columns=["P2018", "P2019", "D2019"])
    o["A2019"] = o["P2019"] - o["P2018"] + o["D2019"]
    o["growth_actual"] = np.log(o["P2019"] / o["P2018"])
    o["cr_actual"] = np.where(o["A2019"] > 0, o["D2019"] / o["A2019"], np.nan)
    h = pd.DataFrame.from_dict(OFFICIAL_2017_D2018, orient="index", columns=["P2017", "D2018"])
    o = o.join(h)
    a18 = o["P2018"] - o["P2017"] + o["D2018"]
    o["growth_official_last_year"] = np.log(o["P2018"] / o["P2017"])
    o["cr_official_last_year"] = np.where(a18 > 0, o["D2018"] / a18, np.nan)
    return o


def state_forecasts(fc: pd.DataFrame, origin: pd.DataFrame) -> pd.DataFrame:
    df = fc.merge(origin[["district_id", "pending", "instituted_12"]], on="district_id")
    df["state"] = (df["district_id"] // 1000).map(STATE_OF_CODE)
    df = df.loc[df["state"].notna()]
    out = {}
    for name, pre in FORECASTERS.items():
        g, c = f"growth_12_{pre}q50", f"cr_12_{pre}q50"
        if g not in df:
            continue
        p1 = df["pending"] * np.exp(df[g])
        w = df["instituted_12"].where(df[c].notna(), 0.0)
        agg = pd.DataFrame({"p0": df["pending"], "p1": p1, "cw": df[c].fillna(0) * w, "w": w, "state": df["state"]})
        s = agg.groupby("state").sum()
        out[f"growth_{name}"] = np.log(s["p1"] / s["p0"])
        out[f"cr_{name}"] = s["cw"] / s["w"]
    out["ddl_pending_2018"] = df.groupby("state")["pending"].sum()
    return pd.DataFrame(out)


def score(t: pd.DataFrame, target: str, names: list[str], subset: str, draws: int = 5000,
          seed: int = 0) -> list[dict[str, object]]:
    """MAE, bias, rank correlation; error reduction vs the trend baseline with a 95% bootstrap range over states."""
    rows = []
    y = t[f"{target}_actual"].to_numpy()
    err = {n: np.abs(t[f"{target}_{n}"].to_numpy() - y) for n in names}
    idx = np.random.default_rng(seed).integers(0, len(y), (draws, len(y)))
    for n in names:
        e = t[f"{target}_{n}"].to_numpy() - y
        row: dict[str, object] = {
            "subset": subset, "target": target, "forecaster": n, "states": int(len(y)),
            "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
            "mae_bias_removed": float(np.abs(e - e.mean()).mean()),
            "spearman_vs_actual": float(t[f"{target}_{n}"].rank().corr(t[f"{target}_actual"].rank())),
        }
        for ref in ("drift", "official_common", "official_last_year"):
            if n == ref or ref not in err or n in BASELINES:
                continue
            boot = 1.0 - err[n][idx].sum(1) / err[ref][idx].sum(1)
            row |= {f"error_reduction_vs_{ref}": float(1.0 - err[n].sum() / err[ref].sum()),
                    f"error_reduction_vs_{ref}_lo": float(np.quantile(boot, 0.025)),
                    f"error_reduction_vs_{ref}_hi": float(np.quantile(boot, 0.975))}
        w = t["P2018"].to_numpy()
        row["mae_case_weighted"] = float((np.abs(e) * w).sum() / w.sum())
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results" / "ddl_2010_2018")
    ap.add_argument("--forecasts", type=Path, default=None)
    ap.add_argument("--min-coverage", type=float, default=0.3,
                    help="drop states where DDL's post-2010 stock is < this share of the official 2018 stock")
    args = ap.parse_args()
    out = args.results / "grading_2019_state"
    out.mkdir(parents=True, exist_ok=True)
    fc = pd.read_csv(args.forecasts or args.results / "forecasts.csv")
    t = official_frame().join(state_forecasts(fc, pd.read_csv(args.results / "origin_state.csv")), how="inner")
    t["ddl_coverage_2018"] = t["ddl_pending_2018"] / t["P2018"]
    keep = t["ddl_coverage_2018"].between(args.min_coverage, 1.5) & t["cr_actual"].between(0.0, 3.0)
    t["graded"] = keep
    t.to_csv(out / "state_table.csv", index_label="state")
    g = t.loc[keep].copy()
    for target in ("growth", "cr"):  # one number for every state: the 2018 cross-state median
        g[f"{target}_official_common"] = g[f"{target}_official_last_year"].median()
    names = [n for n in FORECASTERS if f"growth_{n}" in g] + list(BASELINES)
    # robustness: drop states whose official series jumps implausibly (data clean-ups, not court activity)
    robust = g.loc[g["growth_actual"] >= -0.10]  # official stock falling >10% in a year
    rows = []
    for sub, lab in ((g, "all_graded"), (robust, "excluding_official_jumps")):
        rows += score(sub, "growth", names, lab) + score(sub, "cr", names, lab)
    summary = pd.DataFrame(rows)
    nat = {f"national_growth_{n}": float(np.log((g["ddl_pending_2018"] * np.exp(g[f"growth_{n}"])).sum()
                                                 / g["ddl_pending_2018"].sum())) for n in FORECASTERS}
    nat["national_growth_actual"] = float(np.log(g["P2019"].sum() / g["P2018"].sum()))
    summary.to_csv(out / "summary.csv", index=False)
    meta = {"source": SOURCE, "states_graded": int(keep.sum()),
            "states_dropped": t.index[~keep].tolist(),
            "official_cases_in_graded_states_2018": int(g["P2018"].sum()),
            "share_of_national_2018": float(g["P2018"].sum() / 30074590),
            "states_excluded_in_robustness": g.index[g["growth_actual"] < -0.10].tolist(), **nat}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 200)
    print(json.dumps(meta, indent=2))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
