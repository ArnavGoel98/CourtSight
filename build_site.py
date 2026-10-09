#!/usr/bin/env python3
"""Build the static project site in docs/ (served by GitHub Pages) from the committed results.

  docs/index.html       the findings, with charts: national backlog, COVID excess, the locked 2026 forecast,
                        the backlog clock by state, time to decision by case type, and the judge-handover placebo
  docs/calculator.html  "how long will my case take?": survival curves for every district x case type

Both pages are single files: the data is embedded as JSON (aggregates only, no case-level rows) and the charts are
drawn in the browser as SVG, with no external scripts. Templates: site/*.html, shared styles: site/style.css.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from official_series import pending

HERE: Final = Path(__file__).resolve().parent
RES: Final = HERE / "results"
SITE: Final = HERE / "site"
LABELS: Final = {
    "civ_suit": "Civil suit (property, contract, money)",
    "civ_execution": "Execution (enforcing a decree)",
    "civ_family": "Divorce / family",
    "civ_mact": "Motor accident claim",
    "civ_misc": "Other civil application",
    "crim_bail": "Bail application",
    "crim_magisterial": "Criminal case before a magistrate",
    "crim_sessions": "Sessions trial (serious crime)",
    "crim_ni138": "Cheque bounce (NI Act s.138)",
    "other": "Other / unclassified",
}
SHORT: Final = {"civ_suit": "Civil suit", "civ_execution": "Execution", "civ_family": "Family", "civ_mact": "Motor accident",
                "civ_misc": "Civil misc.", "crim_bail": "Bail", "crim_magisterial": "Magistrate criminal",
                "crim_sessions": "Sessions trial", "crim_ni138": "Cheque bounce", "other": "Other"}
STATS: Final = ["cases", "median_months", "p90_months", "pending_after_1y", "pending_after_3y", "pending_after_5y"]


def _num(v: object, nd: int = 3) -> object:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    return round(float(v), nd)


def _curve(s: str | float) -> list[float | None]:
    if not isinstance(s, str):
        return []
    return [float(x) if x else None for x in s.split(" ")]


def site_data() -> dict[str, object]:
    p = pending().sum()
    covid = pd.read_csv(RES / "covid_excess" / "national.csv")
    fmeta = json.loads((RES / "forecast_2026" / "forecast_meta.json").read_text())
    clock = pd.read_csv(RES / "backlog_clock" / "states.csv")
    nat_clock = pd.read_csv(RES / "backlog_clock" / "national.csv").iloc[0]
    ttd = pd.read_csv(RES / "time_to_decision" / "national_case_type.csv")
    curves = pd.read_csv(RES / "time_to_decision" / "curves_national_case_type.csv")
    jt = json.loads((RES / "judge_transfer" / "summary.json").read_text())
    bench = pd.read_csv(RES / "ddl_2010_2018" / "naive_benchmarks.csv")
    locked = {n: hashlib.sha256((RES / "forecast_2026" / n).read_bytes()).hexdigest()
              for n in ("forecast_states_2026.csv", "forecast_meta.json")}
    med = bench.loc[bench["tau"] == 0.5]
    return {
        "pending": {str(k): int(v) for k, v in p.items()},
        "covid": [{"year": int(r.year), "actual": int(r.actual), "cf": int(r.counterfactual),
                   "lo": int(r.counterfactual_low), "hi": int(r.counterfactual_high), "excess": int(r.excess),
                   "excess_lo": int(r.excess_low),
                   "excess_hi": int(r.excess_high)} for r in covid.itertuples()],
        "forecast": {**fmeta["national"], "method": fmeta["chosen_method"], "sha": locked, "commit": "84732fc",
                     "locked_on": "8 October 2026"},
        "clock": [{"state": r.state, "pending": int(r.pending_2025), "pct": _num(r.change_per_year_pct, 4),
                   "per_year": int(r.change_per_year), "years": _num(r.years_to_clear, 1)}
                  for r in clock.itertuples()],
        "clock_nat": {"per_year": int(nat_clock["change_per_year"]),
                      "clear10": int(nat_clock["extra_disposals_to_clear_in_10y"]),
                      "growing": int((clock["status"] == "growing").sum()), "units": int(len(clock))},
        "ttd": [{"type": r.case_type, "label": LABELS.get(r.case_type, r.case_type), "short": SHORT.get(r.case_type, r.case_type),
                 **{c: _num(getattr(r, c)) for c in STATS}} for r in ttd.itertuples()],
        "curves": {r.case_type: _curve(r.curve_q) for r in curves.itertuples()},
        "judge": {"events": jt["events"], "districts": jt["districts"], "link": _num(jt["courtroom_link_rate"]),
                  "true": [v[0] for v in jt["effect_by_month"].values()],
                  "lo": [v[1] for v in jt["effect_by_month"].values()], "hi": [v[2] for v in jt["effect_by_month"].values()],
                  "placebo": jt.get("placebo_effect_by_month", {}), "means": jt.get("placebo_mean_post_effect", {})},
        "model": {"skill_vs_best_naive": [_num(med["skill_vs_best_naive"].min(), 2), _num(med["skill_vs_best_naive"].max(), 2)]},
    }


def calc_data() -> dict[str, object]:
    t = pd.read_csv(RES / "time_to_decision" / "by_district_case_type.csv").dropna(subset=["state_name", "district_name"])
    cv = pd.read_csv(RES / "time_to_decision" / "curves_by_district_case_type.csv.gz")
    t = t.merge(cv, on=["district_id", "case_type"], how="left")
    nat = pd.read_csv(RES / "time_to_decision" / "national_case_type.csv").merge(
        pd.read_csv(RES / "time_to_decision" / "curves_national_case_type.csv"), on="case_type")

    def row(r: object) -> list[object]:
        return [r.case_type] + [_num(getattr(r, c)) for c in STATS] + [_curve(r.curve_q)]

    states: dict[str, dict[str, list[list[object]]]] = {}
    for (s, d), g in t.sort_values(["state_name", "district_name"]).groupby(["state_name", "district_name"], sort=False):
        states.setdefault(s, {})[d] = [row(r) for r in g.itertuples()]
    return {"labels": LABELS, "cols": ["type"] + STATS + ["curve"], "national": [row(r) for r in nat.itertuples()],
            "states": states}


def render(template: str, data: dict[str, object], out: Path) -> None:
    css = (SITE / "style.css").read_text()
    charts = (SITE / "charts.js").read_text()
    page = ((SITE / template).read_text().replace("/*CSS*/", css).replace("/*CHARTS*/", charts)
            .replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    out.write_text(page)
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size / 1e3:.0f} kB)")


def main() -> None:
    (HERE / "docs").mkdir(exist_ok=True)
    render("index.html", site_data(), HERE / "docs" / "index.html")
    render("calculator.html", calc_data(), HERE / "docs" / "calculator.html")


if __name__ == "__main__":
    main()
