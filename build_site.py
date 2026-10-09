#!/usr/bin/env python3
"""Build the static project site in docs/ (served by GitHub Pages) from the committed results.

  docs/index.html       the findings, with charts: national backlog, COVID excess, the locked 2026 forecast,
                        the backlog clock by state, time to decision by case type, and the judge-handover placebo
  docs/calculator.html  "how long will my case take?": survival curves for every district x case type
  docs/paper.html       the working paper (make_paper_pdf.py prints it to docs/courtsight-paper.pdf)
  docs/calculator-hi.html  the same calculator in Hindi

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
    wy = json.loads((RES / "waiting_years" / "national.json").read_text())
    age = pd.read_csv(RES / "waiting_years" / "age_profile.csv")
    dd = pd.read_csv(RES / "district_drivers" / "districts.csv").dropna(subset=["district_name", "pending_after_3y_civil"])
    dd = dd.merge(pd.read_csv(RES / "waiting_years" / "districts.csv")[["district_id", "case_years"]], on="district_id")
    co = pd.read_csv(RES / "district_drivers" / "coefficients.csv")
    dsum = json.loads((RES / "district_drivers" / "summary.json").read_text())
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
        "waiting": {**{k: v for k, v in wy.items() if k != "collection_months"},
                    "age": {"civil": age["civil"].tolist(), "criminal": age["criminal"].tolist()}},
        "districts": [[r.state_name, r.district_name, _num(r.pending_after_3y_civil), _num(r.pending_after_3y_criminal),
                       _num(r.median_months_civil), _num(r.judge_tenure_months, 1), int(r.case_years)]
                      for r in dd.sort_values(["pending_after_3y_civil", "state_name", "district_name"], kind="mergesort").itertuples()],
        "drivers": {"coef": [{k: (_num(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                             for r in co.to_dict("records")], "summary": dsum},
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


def paper_vars(d: dict) -> dict[str, str]:
    """Numbers quoted in the paper's prose, all taken from the same data as its figures."""
    P, ttd, W, F = d["pending"], {r["type"]: r for r in d["ttd"]}, d["waiting"], d["forecast"]
    nat = pd.read_csv(RES / "time_to_decision" / "national_civil_criminal.csv").set_index("case_type")
    wy = json.loads((RES / "waiting_years" / "national.json").read_text())
    c21 = next(r for r in d["covid"] if r["year"] == 2021)
    cov = pd.read_csv(RES / "covid_excess" / "national.csv").set_index("year")
    clock = pd.read_csv(RES / "backlog_clock" / "national.csv").iloc[0]
    sm, co = d["drivers"]["summary"], {(r["outcome"], r["model"], r["feature"]): r for r in d["drivers"]["coef"]}
    c = lambda f: co[("civil", "structural", f)]
    pp = lambda v: f"{v * 100:+.1f}".replace("-", "−")
    yrs = sorted(P, key=int)
    cr = lambda v: f"{v / 1e7:.2f}"
    J = d["judge"]["means"]
    return {
        "date": "October 2026", "first": yrs[0], "last": yrs[-1], "pending_first": cr(P[yrs[0]]), "pending_last": cr(P[yrs[-1]]),
        "cases_m": f"{W['cases_filed'] / 1e6:.1f}", "districts_ttd": "626",
        "coll_first": "December 2018", "coll_last": "July 2020",
        "civ_med": f"{ttd['civ_suit']['median_months']:.0f}", "civ_5y": f"{ttd['civ_suit']['pending_after_5y'] * 100:.0f}",
        "fam_med": f"{ttd['civ_family']['median_months']:.0f}",
        "civil_med": f"{nat.loc['civil', 'median_months']:.0f}", "crim_med": f"{nat.loc['criminal', 'median_months']:.0f}",
        "crim_5y": f"{nat.loc['criminal', 'pending_after_5y'] * 100:.0f}",
        "cy_all": f"{W['case_years_all'] / 1e7:.1f}", "cy_pending": f"{W['case_years_pending'] / 1e7:.1f}",
        "pend_cr": f"{W['pending_at_collection'] / 1e7:.1f}", "age_med": f"{W['median_age_years'] + 1e-9:.1f}",
        "age_mean": f"{W['mean_age_years']:.1f}", "over3": f"{W['share_over_3y'] * 100:.0f}", "over5": f"{W['share_over_5y'] * 100:.0f}",
        "covid21": f"{c21['excess'] / 1e5:.0f}", "covid21_lo": f"{c21['excess_lo'] / 1e5:.0f}", "covid21_hi": f"{c21['excess_hi'] / 1e5:.0f}",
        "covid21_share": f"{cov.loc[2021, 'excess_share_of_backlog'] * 100:.0f}",
        "per_year": f"{clock['change_per_year'] / 1e5:.0f}", "clear10": f"{clock['extra_disposals_to_clear_in_10y'] / 1e5:.0f}",
        "clear10_pct": f"{clock['clear_in_10y_vs_2021_disposals'] * 100:.0f}",
        "drv_n": str(sm["civil"]["districts"]), "drv_p10": f"{sm['civil']['p10_p90_pending_3y'][0] * 100:.0f}",
        "drv_p90": f"{sm['civil']['p10_p90_pending_3y'][1] * 100:.0f}",
        "load_b": pp(c("log_filings_per_court")["coef_per_sd"]), "load_lo": pp(c("log_filings_per_court")["lo"]),
        "load_hi": pp(c("log_filings_per_court")["hi"]), "ten_b": pp(c("judge_tenure_months")["coef_per_sd"]),
        "ten_sd": f"{sm['feature_sds']['judge_tenure_months']:.0f}", "crim_b": pp(c("criminal_share")["coef_per_sd"]),
        "r2s": f"{sm['civil']['r2_structural'] * 100:.0f}", "r2t": f"{sm['civil']['r2_with_throughput'] * 100:.0f}",
        "j_events": f"{d['judge']['events']:,}", "j_districts": str(d["judge"]["districts"]),
        "j_true": f"{-J['true_date'] * 100:.0f}", "j_p18": f"{-J['18_months_earlier'] * 100:.0f}",
        "j_p30": f"{-J['30_months_earlier'] * 100:.0f}", "j_link": f"{d['judge']['link'] * 100:.0f}",
        "skill_lo": f"{d['model']['skill_vs_best_naive'][0] * 100:.0f}", "skill_hi": f"{d['model']['skill_vs_best_naive'][1] * 100:.0f}",
        "f50": cr(F["q50"]), "f10": cr(F["q10"]), "f90": cr(F["q90"]), "locked_on": F["locked_on"], "commit": F["commit"],
        "sha": F["sha"]["forecast_states_2026.csv"],
    }


def render(template: str, data: dict[str, object], out: Path, lang: str = "en", fill: dict[str, str] | None = None) -> None:
    css = (SITE / "style.css").read_text()
    charts = (SITE / "charts.js").read_text()
    page = ((SITE / template).read_text().replace("/*CSS*/", css).replace("/*CHARTS*/", charts)
            .replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))).replace('"/*LANG*/en"', json.dumps(lang)))
    for k, v in (fill or {}).items():
        page = page.replace("{{" + k + "}}", v)
    assert "{{" not in page, f"unfilled placeholder in {template}"
    if lang != "en":
        page = page.replace('<html lang="en">', f'<html lang="{lang}">')
    out.write_text(page)
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size / 1e3:.0f} kB)")


def main() -> None:
    (HERE / "docs").mkdir(exist_ok=True)
    sd = site_data()
    render("index.html", sd, HERE / "docs" / "index.html")
    render("paper.html", sd, HERE / "docs" / "paper.html", fill=paper_vars(sd))
    cd = calc_data()
    render("calculator.html", cd, HERE / "docs" / "calculator.html")
    render("calculator.html", cd, HERE / "docs" / "calculator-hi.html", lang="hi")


if __name__ == "__main__":
    main()
