#!/usr/bin/env python3
"""Build the static project site in docs/ (served by GitHub Pages) from the committed results.

  docs/index.html       the findings, with charts: national backlog, COVID excess, the locked 2026 forecast,
                        the backlog clock by state, time to decision by case type, and the judge-handover placebo
  docs/calc-data.json   district x case type figures shared by the calculator (both languages) and the rankings page;
                        curves are stored as per-mille drops (see _delta)
  docs/districts.html   sortable, filterable rankings of every district, with CSV download
  docs/map.html, docs/map-data.json  the same figures on a map of India's 2011 Census districts (make_map.py);
                        docs/map-civil-suits.svg is a static version for the paper and README
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
BASE: Final = "https://arnavgoel98.github.io/CourtSight/"
PAGES: Final = ["", "calculator.html", "districts.html", "map.html", "paper.html", "calculator-hi.html",
                "courtsight-paper.pdf"]
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
        "audit": {**{k: v for k, v in json.loads((RES / "audit" / "summary.json").read_text()).items()},
                  "cohort": pd.read_csv(RES / "audit" / "cohort_speed.csv").replace({np.nan: None}).to_dict("list"),
                  "by_type": pd.read_csv(RES / "audit" / "correction_by_type.csv").set_index("case_type")
                  .loc[["civ_suit", "crim_magisterial", "crim_ni138", "crim_sessions", "civ_family"]]
                  .round(3).reset_index().to_dict("records"),
                  "coverage": json.loads((RES / "audit" / "coverage.json").read_text()),
                  "speedup": pd.read_csv(RES / "audit" / "speedup_by_year.csv").round(3).to_dict("list"),
                  "sens": pd.read_csv(RES / "audit" / "national_sensitivity.csv").set_index("case_type")
                  .loc[["civ_suit", "crim_magisterial"]].round(3).reset_index().to_dict("records")},
        "slow5": _slowest_fifth(),
        "district_keys": sorted({(r.state_name, r.district_name) for r in pd.read_csv(
            RES / "time_to_decision" / "by_district_case_type.csv").dropna(subset=["state_name", "district_name"]).itertuples()}),
        "model": {"skill_vs_best_naive": [_num(med["skill_vs_best_naive"].min(), 2), _num(med["skill_vs_best_naive"].max(), 2)]},
    }


def _slowest_fifth() -> dict[str, object]:
    """The state with the largest share of its districts in the slowest fifth for civil suits (pending at 3 years)."""
    t = pd.read_csv(RES / "time_to_decision" / "by_district_case_type.csv")
    t = t.loc[t["case_type"] == "civ_suit"].dropna(subset=["pending_after_3y", "state_name"])
    top = t["pending_after_3y"] >= t["pending_after_3y"].quantile(0.8)
    g = t.assign(top=top).groupby("state_name")["top"].agg(["sum", "count"])
    g = g.loc[g["count"] >= 10].assign(share=lambda x: x["sum"] / x["count"]).sort_values(["share", "count"], ascending=False)
    return {"state": g.index[0], "n": int(g["sum"].iloc[0]), "of": int(g["count"].iloc[0])}


def _delta(c: list[float | None]) -> list[int | None]:
    """Survival curve -> per-mille drops from 1000 (trailing gaps dropped). About half the size of the raw floats."""
    while c and c[-1] is None:
        c = c[:-1]
    out, prev = [], 1000
    for x in c:
        if x is None:
            out.append(None)
            continue
        v = round(x * 1000)
        out.append(prev - v)
        prev = v
    return out


def calc_data() -> dict[str, object]:
    t = pd.read_csv(RES / "time_to_decision" / "by_district_case_type.csv").dropna(subset=["state_name", "district_name"])
    cv = pd.read_csv(RES / "time_to_decision" / "curves_by_district_case_type.csv.gz")
    t = t.merge(cv, on=["district_id", "case_type"], how="left")
    t = t.merge(pd.read_csv(RES / "audit" / "rank_intervals.csv"), on=["district_id", "case_type"], how="left")
    dig = pd.read_csv(RES / "time_to_decision" / "digitisation.csv").set_index("district_id")["first_complete_year"]
    special = set(json.loads((RES / "audit" / "summary.json").read_text())["units_that_are_court_complexes"])
    nat = pd.read_csv(RES / "time_to_decision" / "national_case_type.csv").merge(
        pd.read_csv(RES / "time_to_decision" / "curves_national_case_type.csv"), on="case_type")
    rank_cols = ["rank", "rank_lo", "rank_hi", "of"]

    def row(r: object) -> list[object]:
        ranks = [int(getattr(r, c)) if hasattr(r, "rank") and pd.notna(getattr(r, "rank")) else None for c in rank_cols]
        return [r.case_type] + [_num(getattr(r, c)) for c in STATS] + ranks + [_delta(_curve(r.curve_q))]

    states: dict[str, dict[str, list[list[object]]]] = {}
    meta: dict[str, dict[str, list[object]]] = {}
    for (s, d), g in t.sort_values(["state_name", "district_name"]).groupby(["state_name", "district_name"], sort=False):
        states.setdefault(s, {})[d] = [row(r) for r in g.itertuples()]
        y = dig.get(int(g["district_id"].iloc[0]))
        meta.setdefault(s, {})[d] = [None if y is None or pd.isna(y) else int(y), int(d in special)]
    return {"labels": LABELS, "short": SHORT, "cols": ["type"] + STATS + rank_cols + ["curve"],
            "national": [row(r) for r in nat.itertuples()], "states": states, "meta": meta}


def map_data() -> dict[str, object]:
    """Census 2011 district shapes, and which DDL units (by the names used in calc-data.json) fall in each."""
    geo = json.loads((HERE / "sources" / "india_districts_2011.json").read_text())
    t = pd.read_csv(RES / "time_to_decision" / "by_district_case_type.csv").dropna(subset=["state_name", "district_name"])
    names = t.drop_duplicates("district_id").set_index("district_id")[["state_name", "district_name"]]
    key = pd.read_csv(HERE / "ddl_compact" / "district_key.csv").sort_values("year")
    key = key.assign(district_id=key["state_code"] * 1000 + key["dist_code"]).groupby("district_id").last()
    links = [(int(d), int(c), "") for d, c in key["pc11_district_id"].dropna().items()]
    cw = pd.read_csv(HERE / "sources" / "map_crosswalk.csv")
    links += [(int(r.district_id), int(r.census_2011_code), r.relation) for r in cw.itertuples()]
    units: dict[str, list[list[str]]] = {}
    for d, code, rel in sorted(links):
        if d in names.index:
            units.setdefault(str(code), []).append([*names.loc[d], rel])
    cov = pd.read_csv(RES / "audit" / "coverage_by_state.csv").set_index("state")
    cov = cov.loc[cov["group"] != "too small to classify", "coverage_2015_18"]   # official figures too noisy
    from coverage_audit import state_units
    su = state_units()
    ddl_states = key.groupby("state_code")["state_name"].first()
    coverage = {ddl_states[k]: _num(cov.get(v), 2) for k, v in su.items() if v in cov.index}
    return {"w": geo["w"], "h": geo["h"], "source": geo["source"], "names": geo["names"], "paths": geo["paths"],
            "units": units, "coverage": coverage}


def map_svg(m: dict[str, object], case_type: str = "civ_suit") -> str:
    """Static map of the share still pending after 3 years (paper and README figure). Shapes with several court
    units take their cases-weighted mean, which equals the pooled survival curve at 36 months, as on map.html."""
    t = pd.read_csv(RES / "time_to_decision" / "by_district_case_type.csv").dropna(subset=["state_name", "district_name"])
    t = t.loc[t["case_type"] == case_type].set_index(["state_name", "district_name"])
    val = {}
    for code, units in m["units"].items():
        r = t.reindex([(u[0], u[1]) for u in units]).dropna(subset=["pending_after_3y"])
        if len(r):
            val[code] = float((r["pending_after_3y"] * r["cases"]).sum() / r["cases"].sum())
    cuts = np.quantile(list(val.values()), [0.2, 0.4, 0.6, 0.8])
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
    shapes = "".join(f'<path d="{d}" fill="{ramp[int(np.searchsorted(cuts, val[c], side="right"))] if c in val else "#e9e7e1"}"/>'
                     for c, d in m["paths"].items())
    edges = [min(val.values()), *cuts, max(val.values())]
    w, h = m["w"], m["h"]
    legend = "".join(f'<rect x="{80 + i * 300}" y="{h + 40}" width="60" height="40" rx="8" fill="{ramp[i]}"/>'
                     f'<text x="{150 + i * 300}" y="{h + 72}">{edges[i] * 100:.0f}–{edges[i + 1] * 100:.0f}%</text>'
                     for i in range(5))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h + 280}" font-family="system-ui,sans-serif" '
            f'font-size="40" role="img"><title>Share of civil suits still pending 3 years after filing, by district, '
            f'2010–2018 filings</title><rect width="{w}" height="{h + 280}" fill="#fcfcfb"/>'
            f'<g stroke="#fcfcfb" stroke-width="1.6" stroke-linejoin="round">{shapes}</g>'
            f'<g fill="#52514e">{legend}<rect x="1580" y="{h + 40}" width="60" height="40" rx="8" fill="#e9e7e1"/>'
            f'<text x="1650" y="{h + 72}">no records</text>'
            f'<text x="80" y="{h + 160}" fill="#0b0b0b">Civil suits still pending 3 years after filing (fifths of districts)</text>'
            f'<text x="80" y="{h + 220}" font-size="34">Cases filed 2010–2018, DDL eCourts records. Boundaries: Census 2011 '
            f'via DataMeet (CC BY 4.0).</text></g></svg>\n')


def unit_counts() -> dict[str, str]:
    """How many districts and special courts have results (quoted on several pages)."""
    t = pd.read_csv(RES / "time_to_decision" / "by_district_case_type.csv").dropna(subset=["district_name"])
    special = len(json.loads((RES / "audit" / "summary.json").read_text())["units_that_are_court_complexes"])
    return {"n_dist": str(t["district_id"].nunique() - special), "n_special": str(special)}


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
        "cases_m": f"{W['cases_filed'] / 1e6:.1f}", "districts_ttd": unit_counts()["n_dist"],
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
        **audit_vars(d["audit"]),
    }


def audit_vars(a: dict) -> dict[str, str]:
    t = {r["case_type"]: r for r in a["by_type"]}["civ_suit"]
    pct = lambda v: f"{v * 100:.0f}"
    return {
        "au_f10": f"{a['recorded_filings_2010'] / 1e6:.1f}", "au_f18": f"{a['recorded_filings_2018'] / 1e6:.1f}",
        "au_b10": pct(a["cohort"]["before"][0]), "au_a10": pct(a["cohort"]["after_records_complete"][0]),
        "au_a17": pct(a["cohort"]["after_records_complete"][7]),
        "au_share": pct(a["share_of_cases_filed_before_records_complete"]),
        "au_civ_med_naive": f"{t['median_months__uncorrected']:.0f}", "au_civ_med": f"{t['median_months__corrected_50']:.0f}",
        "au_med30": f"{t['median_months__corrected_30']:.0f}", "au_med70": f"{t['median_months__corrected_70']:.0f}",
        "au_medc10": f"{t['median_months__collection_10']:.0f}", "au_medc40": f"{t['median_months__collection_40']:.0f}",
        "au_rows5": f"{a['rows_shift_over_5pp']:,}", "au_rows": f"{a['rows_compared']:,}",
        "au_rshift": str(a["civil_suit_median_rank_shift"]), "au_rmax": str(a["civil_suit_max_rank_shift"]),
        "au_rwidth": str(a["civil_suit_median_rank_interval_width"]),
        "au_rwide": pct(a["civil_suit_share_rank_interval_over_50"]),
        "au_ncourts": str(len(a["units_that_are_court_complexes"])),
        **coverage_vars(a),
    }


def coverage_vars(a: dict) -> dict[str, str]:
    cv, sp = a["coverage"], a["speedup"]
    sn = {r["case_type"]: r for r in a["sens"]}["civ_suit"]
    pct = lambda v: f"{v * 100:.0f}"
    low = next(iter(cv["lowest_coverage"]))
    return {
        "cv_all": pct(cv["coverage_2015_18"]), "cv_15": pct(cv["coverage_by_year"]["2015"]),
        "cv_18": pct(cv["coverage_by_year"]["2018"]), "cv_low": pct(cv["lowest_coverage"][low]), "cv_low_state": low,
        "cv_none": str(len(cv["official_units_with_no_records"])),
        "cv_w": f"{sn['median_months__weighted_to_official_filings']:.0f}", "cv_14": f"{sn['median_months__records_from_2014']:.0f}",
        "cv_n": str(cv["states_in_spearman"]), "cv_rho": f"{cv['spearman_state_coverage_vs_civil_suit_pending_3y']:.2f}".replace("-", "−"),
        "pa_12": f"{cv['pending_not_listed_in_12m'] * 100:.1f}", "pa_24": f"{cv['pending_not_listed_in_24m'] * 100:.1f}",
        "pa_bihar": f"{cv['pending_not_listed_in_12m_bihar'] * 100:.1f}",
        "pa_wb": pct(cv["pending_not_listed_in_12m_worst"]["West Bengal"]),
        "pa_od": pct(cv["pending_not_listed_in_12m_worst"]["Orissa"]),
        "sp_m10": pct(sp["fixed_case_mix"][0]), "sp_m17": pct(sp["fixed_case_mix"][7]),
        "sp_s10": pct(sp["stable_coverage_states"][0]), "sp_s17": pct(sp["stable_coverage_states"][7]),
        "sp_g10": pct(sp["growing_coverage_states"][0]), "sp_g17": pct(sp["growing_coverage_states"][7]),
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
        page = page.replace('<html lang="en">', f'<html lang="{lang}">').replace(
            f'<link rel="canonical" href="{BASE}{template}">', f'<link rel="canonical" href="{BASE}{out.name}">')
    out.write_text(page)
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size / 1e3:.0f} kB)")


def main() -> None:
    (HERE / "docs").mkdir(exist_ok=True)
    sd, n = site_data(), unit_counts()
    render("index.html", sd, HERE / "docs" / "index.html", fill=n)
    render("paper.html", {k: v for k, v in sd.items() if k != "district_keys"}, HERE / "docs" / "paper.html",
           fill=paper_vars(sd) | n)
    cd = calc_data()
    (HERE / "docs" / "calc-data.json").write_text(json.dumps(cd, separators=(",", ":")))
    print(f"wrote docs/calc-data.json ({(HERE / 'docs' / 'calc-data.json').stat().st_size / 1e3:.0f} kB)")
    render("calculator.html", {}, HERE / "docs" / "calculator.html", fill=n)
    render("calculator.html", {}, HERE / "docs" / "calculator-hi.html", lang="hi", fill=n)
    render("districts.html", {}, HERE / "docs" / "districts.html", fill=n)
    md = map_data()
    (HERE / "docs" / "map-data.json").write_text(json.dumps(md, separators=(",", ":"), ensure_ascii=False))
    (HERE / "docs" / "map-civil-suits.svg").write_text(map_svg(md))
    print(f"wrote docs/map-data.json ({(HERE / 'docs' / 'map-data.json').stat().st_size / 1e3:.0f} kB)")
    render("map.html", {}, HERE / "docs" / "map.html", fill=n)
    (HERE / "docs" / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url><loc>{BASE}{p}</loc></url>\n" for p in PAGES) + "</urlset>\n")


if __name__ == "__main__":
    main()
