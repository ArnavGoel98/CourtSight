#!/usr/bin/env python3
"""How much did COVID-19 add to India's district-court backlog? Official state series vs a pre-2020 trend.

Method (fixed before looking at post-2019 results):
  counterfactual_s(t) = P_s(2019) * exp(g_s * (t - 2019)),  g_s = mean log growth of state s over 2015-2019
  excess(t) = sum_s [P_s(t) - counterfactual_s(t)]
Calibration: the same projection is backtested before COVID (origins 2016-2018, every horizon up to 2019). Its
national error per year of horizon, r, ranged from about -3% to +2% and was mostly negative (it under-predicted a
backlog that was already accelerating). The headline counterfactual is corrected by the median r, and its range
spans the backtest r values, so the reported uncertainty is the method's own observed error, not a model of it.
A year-resampling bootstrap is also reported; it ignores that bias and is too narrow (backtest values fall outside).
Caveat: everything else that changed after 2019 (eCourts coverage, virtual traffic courts, new criminal codes from
July 2024) is folded into "excess"; the estimate is cleanest for 2020-2022.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from official_series import SOURCES, pending

HERE: Final = Path(__file__).resolve().parent
BASE_YEARS: Final = (2015, 2016, 2017, 2018, 2019)


def growth(p: pd.DataFrame) -> pd.DataFrame:
    return np.log(p).diff(axis=1)


def counterfactual(p: pd.DataFrame, origin: int, years: list[int], base: tuple[int, ...]) -> pd.DataFrame:
    g = growth(p)[list(base)].mean(axis=1)
    return pd.DataFrame({t: p[origin] * np.exp(g * (t - origin)) for t in years})


def bootstrap_national(p: pd.DataFrame, origin: int, years: list[int], base: tuple[int, ...], draws: int,
                       seed: int = 0) -> np.ndarray:
    """draws x len(years) national counterfactual totals."""
    g = growth(p)[list(base)].to_numpy()  # states x base years
    rng = np.random.default_rng(seed)
    out = np.empty((draws, len(years)))
    for d in range(draws):
        pick = rng.integers(0, g.shape[1], len(years))
        path = np.cumsum(g[:, pick], axis=1)  # states x horizon
        out[d] = (p[origin].to_numpy()[:, None] * np.exp(path)).sum(0)
    return out


def backtest_error_rates(p: pd.DataFrame) -> list[float]:
    """Per-year log error of the national projection, over all pre-COVID origins and horizons."""
    rates = []
    for origin in (2016, 2017, 2018):
        base = tuple(range(2015, origin + 1))
        for t in range(origin + 1, 2020):
            proj = counterfactual(p, origin, [t], base)[t].sum()
            rates.append(float(np.log(proj / p[t].sum()) / (t - origin)))
    return rates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "covid_excess")
    ap.add_argument("--draws", type=int, default=10000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p = pending()

    # 1. backtest the method before COVID: origin 2017, base growth 2015-2017, predict 2018-2019
    bt_years = [2018, 2019]
    bt = counterfactual(p, 2017, bt_years, (2015, 2016, 2017)).sum()
    bt_boot = bootstrap_national(p, 2017, bt_years, (2015, 2016, 2017), args.draws)
    backtest = {str(t): {"actual": float(p[t].sum()), "projected": float(bt[t]),
                         "error_pct": float(bt[t] / p[t].sum() - 1),
                         "interval_90": [float(x) for x in np.quantile(bt_boot[:, i], [0.05, 0.95])],
                         "actual_inside_interval": bool(np.quantile(bt_boot[:, i], 0.05) <= p[t].sum()
                                                         <= np.quantile(bt_boot[:, i], 0.95))}
                for i, t in enumerate(bt_years)}

    # 2. COVID excess
    years = list(range(2020, 2026))
    cf = counterfactual(p, 2019, years, BASE_YEARS)
    boot = bootstrap_national(p, 2019, years, BASE_YEARS, args.draws)
    actual = p[years].sum()
    rates = backtest_error_rates(p)
    r_med, r_lo, r_hi = float(np.median(rates)), float(min(rates)), float(max(rates))
    rows = []
    for i, t in enumerate(years):
        h = t - 2019
        raw = float(cf[t].sum())
        # projection = truth * exp(r h)  =>  truth = projection * exp(-r h)
        c_mid, c_hi, c_lo = (raw * np.exp(-r * h) for r in (r_med, r_lo, r_hi))
        b_lo, b_hi = np.quantile(boot[:, i], [0.05, 0.95])
        rows.append({"year": t, "actual": float(actual[t]), "counterfactual": c_mid,
                     "counterfactual_low": c_lo, "counterfactual_high": c_hi,
                     "excess": float(actual[t]) - c_mid, "excess_low": float(actual[t]) - c_hi,
                     "excess_high": float(actual[t]) - c_lo,
                     "excess_share_of_backlog": (float(actual[t]) - c_mid) / float(actual[t]),
                     "uncorrected_counterfactual": raw, "uncorrected_excess": float(actual[t]) - raw,
                     "bootstrap_excess_lo90": float(actual[t] - b_hi), "bootstrap_excess_hi90": float(actual[t] - b_lo)})
    national = pd.DataFrame(rows)
    national.to_csv(args.out / "national.csv", index=False)
    by_state = pd.DataFrame({"pending_2019": p[2019], "pending_2022": p[2022], "counterfactual_2022": cf[2022],
                             "excess_2022": p[2022] - cf[2022],
                             "excess_2022_pct": p[2022] / cf[2022] - 1}).sort_values("excess_2022", ascending=False)
    by_state.to_csv(args.out / "by_state_2022.csv", index_label="state")

    # 3. sensitivity: other reasonable baselines for the same counterfactual
    sens = {}
    for name, base in (("mean_growth_2015_2019", BASE_YEARS), ("mean_growth_2017_2019", (2017, 2018, 2019)),
                       ("mean_growth_2016_2019", (2016, 2017, 2018, 2019))):
        c = counterfactual(p, 2019, [2022], base).sum()
        sens[name] = float(p[2022].sum() - c[2022])
    meta = {"sources": SOURCES, "method": __doc__.split("Method")[1].split("Caveat")[0].strip(),
            "backtest_pre_covid": backtest, "backtest_error_rates_per_year": rates,
            "correction_rate_median": r_med, "sensitivity_excess_2022_uncorrected": sens}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 200)
    print(json.dumps(backtest, indent=2))
    print("error rates/yr:", [round(r * 100, 2) for r in rates])
    show = national.set_index("year")[["actual", "counterfactual", "counterfactual_low", "counterfactual_high", "excess",
                                       "excess_low", "excess_high", "uncorrected_excess"]] / 1e5
    print(show.round(1).to_string())
    print("excess share of backlog:", national.set_index("year")["excess_share_of_backlog"].round(3).to_dict())
    print("sensitivity (lakh):", {k: round(v / 1e5, 1) for k, v in sens.items()})
    print((by_state.head(8)[["excess_2022", "excess_2022_pct"]] / [1e5, 0.01]).round(1).to_string())


if __name__ == "__main__":
    main()
