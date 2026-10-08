#!/usr/bin/env python3
"""Prospective forecast: pending cases in each state's District & Subordinate Courts on 31 December 2026.

The outcome does not exist yet. The forecast, the method-selection rule and the scoring rule are committed to git
before it does, and grade_2026.py scores them once the Ministry of Law & Justice publishes 31.12.2026 figures
(they appear in Parliament answers, usually by February/March of the next year).

Candidate methods (one-year-ahead log growth per state):
  zero        no change
  own_last    the state's own growth in the latest year
  own_mean3   the state's mean growth over the latest three years
  national    the national growth of the latest year, for every state
  blend       half own_mean3, half national
Selection rule, fixed in code before any backtest was run: the lowest case-weighted mean absolute error of state log
growth over one-year backtests whose target year is outside the COVID disruption (targets 2017-2019 and 2023-2025).
Intervals: q10/q90 of the chosen method's pooled backtest errors (state level) and of its national errors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Final

import numpy as np
import pandas as pd

from official_series import SOURCES, pending

HERE: Final = Path(__file__).resolve().parent
TARGET: Final = 2026
BACKTEST_TARGETS: Final = (2017, 2018, 2019, 2023, 2024, 2025)
Method = Callable[[pd.DataFrame, int], pd.Series]


def _g(p: pd.DataFrame) -> pd.DataFrame:
    return np.log(p).diff(axis=1)


def _national(p: pd.DataFrame, origin: int) -> float:
    return float(np.log(p[origin].sum() / p[origin - 1].sum()))


METHODS: Final[dict[str, Method]] = {
    "zero": lambda p, o: pd.Series(0.0, index=p.index),
    "own_last": lambda p, o: _g(p)[o],
    "own_mean3": lambda p, o: _g(p)[[o - 2, o - 1, o]].mean(axis=1),
    "national": lambda p, o: pd.Series(_national(p, o), index=p.index),
    "blend": lambda p, o: 0.5 * _g(p)[[o - 2, o - 1, o]].mean(axis=1) + 0.5 * _national(p, o),
}


def backtest(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    g = _g(p)
    for t in BACKTEST_TARGETS:
        o = t - 1
        w = p[o] / p[o].sum()
        for name, f in METHODS.items():
            pred = f(p, o)
            err = pred - g[t]
            nat_pred = float(np.log((p[o] * np.exp(pred)).sum() / p[o].sum()))
            nat_act = float(np.log(p[t].sum() / p[o].sum()))
            rows.append({"target": t, "method": name, "mae": float(err.abs().mean()),
                         "mae_case_weighted": float((err.abs() * w).sum()), "national_error": nat_pred - nat_act,
                         "errors": err.to_dict()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "forecast_2026")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p = pending()
    bt = backtest(p)
    summary = bt.groupby("method").agg(mae=("mae", "mean"), mae_case_weighted=("mae_case_weighted", "mean"),
                                       national_abs_error=("national_error", lambda e: float(np.abs(e).mean())))
    summary = summary.sort_values("mae_case_weighted")
    chosen = str(summary.index[0])
    errs = np.concatenate([list(e.values()) for e in bt.loc[bt["method"] == chosen, "errors"]])
    nat_errs = bt.loc[bt["method"] == chosen, "national_error"].to_numpy()
    e_lo, e_hi = np.quantile(errs, [0.1, 0.9])  # error = predicted - actual
    origin = TARGET - 1
    g_hat = METHODS[chosen](p, origin)
    fc = pd.DataFrame({
        "pending_2025": p[origin],
        "q50": p[origin] * np.exp(g_hat),
        "q10": p[origin] * np.exp(g_hat - e_hi),
        "q90": p[origin] * np.exp(g_hat - e_lo),
    }).round(0).astype(np.int64)
    nat50 = float(fc["q50"].sum())
    nat = {"pending_2025": int(p[origin].sum()), "q50": int(round(nat50)),
           "q10": int(round(nat50 * np.exp(-nat_errs.max()))), "q90": int(round(nat50 * np.exp(-nat_errs.min())))}
    fc.to_csv(args.out / "forecast_states_2026.csv", index_label="state")
    summary.to_csv(args.out / "method_backtest.csv")
    bt.drop(columns="errors").to_csv(args.out / "method_backtest_by_year.csv", index=False)
    meta = {"target": "pending cases in District & Subordinate Courts on 31.12.2026, by state and national",
            "made_on_data_through": "31.12.2025", "sources": SOURCES, "chosen_method": chosen,
            "selection_rule": "lowest case-weighted MAE of state log growth, one-year backtests, targets "
                              + ", ".join(map(str, BACKTEST_TARGETS)),
            "national": nat, "state_error_quantiles_q10_q90": [float(e_lo), float(e_hi)],
            "locked_by": "the git commit that first added this file"}
    (args.out / "forecast_meta.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string())
    print(json.dumps(meta, indent=2))
    print((fc.sort_values("pending_2025", ascending=False).head(10) / 1e5).round(2).to_string())


if __name__ == "__main__":
    main()
