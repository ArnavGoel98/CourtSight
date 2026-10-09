#!/usr/bin/env python3
"""When does each state clear its district-court backlog at today's pace?

Uses only the official year-end pending counts (official_series.py). Over the latest two years (31.12.2023 to
31.12.2025) each state's backlog changed by an average of dP cases a year:
  dP > 0   the backlog is growing: at this pace it is never cleared
  dP < 0   it is shrinking: years to clear = P(2025) / -dP
Also reported: the extra cases a state would have to dispose of each year just to stop growing (dP), and to clear
everything within 10 years (dP + P/10), shown against the state's 2021 disposals for scale (the latest year with
official disposal figures, so the scale is approximate).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from official_series import disposed, pending

HERE: Final = Path(__file__).resolve().parent


def clock(p: pd.DataFrame, d: pd.DataFrame, start: int = 2023, end: int = 2025) -> pd.DataFrame:
    dp = (p[end] - p[start]) / (end - start)
    out = pd.DataFrame({"pending_2025": p[end], "change_per_year": dp.round(0),
                        "change_per_year_pct": dp / p[start]})
    out["status"] = np.where(dp > 0, "growing", "shrinking")
    out["years_to_clear"] = np.where(dp < 0, p[end] / -dp, np.inf)
    out["extra_disposals_to_stop_growth"] = dp.clip(lower=0).round(0)
    out["extra_disposals_to_clear_in_10y"] = (dp + p[end] / 10).clip(lower=0).round(0)
    out["disposed_2021"] = d[2021]
    out["clear_in_10y_vs_2021_disposals"] = out["extra_disposals_to_clear_in_10y"] / d[2021]
    return out.sort_values("pending_2025", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "backlog_clock")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p, d = pending(), disposed()
    t = clock(p, d)
    t.to_csv(args.out / "states.csv", index_label="state")
    nat = clock(p.sum().to_frame("India").T, d.sum().to_frame("India").T)
    nat.to_csv(args.out / "national.csv", index_label="unit")
    pd.set_option("display.width", 200)
    print(nat.round(3).to_string())
    print(t.head(15).round(3).to_string())
    print(t["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
