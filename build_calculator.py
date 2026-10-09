#!/usr/bin/env python3
"""Build docs/index.html: a static "how long will my case take?" page from results/time_to_decision.

The page embeds the district x case-type survival summaries (no case-level data) and runs entirely in the browser,
so it can be served by GitHub Pages from the docs/ folder.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd

HERE: Final = Path(__file__).resolve().parent
RES: Final = HERE / "results" / "time_to_decision"
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
COLS: Final = ["cases", "median_months", "p90_months", "pending_after_1y", "pending_after_3y", "pending_after_5y"]


def rows(t: pd.DataFrame) -> list[list[object]]:
    out = []
    for r in t.itertuples(index=False):
        out.append([r.case_type] + [None if pd.isna(getattr(r, c)) else
                                    (int(getattr(r, c)) if c in ("cases", "median_months", "p90_months")
                                     else round(float(getattr(r, c)), 3)) for c in COLS])
    return out


def main() -> None:
    t = pd.read_csv(RES / "by_district_case_type.csv").dropna(subset=["state_name", "district_name"])
    nat = pd.read_csv(RES / "national_case_type.csv")
    data = {"labels": LABELS, "cols": ["type"] + COLS, "national": rows(nat), "states": {}}
    for (s, d), g in t.sort_values(["state_name", "district_name"]).groupby(["state_name", "district_name"], sort=False):
        data["states"].setdefault(s, {})[d] = rows(g)
    page = (HERE / "calculator_template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    out = HERE / "docs" / "index.html"
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB): {len(data['states'])} states, {len(t)} rows")


if __name__ == "__main__":
    main()
