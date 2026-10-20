"""Kaplan-Meier on a hand-built cube: 100 cases filed Jan 2015, 50 decided after exactly 12 months, 50 still
pending when the state's data ends (Dec 2018). Expected: 50% pending after 1 year; median 12 months."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time_to_decision as ttd  # noqa: E402


def test_km_simple() -> None:
    rows = [(1001, 201501, 201601, "civil", 50), (1001, 201501, 0, "civil", 50)]
    # state-wide decisions every month of 2018 so the collection month is Dec 2018
    rows += [(1002, 201701, 201800 + m, "civil", 10) for m in range(1, 13)]
    c = pd.DataFrame(rows, columns=["district_id", "f", "d", "case_type", "n"])
    t = ttd.km_table(c, min_cases=50).set_index("district_id")
    assert abs(t.loc[1001, "pending_after_1y"] - 0.5) < 1e-9
    assert t.loc[1001, "median_months"] == 12
    assert np.isclose(t.loc[1001, "pending_after_3y"], 0.5)



def test_delayed_entry() -> None:
    """Records complete from Jan 2016. 100 cases filed Jan 2016: 50 decided at 12 months, 50 pending. 100 cases filed
    Jan 2014 were only recorded if still pending in Jan 2016 (age 24 months): 50 of those, 25 decided at 30 months,
    25 pending. Truth: half decided by 12 months. Uncorrected KM wrongly counts the old survivors from filing.
    Delayed entry: the old cases join at 24 months, so S(12) = 0.5 exactly and S(30) = 0.5 * (1 - 25/(50+50)) = 0.375."""
    rows = [(1001, 201601, 201701, "civil", 50), (1001, 201601, 0, "civil", 50),
            (1001, 201401, 201607, "civil", 25), (1001, 201401, 0, "civil", 25)]
    rows += [(1002, 201701, 201800 + m, "civil", 10) for m in range(1, 13)]
    c = pd.DataFrame(rows, columns=["district_id", "f", "d", "case_type", "n"])
    entry = pd.Series({1001: ttd.months(pd.Series([201601])).iloc[0]})
    naive = ttd.km_table(c, min_cases=50).set_index("district_id")
    fixed = ttd.km_table(c, min_cases=50, entry=entry).set_index("district_id")
    assert naive.loc[1001, "pending_after_1y"] > 0.6          # survivors make it look slow
    assert np.isclose(fixed.loc[1001, "pending_after_1y"], 0.5)
    curve = [float(x) for x in fixed.loc[1001, "curve_q"].split(" ") if x]
    assert np.isclose(curve[10], 0.375)                       # month 30


def test_missing_decisions() -> None:
    """Two districts file 100 cases every month of 2010-2017. District 1001 decides half of each month's filings six
    months later; district 1002 records no decision at all until 2018 (like Gaya). Filings look complete in both,
    but only 1001's records are: 1002 must never count as complete, and its cases must not reach the estimates."""
    ym = [y * 100 + m for y in range(2010, 2018) for m in range(1, 13)]
    later = lambda v, k: (v // 100 + (v % 100 + k - 1) // 12) * 100 + (v % 100 + k - 1) % 12 + 1
    rows = []
    for f in ym:
        rows += [(1001, f, later(f, 6), "civil", 50), (1001, f, 0, "civil", 50), (1002, f, 0, "civil", 100)]
    rows += [(1002, 201601, 201800 + m, "civil", 1) for m in range(1, 13)]
    c = pd.DataFrame(rows, columns=["district_id", "f", "d", "case_type", "n"])
    e = ttd.digitisation_month(c)
    assert e[1001] == ttd.months(pd.Series([201001])).iloc[0]
    assert np.isnan(e[1002])
    t = ttd.km_table(c, min_cases=50, entry=e)
    assert set(t["district_id"]) == {1001}


if __name__ == "__main__":
    test_km_simple()
    test_delayed_entry()
    test_missing_decisions()
    print("time-to-decision tests passed")
