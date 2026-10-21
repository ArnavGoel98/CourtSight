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


def test_curve_stops_when_few_remain() -> None:
    """30 cases filed Jan 2015: 15 decided after 6 months, 15 still pending. From month 7 only 15 are followed, fewer
    than MIN_AT_RISK (20), so the curve stops at 6 months: the median (6) is known, the 1-year share is not."""
    rows = [(1001, 201501, 201507, "civil", 15), (1001, 201501, 0, "civil", 15)]
    rows += [(1002, 201701, 201800 + m, "other", 10) for m in range(1, 13)]
    c = pd.DataFrame(rows, columns=["district_id", "f", "d", "case_type", "n"])
    t = ttd.km_table(c, min_cases=10).set_index(["district_id", "case_type"]).loc[(1001, "civil")]
    assert ttd.MIN_AT_RISK == 20 and t["followed_months"] == 6
    assert t["median_months"] == 6 and np.isnan(t["pending_after_1y"]) and np.isnan(t["p90_months"])


def test_decided_only() -> None:
    """100 civil cases filed Jan 2015: 40 decided after 6 months, 60 still pending. Counting only decided cases says the
    median is 6 months; Kaplan-Meier says over half are still pending."""
    rows = [(1001, 201501, 201507, "civil", 40), (1001, 201501, 0, "civil", 60)]
    rows += [(1002, 201701, 201800 + m, "other", 10) for m in range(1, 13)]   # sets the collection month
    c = pd.DataFrame(rows, columns=["district_id", "f", "d", "case_type", "n"])
    d = ttd.decided_only(c).set_index("case_type")
    assert d.loc["civil", "decided_cases"] == 40 and d.loc["civil", "median_months_decided_only"] == 6
    km = ttd.km_table(c, min_cases=50).set_index(["district_id", "case_type"])
    assert np.isnan(km.loc[(1001, "civil"), "median_months"])


def test_national_keeps_each_state_collection_month() -> None:
    """State 1 is collected in Dec 2018, state 2 in Jun 2020. Each has 40 civil cases filed Jan 2016: 20 decided after
    5 months; the other 20 are still pending at state 1's collection (censored at 35 months) and decided after 48
    months in state 2. Pooled with each state's own collection month, every case still followed at 48 months is
    decided then, so 90% are decided by 48 months. Pooling with one national collection month (Jun 2020 here) would
    keep state 1's 20 cases at risk with no decisions recorded, and the curve would stall at 25%."""
    rows = [(1001, 201601, 201606, "civil", 20), (1001, 201601, 0, "civil", 20),
            (2001, 201601, 201606, "civil", 20), (2001, 201601, 202001, "civil", 20)]
    rows += [(1001, 201601, 201800 + m, "other", 10) for m in range(1, 13)]                  # state 1: data ends Dec 2018
    rows += [(2001, 201601, y * 100 + m, "other", 10) for y in (2018, 2019, 2020) for m in range(1, 13)
             if y * 100 + m <= 202006]                                                       # state 2: data ends Jun 2020
    c = pd.DataFrame(rows, columns=["district_id", "f", "d", "case_type", "n"])
    t = ttd.km_table(ttd.as_national(c), min_cases=10).set_index("case_type").loc["civil"]
    assert t["median_months"] == 5 and t["p90_months"] == 48 and t["followed_months"] == 48
    one = ttd.km_table(c.assign(district_id=0), min_cases=10).set_index("case_type").loc["civil"]
    assert np.isnan(one["p90_months"])   # the old pooling: state 1's pending cases never resolve


if __name__ == "__main__":
    test_km_simple()
    test_delayed_entry()
    test_missing_decisions()
    test_curve_stops_when_few_remain()
    test_decided_only()
    test_national_keeps_each_state_collection_month()
    print("time-to-decision tests passed")
