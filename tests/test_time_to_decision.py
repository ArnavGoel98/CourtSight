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


if __name__ == "__main__":
    test_km_simple()
    print("time-to-decision tests passed")
