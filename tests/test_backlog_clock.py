"""Backlog clock arithmetic on a two-state toy series."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backlog_clock import clock  # noqa: E402


def test_clock() -> None:
    p = pd.DataFrame({2023: [100.0, 100.0], 2025: [120.0, 80.0]}, index=["grows", "shrinks"])
    d = pd.DataFrame({2021: [50.0, 50.0]}, index=["grows", "shrinks"])
    t = clock(p, d)
    assert t.loc["grows", "status"] == "growing" and np.isinf(t.loc["grows", "years_to_clear"])
    assert t.loc["shrinks", "years_to_clear"] == 8.0  # 80 left, falling 10 a year
    assert t.loc["grows", "extra_disposals_to_clear_in_10y"] == 22.0  # 10 to stop growth + 12 a year to clear 120


if __name__ == "__main__":
    test_clock()
    print("backlog clock tests passed")
