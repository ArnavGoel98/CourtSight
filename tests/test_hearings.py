"""Hearing-gap arithmetic: the stage table and the typical gap on a hand-made snapshot."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearings import gm, stage_table  # noqa: E402


def test_stage_table() -> None:
    # 30 cases at summons with gaps of 30 and 120 days (geometric mean 60), 10 at evidence 20 days apart
    s = pd.DataFrame({"district_id": [1001, 1001, 1001], "t_ym": [201901] * 3,
                      "stage": ["service", "service", "evidence"], "n": [15, 15, 10], "n_gap": [15, 15, 10],
                      "sum_log_gap": [15 * np.log(30), 15 * np.log(120), 10 * np.log(20)]})
    t = stage_table(s).set_index("group")
    assert np.isclose(t["share"].sum(), 1.0)
    assert np.isclose(t.loc["service", "share"], 0.75) and np.isclose(t.loc["service", "gap_days"], 60.0)
    assert np.isclose(t.loc["evidence", "gap_days"], 20.0)
    assert np.isclose(gm(s["sum_log_gap"].sum(), s["n_gap"].sum()), np.exp((30 * np.log(60) + 10 * np.log(20)) / 40))


if __name__ == "__main__":
    test_stage_table()
    print("ok")
