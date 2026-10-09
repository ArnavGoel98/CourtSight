"""Years-of-waiting arithmetic on a tiny hand-made cube."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from waiting_years import weighted_median  # noqa: E402


def test_weighted_median() -> None:
    assert weighted_median(pd.Series([1, 5, 9]), pd.Series([1, 1, 1])) == 5
    assert weighted_median(pd.Series([1, 5, 9]), pd.Series([10, 1, 1])) == 1
    assert weighted_median(pd.Series([9, 1, 5]), pd.Series([1, 1, 10])) == 5


if __name__ == "__main__":
    test_weighted_median()
    print("ok")
