"""Transcribed official figures must reproduce the totals printed in the Parliament answers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import official_series as o  # noqa: E402


def test_pending_totals_match_print() -> None:
    assert o.check_totals(tol=0.0) == {}


def test_disposed_totals_match_print() -> None:
    tot = o.disposed().sum()
    for year, printed in o.PRINTED_DISPOSED_TOTALS.items():
        expected = printed - (o.KERALA_2016_AS_PRINTED if year == 2016 else 0)
        assert int(tot[year]) == expected, year


if __name__ == "__main__":
    test_pending_totals_match_print()
    test_disposed_totals_match_print()
    print("official series match the printed totals")
