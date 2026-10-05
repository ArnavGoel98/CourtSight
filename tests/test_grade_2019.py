"""Identity test for grade_2019: NJDG-style (pending by filing year, disposed by year x filing year) tables built
from known case histories must reproduce the exact year-end stocks and flows.  Run: python3 tests/test_grade_2019.py"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from grade_2019 import crosswalk, load_ddl_key, reconstruct  # noqa: E402


def cases(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 60_000
    dist = rng.choice(["Pune", "Nashik", "Thane Rural", "Thane City"], n)
    fy = rng.integers(2005, 2020, n)  # includes pre-2010 cohorts, which must be ignored
    dur = rng.exponential(3.0, n)
    dy = np.floor(fy + rng.random(n) + dur).astype(int)
    dy = np.where(rng.random(n) < 0.1, 9999, dy)  # never decided
    return pd.DataFrame({"state": "Maharashtra", "district_as_per_source": dist, "fy": fy, "dy": dy})


def njdg_tables(c: pd.DataFrame, as_of_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    live = c.loc[(c["fy"] <= as_of_year) & (c["dy"] > as_of_year)]
    pend = live.groupby(["state", "district_as_per_source", "fy"]).size().rename("pending_cases").reset_index()
    done = c.loc[(c["dy"] >= 2018) & (c["dy"] <= as_of_year)]
    disp = done.groupby(["state", "district_as_per_source", "fy", "dy"]).size().rename("cases_disposed").reset_index()
    pend = pend.rename(columns={"fy": "instituted_year"}).assign(data_as_of="2025-06-30")
    disp = disp.rename(columns={"fy": "instituted_year", "dy": "disposed_year"}).assign(data_as_of="2025-06-30")
    return pend, disp


def test_reconstruct_identity() -> None:
    c = cases()
    pend, disp = njdg_tables(c, 2025)
    got = reconstruct(pend, disp, 2018).set_index("district_as_per_source")
    c = c.loc[c["fy"] >= 2010]
    for d, g in c.groupby("district_as_per_source"):
        exp = {"P0": ((g["fy"] <= 2018) & (g["dy"] > 2018)).sum(), "P1": ((g["fy"] <= 2019) & (g["dy"] > 2019)).sum(),
               "D1": (g["dy"] == 2019).sum(), "A1": (g["fy"] == 2019).sum()}
        for k, v in exp.items():
            assert got.loc[d, k] == v, (d, k, got.loc[d, k], v)


def test_crosswalk_and_split() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        key = Path(tmp) / "key.csv"
        pd.DataFrame({"year": 2018, "state_code": 1, "state_name": "Maharashtra", "dist_code": [1, 2, 3],
                      "district_name": ["Pune District", "Nasik", "Thane"]}).to_csv(key, index=False)
        nj = pd.DataFrame({"state": "Maharashtra", "district_as_per_source": ["Pune", "Nashik", "Thane Rural", "Thane City"]})
        ov = Path(tmp) / "ov.csv"
        pd.DataFrame({"state": "Maharashtra", "district_as_per_source": ["Thane Rural", "Thane City"],
                      "district_id": [1003, 1003]}).to_csv(ov, index=False)
        cw = crosswalk(load_ddl_key(key), nj, ov).set_index("district_as_per_source")
    assert cw.loc["Pune", "method"] == "exact" and cw.loc["Pune", "district_id"] == 1001
    assert cw.loc["Nashik", "method"] == "fuzzy" and cw.loc["Nashik", "district_id"] == 1002
    assert (cw.loc[["Thane Rural", "Thane City"], "district_id"] == 1003).all()


def test_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "dataful").mkdir()
        pend, disp = njdg_tables(cases(), 2025)
        pend.to_parquet(t / "dataful" / "pending_by_year__all.parquet")
        disp.to_parquet(t / "dataful" / "disposed_by_year__all.parquet")
        pd.DataFrame({"year": 2018, "state_code": 1, "state_name": "Maharashtra", "dist_code": [1, 2, 3],
                      "district_name": ["Pune", "Nashik", "Thane"]}).to_csv(t / "key.csv", index=False)
        pd.DataFrame({"state": "Maharashtra", "district_as_per_source": ["Thane Rural", "Thane City"],
                      "district_id": [1003, 1003]}).to_csv(t / "ov.csv", index=False)
        res = t / "results"
        res.mkdir()
        ids = [1001, 1002, 1003]
        pd.DataFrame({"district_id": ids, "pending": [1500.0, 1500.0, 3000.0],
                      "instituted_12": [1000.0, 1000.0, 2000.0]}).to_csv(res / "origin_state.csv", index=False)
        fc = pd.DataFrame({"district_id": ids, "origin": "2018-12"})
        for tgt, base in (("growth", 0.0), ("cr", 1.0)):
            for kind in ("", "linear_", "drift_"):
                for q, off in ((10, -0.2), (50, 0.0), (90, 0.2)):
                    fc[f"{tgt}_12_{kind}q{q}"] = base + off
        fc.to_csv(res / "forecasts.csv", index=False)
        subprocess.run([sys.executable, str(HERE / "grade_2019.py"), "--dataful", str(t / "dataful"),
                        "--district-key", str(t / "key.csv"), "--overrides", str(t / "ov.csv"), "--results", str(res),
                        "--private-out", str(t / "private"), "--draws", "50"], check=True, capture_output=True)
        summary = pd.read_csv(res / "grading_2019" / "grading_summary.csv")
        assert {"model", "linear", "drift"} <= set(summary["forecaster"])
        assert (summary.loc[summary["tau"] != "share_below_1", "n"] == 3).all()


if __name__ == "__main__":
    test_reconstruct_identity()
    test_crosswalk_and_split()
    test_end_to_end()
    print("grade_2019 tests passed")
