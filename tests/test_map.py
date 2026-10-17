"""The district map: every hand-matched unit points at a real court unit and a real 2011 shape, and the map shows
every district that has results."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_site import map_data  # noqa: E402

RELATIONS = {"renamed", "carved_from", "city_courts", "split_court"}


def test_crosswalk() -> None:
    cw = pd.read_csv(ROOT / "sources" / "map_crosswalk.csv")
    key = pd.read_csv(ROOT / "ddl_compact" / "district_key.csv")
    ids = set(key["state_code"] * 1000 + key["dist_code"])
    m = map_data()
    assert set(cw["relation"]) <= RELATIONS, set(cw["relation"]) - RELATIONS
    assert set(cw["district_id"]) <= ids, set(cw["district_id"]) - ids
    assert {str(c) for c in cw["census_2011_code"]} <= set(m["paths"]), "crosswalk points at a missing shape"
    # a unit DDL already links to a 2011 district must not be matched again by hand
    linked = set((key["state_code"] * 1000 + key["dist_code"])[key["pc11_district_id"].notna()])
    assert not set(cw["district_id"]) & linked, set(cw["district_id"]) & linked


def test_every_district_on_the_map() -> None:
    m = map_data()
    assert len(m["paths"]) == 641 and set(m["paths"]) == set(m["names"])
    shown = {(u[0], u[1]) for us in m["units"].values() for u in us}
    t = pd.read_csv(ROOT / "results" / "time_to_decision" / "by_district_case_type.csv").dropna(subset=["district_name"])
    t = t.loc[t["case_type"] == "civ_suit"]
    special = set(json.loads((ROOT / "results" / "audit" / "summary.json").read_text())["units_that_are_court_complexes"])
    t = t.loc[~t["district_name"].isin(special)]   # state-wide tribunals and courts have no district shape
    missing = {(s, d) for s, d in zip(t["state_name"], t["district_name"])} - shown
    assert not missing, f"districts with civil-suit results but no shape: {sorted(missing)}"


if __name__ == "__main__":
    test_crosswalk()
    test_every_district_on_the_map()
    print("ok")
