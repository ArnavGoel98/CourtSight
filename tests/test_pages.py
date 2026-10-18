"""The generated district pages and JSON files (run after build_site.py): one of each for every district with
results, a page for every withheld district, no unfilled or missing values, and every page in the sitemap."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from district_pages import slug  # noqa: E402

DOCS = ROOT / "docs"


def test_pages_and_api() -> None:
    t = pd.read_csv(ROOT / "results" / "time_to_decision" / "by_district_case_type.csv").dropna(subset=["district_name"])
    pairs = set(zip(t["state_name"], t["district_name"]))
    for s, d in pairs:
        assert (DOCS / "d" / slug(s) / f"{slug(d)}.html").exists(), (s, d)
        j = json.loads((DOCS / "api" / "v1" / slug(s) / f"{slug(d)}.json").read_text())
        assert j["district"] == d and j["case_types"], (s, d)
    held = json.loads((ROOT / "results" / "audit" / "summary.json").read_text())["withheld_detail"]
    for d, v in held.items():
        if v["filed_2010_2017"] >= 1000:
            assert (DOCS / "d" / slug(v["state"]) / f"{slug(d)}.html").exists(), d
    index = json.loads((DOCS / "api" / "v1" / "index.json").read_text())
    assert len(index["districts"]) == len(pairs)


def test_no_broken_values() -> None:
    bad = re.compile(r"\bnan\b|\bNone\b|\{\{|>undefined<|NaN%")
    for f in (DOCS / "d").rglob("*.html"):
        text = f.read_text()
        assert not bad.search(re.sub(r"<script.*?</script>", "", text, flags=re.S)), f
        assert 'd=""' not in text, f"empty map shape in {f}"


def test_sitemap_lists_every_page() -> None:
    sm = (DOCS / "sitemap.xml").read_text()
    for f in (DOCS / "d").rglob("*.html"):
        assert f"/{f.relative_to(DOCS).as_posix()}<" in sm, f


def test_slug_matches_the_browser() -> None:
    # embed.html and calculator.html build the same slugs in JavaScript:
    # s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "")
    assert slug("Saran at Chapra") == "saran-at-chapra"
    assert slug("Mumbai CMM Courts") == "mumbai-cmm-courts"
    assert slug("Unakoti  Tripura") == "unakoti-tripura"
    assert slug("Jammu and Kashmir") == "jammu-and-kashmir"


if __name__ == "__main__":
    test_pages_and_api()
    test_no_broken_values()
    test_sitemap_lists_every_page()
    test_slug_matches_the_browser()
    print("ok")
