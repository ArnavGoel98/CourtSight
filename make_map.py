#!/usr/bin/env python3
"""Build the district map geometry used by the site (sources/india_districts_2011.json). Run once; the output is
committed, so building the site does not need any of the tools below.

Boundaries: Census of India 2011 districts (641), from the DataMeet community maps repository,
https://github.com/datameet/maps (Districts/Census_2011/2011_Dist.*), licensed CC BY 4.0. The boundaries are those
of the 2011 Census, which show the whole of India as claimed by the Government of India.

Steps: download the shapefile into data/ (gitignored) and check its SHA-256; project it (Lambert conformal conic
for India) and simplify it to 5% of its vertices with mapshaper, which keeps shared borders shared; then write each
district as a compact SVG path (relative moves, integer coordinates on a 2000-unit-wide canvas).

Linking DDL districts to these shapes: DDL's own key gives a 2011 Census district for 534 of 630 units
(ddl_compact/district_key.csv). The rest are matched by hand in sources/map_crosswalk.csv: renamed districts
(Gulbarga -> Kalaburagi), districts carved out after 2011 (shown inside their 2011 parent, pooled with it), cities
whose courts DDL records as city-wide units (Mumbai), and cities split into several court units (Jaipur, Jodhpur).

Requires: node (npx mapshaper@0.6.102), pyshp. Usage: python3 make_map.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Final

import shapefile  # pyshp

HERE: Final = Path(__file__).resolve().parent
RAW: Final = HERE / "data" / "census2011"
OUT: Final = HERE / "sources" / "india_districts_2011.json"
URL: Final = "https://raw.githubusercontent.com/datameet/maps/master/Districts/Census_2011/2011_Dist."
PROJ: Final = "+proj=lcc +lat_1=12 +lat_2=28 +lat_0=22 +lon_0=82 +datum=WGS84"
WIDTH: Final = 2000


def download() -> dict[str, str]:
    """Fetch the shapefile once; its SHA-256 digests are written into the output, and a rerun stops if they change."""
    RAW.mkdir(parents=True, exist_ok=True)
    digests = {}
    for ext in ("shp", "dbf", "shx", "prj"):
        f = RAW / f"2011_Dist.{ext}"
        if not f.exists():
            urllib.request.urlretrieve(URL + ext, f)
        digests[ext] = hashlib.sha256(f.read_bytes()).hexdigest()
    if OUT.exists() and json.loads(OUT.read_text())["sha256"] != digests:
        raise SystemExit(f"the downloaded shapefile differs from the one {OUT.name} was built from; check it first")
    return digests


def rings(g: dict) -> list[list[list[float]]]:
    return g["coordinates"] if g["type"] == "Polygon" else [r for p in g["coordinates"] for r in p]


def main() -> None:
    digests = download()
    tmp = RAW / "projected.json"
    subprocess.run(["npx", "--yes", "mapshaper@0.6.102", str(RAW / "2011_Dist.shp"), "-proj", PROJ,
                    "-simplify", "5%", "keep-shapes", "-clean", "-filter-fields", "censuscode",
                    "-o", str(tmp), "format=geojson", "precision=100", "force"], check=True)
    feats = json.loads(tmp.read_text())["features"]
    pts = [p for f in feats for r in rings(f["geometry"]) for p in r]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    s = WIDTH / (x1 - x0)
    paths = {}
    for f in feats:
        parts = []
        for r in rings(f["geometry"]):
            q = [(round((x - x0) * s), round((y1 - y) * s)) for x, y in r]
            q = [q[0]] + [b for a, b in zip(q, q[1:]) if b != a]
            if len(q) < 3:
                continue
            rel = " ".join(f"{b[0] - a[0]} {b[1] - a[1]}" for a, b in zip(q, q[1:]))
            parts.append((f"M{q[0][0]} {q[0][1]}l{rel}z").replace(" -", "-"))
        paths[str(f["properties"]["censuscode"])] = "".join(parts)
    names = {str(r["censuscode"]): [r["DISTRICT"], r["ST_NM"]] for r in shapefile.Reader(str(RAW / "2011_Dist")).records()}
    OUT.write_text(json.dumps({"source": "Census of India 2011 district boundaries via DataMeet "
                                         "(github.com/datameet/maps), CC BY 4.0",
                               "sha256": digests, "w": WIDTH, "h": round((y1 - y0) * s),
                               "names": names, "paths": paths}, separators=(",", ":")) + "\n")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size / 1e3:.0f} kB, {len(paths)} districts)")


if __name__ == "__main__":
    main()
