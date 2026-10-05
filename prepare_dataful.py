#!/usr/bin/env python3
"""Pack Dataful (Factly) NJDG district-court downloads into small parquet files for the model.

Download these datasets from dataful.in in your browser (CSV, XLSX or PARQUET), put them in one folder:
    21265  pending cases (civil + criminal) by district x filing year
    21266  pending civil cases by district x filing year
    21282  disposed cases (civil + criminal) by district x disposal year x filing year
    21280  disposed civil cases by district x disposal year x filing year
    21277  pending cases by district x case stage            (optional)
Then:
    python3 court_pendency/prepare_dataful.py --src ~/Downloads/dataful --out court_pendency/dataful
The output is a few parquet files plus report.json. Dataful data is paid and this repo is public, so the output
folder is git-ignored: keep it on your computer and run grade_2019.py there; publish only its aggregate summary.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Final

import pandas as pd

LOG: Final = logging.getLogger("prepare_dataful")
GEO: Final = ("data_as_of", "state", "district_as_per_source", "district_as_per_lgd", "district_lgd_code")
# dataset -> (output name, value column, extra key columns); identified by columns, not file name
KINDS: Final = {
    "pending_by_year": ("pending_cases", ("instituted_year",)),
    "disposed_by_year": ("cases_disposed", ("disposed_year", "instituted_year")),
    "pending_by_stage": ("pending_cases", ("stage_category",)),
}


def read_any(p: Path) -> pd.DataFrame:
    suf = "".join(p.suffixes).lower()
    if suf.endswith(".parquet"):
        return pd.read_parquet(p)
    if suf.endswith((".xlsx", ".xls")):
        return pd.read_excel(p)
    return pd.read_csv(p, low_memory=False)


def kind_of(df: pd.DataFrame) -> str | None:
    cols = set(df.columns)
    if {"disposed_year", "instituted_year", "cases_disposed"} <= cols:
        return "disposed_by_year"
    if {"instituted_year", "pending_cases"} <= cols:
        return "pending_by_year"
    if {"stage_category", "pending_cases"} <= cols:
        return "pending_by_stage"
    return None


def scope_of(p: Path, df: pd.DataFrame) -> str:
    """'all' (civil + criminal) or 'civil'; from the dataset id or file name."""
    name = p.name.lower()
    if any(k in name for k in ("21266", "21280", "civil")) and "criminal" not in name:
        return "civil"
    if any(k in name for k in ("21281", "criminal")) and "civil" not in name:
        return "criminal"
    return "all"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("court_pendency/dataful"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    files = sorted(p for p in args.src.expanduser().iterdir()
                   if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls", ".parquet"))
    if not files:
        raise SystemExit(f"no CSV/XLSX/PARQUET files in {args.src}")
    for p in files:
        df = read_any(p)
        df.columns = [str(c).strip().lower() for c in df.columns]
        kind = kind_of(df)
        if kind is None:
            LOG.warning("skipping %s: unrecognised columns %s", p.name, list(df.columns))
            report[p.name] = {"skipped": list(df.columns)}
            continue
        value, keys = KINDS[kind]
        scope = scope_of(p, df)
        keep = [c for c in (*GEO, *keys, value) if c in df.columns]
        out = df[keep].copy()
        out["data_as_of"] = pd.to_datetime(out["data_as_of"], errors="coerce").dt.date.astype("string")
        dest = args.out / f"{kind}__{scope}.parquet"
        if any(isinstance(v, dict) and v.get("out") == dest.name for v in report.values()):
            raise SystemExit(f"{p.name} and an earlier file both look like {kind} ({scope}); keep the dataset id "
                             "(e.g. 21266) or 'civil' in the civil-only file names")
        out.to_parquet(dest, index=False, compression="zstd")
        report[p.name] = {"kind": kind, "scope": scope, "rows": len(out), "columns": keep,
                          "snapshots": sorted(out["data_as_of"].dropna().unique().tolist()),
                          "states": int(out["state"].nunique()), "districts": int(out["district_as_per_source"].nunique()),
                          "total": float(out[value].sum()), "out": dest.name, "mb": round(dest.stat().st_size / 1e6, 2)}
        LOG.info("%s -> %s (%d rows, %.1f MB)", p.name, dest.name, len(out), dest.stat().st_size / 1e6)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    LOG.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
