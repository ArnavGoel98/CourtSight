#!/usr/bin/env python3
"""Shrink the unzipped DDL judicial-data folder (~81M case rows) into a small package for the model.

Run on the machine that has the data:
    pip install pandas pyarrow
    python compress_ddl.py --src "PATH/TO/UNZIPPED/FOLDER" --out ddl_compact

Output (ddl_compact/, typically tens of MB, every file < 24 MB so GitHub web upload accepts it):
    cube/state=<S>[_<k>].parquet  case counts by district x filing month x decision month x criminal flag
    snapshot.parquet              hearing-gap and stage counts of cases still pending at collection
    judges.csv.gz                 judge postings (start/end dates per court seat)
    audit_type_labels.csv         every case-type label, its count, and the category it was mapped to
    audit_purpose_labels.csv      same for hearing purpose (stage) labels
    report.json                   files found, columns, row counts, date ranges -> lets the modeller check the schema
No case-level or personal fields leave the machine: only aggregated counts.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import IO, Final, Iterator

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import CASE_CATEGORIES, DDL_CASE_COLS, DDL_JUDGE_COLS, STAGE_PATTERNS, classify  # noqa: E402

LOG: Final = logging.getLogger("compress_ddl")
CHUNK: Final = 1_000_000
MAX_PART_BYTES: Final = 24 * 1024 * 1024
CASE_FILE_RE: Final = re.compile(r"cases_(\d{4})\.(csv|csv\.gz|dta|parquet)$", re.I)
ARCHIVE_RE: Final = re.compile(r"\.(tar\.gz|tgz|tar)$", re.I)
WANTED: Final = frozenset({*DDL_CASE_COLS, "year"})
LABEL_COLS: Final = ("type_name", "purpose_name")


# ----------------------------------------------------------------------------------------- discovery
def discover(src: Path) -> dict[str, list[Path]]:
    files = [p for p in src.rglob("*") if p.is_file()]
    out = {
        "cases": sorted(p for p in files if CASE_FILE_RE.search(p.name)),
        "archives": sorted(p for p in files if ARCHIVE_RE.search(p.name) and "case" in p.name.lower()),
        "judges": sorted(p for p in files if "judge" in p.name.lower() and "key" not in p.name.lower()
                         and "case" not in p.name.lower() and re.search(r"\.(csv|csv\.gz|dta|parquet)$", p.name, re.I)),
        "keys": sorted(p for p in files if "key" in p.name.lower() and re.search(r"\.(csv|csv\.gz|dta)$", p.name, re.I)),
    }
    out["other"] = sorted(set(files) - {p for v in out.values() for p in v})
    return out


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".dta":
        return pd.read_stata(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def iter_case_chunks(paths: list[Path], archives: list[Path]) -> Iterator[tuple[str, pd.DataFrame]]:
    def usecols(c: str) -> bool:
        return c in WANTED

    for p in paths:
        name = p.name.lower()
        if name.endswith(".dta"):
            for ch in pd.read_stata(p, chunksize=CHUNK, columns=None):
                yield p.name, ch[[c for c in ch.columns if c in WANTED]]
        elif name.endswith(".parquet"):
            yield p.name, pd.read_parquet(p, columns=None).pipe(lambda d: d[[c for c in d.columns if c in WANTED]])
        else:
            for ch in pd.read_csv(p, usecols=usecols, chunksize=CHUNK, dtype=str):
                yield p.name, ch
    for a in archives:
        with tarfile.open(a, "r:*") as tf:
            for m in tf:
                if m.isfile() and CASE_FILE_RE.search(Path(m.name).name):
                    fh: IO[bytes] | None = tf.extractfile(m)
                    if fh is None:
                        continue
                    comp = "gzip" if m.name.lower().endswith(".gz") else None
                    for ch in pd.read_csv(fh, usecols=usecols, chunksize=CHUNK, dtype=str, compression=comp):
                        yield Path(m.name).name, ch


# ----------------------------------------------------------------------------------------- decoding
class KeyDecoder:
    """Maps numeric label codes (type_name=17) to strings via DDL *_key files when the case files hold codes."""

    def __init__(self, key_paths: list[Path]) -> None:
        self.maps: dict[str, tuple[list[str], dict[tuple, str]]] = {}
        for col in LABEL_COLS:
            cands = [p for p in key_paths if col in p.name.lower()]
            if not cands:
                continue
            k = _read_table(cands[0])
            k.columns = [c.lower() for c in k.columns]
            if col not in k.columns:
                continue
            label_cols = [c for c in k.columns if c.endswith("_s")] or [
                c for c in k.columns if c not in (col, "year", "count") and k[c].dtype == object]
            if not label_cols:
                continue
            on = ["year", col] if "year" in k.columns else [col]
            keys = [tuple(str(v).split(".")[0] for v in row) for row in k[on].itertuples(index=False)]
            self.maps[col] = (on, dict(zip(keys, k[label_cols[0]].astype(str))))
            LOG.info("key file %s: %d codes for %s (join on %s)", cands[0].name, len(keys), col, on)

    def decode(self, df: pd.DataFrame, col: str) -> pd.Series:
        s = df[col]
        numeric = s.dropna().astype(str).str.fullmatch(r"\d+(\.0)?").mean() if s.notna().any() else 0.0
        if col not in self.maps or numeric < 0.9:
            return s.astype("string")
        on, mapping = self.maps[col]
        if any(c not in df.columns for c in on):
            on = [col]
            mapping = {(k[-1],): v for k, v in mapping.items()}
        keys = pd.MultiIndex.from_arrays([df[c].astype(str).str.split(".").str[0] for c in on])
        return pd.Series([mapping.get(k if isinstance(k, tuple) else (k,)) for k in keys], index=df.index, dtype="string")


def _ym(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'YYYY-MM-DD' (or similar) -> YYYYMM int; unparseable -> 0."""
    d = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if s.notna().sum() and d.notna().sum() < 0.5 * s.notna().sum():
        d = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return (d.dt.year * 100 + d.dt.month).fillna(0).astype(np.int32), d


# ----------------------------------------------------------------------------------------- main pass
def compress(src: Path, out: Path) -> None:
    t0 = time.time()
    found = discover(src)
    report: dict[str, object] = {"src": str(src), "files": {k: [f"{p.relative_to(src)} ({p.stat().st_size / 1e6:.1f} MB)"
                                                             for p in v][:200] for k, v in found.items()}}
    if not found["cases"] and not found["archives"]:
        raise SystemExit(f"No cases_YYYY.csv / .csv.gz / .dta / archive found under {src}. Files seen: "
                         f"{[p.name for p in found['other'][:30]]}")
    LOG.info("found %d case files, %d case archives, %d judge files, %d key files",
             len(found["cases"]), len(found["archives"]), len(found["judges"]), len(found["keys"]))
    dec = KeyDecoder(found["keys"])
    out.mkdir(parents=True, exist_ok=True)

    cube_parts: list[pd.DataFrame] = []
    snap_parts: list[pd.DataFrame] = []
    type_counts: Counter[str] = Counter()
    purpose_counts: Counter[str] = Counter()
    columns_seen: dict[str, list[str]] = {}
    stats = {"rows": 0, "rows_bad_filing": 0, "rows_decided_before_filing": 0}
    filing_years: Counter[int] = Counter()
    max_decided = pd.Timestamp.min
    for fname, ch in iter_case_chunks(found["cases"], found["archives"]):
        columns_seen.setdefault(fname, list(ch.columns))
        need = {"state_code", "dist_code", "date_of_filing", "date_of_decision", "type_name"}
        missing = need - set(ch.columns)
        if missing:
            raise SystemExit(f"{fname} lacks required columns {sorted(missing)}; columns are {list(ch.columns)}")
        stats["rows"] += len(ch)
        f, _ = _ym(ch["date_of_filing"])
        d, d_dt = _ym(ch["date_of_decision"])
        ok = (f > 0) & ~((d > 0) & (d < f))
        stats["rows_bad_filing"] += int((f == 0).sum())
        stats["rows_decided_before_filing"] += int(((d > 0) & (d < f)).sum())
        if d_dt.notna().any():
            max_decided = max(max_decided, d_dt.max())
        tlab = dec.decode(ch, "type_name")
        type_counts.update(tlab.fillna("<NA>").value_counts().to_dict())
        crim = np.char.startswith(classify(tlab, CASE_CATEGORIES, "other").astype(str), "crim_").astype(np.int8)
        dist = (pd.to_numeric(ch["state_code"], errors="coerce").fillna(-1).astype(np.int64) * 1000
                + pd.to_numeric(ch["dist_code"], errors="coerce").fillna(-1).astype(np.int64))
        frame = pd.DataFrame({"district_id": dist.to_numpy(), "f": f.to_numpy(), "d": d.to_numpy(), "crim": crim})[ok.to_numpy()]
        filing_years.update((frame["f"] // 100).value_counts().to_dict())
        cube_parts.append(frame.groupby(["district_id", "f", "d", "crim"], sort=False).size().rename("n").reset_index())

        if {"date_last_list", "purpose_name"} <= set(ch.columns):
            live = ok & (d == 0)
            ll, ll_dt = _ym(ch["date_last_list"])
            live &= ll > 0
            sub = ch.loc[live.to_numpy()]
            plab = dec.decode(sub, "purpose_name")
            purpose_counts.update(plab.fillna("<NA>").value_counts().to_dict())
            nxt = pd.to_datetime(sub.get("date_next_list"), format="%Y-%m-%d", errors="coerce") if "date_next_list" in sub else None
            gap = (nxt - ll_dt[live.to_numpy()]).dt.days.to_numpy(dtype=np.float64) if nxt is not None else np.full(len(sub), np.nan)
            has = np.isfinite(gap) & (gap > 0)
            snap_parts.append(pd.DataFrame({
                "district_id": dist[live.to_numpy()].to_numpy(), "t_ym": ll[live.to_numpy()].to_numpy(),
                "stage": classify(plab, STAGE_PATTERNS, "unknown"), "n": 1,
                "n_gap": has.astype(np.int32), "sum_log_gap": np.where(has, np.log(np.clip(gap, 1, 730)), 0.0),
            }).groupby(["district_id", "t_ym", "stage"], sort=False).sum().reset_index())
        if len(cube_parts) >= 20:  # keep memory flat
            cube_parts = [pd.concat(cube_parts).groupby(["district_id", "f", "d", "crim"], sort=False)["n"].sum().reset_index()]
        LOG.info("%s: %s rows so far (%.0fs)", fname, f"{stats['rows']:,}", time.time() - t0)

    cube = pd.concat(cube_parts).groupby(["district_id", "f", "d", "crim"])["n"].sum().reset_index()
    cube = cube.astype({"district_id": np.int32, "f": np.int32, "d": np.int32, "crim": np.int8, "n": np.int32})
    (out / "cube").mkdir(exist_ok=True)
    for state, g in cube.groupby(cube["district_id"] // 1000):
        _write_parts(g.sort_values(["district_id", "f", "d"]), out / "cube", f"state={state}")
    if snap_parts:
        snap = pd.concat(snap_parts).groupby(["district_id", "t_ym", "stage"])[["n", "n_gap", "sum_log_gap"]].sum().reset_index()
        snap.to_parquet(out / "snapshot.parquet", index=False, compression="zstd")
    _audit(type_counts, CASE_CATEGORIES, out / "audit_type_labels.csv")
    _audit(purpose_counts, STAGE_PATTERNS, out / "audit_purpose_labels.csv")

    judge_cols = {}
    if found["judges"]:
        j = _read_table(found["judges"][0])
        judge_cols = {found["judges"][0].name: list(j.columns)}
        keep = [c for c in DDL_JUDGE_COLS if c in j.columns]
        j[keep].to_csv(out / "judges.csv.gz", index=False, compression="gzip")
        LOG.info("judges: %d rows from %s", len(j), found["judges"][0].name)
    else:
        LOG.warning("no judges file found; working-judge features will be missing")

    report |= {
        "rows": stats, "filing_years": dict(sorted(filing_years.items())),
        "max_decision_date (collection date proxy)": str(max_decided.date()) if max_decided != pd.Timestamp.min else None,
        "districts": int(cube["district_id"].nunique()), "cube_rows": len(cube),
        "case_columns": columns_seen, "judge_columns": judge_cols,
        "decoded_with_key_files": sorted(dec.maps), "seconds": round(time.time() - t0),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    LOG.info("done: %d districts, %s case rows -> %s (%.1f MB total) in %.0fs",
             report["districts"], f"{stats['rows']:,}", out, total / 1e6, time.time() - t0)


def _write_parts(g: pd.DataFrame, folder: Path, stem: str) -> None:
    n_parts = 1
    while True:
        paths = []
        bounds = np.linspace(0, len(g), n_parts + 1).astype(int)
        for k, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            part = g.iloc[lo:hi]
            p = folder / (f"{stem}.parquet" if n_parts == 1 else f"{stem}_{k}.parquet")
            part.to_parquet(p, index=False, compression="zstd", compression_level=19)
            paths.append(p)
        if all(p.stat().st_size < MAX_PART_BYTES for p in paths):
            return
        for p in paths:
            p.unlink()
        n_parts *= 2


def _audit(counts: Counter[str], rules: tuple[tuple[str, str], ...], path: Path) -> None:
    if not counts:
        return
    a = pd.DataFrame(counts.most_common(), columns=["label", "n"])
    a["assigned"] = classify(a["label"], rules, "other/unknown")
    a.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="unzipped DDL folder")
    ap.add_argument("--out", type=Path, default=Path("ddl_compact"))
    ap.add_argument("--list-only", action="store_true", help="only list what was found, then stop")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.list_only:
        for k, v in discover(args.src).items():
            print(f"\n== {k} ({len(v)})")
            for p in v[:50]:
                print(f"  {p.relative_to(args.src)}  {p.stat().st_size / 1e6:.1f} MB")
        return
    compress(args.src, args.out)


if __name__ == "__main__":
    main()
