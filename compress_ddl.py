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
import io
import json
import logging
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import IO, Callable, Final, Iterator

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import CASE_CATEGORIES, DDL_CASE_COLS, DDL_JUDGE_COLS, STAGE_PATTERNS, classify  # noqa: E402

LOG: Final = logging.getLogger("compress_ddl")
CHUNK: Final = 1_000_000
MAX_PART_BYTES: Final = 24 * 1024 * 1024
TABLE_RE: Final = re.compile(r"\.(csv|csv\.gz|dta)$", re.I)
ARCHIVE_RE: Final = re.compile(r"\.(tar\.gz|tgz|tar)$", re.I)
WANTED: Final = frozenset({*DDL_CASE_COLS, "year"})
LABEL_COLS: Final = ("type_name", "purpose_name")


# ----------------------------------------------------------------------------------------- discovery
def _is_case(name: str) -> bool:
    n = name.lower()
    return "case" in n and not any(x in n for x in ("key", "act", "judge"))


def _is_judge(name: str) -> bool:
    n = name.lower()
    return "judge" in n and not any(x in n for x in ("key", "case"))


def _is_key(name: str) -> bool:
    return "key" in name.lower()


def _prefer_csv(paths: list[Path]) -> list[Path]:
    """DDL ships every dataset twice (csv/ and dta/); keep the CSV copies when both exist."""
    csv = [p for p in paths if "csv" in (q.lower() for q in p.parts) or re.search(r"\.csv(\.gz)?$", p.name, re.I)]
    return sorted(csv) if csv else sorted(paths)


def discover(src: Path) -> dict[str, list[Path]]:
    files = [p for p in src.rglob("*") if p.is_file() and not p.name.startswith(".")]
    tables = [p for p in files if TABLE_RE.search(p.name)]
    archives = [p for p in files if ARCHIVE_RE.search(p.name)]
    out = {
        "cases": _prefer_csv([p for p in tables + archives if _is_case(p.name)]),
        "judges": _prefer_csv([p for p in tables + archives if _is_judge(p.name)]),
        "keys": _prefer_csv([p for p in tables + archives if _is_key(p.name)]),
    }
    out["skipped"] = sorted(set(files) - {p for v in out.values() for p in v})
    return out


def _read_bytes_table(fh: IO[bytes], name: str) -> pd.DataFrame:
    if name.lower().endswith(".dta"):
        return pd.read_stata(io.BytesIO(fh.read()))
    return pd.read_csv(fh, low_memory=False, compression="gzip" if name.lower().endswith(".gz") else None)


def read_small_tables(sources: list[Path], pred: Callable[[str], bool]) -> list[tuple[str, pd.DataFrame]]:
    """Small tables (keys, judges) from plain files or from inside .tar.gz archives."""
    out: list[tuple[str, pd.DataFrame]] = []
    for p in sources:
        if ARCHIVE_RE.search(p.name):
            with tarfile.open(p, "r:*") as tf:
                for m in tf.getmembers():
                    base = Path(m.name).name
                    if m.isfile() and not base.startswith(".") and TABLE_RE.search(base) and pred(base):
                        fh = tf.extractfile(m)
                        if fh is not None:
                            out.append((base, _read_bytes_table(fh, base)))
        else:
            with open(p, "rb") as fh:
                out.append((p.name, _read_bytes_table(fh, p.name)))
    return out


class _Unseekable(io.RawIOBase):
    """Adapter so pandas can read a member of a streamed (non-seekable) tar.gz."""

    def __init__(self, fh: IO[bytes]) -> None:
        self._fh = fh

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, b: bytearray) -> int:  # type: ignore[override]
        data = self._fh.read(len(b))
        b[: len(data)] = data
        return len(data)


def iter_case_chunks(sources: list[Path]) -> Iterator[tuple[str, pd.DataFrame]]:
    """Stream case rows in 1M-row chunks from plain files or straight out of .tar.gz (no unpacking)."""

    def usecols(c: str) -> bool:
        return c in WANTED

    def from_handle(fh: IO[bytes], name: str) -> Iterator[pd.DataFrame]:
        if name.lower().endswith(".dta"):
            for ch in pd.read_stata(io.BytesIO(fh.read()), chunksize=CHUNK):
                yield ch[[c for c in ch.columns if c in WANTED]].astype("string")
        else:
            comp = "gzip" if name.lower().endswith(".gz") else None
            yield from pd.read_csv(fh, usecols=usecols, chunksize=CHUNK, dtype=str, compression=comp)

    for p in sources:
        if ARCHIVE_RE.search(p.name):
            LOG.info("streaming %s ...", p.name)
            with tarfile.open(p, "r|*") as tf:  # sequential: never extracts to disk
                for m in tf:
                    base = Path(m.name).name
                    if m.isfile() and not base.startswith(".") and TABLE_RE.search(base) and _is_case(base):
                        fh = tf.extractfile(m)
                        if fh is not None:
                            stream = io.BufferedReader(_Unseekable(fh), buffer_size=1 << 20)
                            for ch in from_handle(stream, base):
                                yield base, ch
        else:
            with open(p, "rb") as fh:
                for ch in from_handle(fh, p.name):
                    yield p.name, ch


# ----------------------------------------------------------------------------------------- decoding
class KeyDecoder:
    """Maps numeric label codes (type_name=17) to strings via DDL *_key files when the case files hold codes."""

    def __init__(self, key_tables: list[tuple[str, pd.DataFrame]]) -> None:
        self.maps: dict[str, tuple[list[str], dict[tuple, str]]] = {}
        for col in LABEL_COLS:
            cands = [(n, k) for n, k in key_tables if col in n.lower()]
            if not cands:
                continue
            name, k = cands[0]
            k = k.copy()
            k.columns = [c.lower() for c in k.columns]
            if col not in k.columns:
                continue
            label_cols = [c for c in k.columns if c.endswith("_s")] or [
                c for c in k.columns if c not in (col, "year", "count") and not pd.api.types.is_numeric_dtype(k[c])]
            if not label_cols:
                continue
            on = ["year", col] if "year" in k.columns else [col]
            keys = [tuple(str(v).split(".")[0] for v in row) for row in k[on].itertuples(index=False)]
            self.maps[col] = (on, dict(zip(keys, k[label_cols[0]].astype(str))))
            LOG.info("key file %s: %d codes for %s (join on %s)", name, len(keys), col, on)

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
def compress(src: Path, out: Path, detail: bool = False) -> None:
    t0 = time.time()
    found = discover(src)
    report: dict[str, object] = {"src": str(src), "files": {k: [f"{p.relative_to(src)} ({p.stat().st_size / 1e6:.1f} MB)"
                                                             for p in v][:200] for k, v in found.items()}}
    if not found["cases"]:
        raise SystemExit(f"No case files or case archives found under {src}. Files seen: "
                         f"{[p.name for p in found['skipped'][:30]]}")
    LOG.info("using case sources %s, judge sources %s, key sources %s",
             [p.name for p in found["cases"]], [p.name for p in found["judges"]], [p.name for p in found["keys"]])
    key_tables = read_small_tables(found["keys"], _is_key)
    report["key_tables"] = {n: list(k.columns) for n, k in key_tables}
    out.mkdir(parents=True, exist_ok=True)
    for name, k in key_tables:
        if "district" in name.lower() and {"state_code", "dist_code"} <= set(k.columns):
            # district names <-> eCourts codes (+ Census 2011 ids): needed to match NJDG exports, which use names
            keep = [c for c in ("year", "state_code", "state_name", "dist_code", "district_name",
                                "pc11_state_id", "pc11_district_id", "pc11_district_name") if c in k.columns]
            k[keep].drop_duplicates().to_csv(out / "district_key.csv", index=False)
            LOG.info("district key: %d rows from %s", len(k), name)
    dec = KeyDecoder(key_tables)
    out.mkdir(parents=True, exist_ok=True)

    cube_parts: list[pd.DataFrame] = []
    # --detail: case-type cube (time-to-decision by district x case type) and courtroom-month flows (judge transfers)
    cat_parts: list[pd.DataFrame] = []
    court_parts: list[pd.DataFrame] = []
    snap_parts: list[pd.DataFrame] = []
    type_counts: Counter[str] = Counter()
    purpose_counts: Counter[str] = Counter()
    columns_seen: dict[str, list[str]] = {}
    stats = {"rows": 0, "rows_bad_filing": 0, "rows_decided_before_filing": 0}
    filing_years: Counter[int] = Counter()
    max_decided = pd.Timestamp.min
    for fname, ch in iter_case_chunks(found["cases"]):
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
        cat = classify(tlab, CASE_CATEGORIES, "other").astype(str)
        crim = np.char.startswith(cat, "crim_").astype(np.int8)
        dist = (pd.to_numeric(ch["state_code"], errors="coerce").fillna(-1).astype(np.int64) * 1000
                + pd.to_numeric(ch["dist_code"], errors="coerce").fillna(-1).astype(np.int64))
        frame = pd.DataFrame({"district_id": dist.to_numpy(), "f": f.to_numpy(), "d": d.to_numpy(), "crim": crim})[ok.to_numpy()]
        filing_years.update((frame["f"] // 100).value_counts().to_dict())
        cube_parts.append(frame.groupby(["district_id", "f", "d", "crim"], sort=False).size().rename("n").reset_index())
        if detail:
            okv = ok.to_numpy()
            fc = frame.assign(category=cat[okv])
            cat_parts.append(fc.groupby(["district_id", "f", "d", "category"], sort=False).size().rename("n").reset_index())
            court = pd.to_numeric(ch["court_no"], errors="coerce").fillna(-1).astype(np.int32).to_numpy()[okv] \
                if "court_no" in ch else np.full(int(okv.sum()), -1, np.int32)
            cf = frame.assign(court_no=court)
            filed = cf.groupby(["district_id", "court_no", "f"], sort=False).size().rename("filed")
            disp = cf.loc[cf["d"] > 0].groupby(["district_id", "court_no", "d"], sort=False).size().rename("disposed")
            filed.index = filed.index.set_names("ym", level=2)
            disp.index = disp.index.set_names("ym", level=2)
            court_parts.append(pd.concat([filed, disp], axis=1).fillna(0).reset_index())
            if len(cat_parts) >= 20:
                cat_parts = [pd.concat(cat_parts).groupby(["district_id", "f", "d", "category"], sort=False)["n"].sum().reset_index()]
                court_parts = [pd.concat(court_parts).groupby(["district_id", "court_no", "ym"], sort=False)[["filed", "disposed"]].sum().reset_index()]

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
    if detail and cat_parts:
        (out / "detail" / "cube_cat").mkdir(parents=True, exist_ok=True)
        cc = pd.concat(cat_parts).groupby(["district_id", "f", "d", "category"])["n"].sum().reset_index()
        cc = cc.astype({"district_id": np.int32, "f": np.int32, "d": np.int32, "n": np.int32})
        for state, g in cc.groupby(cc["district_id"] // 1000):
            _write_parts(g.sort_values(["district_id", "category", "f", "d"]), out / "detail" / "cube_cat", f"state={state}")
        cm = pd.concat(court_parts).groupby(["district_id", "court_no", "ym"])[["filed", "disposed"]].sum().reset_index()
        cm = cm.astype({"district_id": np.int32, "court_no": np.int32, "ym": np.int32, "filed": np.int32, "disposed": np.int32})
        cm.to_parquet(out / "detail" / "court_month.parquet", index=False, compression="zstd")
        LOG.info("detail: %d case-type cube rows, %d courtroom-months", len(cc), len(cm))
    if snap_parts:
        snap = pd.concat(snap_parts).groupby(["district_id", "t_ym", "stage"])[["n", "n_gap", "sum_log_gap"]].sum().reset_index()
        snap.to_parquet(out / "snapshot.parquet", index=False, compression="zstd")
    _audit(type_counts, CASE_CATEGORIES, out / "audit_type_labels.csv")
    _audit(purpose_counts, STAGE_PATTERNS, out / "audit_purpose_labels.csv")

    judge_cols = {}
    judge_tables = read_small_tables(found["judges"], _is_judge)
    if judge_tables:
        jname, j = judge_tables[0]
        judge_cols = {jname: list(j.columns)}
        keep = [c for c in DDL_JUDGE_COLS if c in j.columns]
        j[keep].to_csv(out / "judges.csv.gz", index=False, compression="gzip")
        LOG.info("judges: %d rows from %s", len(j), jname)
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
    ap.add_argument("--detail", action="store_true",
                    help="also write detail/: counts by case type and courtroom-month filings/disposals")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.list_only:
        found = discover(args.src)
        for k, v in found.items():
            print(f"\n== {k} ({len(v)})")
            for p in v[:50]:
                print(f"  {p.relative_to(args.src)}  {p.stat().st_size / 1e6:.1f} MB")
        for k in ("judges", "keys"):
            for p in found[k]:
                if ARCHIVE_RE.search(p.name):
                    with tarfile.open(p, "r:*") as tf:
                        print(f"\n-- inside {p.relative_to(args.src)}:", [m.name for m in tf.getmembers() if m.isfile()][:40])
        for p in found["cases"][:1]:
            if ARCHIVE_RE.search(p.name):
                print(f"\n-- first files inside {p.relative_to(args.src)} (scanning, may take a minute):")
                with tarfile.open(p, "r|*") as tf:
                    for i, m in enumerate(tf):
                        print(f"   {m.name}  {m.size / 1e6:.1f} MB")
                        if i >= 12:
                            break
        return
    compress(args.src, args.out, detail=args.detail)


if __name__ == "__main__":
    main()
