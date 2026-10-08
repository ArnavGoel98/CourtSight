#!/usr/bin/env python3
"""Grade the committed Dec-2018-origin forecasts against what actually happened in 2019, using NJDG data.

The forecasts in results/ddl_2010_2018/forecasts.csv were committed before any 2019 data was looked at, so this
is a genuine out-of-sample test. NJDG (via Dataful datasets 21265 + 21282, packed by prepare_dataful.py) gives
  pending[y]       cases filed in year y still pending at the snapshot date
  disposed[y, d]   cases filed in year y disposed in year d (d >= 2018)
from which the forecast targets for the cases the model can see (filed since 2010) are rebuilt exactly:
  P(end of year Y) = sum_{y=2010..Y} ( pending[y] + sum_{d=Y+1..snapshot} disposed[y, d] )
  D(year Y+1)      = sum_{y=2010..Y+1} disposed[y, Y+1]
  A(year Y+1)      = pending[Y+1] + sum_{d=Y+1..snapshot} disposed[Y+1, d]
  growth = log P(2019) - log P(2018),  clearance ratio = D(2019) / A(2019)
Restored / transferred cases and NJDG revisions make this approximate; the coverage ratio kappa (NJDG / DDL
pending at Dec 2018) and the filing-jump flag mark districts where the two sources disagree.

Dataful data is paid and must not be published: run this locally, keep --private-out out of git, and publish only
the aggregate grading_summary.csv (and crosswalk_review.csv, which holds district names only).

    python3 grade_2019.py --dataful dataful \\
        --district-key ddl_compact/district_key.csv
"""
from __future__ import annotations

import argparse
import difflib
import logging
import re
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pendency_forecast import cluster_bootstrap_skill, pinball_rows  # noqa: E402

LOG: Final = logging.getLogger("grade_2019")
HERE: Final = Path(__file__).resolve().parent
FIRST_COHORT: Final = 2010
QUANTILES: Final = (0.1, 0.5, 0.9)
STATE_ALIASES: Final = {"orissa": "odisha", "uttaranchal": "uttarakhand", "pondicherry": "puducherry",
                        "jammu kashmir": "jammu and kashmir", "nct of delhi": "delhi", "andaman nicobar": "andaman and nicobar"}
DROP_WORDS: Final = re.compile(r"\b(district|dist|court|courts|judgeship|sessions|division|zilla|the)\b")


def norm(s: pd.Series) -> pd.Series:
    out = (s.astype("string").str.lower().str.replace("&", " and ", regex=False)
           .str.replace(r"[^a-z ]", " ", regex=True).str.replace(DROP_WORDS, " ", regex=True)
           .str.replace(r"\s+", " ", regex=True).str.strip())
    return out.replace(STATE_ALIASES)


def load_ddl_key(path: Path) -> pd.DataFrame:
    k = pd.read_csv(path)
    if "year" in k:
        k = k.sort_values("year").drop_duplicates(["state_code", "dist_code"], keep="last")
    k["district_id"] = k["state_code"].astype(np.int64) * 1000 + k["dist_code"].astype(np.int64)
    k["state_n"], k["dist_n"] = norm(k["state_name"]), norm(k["district_name"])
    return k[["district_id", "state_name", "district_name", "state_n", "dist_n"]]


def crosswalk(key: pd.DataFrame, nj: pd.DataFrame, overrides: Path | None, cutoff: float = 0.88) -> pd.DataFrame:
    """NJDG (state, district_as_per_source) -> DDL district_id; several NJDG districts may share one DDL parent
    (districts split after 2018). Exact normalised match, then a close fuzzy match within the state, then overrides."""
    names = nj[["state", "district_as_per_source"]].drop_duplicates().reset_index(drop=True)
    names["state_n"], names["dist_n"] = norm(names["state"]), norm(names["district_as_per_source"])
    exact = names.merge(key[["district_id", "state_n", "dist_n"]], on=["state_n", "dist_n"], how="left")
    exact["method"] = np.where(exact["district_id"].notna(), "exact", "")
    by_state = {s: g for s, g in key.groupby("state_n")}
    cands: list[str] = []
    for i, r in exact.iterrows():
        pool = by_state.get(r["state_n"])
        if pool is None:
            cands.append("")
            continue
        close = difflib.get_close_matches(str(r["dist_n"]), pool["dist_n"].tolist(), n=3, cutoff=0.6)
        cands.append("; ".join(f"{c} ({int(pool.loc[pool['dist_n'] == c, 'district_id'].iloc[0])})" for c in close))
        if r["method"] == "" and close and difflib.SequenceMatcher(None, str(r["dist_n"]), close[0]).ratio() >= cutoff:
            exact.at[i, "district_id"] = pool.loc[pool["dist_n"] == close[0], "district_id"].iloc[0]
            exact.at[i, "method"] = "fuzzy"
    exact["candidates"] = cands
    if overrides is not None and overrides.exists():
        ov = pd.read_csv(overrides)  # columns: state, district_as_per_source, district_id
        ov = ov.set_index(["state", "district_as_per_source"])["district_id"]
        idx = pd.MultiIndex.from_frame(exact[["state", "district_as_per_source"]])
        hit = idx.isin(ov.index)
        exact.loc[hit, "district_id"] = ov.reindex(idx[hit]).to_numpy()
        exact.loc[hit, "method"] = "override"
    exact["method"] = exact["method"].replace("", "unmatched")
    return exact


def _latest_common(p: pd.DataFrame, d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    common = sorted(set(p["data_as_of"].dropna()) & set(d["data_as_of"].dropna()))
    if common:
        as_of = common[-1]
        return p.loc[p["data_as_of"] == as_of], d.loc[d["data_as_of"] == as_of], as_of
    ap, ad = p["data_as_of"].max(), d["data_as_of"].max()
    LOG.warning("no common snapshot date: pending as of %s, disposed as of %s; disposals in between are missed", ap, ad)
    return p.loc[p["data_as_of"] == ap], d.loc[d["data_as_of"] == ad], f"{ap} / {ad}"


def reconstruct(pending: pd.DataFrame, disposed: pd.DataFrame, origin_year: int) -> pd.DataFrame:
    """Per (state, district_as_per_source): P at end of origin_year and origin_year+1, D and A in origin_year+1."""
    y1 = origin_year + 1
    key = ["state", "district_as_per_source"]
    p = pending.assign(y=pd.to_numeric(pending["instituted_year"], errors="coerce"))
    d = disposed.assign(y=pd.to_numeric(disposed["instituted_year"], errors="coerce"),
                        dy=pd.to_numeric(disposed["disposed_year"], errors="coerce"))
    p, d = p.loc[p["y"].between(FIRST_COHORT, y1)], d.loc[d["y"].between(FIRST_COHORT, y1)]
    if d["dy"].min() > y1 or not set(range(FIRST_COHORT, y1 + 1)) <= set(p["y"].dropna().astype(int)):
        raise SystemExit(f"need pending by single filing year {FIRST_COHORT}..{y1} and disposals from {y1} onwards")
    pend = p.pivot_table(index=key, columns="y", values="pending_cases", aggfunc="sum", fill_value=0.0)
    disp = d.pivot_table(index=key + ["y"], columns="dy", values="cases_disposed", aggfunc="sum", fill_value=0.0)
    idx = pend.index.union(disp.index.droplevel("y").unique())
    pend = pend.reindex(idx, fill_value=0.0)

    def after(year: int) -> pd.DataFrame:  # disposals strictly after `year`, by cohort
        cols = [c for c in disp.columns if c > year]
        return disp[cols].sum(axis=1).unstack("y").reindex(index=idx, columns=pend.columns, fill_value=0.0).fillna(0.0)

    stock_end = {yr: (pend + after(yr)).loc[:, [c for c in pend.columns if c <= yr]].sum(axis=1) for yr in (origin_year, y1)}
    d_y1 = disp[y1].unstack("y").reindex(index=idx, fill_value=0.0).fillna(0.0).sum(axis=1) if y1 in disp.columns else 0.0
    later = after(y1 - 1)
    a_y1 = pend[y1] + later[y1]
    return pd.DataFrame({"P0": stock_end[origin_year], "P1": stock_end[y1], "D1": d_y1, "A1": a_y1}).reset_index()


def grade(fc: pd.DataFrame, truth: pd.DataFrame, label: str, draws: int, seed: int) -> list[dict[str, object]]:
    """Pinball / coverage for each forecaster in forecasts.csv (model, linear, drift); skill vs drift and vs linear
    with High-Court block-bootstrap intervals."""
    rows: list[dict[str, object]] = []
    m = fc.merge(truth, on="district_id")
    hc = m["district_id"].to_numpy() // 1000
    for target, y_all in (("growth", m["y_growth"].to_numpy()), ("cr", m["y_cr"].to_numpy())):
        ok = np.isfinite(y_all)
        y = y_all[ok]
        for tau in QUANTILES:
            q = int(round(tau * 100))
            cols = {"model": f"{target}_12_q{q}", "linear": f"{target}_12_linear_q{q}", "drift": f"{target}_12_drift_q{q}"}
            loss = {k: pinball_rows(y, m.loc[ok, c].to_numpy(), tau) for k, c in cols.items() if c in m}
            for name, lm in loss.items():
                row: dict[str, object] = {"subset": label, "forecaster": name, "target": target, "tau": tau,
                                          "n": int(ok.sum()), "pinball": lm.mean(),
                                          "coverage": float(np.mean(y <= m.loc[ok, cols[name]].to_numpy()))}
                for ref in ("drift", "linear"):
                    if ref != name and ref in loss:
                        lo, hi = cluster_bootstrap_skill(lm, loss[ref], hc[ok], draws, seed)
                        row |= {f"skill_vs_{ref}": 1.0 - lm.sum() / loss[ref].sum(),
                                f"skill_vs_{ref}_lo": lo, f"skill_vs_{ref}_hi": hi}
                rows.append(row)
        if target == "cr":
            rows.append({"subset": label, "forecaster": "model", "target": "cr", "tau": "share_below_1",
                         "n": int(ok.sum()), "forecast_q50": float(np.mean(m.loc[ok, "cr_12_q50"] < 1)),
                         "actual": float(np.mean(y < 1))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataful", type=Path, default=HERE / "dataful", help="output folder of prepare_dataful.py")
    ap.add_argument("--district-key", type=Path, default=HERE / "ddl_compact" / "district_key.csv")
    ap.add_argument("--results", type=Path, default=HERE / "results" / "ddl_2010_2018")
    ap.add_argument("--forecasts", type=Path, default=None, help="default: <results>/forecasts.csv")
    ap.add_argument("--overrides", type=Path, default=HERE / "crosswalk_overrides.csv")
    ap.add_argument("--out", type=Path, default=None, help="publishable outputs; default <results>/grading_2019")
    ap.add_argument("--private-out", type=Path, default=HERE / "dataful" / "private", help="per-district actuals (never commit)")
    ap.add_argument("--origin-year", type=int, default=2018)
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = args.out or args.results / "grading_2019"
    out.mkdir(parents=True, exist_ok=True)
    args.private_out.mkdir(parents=True, exist_ok=True)

    pend = pd.read_parquet(args.dataful / "pending_by_year__all.parquet")
    disp = pd.read_parquet(args.dataful / "disposed_by_year__all.parquet")
    pend, disp, as_of = _latest_common(pend, disp)
    LOG.info("NJDG snapshot %s: %d pending rows, %d disposed rows", as_of, len(pend), len(disp))
    nj = reconstruct(pend, disp, args.origin_year)

    cw = crosswalk(load_ddl_key(args.district_key), nj, args.overrides)
    cw.to_csv(out / "crosswalk_review.csv", index=False)
    LOG.info("crosswalk: %s", cw["method"].value_counts().to_dict())
    nj = nj.merge(cw[["state", "district_as_per_source", "district_id"]], on=["state", "district_as_per_source"])
    nj = nj.loc[nj["district_id"].notna()].astype({"district_id": np.int64})
    truth = nj.groupby("district_id")[["P0", "P1", "D1", "A1"]].sum()  # NJDG splits of one DDL district re-aggregated
    with np.errstate(divide="ignore", invalid="ignore"):
        truth["y_growth"] = np.log(truth["P1"] / truth["P0"])
        truth["y_cr"] = truth["D1"] / truth["A1"]
    origin = pd.read_csv(args.results / "origin_state.csv").set_index("district_id")
    truth["kappa_2018"] = truth["P0"] / origin["pending"].reindex(truth.index)
    truth["filing_jump"] = truth["A1"] / origin["instituted_12"].reindex(truth.index)
    truth = truth.reset_index()
    truth.to_csv(args.private_out / f"actuals_{args.origin_year + 1}.csv", index=False)

    fc = pd.read_csv(args.forecasts or args.results / "forecasts.csv")
    clean = truth.loc[truth["kappa_2018"].between(0.5, 2.0) & truth["filing_jump"].between(0.5, 2.0)]
    LOG.info("graded districts: %d matched, %d clean (NJDG/DDL coverage and filings within 2x)", len(truth), len(clean))
    rows = [r for sub, lab in ((truth, "all_matched"), (clean, "clean")) if len(sub) for r in grade(fc, sub, lab, args.draws, 0)]
    summary = pd.DataFrame(rows)
    summary.insert(0, "njdg_snapshot", as_of)
    summary["kappa_median"] = float(truth["kappa_2018"].median())
    summary.to_csv(out / "grading_summary.csv", index=False)
    LOG.info("grading summary:\n%s", summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
