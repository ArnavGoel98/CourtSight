#!/usr/bin/env python3
"""District-level court pendency forecasting for Indian district judiciaries (DDL eCourts x NJDG).

Stages
  1. ingest    DDL case CSV -> hive-partitioned parquet -> per-state (filing x decision) cohort cubes
  2. harmonise cube -> NJDG aggregate schema; legacy-stock backcast + coverage calibration at the seam
  3. features  leakage-safe temporal, hierarchical (High Court leave-one-out) and spatial-lag features
  4. model     direct multi-horizon LightGBM quantile suites (pinball loss), purged temporal splits,
               split-conformal recalibration, non-crossing by monotone rearrangement
  5. explain   TreeSHAP family decomposition: structural capacity deficit vs transient filing shock
  6. policy    two-way FE + shift-share IV disposal elasticities -> bench / hearing-cadence / surge levers
               (levers are written only when the judge elasticity is identified: strong first stage, plausible value)

DDL-only caveats handled in code
  - DDL stores each pending case's *latest* hearing (as of the 2019-20 scrape), so hearing-stage / hearing-gap
    snapshots cannot be dated back in time without look-ahead; they are descriptive only (scrape_summary) and
    never model features. NJDG monthly aggregates (contemporaneous) are the only source of those features.
  - Pre-2010 filings are absent, so the observed stock, age shape and workload are left-truncated; DDL-only runs
    drop truncation-sensitive features and use age/workload measured inside the fully observed < 3-year window.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterator, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.dataset as ds
import scipy.sparse as sp
from scipy.optimize import linprog

from taxonomy import CASE_CATEGORIES, DDL_CASE_COLS, DDL_JUDGE_COLS, JUDGE_CADRES, STAGE_PATTERNS, STAGES, classify

LOG: Final = logging.getLogger("pendency")
FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]

# ----------------------------------------------------------------------------------------- schema
AGE_EDGES_Y: Final[tuple[float, ...]] = (0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0)  # 30+ capped at 40
AGE_BUCKETS: Final[tuple[str, ...]] = (
    "age_0_1", "age_1_3", "age_3_5", "age_5_10", "age_10_20", "age_20_30", "age_30p",
)
PRETRIAL_STAGES: Final[tuple[str, ...]] = ("service", "charge", "interlocutory")
STAGE_SHARE_COLS: Final[tuple[str, ...]] = tuple(f"share_{s}" for s in STAGES)

DATE_COLS: Final[tuple[str, ...]] = ("filed", "decided", "first_list", "last_list", "next_list")

NJDG_REQUIRED: Final[tuple[str, ...]] = (
    "state_code", "dist_code", "month", "pending", "pending_crim", "instituted", "disposed",
    *AGE_BUCKETS, "working_judges", "sanctioned_judges",
)
NJDG_OPTIONAL: Final[tuple[str, ...]] = ("hearing_gap_days", *STAGE_SHARE_COLS)

FEATURES: Final[tuple[str, ...]] = (
    # top-8 engineered
    "idvr_12", "vacancy_rate", "log_hearing_gap", "age_bowley_skew", "workload_per_judge",
    "pretrial_share", "filing_shock_z", "state_loo_growth_12_past",
    # supporting
    "w_lag_growth_12_past", "state_loo_idvr_12", "state_loo_vacancy_rate", "state_loo_filing_shock_z",
    "log_pending", "growth_12_past", "share_age_gt5y", "share_criminal", "log_working_judges",
    "inflow_trend_growth", "disposal_momentum", "is_njdg", "month_of_year",
    # truncation-robust: measured inside the < 3-year age window, fully observed at every DDL origin >= 2013-01
    "share_age_1_3_of_lt3", "pending_lt3_per_judge",
)
# DDL-only: these depend on the unobserved pre-2010 stock (age > 5y is empty before 2015 by construction)
TRUNCATION_SENSITIVE: Final[tuple[str, ...]] = ("share_age_gt5y", "age_bowley_skew", "workload_per_judge")
# DDL-only: judge postings are recorded mostly for recent years (2.8k seats in Dec 2010 vs 16.3k in Dec 2018 while
# real strength was ~flat), so within-district judge counts trace record coverage, not capacity
JUDGE_RECORD_SENSITIVE: Final[tuple[str, ...]] = (
    "vacancy_rate", "log_working_judges", "pending_lt3_per_judge", "state_loo_vacancy_rate",
)
# small, transparent benchmark: linear quantile regression on the innovation over the same drift anchor
LINEAR_BASELINE_FEATURES: Final[tuple[str, ...]] = (
    "growth_12_past", "idvr_12", "log_pending", "filing_shock_z", "inflow_trend_growth", "disposal_momentum",
    "state_loo_growth_12_past",
)
CADRES: Final[tuple[str, ...]] = tuple(n for n, _ in JUDGE_CADRES)
FEATURE_FAMILY: Final[Mapping[str, str]] = {
    "vacancy_rate": "capacity", "log_working_judges": "capacity", "workload_per_judge": "capacity",
    "log_hearing_gap": "capacity", "disposal_momentum": "capacity",
    "pretrial_share": "procedural", "age_bowley_skew": "procedural", "share_age_gt5y": "procedural",
    "share_criminal": "procedural", "share_age_1_3_of_lt3": "procedural", "pending_lt3_per_judge": "capacity",
    "filing_shock_z": "transient_shock", "state_loo_filing_shock_z": "transient_shock",
    "month_of_year": "transient_shock",
    "inflow_trend_growth": "demand_trend",
    "idvr_12": "flow_state", "log_pending": "flow_state", "growth_12_past": "flow_state", "is_njdg": "flow_state",
    "state_loo_growth_12_past": "spatial", "w_lag_growth_12_past": "spatial", "state_loo_idvr_12": "spatial",
    "state_loo_vacancy_rate": "spatial",
}
STRUCTURAL_FAMILIES: Final[frozenset[str]] = frozenset({"capacity", "procedural"})
TRANSIENT_FAMILIES: Final[frozenset[str]] = frozenset({"transient_shock"})


# ----------------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class Config:
    ddl_start: pd.Period = pd.Period("2010-01", "M")
    ddl_end: pd.Period = pd.Period("2018-12", "M")
    val_start: pd.Period = pd.Period("2022-01", "M")
    cal_months: int = 12
    horizons: tuple[int, ...] = (12, 24, 36)
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    policy_horizon: int = 24
    rmst_horizon_m: int = 96
    min_snapshot_obs: int = 30
    flag_growth_q90: float = float(np.log(1.10))
    backlog_reduction: float = 0.0  # extra fraction of P_t to clear within policy_horizon
    appointment_lag_m: int = 3
    ramp_tau_m: float = 4.0
    seed: int = 20260928
    # structural breaks: rows whose origin-to-target window (t, t+h] touches one are excluded from fit/calibration
    break_windows: tuple[tuple[str, str], ...] = (("2020-03", "2021-06"),)  # COVID physical-court closure
    # DDL-only runs: origins before this are dropped (stock is left-truncated: pre-2010 filings are absent)
    min_origin: pd.Period | None = None
    min_months_on_ecourts: int = 24  # training rows; forecasts require 36
    # rolling-origin validation: (first origin of a 12-month validation year, calibration months); empty -> val_start
    val_folds: tuple[tuple[str, int], ...] = ()
    exclude_features: tuple[str, ...] = ()
    bootstrap_draws: int = 1000
    # judge-elasticity identification gate for policy levers
    min_first_stage_f: float = 10.0
    iv_base_year: int = 2013  # cadre shares fixed in this year; IV sample starts 12 months later


def abs_month(p: pd.Period) -> int:
    return p.year * 12 + p.month - 1


def period_of(t: int) -> pd.Period:
    return pd.Period(year=t // 12, month=t % 12 + 1, freq="M")


@dataclass(frozen=True)
class MonthGrid:
    start: pd.Period
    end: pd.Period

    @property
    def t0(self) -> int:
        return abs_month(self.start)

    @property
    def n(self) -> int:
        return abs_month(self.end) - self.t0 + 1

    def index(self, ts: pd.Series) -> FloatArr:
        ts = pd.to_datetime(ts, errors="coerce")
        return ((ts.dt.year * 12 + ts.dt.month - 1) - self.t0).to_numpy(dtype=np.float64)


def district_key(state: pd.Series | IntArr, dist: pd.Series | IntArr) -> IntArr:
    return np.asarray(state, dtype=np.int64) * 1000 + np.asarray(dist, dtype=np.int64)


# ----------------------------------------------------------------------------------------- 1. ingest
def convert_ddl_csv_to_parquet(csv_glob: str, out_dir: Path, rename: Mapping[str, str] = DDL_CASE_COLS) -> None:
    """One-off: per-year DDL CSVs -> hive-partitioned (state_code) parquet with column projection."""
    files = sorted(glob.glob(csv_glob))
    if not files:
        raise FileNotFoundError(csv_glob)
    types = {c: pa.string() for c in rename if c not in ("state_code", "dist_code", "court_no")}
    types |= {"state_code": pa.int32(), "dist_code": pa.int32(), "court_no": pa.int32()}
    fmt = ds.CsvFileFormat(convert_options=pacsv.ConvertOptions(column_types=types, strings_can_be_null=True))
    src = ds.dataset(files, format=fmt)
    cols = [c for c in rename if c in src.schema.names]
    LOG.info("converting %d CSV files (%d columns) -> %s", len(files), len(cols), out_dir)
    ds.write_dataset(
        src.scanner(columns=cols, batch_size=1 << 20),
        out_dir,
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("state_code", pa.int32())]), flavor="hive"),
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_group=1 << 20,
    )


def iter_ddl_partitions(parquet_dir: Path, rename: Mapping[str, str] = DDL_CASE_COLS) -> Iterator[pd.DataFrame]:
    """Stream one state at a time; peak memory bounded by the largest state (UP ~ 15-20M rows)."""
    dataset = ds.dataset(parquet_dir, format="parquet", partitioning="hive")
    states = sorted(int(p.name.split("=", 1)[1]) for p in parquet_dir.glob("state_code=*"))
    cols = [c for c in rename if c in dataset.schema.names]
    for s in states:
        tbl = dataset.to_table(columns=cols, filter=ds.field("state_code") == s)
        LOG.info("state %s: %d case records", s, tbl.num_rows)
        yield normalise_ddl_cases(tbl.to_pandas(), rename)


def normalise_ddl_cases(raw: pd.DataFrame, rename: Mapping[str, str] = DDL_CASE_COLS) -> pd.DataFrame:
    df = raw.rename(columns=dict(rename))
    for c in DATE_COLS:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce", format="%Y-%m-%d")
    bad = df["decided"].notna() & (df["decided"] < df["filed"])
    df = df.loc[df["filed"].notna() & ~bad].copy()
    df["district_id"] = district_key(df["state_code"], df["dist_code"])
    df["category"] = classify(df["type_name"], CASE_CATEGORIES, "other")
    df["is_criminal"] = np.char.startswith(df["category"].to_numpy(dtype=str), "crim_")
    df["stage"] = classify(df["purpose_name"], STAGE_PATTERNS, "unknown")
    return df


def load_ddl_judges(path: Path, rename: Mapping[str, str] = DDL_JUDGE_COLS) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=dict(rename))
    for c in ("start", "end"):
        raw = df[c].astype("string")
        parsed = pd.to_datetime(raw, format="%d-%m-%Y", errors="coerce")
        if parsed.notna().sum() < 0.5 * raw.notna().sum():
            parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        df[c] = parsed
    if "judge_position" not in df:
        df["judge_position"] = "na"
    df["cadre"] = classify(df["judge_position"], JUDGE_CADRES, "other")
    df["district_id"] = district_key(df["state_code"], df["dist_code"])
    return df.loc[df["start"].notna()]


def load_njdg(path: Path) -> pd.DataFrame:
    """NJDG district-month aggregates (accumulated monthly snapshots; see NJDG schema map)."""
    df = pd.read_csv(path)
    missing = [c for c in NJDG_REQUIRED if c not in df]
    if missing:
        raise ValueError(f"NJDG file missing columns: {missing}")
    for c in NJDG_OPTIONAL:
        if c not in df:
            df[c] = np.nan
    df["t"] = pd.PeriodIndex(df["month"].astype(str), freq="M").map(abs_month).astype(np.int64)
    df["district_id"] = district_key(df["state_code"], df["dist_code"])
    df["source"] = "njdg"
    return df.drop(columns=["month"])


# ------------------------------------------------------------------- 2a. cohort cube -> aggregates
def _bucket_masks(m: int) -> FloatArr:
    """B[f, m, b] = 1 if cohort f is at-risk-eligible (f<=m) and its age m-f falls in bucket b."""
    age = np.arange(m)[None, :] - np.arange(m)[:, None]
    edges = np.asarray(AGE_EDGES_Y) * 12.0
    edges[-1] = np.inf
    return np.stack([(age >= lo) & (age < hi) for lo, hi in zip(edges[:-1], edges[1:])], axis=-1).astype(np.float64)


def cube_to_panel(cases: pd.DataFrame, grid: MonthGrid) -> pd.DataFrame:
    """Case-level rows -> district-month panel (one case = weight 1)."""
    return counts_to_panel(cases["district_id"].to_numpy(), grid.index(cases["filed"]), grid.index(cases["decided"]),
                           cases["is_criminal"].to_numpy(dtype=bool), np.ones(len(cases)), grid)


def counts_to_panel(district: IntArr, f: FloatArr, d: FloatArr, crim: npt.NDArray[np.bool_], w: FloatArr,
                    grid: MonthGrid) -> pd.DataFrame:
    """Exact stock-flow reconstruction from N[i, f, d] = #cases filed in month f, decided in month d.

    pending_i(m) = sum_{f<=m} sum_{d>m} N[i,f,d];  age buckets by (m - f);  identity
    pending(m) - pending(m-1) = instituted(m) - disposed(m) holds by construction.
    f, d are month indices on `grid` (d = NaN when undecided); w = case counts per row.
    """
    m_n = grid.n
    ok = (f >= 0) & (f < m_n)
    codes, dists = pd.factorize(district[ok])
    fi = f[ok].astype(np.int64)
    di = d[ok]
    di = np.where(np.isnan(di) | (di >= m_n), m_n, di).astype(np.int64)  # m_n = censored / beyond panel
    cr, wt = crim[ok], w[ok].astype(np.float64)
    shape = (len(dists), m_n, m_n + 1)
    flat = np.ravel_multi_index((codes.astype(np.int64), fi, di), shape)
    size = int(np.prod(shape))
    cube = np.bincount(flat, weights=wt, minlength=size).reshape(shape)
    cube_c = np.bincount(flat[cr], weights=wt[cr], minlength=size).reshape(shape)

    masks = _bucket_masks(m_n)

    def stock(n: FloatArr) -> FloatArr:
        tail = np.flip(np.cumsum(np.flip(n, axis=2), axis=2), axis=2)  # sum_{d >= j}
        return np.einsum("ifm,fmb->imb", tail[:, :, 1:], masks)  # j = m + 1  ->  d > m

    buckets = stock(cube)
    out = pd.DataFrame(
        {
            "district_id": np.repeat(np.asarray(dists, dtype=np.int64), m_n),
            "t": np.tile(grid.t0 + np.arange(m_n, dtype=np.int64), len(dists)),
            "instituted": cube.sum(axis=2).reshape(-1),
            "disposed": cube.sum(axis=1)[:, :m_n].reshape(-1),
            "pending": buckets.sum(-1).reshape(-1),
            "pending_crim": stock(cube_c).sum(-1).reshape(-1),
        }
    )
    for k, b in enumerate(AGE_BUCKETS):
        out[b] = buckets[:, :, k].reshape(-1)
    return out


@dataclass
class SurvivalStats:
    """Additive sufficient statistics for monthly Kaplan-Meier per class (0=civil, 1=criminal)."""

    horizon: int
    dur_hist: FloatArr  # (2, H+2): censoring/decision month-age histogram
    evt_hist: FloatArr  # (2, H+2): decisions by month-age

    @classmethod
    def empty(cls, horizon: int) -> SurvivalStats:
        z = np.zeros((2, horizon + 2))
        return cls(horizon, z.copy(), z.copy())

    def update(self, cases: pd.DataFrame, grid: MonthGrid, fit_end_idx: int) -> None:
        self.update_counts(grid.index(cases["filed"]), grid.index(cases["decided"]),
                           cases["is_criminal"].to_numpy(dtype=bool), np.ones(len(cases)), fit_end_idx)

    def update_counts(self, f: FloatArr, d: FloatArr, crim: npt.NDArray[np.bool_], w: FloatArr, fit_end_idx: int) -> None:
        ok = (f >= 0) & (f < fit_end_idx)
        f, d, w = f[ok], d[ok], w[ok].astype(np.float64)
        cls = crim[ok].astype(np.int64)
        event = ~np.isnan(d) & (d < fit_end_idx)
        dur = np.where(event, d, fit_end_idx) - f
        dur_c = np.minimum(dur, self.horizon + 1).astype(np.int64)
        evt = event & (dur <= self.horizon)
        for k in (0, 1):
            m = cls == k
            self.dur_hist[k] += np.bincount(dur_c[m], weights=w[m], minlength=self.horizon + 2)
            self.evt_hist[k] += np.bincount(dur_c[m & evt], weights=w[m & evt], minlength=self.horizon + 2)


@dataclass(frozen=True)
class WorkloadWeights:
    """omega[k, b]: mean residual life of a class-k case aged at bucket-b lower edge, / pooled MRL(0)."""

    omega: FloatArr  # (2, n_buckets)
    tail_hazard: float  # monthly hazard for ages beyond the observable horizon (legacy decay)


def fit_workload_weights(stats: SurvivalStats) -> WorkloadWeights:
    h = stats.horizon

    def km(dur: FloatArr, evt: FloatArr) -> FloatArr:
        at_risk = np.flip(np.cumsum(np.flip(dur)))[: h + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            haz = np.where(at_risk > 0, evt[: h + 1] / at_risk, 0.0)
        return np.cumprod(1.0 - haz)  # S(u): undecided after month-age u

    def mrl(s: FloatArr) -> tuple[FloatArr, float]:
        lo = max(h - 12, 0)
        tail = -np.log(max(s[h], 1e-9) / max(s[lo], 1e-9)) / max(h - lo, 1)
        tail = float(np.clip(tail, 1e-3, 1.0))
        s_prev = np.concatenate([[1.0], s[:-1]])
        rest = np.flip(np.cumsum(np.flip(s)))  # sum_{u>=a} S(u)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (rest + s[h] / tail) / np.maximum(s_prev, 1e-9)
        return out, tail

    edges_m = (np.asarray(AGE_EDGES_Y[:-1]) * 12).astype(np.int64)
    pooled_s = km(stats.dur_hist.sum(0), stats.evt_hist.sum(0))
    pooled_mrl, pooled_tail = mrl(pooled_s)
    omega = np.empty((2, len(edges_m)))
    for k in (0, 1):
        m_k, tail_k = mrl(km(stats.dur_hist[k], stats.evt_hist[k]))
        omega[k] = np.where(edges_m <= h, m_k[np.minimum(edges_m, h)], 1.0 / tail_k)
    omega /= pooled_mrl[0]
    LOG.info("workload weights omega (civ/crim x age bucket):\n%s", np.round(omega, 3))
    return WorkloadWeights(omega=omega, tail_hazard=pooled_tail)


def snapshot_observations(cases: pd.DataFrame) -> pd.DataFrame:
    """Cases undecided at scrape, dated at month(last_list): same long schema as compress_ddl's snapshot.parquet.

    Only the *latest* hearing of each still-pending case survives in DDL, so a month-t count depends on what
    happened after t (cases heard again later move out of t). Use for scrape-time description, never as a feature.
    """
    live = cases.loc[cases["decided"].isna() & cases["last_list"].notna()]
    gap = (live["next_list"] - live["last_list"]).dt.days.to_numpy(dtype=np.float64)
    has_gap = np.isfinite(gap) & (gap > 0)
    frame = pd.DataFrame(
        {
            "district_id": live["district_id"].to_numpy(),
            "t_ym": (live["last_list"].dt.year * 100 + live["last_list"].dt.month).to_numpy(dtype=np.int64),
            "stage": live["stage"].to_numpy(),
            "n": 1.0,
            "n_gap": has_gap.astype(np.float64),
            "sum_log_gap": np.where(has_gap, np.log(np.clip(gap, 1, 730)), 0.0),
        }
    )
    return frame.groupby(["district_id", "t_ym", "stage"], as_index=False).sum()


def scrape_summary(snap: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Per district: hearing gap and stage mix of live cases in the 12 months up to that district's scrape month.

    The scrape month is the last month holding >= 20% of the district's peak monthly count (drops far-future typos).
    Describes the court at data collection (2019-20 for DDL); it is not a time series.
    """
    cols = ["scrape_month", "hearing_gap_at_scrape_days", *[f"share_{s}" for s in STAGES]]
    if snap.empty:
        return pd.DataFrame(columns=cols, index=pd.Index([], name="district_id"))
    s = snap.loc[(snap["t_ym"] >= 201001) & (snap["t_ym"] <= 202112)].astype({"n": np.float64})
    s["m"] = (s["t_ym"] // 100) * 12 + s["t_ym"] % 100 - 1
    per_m = s.groupby(["district_id", "m"])["n"].sum().reset_index()
    peak = per_m.groupby("district_id")["n"].transform("max")
    scrape_m = per_m.loc[per_m["n"] >= 0.2 * peak].groupby("district_id")["m"].max()
    s = s.join(scrape_m.rename("m_star"), on="district_id")
    s = s.loc[(s["m"] > s["m_star"] - 12) & (s["m"] <= s["m_star"])]
    g = s.groupby("district_id")[["n_gap", "sum_log_gap"]].sum()
    st = s.pivot_table(index="district_id", columns="stage", values="n", aggfunc="sum", fill_value=0.0)
    st = st.reindex(columns=list(STAGES), fill_value=0.0)
    n_st = st.sum(axis=1)
    out = pd.DataFrame(index=g.index)
    out["scrape_month"] = scrape_m.reindex(out.index).map(lambda m: str(period_of(int(m))))
    out["hearing_gap_at_scrape_days"] = np.where(g["n_gap"] >= min_obs, np.exp(g["sum_log_gap"] / g["n_gap"].clip(lower=1)), np.nan)
    for c in STAGES:
        out[f"share_{c}"] = np.where(n_st.reindex(out.index) >= min_obs, st[c].reindex(out.index) / n_st.reindex(out.index).clip(lower=1), np.nan)
    return out[cols]


def judges_panel(judges: pd.DataFrame, grid: MonthGrid, lookback: int = 36) -> pd.DataFrame:
    """Occupied judicial seats per district-month; sanctioned proxy = seats staffed within `lookback`."""
    m_n = grid.n
    s = grid.index(judges["start"])
    e = grid.index(judges["end"])
    e = np.where(np.isnan(e), m_n - 1, e)
    ok = (s <= m_n - 1) & (e >= 0) & (e >= s)
    j = judges.loc[ok]
    s = np.clip(s[ok], 0, m_n - 1).astype(np.int64)
    e = np.clip(e[ok], 0, m_n - 1).astype(np.int64)
    seat_key = j["district_id"].astype(str) + "|" + j["court_no"].astype(str) + "|" + j["judge_position"].astype(str)
    seat, _ = pd.factorize(seat_key)
    seat_dist = pd.Series(j["district_id"].to_numpy()).groupby(seat).first()
    diff = np.zeros((seat.max() + 1, m_n + 1))
    np.add.at(diff, (seat, s), 1.0)
    np.add.at(diff, (seat, e + 1), -1.0)
    occ = (np.cumsum(diff, axis=1)[:, :m_n] > 0).astype(np.float64)
    cs = np.cumsum(occ, axis=1)
    staffed = ((cs - np.pad(cs, ((0, 0), (lookback, 0)))[:, :m_n]) > 0).astype(np.float64)
    dcode, dists = pd.factorize(seat_dist.to_numpy())
    working = np.zeros((len(dists), m_n))
    sanctioned = np.zeros((len(dists), m_n))
    np.add.at(working, dcode, occ)
    np.add.at(sanctioned, dcode, staffed)
    out = pd.DataFrame(
        {
            "district_id": np.repeat(np.asarray(dists, dtype=np.int64), m_n),
            "t": np.tile(grid.t0 + np.arange(m_n, dtype=np.int64), len(dists)),
            "working_judges": working.reshape(-1),
            "sanctioned_judges": sanctioned.reshape(-1),
        }
    )
    # occupied seats by cadre (a seat takes the cadre of its posting); feeds the shift-share instrument
    seat_cadre = pd.Series(j["cadre"].to_numpy()).groupby(seat).first().to_numpy() if "cadre" in j else None
    for name in CADRES:
        by = np.zeros((len(dists), m_n))
        if seat_cadre is not None:
            m = seat_cadre == name
            np.add.at(by, dcode[m], occ[m])
        out[f"working_{name}"] = by.reshape(-1)
    return out


def ym_index(ym: npt.ArrayLike, grid: MonthGrid) -> FloatArr:
    """YYYYMM ints (0 = missing) -> month index on `grid`."""
    a = np.asarray(ym, dtype=np.int64)
    return np.where(a > 0, (a // 100) * 12 + (a % 100) - 1 - grid.t0, np.nan).astype(np.float64)


def compact_snapshot(path: Path) -> pd.DataFrame:
    """snapshot.parquet from compress_ddl.py (same long schema as snapshot_observations())."""
    if not path.exists():
        return pd.DataFrame(columns=["district_id", "t_ym", "stage", "n", "n_gap", "sum_log_gap"])
    return pd.read_parquet(path)


def iter_compact_states(compact_dir: Path) -> Iterator[pd.DataFrame]:
    files = sorted((compact_dir / "cube").glob("state=*.parquet"))
    if not files:
        raise FileNotFoundError(f"no cube/state=*.parquet under {compact_dir}")
    by_state: dict[str, list[Path]] = {}
    for f in files:
        by_state.setdefault(f.stem.split("_")[0], []).append(f)
    for state, parts in sorted(by_state.items()):
        cube = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        LOG.info("%s: %d cube cells, %d cases", state, len(cube), int(cube["n"].sum()))
        yield cube


def build_ddl_panel(
    ddl_dir: Path, judges_csv: Path, cfg: Config, fit_end_t: int, compact: bool = False
) -> tuple[pd.DataFrame, WorkloadWeights, pd.DataFrame]:
    """ddl_dir = hive parquet of raw cases, or (compact=True) the output folder of compress_ddl.py."""
    grid = MonthGrid(cfg.ddl_start, cfg.ddl_end)
    fit_end_idx = int(np.clip(fit_end_t - grid.t0, 12, grid.n))
    horizon = int(min(cfg.rmst_horizon_m, fit_end_idx - 1))
    stats = SurvivalStats.empty(horizon)
    panels, snaps = [], []
    if compact:
        for cube in iter_compact_states(ddl_dir):
            f, d = ym_index(cube["f"], grid), ym_index(cube["d"], grid)
            crim, w = cube["crim"].to_numpy(dtype=bool), cube["n"].to_numpy(dtype=np.float64)
            dist = cube["district_id"].to_numpy(dtype=np.int64)
            panels.append(counts_to_panel(dist, f, d, crim, w, grid))
            stats.update_counts(f, d, crim, w, fit_end_idx)
        snaps.append(compact_snapshot(ddl_dir / "snapshot.parquet"))
    else:
        for part in iter_ddl_partitions(ddl_dir):
            panels.append(cube_to_panel(part, grid))
            snaps.append(snapshot_observations(part))
            stats.update(part, grid, fit_end_idx)
    panel = pd.concat(panels, ignore_index=True).sort_values(["district_id", "t"])
    scrape = scrape_summary(pd.concat(snaps, ignore_index=True), cfg.min_snapshot_obs)
    # DDL cannot date hearing stage / gap without look-ahead: model features come from NJDG rows only
    panel["hearing_gap_days"] = np.nan
    for s in STAGES:
        panel[f"share_{s}"] = np.nan
    judges = judges_panel(load_ddl_judges(judges_csv), grid)
    panel = panel.merge(judges, on=["district_id", "t"], how="left")
    panel["source"] = "ddl"
    LOG.info("DDL panel: %d districts x %d months", panel["district_id"].nunique(), grid.n)
    return panel.reset_index(drop=True), fit_workload_weights(stats), scrape


# ------------------------------------------------------------- 2b. seam: legacy stock + coverage
def harmonise(ddl: pd.DataFrame, njdg: pd.DataFrame, cfg: Config, weights: WorkloadWeights) -> pd.DataFrame:
    """Align DDL (filings >= ddl_start, left-truncated stock) to NJDG stock at the seam month T0.

    L_b(T0)  = P^njdg_b(T0) * lambda_b,  lambda_b = share of bucket b older than DDL coverage age A0
    L_b(t)   = L_b(T0) * exp(h_tail * (T0 - t)),  re-bucketed at age mid_b - (T0 - t)/12
    kappa_i  = (P^njdg(T0) e^{-g (T0-T_end)} - L(T_end)) / P^ddl(T_end)     (establishment coverage)
    P^corr   = kappa * P^ddl + L;  D^corr = kappa * D^ddl + (L(t-1) - L(t));  A^corr = kappa * A^ddl
    """
    t_end = abs_month(cfg.ddl_end)
    common = np.intersect1d(ddl["district_id"].unique(), njdg["district_id"].unique())
    LOG.info("seam: %d districts in both sources (DDL-only dropped: %d)", len(common),
             ddl["district_id"].nunique() - len(common))
    ddl = ddl.loc[ddl["district_id"].isin(common)].sort_values(["district_id", "t"]).reset_index(drop=True)
    nj = njdg.loc[njdg["district_id"].isin(common)].sort_values(["district_id", "t"])
    first = nj.drop_duplicates("district_id", keep="first").set_index("district_id")
    t_first = first["t"].to_numpy(dtype=np.int64)
    later = nj.merge(first[["t"]].rename(columns={"t": "t0"}), left_on="district_id", right_index=True)
    p12 = later.loc[later["t"] == later["t0"] + 12].set_index("district_id")["pending"].reindex(first.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_m = np.log(p12.to_numpy() / first["pending"].to_numpy()) / 12.0
    g_m = np.clip(np.where(np.isfinite(g_m), g_m, 0.0), -0.1, 0.1)

    edges = np.asarray(AGE_EDGES_Y)
    a0 = (t_first - abs_month(cfg.ddl_start)) / 12.0  # coverage age at T0 (years)
    lo, hi = edges[:-1][None, :], edges[1:][None, :]
    lam = np.clip((hi - a0[:, None]) / (hi - lo), 0.0, 1.0)
    legacy0 = first[list(AGE_BUCKETS)].to_numpy(dtype=np.float64) * lam  # (n_d, 7)
    mid = (np.maximum(lo, a0[:, None]) + hi) / 2.0
    s_crim0 = (first["pending_crim"] / first["pending"]).to_numpy(dtype=np.float64)

    n_d, n_t = len(common), ddl["t"].nunique()
    t_grid = ddl["t"].to_numpy().reshape(n_d, n_t)[0]
    dt = (t_first[:, None] - t_grid[None, :]).astype(np.float64)  # (n_d, n_t) months before T0
    decay = np.exp(weights.tail_hazard * dt)
    legacy_b = np.zeros((n_d, n_t, len(AGE_BUCKETS)))
    for b in range(len(AGE_BUCKETS)):
        tgt = np.digitize(mid[:, b][:, None] - dt / 12.0, edges[1:-1])
        legacy_b += np.eye(len(AGE_BUCKETS))[tgt] * (legacy0[:, b][:, None] * decay)[..., None]
    legacy = legacy_b.sum(-1)
    legacy_prev = legacy * np.exp(weights.tail_hazard)

    p_ddl = ddl["pending"].to_numpy().reshape(n_d, n_t)
    gap_m = (t_first - t_end).astype(np.float64)
    l_end = legacy[:, t_grid == t_end][:, 0] if (t_grid == t_end).any() else legacy[:, -1]
    kappa_raw = (first["pending"].to_numpy() * np.exp(-g_m * gap_m) - l_end) / np.maximum(p_ddl[:, -1], 1.0)
    kappa = np.clip(np.nan_to_num(kappa_raw, nan=1.0), 0.5, 2.0)
    n_clip = int(((kappa_raw < 0.5) | (kappa_raw > 2.0)).sum())
    LOG.info("coverage kappa: median %.3f, IQR [%.3f, %.3f], clipped %d", np.median(kappa),
             *np.quantile(kappa, [0.25, 0.75]), n_clip)

    k = np.repeat(kappa, n_t)
    out = ddl.copy()
    for c in ("instituted", "pending", "pending_crim", *AGE_BUCKETS):
        out[c] = out[c] * k
    out["pending"] += legacy.reshape(-1)
    out["pending_crim"] += (legacy * s_crim0[:, None]).reshape(-1)
    out["disposed"] = out["disposed"] * k + (legacy_prev - legacy).reshape(-1)
    for i, b in enumerate(AGE_BUCKETS):
        out[b] += legacy_b[:, :, i].reshape(-1)
    out["coverage_kappa"] = k

    nj = nj.assign(coverage_kappa=1.0)
    out = out.loc[~out.set_index(["district_id", "t"]).index.isin(nj.set_index(["district_id", "t"]).index)]
    panel = pd.concat([out, nj], ignore_index=True)
    full = pd.MultiIndex.from_product(
        [np.sort(common), np.arange(panel["t"].min(), panel["t"].max() + 1)], names=["district_id", "t"]
    )
    panel = panel.set_index(["district_id", "t"]).reindex(full).reset_index()
    panel["state_code"] = panel["district_id"] // 1000
    return panel


# ----------------------------------------------------------------------------------------- 3. features
def _roll(a: FloatArr, w: int, how: str = "sum", q: float | None = None) -> FloatArr:
    r = pd.DataFrame(a.T).rolling(w, min_periods=w)
    res = r.quantile(q) if how == "quantile" else getattr(r, how)()
    return res.to_numpy().T


def _shift(a: FloatArr, k: int) -> FloatArr:
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:, k:] = a[:, :-k]
    elif k < 0:
        out[:, :k] = a[:, -k:]
    else:
        out[:] = a
    return out


def bowley_skew(buckets: FloatArr) -> FloatArr:
    """Quartile skewness of the pending-age distribution from bucket counts (piecewise-linear CDF)."""
    edges = np.asarray(AGE_EDGES_Y)
    tot = buckets.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = buckets / tot
    cum = np.concatenate([np.zeros_like(pdf[..., :1]), np.cumsum(pdf, -1)], -1)
    qs = []
    for q in (0.25, 0.50, 0.75):
        k = np.clip((cum[..., 1:] < q).sum(-1), 0, len(AGE_BUCKETS) - 1)
        c_prev = np.take_along_axis(cum, k[..., None], -1)[..., 0]
        mass = np.take_along_axis(pdf, k[..., None], -1)[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            qs.append(edges[k] + (q - c_prev) / mass * (edges[k + 1] - edges[k]))
    q1, q2, q3 = qs
    with np.errstate(divide="ignore", invalid="ignore"):
        return (q3 + q1 - 2 * q2) / (q3 - q1)


def _state_loo(x: FloatArr, state_idx: IntArr) -> FloatArr:
    """Leave-one-out mean over the other districts of the same High Court jurisdiction, same month."""
    n_s = state_idx.max() + 1
    x0, nn = np.nan_to_num(x), np.isfinite(x).astype(np.float64)
    s_sum, s_cnt = np.zeros((n_s, x.shape[1])), np.zeros((n_s, x.shape[1]))
    np.add.at(s_sum, state_idx, x0)
    np.add.at(s_cnt, state_idx, nn)
    with np.errstate(divide="ignore", invalid="ignore"):
        loo = (s_sum[state_idx] - x0) / (s_cnt[state_idx] - nn)
    return np.where(np.isfinite(loo), loo, np.nan)


def spatial_weights(edges_csv: Path, districts: IntArr) -> sp.csr_matrix:
    """Row-standardised W from an undirected district edge list (e.g. Queen contiguity on SHRUG pc11 polygons)."""
    e = pd.read_csv(edges_csv)
    pos = pd.Series(np.arange(len(districts)), index=districts)
    src = district_key(e["src_state"], e["src_dist"])
    dst = district_key(e["dst_state"], e["dst_dist"])
    ok = np.isin(src, districts) & np.isin(dst, districts)
    i, j = pos.loc[src[ok]].to_numpy(), pos.loc[dst[ok]].to_numpy()
    w = sp.coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(len(districts),) * 2).tocsr()
    w.data[:] = 1.0
    deg = np.asarray(w.sum(1)).ravel()
    return sp.diags(np.where(deg > 0, 1.0 / deg, 0.0)) @ w


def _spatial_lag(x: FloatArr, w: sp.csr_matrix) -> FloatArr:
    """W x with weights renormalised over neighbours observed at t (x at t is info-set I_t)."""
    ok = np.isfinite(x)
    num, den = w @ np.where(ok, x, 0.0), w @ ok.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def months_on_ecourts(a: FloatArr) -> FloatArr:
    """Leak-free onboarding clock: at origin t, months since the first active month judged with data <= t
    (active = filings >= 10% of the 90th percentile of positive monthly filings so far; -1 before onboarding)."""
    a0 = np.nan_to_num(a)
    pos = np.where(a0 > 0, a0, np.nan)
    out = np.empty(a.shape)
    for t in range(a.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows before any filing
            typical = np.nanpercentile(pos[:, : t + 1], 90, axis=1)
        active = a0[:, : t + 1] >= 0.1 * np.nan_to_num(typical, nan=np.inf)[:, None]
        out[:, t] = np.where(active.any(1), t - active.argmax(1), -1.0)
    return out


def engineer_features(
    panel: pd.DataFrame, weights: WorkloadWeights, cfg: Config, w: sp.csr_matrix | None
) -> pd.DataFrame:
    """Every feature at origin t is a function of data dated <= t only; targets are dated > t."""
    panel = panel.sort_values(["district_id", "t"]).reset_index(drop=True)
    districts = panel["district_id"].unique()
    shape = (len(districts), panel["t"].nunique())
    if shape[0] * shape[1] != len(panel):
        raise ValueError("panel is not a complete district x month grid")

    def g(c: str) -> FloatArr:
        return panel[c].to_numpy(dtype=np.float64).reshape(shape)

    a, d, p, pc = g("instituted"), g("disposed"), g("pending"), g("pending_crim")
    bk = np.stack([g(b) for b in AGE_BUCKETS], -1)
    c, s_cap = g("working_judges"), g("sanctioned_judges")
    f: dict[str, FloatArr] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        a12, d12 = _roll(a, 12), _roll(d, 12)
        s_crim = pc / p
        om = weights.omega
        wu = (bk * (s_crim[..., None] * om[1] + (1 - s_crim)[..., None] * om[0])).sum(-1)
        trend = _shift(_roll(a, 24, "median"), 3)
        iqr = _shift(_roll(a, 24, "quantile", 0.75) - _roll(a, 24, "quantile", 0.25), 3)
        log_p = np.log(p)
        f["idvr_12"] = a12 / d12
        f["vacancy_rate"] = np.clip(1.0 - c / s_cap, 0.0, 1.0)
        f["log_working_judges"] = np.log(c)
        f["workload_per_judge"] = wu / c
        f["age_bowley_skew"] = bowley_skew(bk)
        f["share_age_gt5y"] = bk[..., 3:].sum(-1) / p
        lt3 = bk[..., 0] + bk[..., 1]
        f["share_age_1_3_of_lt3"] = bk[..., 1] / lt3
        f["pending_lt3_per_judge"] = lt3 / c
        f["log_hearing_gap"] = np.log(g("hearing_gap_days"))
        f["pretrial_share"] = sum(g(f"share_{s}") for s in PRETRIAL_STAGES)
        f["filing_shock_z"] = (_roll(a, 3, "mean") - trend) / np.maximum(iqr / 1.349, 1.0)
        f["inflow_trend_growth"] = np.log(trend / _shift(trend, 12))
        f["disposal_momentum"] = np.log(d12 / _shift(d12, 12))
        f["log_pending"] = log_p
        f["growth_12_past"] = log_p - _shift(log_p, 12)
        f["share_criminal"] = s_crim
        f["is_njdg"] = (panel["source"].to_numpy() == "njdg").reshape(shape).astype(np.float64)
        f["month_of_year"] = (panel["t"].to_numpy().reshape(shape) % 12 + 1).astype(np.float64)
        # months since the district's records start on eCourts (onboarding dumps look like fake backlog surges);
        # uses data <= t only: a month is "active" once filings reach 10% of the 90th percentile observed so far
        f["months_on_ecourts"] = months_on_ecourts(a)
        f["d_rate_12"] = d12 / 12.0
        f["a_trend"] = trend
        f["growth_12_past"] = np.where(np.isfinite(f["growth_12_past"]), f["growth_12_past"], np.nan)
        for h in cfg.horizons:
            f[f"anchor_growth_{h}"] = f["growth_12_past"] * h / 12.0  # drift extrapolation
            f[f"anchor_cr_{h}"] = d12 / a12  # trailing clearance ratio
            f[f"y_growth_{h}"] = _shift(log_p, -h) - log_p
            f[f"y_cr_{h}"] = _shift(_roll(d, h), -h) / _shift(_roll(a, h), -h)
        state_idx = pd.factorize(districts // 1000)[0].astype(np.int64)
        for col in ("growth_12_past", "idvr_12", "vacancy_rate", "filing_shock_z"):
            f[f"state_loo_{col}"] = _state_loo(np.where(np.isfinite(f[col]), f[col], np.nan), state_idx)
        f["w_lag_growth_12_past"] = _spatial_lag(f["growth_12_past"], w) if w is not None else np.full(shape, np.nan)
    for k, v in f.items():
        panel[k] = np.where(np.isfinite(v), v, np.nan).reshape(-1)
    return panel


# ----------------------------------------------------------------------------------------- 4. model
def pinball_rows(y: FloatArr, q: FloatArr, tau: float) -> FloatArr:
    u = y - q
    return np.maximum(tau * u, (tau - 1.0) * u)


def pinball(y: FloatArr, q: FloatArr, tau: float) -> float:
    return float(np.mean(pinball_rows(y, q, tau)))


def cluster_bootstrap_skill(loss: FloatArr, loss_ref: FloatArr, clusters: npt.ArrayLike, draws: int,
                            seed: int) -> tuple[float, float]:
    """95% percentile interval of 1 - sum(loss)/sum(loss_ref), resampling whole clusters (High Courts).

    Districts in one High Court share shocks and overlapping 12-month windows share months, so rows are far from
    independent; resampling ~30 High Court blocks gives honest (wide) intervals.
    """
    codes, _ = pd.factorize(np.asarray(clusters))
    if len(codes) == 0:
        return float("nan"), float("nan")
    s_m = np.bincount(codes, weights=loss)
    s_r = np.bincount(codes, weights=loss_ref)
    k = len(s_m)
    w = np.random.default_rng(seed).multinomial(k, np.full(k, 1.0 / k), size=draws).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sk = 1.0 - (w @ s_m) / (w @ s_r)
    lo, hi = np.nanquantile(sk, [0.025, 0.975])
    return float(lo), float(hi)


def linear_quantile_fit(x: FloatArr, y: FloatArr, tau: float, max_rows: int = 20000, seed: int = 0) -> FloatArr:
    """Linear quantile regression (intercept first) as an LP: min tau*1'u + (1-tau)*1'v, Xb + u - v = y."""
    if len(y) > max_rows:
        idx = np.random.default_rng(seed).choice(len(y), max_rows, replace=False)
        x, y = x[idx], y[idx]
    n = len(y)
    xd = np.column_stack([np.ones(n), x])
    k = xd.shape[1]
    a_eq = sp.hstack([sp.csr_matrix(xd), sp.identity(n, format="csr"), -sp.identity(n, format="csr")], format="csr")
    c = np.concatenate([np.zeros(k), np.full(n, tau), np.full(n, 1.0 - tau)])
    bounds = [(None, None)] * k + [(0.0, None)] * (2 * n)
    res = linprog(c, A_eq=a_eq, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"quantile LP failed: {res.message}")
    return np.asarray(res.x[:k], dtype=np.float64)


@dataclass
class LinearQuantileBaseline:
    """Benchmark: y = anchor + X b(tau), X standardised with training medians filling gaps; same conformal step."""

    cols: list[str]
    anchor: str
    med: FloatArr
    scale: FloatArr
    coef: dict[float, FloatArr]
    shifts: dict[float, float]

    @classmethod
    def fit(cls, tr: pd.DataFrame, cq: pd.DataFrame, y_col: str, anchor: str, cfg: Config) -> LinearQuantileBaseline:
        cols = [c for c in LINEAR_BASELINE_FEATURES if c not in cfg.exclude_features and tr[c].notna().any()]
        x = tr[cols].to_numpy(dtype=np.float64)
        med = np.nanmedian(x, axis=0)
        scale = np.nanstd(x, axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        base = cls(cols, anchor, med, scale, {}, {tau: 0.0 for tau in cfg.quantiles})
        z, y = base._x(tr), (tr[y_col] - tr[anchor]).to_numpy(dtype=np.float64)
        base.coef = {tau: linear_quantile_fit(z, y, tau, seed=cfg.seed) for tau in cfg.quantiles}
        if len(cq) >= 50:
            base.shifts = {tau: conformal_shift((cq[y_col] - base.raw(cq, tau)).to_numpy(), tau) for tau in cfg.quantiles}
        return base

    def _x(self, df: pd.DataFrame) -> FloatArr:
        x = df[self.cols].to_numpy(dtype=np.float64)
        return (np.where(np.isfinite(x), x, self.med) - self.med) / self.scale

    def raw(self, df: pd.DataFrame, tau: float) -> FloatArr:
        b = self.coef[tau]
        return b[0] + self._x(df) @ b[1:] + df[self.anchor].to_numpy()

    def predict(self, df: pd.DataFrame) -> FloatArr:
        return np.sort(np.column_stack([self.raw(df, t) + self.shifts[t] for t in sorted(self.coef)]), axis=1)


def conformal_shift(residuals: FloatArr, tau: float) -> float:
    """Split-conformal quantile recalibration: P(y <= q_hat + shift) >= tau (finite-sample corrected)."""
    n = len(residuals)
    if tau >= 0.5:
        return float(np.quantile(residuals, min(1.0, np.ceil((n + 1) * tau) / n), method="higher"))
    return float(np.quantile(residuals, max(0.0, np.floor((n + 1) * tau) / n), method="lower"))


@dataclass
class QuantileSuite:
    """Quantiles of y = anchor + innovation; boosters model the innovation over the drift anchor."""

    target: str
    horizon: int
    features: list[str]
    anchor: str
    boosters: dict[float, lgb.Booster]
    shifts: dict[float, float]
    # drift benchmark: quantiles of (y - anchor) on the training + calibration rows of this suite
    drift_resid_q: dict[float, float] = field(default_factory=dict)
    # linear quantile-regression benchmark fitted on the same rows (None when loaded from an older bundle)
    linear: LinearQuantileBaseline | None = None

    def raw(self, x: pd.DataFrame, tau: float) -> FloatArr:
        return self.boosters[tau].predict(x[self.features]) + x[self.anchor].to_numpy()

    def predict(self, x: pd.DataFrame) -> FloatArr:
        q = np.column_stack([self.raw(x, tau) + self.shifts[tau] for tau in sorted(self.boosters)])
        return np.sort(q, axis=1)  # Chernozhukov-Fernandez-Val-Galichon rearrangement: no crossing


def conservative_train_end(cfg: Config) -> int:
    """Latest origin any horizon's training set could use (bounds the survival-weight fit: no lookahead)."""
    return abs_month(cfg.val_start) - 2 * max(cfg.horizons) - cfg.cal_months


def break_mask(t: pd.Series, h: int, cfg: Config) -> pd.Series:
    m = pd.Series(False, index=t.index)
    for lo, hi in cfg.break_windows:
        m |= (t <= abs_month(pd.Period(hi, "M"))) & (t + h >= abs_month(pd.Period(lo, "M")))
    return m


def purged_split(data: pd.DataFrame, h: int, cfg: Config, v0: int | None = None,
                 cal_months: int | None = None) -> tuple[pd.DataFrame, ...]:
    """Temporal split with an h-month embargo at every boundary and break-window masking.

    cal = last `cal_months` clean origins whose targets realise before val_start;
    train = clean origins whose targets realise before the first cal origin;
    cal is halved: early-stopping half (es) / conformal half (cq).
    """
    v0 = abs_month(cfg.val_start) if v0 is None else v0
    cal_months = cfg.cal_months if cal_months is None else cal_months
    if cfg.min_origin is not None:
        data = data.loc[data["t"] >= abs_month(cfg.min_origin)]
    clean = data.loc[~break_mask(data["t"], h, cfg)]
    eligible = clean.loc[clean["t"] + h < v0]
    origins = np.sort(eligible["t"].unique())
    cal_orig = origins[-cal_months:]
    if len(cal_orig) == 0:
        empty = data.iloc[:0]
        return empty, empty, empty, data.loc[data["t"] >= v0], clean
    cal = eligible.loc[eligible["t"].isin(cal_orig)]
    tr = eligible.loc[eligible["t"] + h < cal_orig.min()]
    half = cal_orig[len(cal_orig) // 2]
    return tr, cal.loc[cal["t"] < half], cal.loc[cal["t"] >= half], data.loc[data["t"] >= v0], clean


def _params(tau: float, seed: int) -> dict[str, object]:
    # LightGBM forbids monotone_constraints with objective="quantile"; directional structure is
    # carried by the FE elasticities in the policy layer, not imposed on the forecaster.
    return {
        "objective": "quantile", "alpha": tau, "learning_rate": 0.03, "num_leaves": 31,
        "min_data_in_leaf": 40, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "seed": seed, "deterministic": True,
        "force_row_wise": True, "verbose": -1,
    }


def _train_boosters(tr: pd.DataFrame, es: pd.DataFrame, feats: list[str], y_col: str, anchor: str,
                    cfg: Config) -> tuple[dict[float, lgb.Booster], dict[float, int]]:
    boosters, best_iter = {}, {}
    for tau in cfg.quantiles:
        params = _params(tau, cfg.seed)
        dtr = lgb.Dataset(tr[feats], tr[y_col] - tr[anchor], free_raw_data=False)
        if len(es) >= 50:
            des = lgb.Dataset(es[feats], es[y_col] - es[anchor], reference=dtr)
            bst = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[des],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        else:
            bst = lgb.train(params, dtr, num_boost_round=400)
        best_iter[tau] = int(bst.best_iteration or bst.current_iteration())
        boosters[tau] = bst
    return boosters, best_iter


def _fit_one(data: pd.DataFrame, target: str, h: int, cfg: Config, v0: int, cal_months: int
             ) -> tuple[QuantileSuite, dict[float, int], pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Train on purged history before v0, early-stop on the first calibration half, conformalise on the second."""
    y_col, anchor = f"y_{target}_{h}", f"anchor_{target}_{h}"
    tr, es, cq, va, _ = purged_split(data, h, cfg, v0, cal_months)
    if len(tr) < 200:
        LOG.warning("%s h=%d v0=%s: only %d training rows; skipped", target, h, period_of(v0), len(tr))
        return None
    feats = [c for c in (*FEATURES, anchor) if c not in cfg.exclude_features and tr[c].notna().any()]
    LOG.info("%s h=%d v0=%s: train=%d [%s..%s] es=%d cq=%d", target, h, period_of(v0), len(tr),
             period_of(int(tr["t"].min())), period_of(int(tr["t"].max())), len(es), len(cq))
    boosters, best_iter = _train_boosters(tr, es, feats, y_col, anchor, cfg)
    suite = QuantileSuite(target, h, feats, anchor, boosters, {tau: 0.0 for tau in cfg.quantiles})
    if len(cq) >= 50:
        suite.shifts = {tau: conformal_shift((cq[y_col] - suite.raw(cq, tau)).to_numpy(), tau) for tau in cfg.quantiles}
    seen = pd.concat([tr, es, cq])
    suite.drift_resid_q = {tau: float(np.quantile(seen[y_col] - seen[anchor], tau)) for tau in cfg.quantiles}
    suite.linear = LinearQuantileBaseline.fit(tr, cq, y_col, anchor, cfg)
    return suite, best_iter, tr, cq, va


def _score(label: str, target: str, h: int, va: pd.DataFrame, y: FloatArr, q: Mapping[str, FloatArr], cfg: Config,
           extra: Mapping[float, Mapping[str, object]]) -> list[dict[str, object]]:
    """Pinball/coverage for the model and both benchmarks, with High-Court block-bootstrap skill intervals."""
    hc = va["district_id"].to_numpy() // 1000
    rows: list[dict[str, object]] = []
    for k, tau in enumerate(sorted(cfg.quantiles)):
        lm, ld, ll = (pinball_rows(y, q[m][:, k], tau) for m in ("model", "drift", "linear"))
        d_lo, d_hi = cluster_bootstrap_skill(lm, ld, hc, cfg.bootstrap_draws, cfg.seed)
        l_lo, l_hi = cluster_bootstrap_skill(lm, ll, hc, cfg.bootstrap_draws, cfg.seed)
        rows.append({
            "fold": label, "target": target, "h": h, "tau": tau, "n_val": int(len(y)),
            "n_districts": int(va["district_id"].nunique()),
            "pinball": lm.mean(), "pinball_naive": ld.mean(), "pinball_linear": ll.mean(),
            "skill_vs_drift": 1.0 - lm.sum() / ld.sum(), "skill_vs_drift_lo": d_lo, "skill_vs_drift_hi": d_hi,
            "skill_vs_linear": 1.0 - lm.sum() / ll.sum(), "skill_vs_linear_lo": l_lo, "skill_vs_linear_hi": l_hi,
            "coverage": float(np.mean(y <= q["model"][:, k])), "coverage_linear": float(np.mean(y <= q["linear"][:, k])),
            **extra.get(tau, {}),
        })
    lo, hi = q["model"][:, 0], q["model"][:, -1]
    rows.append({"fold": label, "target": target, "h": h, "tau": "interval", "n_val": int(len(y)),
                 "coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float(np.mean(hi - lo))})
    return rows


def fit_suite(frame: pd.DataFrame, target: str, h: int, cfg: Config, origin_t: int
              ) -> tuple[QuantileSuite | None, list[dict[str, object]]]:
    """Rolling-origin validation (one 12-month origin year per fold), then the production suite for origin_t.

    Production is refit with the same purged procedure as if validating at origin_t + 1, so its conformal
    shifts come from the most recent realised origins instead of the older calibration years of the folds.
    """
    y_col, anchor = f"y_{target}_{h}", f"anchor_{target}_{h}"
    data = frame.loc[frame[anchor].notna() & (frame["months_on_ecourts"] >= cfg.min_months_on_ecourts)]
    realised = data.loc[data[y_col].notna()]
    folds = cfg.val_folds or ((str(cfg.val_start), cfg.cal_months),)
    metrics: list[dict[str, object]] = []
    pooled: list[tuple[pd.DataFrame, FloatArr, dict[str, FloatArr]]] = []
    for vs, cal_m in folds:
        v0 = abs_month(pd.Period(vs, "M"))
        fit = _fit_one(realised, target, h, cfg, v0, cal_m)
        if fit is None:
            continue
        suite, best_iter, tr, cq, va = fit
        va = va.loc[va["t"] < v0 + 12]
        if va.empty:
            continue
        resid = (tr[y_col] - tr[anchor]).to_numpy()  # drift benchmark sees the same training rows as the model
        q = {"model": suite.predict(va), "linear": suite.linear.predict(va),  # type: ignore[union-attr]
             "drift": va[anchor].to_numpy()[:, None] + np.quantile(resid, sorted(cfg.quantiles))[None, :]}
        y = va[y_col].to_numpy()
        extra = {tau: {"conformal_shift": suite.shifts[tau], "trees": best_iter[tau], "n_train": len(tr),
                       "train_origins": f"{period_of(int(tr['t'].min()))}..{period_of(int(tr['t'].max()))}"}
                 for tau in cfg.quantiles}
        metrics += _score(vs[:4], target, h, va, y, q, cfg, extra)
        pooled.append((va, y, q))
    if len(pooled) > 1:
        va = pd.concat([p[0] for p in pooled])
        y = np.concatenate([p[1] for p in pooled])
        q = {m: np.vstack([p[2][m] for p in pooled]) for m in ("model", "drift", "linear")}
        metrics += _score("pooled", target, h, va, y, q, cfg, {})
    prod = _fit_one(realised, target, h, cfg, origin_t + 1, cfg.cal_months)
    return (prod[0] if prod is not None else None), metrics


# ----------------------------------------------------------------------------------------- 5. explain
def shap_attribution(booster: lgb.Booster, x: pd.DataFrame) -> pd.DataFrame:
    try:
        import shap

        phi = np.asarray(shap.TreeExplainer(booster).shap_values(x))
    except Exception as exc:  # shap/lightgbm version skew: fall back to LightGBM's native TreeSHAP
        LOG.warning("shap.TreeExplainer failed (%s); using LightGBM pred_contrib", exc)
        phi = booster.predict(x, pred_contrib=True)[:, :-1]
    return pd.DataFrame(phi, columns=x.columns, index=x.index)


def decompose_drivers(phi: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    fam = pd.Series({c: FEATURE_FAMILY.get(c, "other") for c in phi.columns})
    by_fam = phi.T.groupby(fam).sum().T
    abs_tot = phi.abs().sum(axis=1).replace(0, np.nan)
    structural_cols = fam.index[fam.isin(STRUCTURAL_FAMILIES)]
    out = pd.DataFrame(index=phi.index)
    out["structural_push"] = by_fam.reindex(columns=sorted(STRUCTURAL_FAMILIES), fill_value=0).sum(axis=1)
    out["transient_push"] = by_fam.reindex(columns=sorted(TRANSIENT_FAMILIES), fill_value=0).sum(axis=1)
    out["structural_share_abs"] = phi[structural_cols].abs().sum(axis=1) / abs_tot
    order = np.argsort(-phi.to_numpy(), axis=1)[:, :top_k]
    names = phi.columns.to_numpy()[order]
    vals = np.take_along_axis(phi.to_numpy(), order, axis=1)
    for k in range(top_k):
        out[f"driver_{k + 1}"] = names[:, k]
        out[f"driver_{k + 1}_phi"] = vals[:, k]
    return out


# ----------------------------------------------------------------------------------------- 6. policy
def _twoway_demean(df: pd.DataFrame, cols: Sequence[str], unit: str, time_group: str, iters: int = 200) -> FloatArr:
    """Partial out unit + time-group fixed effects by alternating projections."""
    z = df[list(cols)].to_numpy(dtype=np.float64).copy()
    gu, _ = pd.factorize(df[unit])
    gt, _ = pd.factorize(df[time_group])

    def demean(a: FloatArr, grp: IntArr) -> FloatArr:
        sums = np.zeros((grp.max() + 1, a.shape[1]))
        np.add.at(sums, grp, a)
        return a - (sums / np.bincount(grp)[:, None])[grp]

    for _ in range(iters):
        z_new = demean(demean(z, gu), gt)
        if np.max(np.abs(z_new - z)) < 1e-10:
            return z_new
        z = z_new
    return z


def _cluster_vcov(xh: FloatArr, u: FloatArr, clusters: npt.ArrayLike) -> FloatArr:
    """Sandwich (X'X)^-1 [sum_g X_g'u_g u_g'X_g] (X'X)^-1 with the usual small-sample factor."""
    gc, _ = pd.factorize(np.asarray(clusters))
    bread = np.linalg.pinv(xh.T @ xh)
    scores = np.zeros((gc.max() + 1, xh.shape[1]))
    np.add.at(scores, gc, xh * u[:, None])
    n, k, n_g = len(u), xh.shape[1], gc.max() + 1
    return bread @ (scores.T @ scores) @ bread * (n_g / (n_g - 1)) * ((n - 1) / (n - k))


def twoway_fe_ols(
    df: pd.DataFrame, y: str, xs: Sequence[str], unit: str, time_group: str, cluster: str, iters: int = 200
) -> pd.DataFrame:
    """Within estimator with unit + (state x month) FE via alternating projections; clustered SE."""
    z = _twoway_demean(df, [y, *xs], unit, time_group, iters)
    yv, xm = z[:, 0], z[:, 1:]
    beta = np.linalg.pinv(xm.T @ xm) @ xm.T @ yv
    vcov = _cluster_vcov(xm, yv - xm @ beta, df[cluster])
    return pd.DataFrame({"coef": beta, "se": np.sqrt(np.diag(vcov))}, index=list(xs))


def twoway_fe_iv(df: pd.DataFrame, y: str, endog: str, instr: str, exog: Sequence[str], unit: str, time_group: str,
                 cluster: str) -> dict[str, float]:
    """Just-identified 2SLS with unit + time-group FE; clustered SE and cluster-robust first-stage F."""
    z = _twoway_demean(df, [y, endog, instr, *exog], unit, time_group)
    yv, x, zi, w = z[:, 0], z[:, 1], z[:, 2], z[:, 3:]
    zm, xm = np.column_stack([zi, w]), np.column_stack([x, w])
    pi = np.linalg.pinv(zm.T @ zm) @ zm.T @ x
    v_fs = _cluster_vcov(zm, x - zm @ pi, df[cluster])
    xh = zm @ (np.linalg.pinv(zm.T @ zm) @ zm.T @ xm)  # first-stage fitted regressors
    beta = np.linalg.pinv(xh.T @ xm) @ xh.T @ yv
    vcov = _cluster_vcov(xh, yv - xm @ beta, df[cluster])
    return {"coef": float(beta[0]), "se": float(np.sqrt(vcov[0, 0])), "first_stage_coef": float(pi[0]),
            "first_stage_f": float(pi[0] ** 2 / v_fs[0, 0]), "n": float(len(yv)),
            "clusters": float(pd.Series(np.asarray(df[cluster])).nunique())}


def shift_share_instrument(frame: pd.DataFrame, base_year: int) -> pd.Series:
    """Predicted log judge strength from cadre mix x state cadre intake (Bartik).

    z_dt = sum_k s_dk * [log J_k,s,t - log J_k,s,base],  s_dk = district's base-year share of cadre k seats,
    J_k,s,t = occupied cadre-k seats in the High Court jurisdiction. Cadres are recruited by separate processes
    (PSC batches for civil judges / magistrates, promotions and HC direct recruitment for district judges), so within
    a High Court-month (absorbed by FE) z varies only through the predetermined shares s_dk. Shifts are NOT
    leave-one-out: with High Court x month FE, J_k,-d,t = J_k,s,t - c_dk,t varies within the cell only through the
    district's own seats, which would make the instrument a mirror of the endogenous regressor. Own-district
    feedback through J_k,s,t is O(1 / districts per High Court).
    Returned as a 12-month rolling mean to match log_c12.
    """
    f = frame.sort_values(["district_id", "t"])
    cad = [f"working_{c}" for c in CADRES]
    seats = f[cad].to_numpy(dtype=np.float64)
    missing = ~np.isfinite(seats).all(1)  # NJDG-period rows carry no cadre detail
    seats = np.column_stack([seats, np.maximum(f["working_judges"].to_numpy(dtype=np.float64) - seats.sum(1), 0.0)])
    k = seats.shape[1]
    keys = pd.DataFrame({"s": f["state_code"].to_numpy(), "t": f["t"].to_numpy()})
    state_tot = pd.DataFrame(seats).groupby([keys["s"], keys["t"]]).transform(lambda v: v.sum(min_count=1)).to_numpy()
    base = (f["t"] // 12 == base_year).to_numpy()
    did = f["district_id"].to_numpy()
    base_seats = pd.DataFrame(seats[base]).groupby(did[base]).mean()
    share = base_seats.div(base_seats.sum(axis=1), axis=0).reindex(did).to_numpy()
    tot_base = pd.DataFrame(state_tot[base]).groupby(did[base]).mean().reindex(did).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.log(state_tot) - np.log(tot_base)
    g = np.where(np.isfinite(g), g, 0.0)
    z = np.where(np.isfinite(share), share, 0.0) * g
    zs = pd.Series(z.sum(1), index=f.index)
    zs[~np.isfinite(share).all(1) | (np.nan_to_num(share).sum(1) == 0) | missing] = np.nan
    return zs.groupby(f["district_id"]).transform(lambda v: v.rolling(12, min_periods=12).mean()).reindex(frame.index)


def judge_coverage(frame: pd.DataFrame) -> dict[str, float]:
    """Occupied seats recorded in DDL each December: a rising total flags judge records thinning out back in time."""
    dec = frame.loc[frame["t"] % 12 == 11]
    return {f"judges_recorded_dec_{int(t // 12)}": float(v) for t, v in dec.groupby("t")["working_judges"].sum().items()}


def estimate_elasticities(frame: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """eps_c = dlogD/dlogc (OLS and shift-share IV) and eps_g = dlogD/dlog(gap); district FE + High Court x month FE.

    Estimated on the capacity-bound regime (lagged IDVR >= 1, i.e. rho >= 1) when large enough: below saturation
    disposals are demand-limited and dlogD/dlogc -> 0. OLS is biased by reverse causality (judges are posted
    where backlogs are high) and by judge records thinning out back in time; the IV addresses the first.
    eps_judges is reported as identified only if the IV first stage is strong and the estimate plausible.
    """
    grp = frame.groupby("district_id")
    df = frame.assign(
        log_d12=np.log(frame["d_rate_12"]),
        log_c12=np.log(grp["working_judges"].transform(lambda s: s.rolling(12, min_periods=12).mean())),
        log_p_lag12=grp["log_pending"].shift(12),
        idvr_lag12=grp["idvr_12"].shift(12),
        z_c12=shift_share_instrument(frame, cfg.iv_base_year),
        hc_month=frame["state_code"].astype(str) + "_" + frame["t"].astype(str),
    ).replace([np.inf, -np.inf], np.nan)
    df = df.loc[df["months_on_ecourts"] >= cfg.min_months_on_ecourts]
    cols = ["log_d12", "log_c12", "log_p_lag12"]
    pooled = df.dropna(subset=cols)
    bound = pooled.loc[pooled["idvr_lag12"] >= 1.0]
    res_pool = twoway_fe_ols(pooled, "log_d12", cols[1:], "district_id", "hc_month", "state_code")
    use = bound if len(bound) >= 300 and bound["district_id"].nunique() >= 10 else pooled
    res = twoway_fe_ols(use, "log_d12", cols[1:], "district_id", "hc_month", "state_code")
    LOG.info("disposal elasticity FE-OLS pooled (n=%d):\n%s\ncapacity-bound regime (n=%d):\n%s",
             len(pooled), res_pool.round(4), len(use), res.round(4))
    out: dict[str, float] = {
        "eps_judges_ols": float(res.loc["log_c12", "coef"]), "eps_judges_ols_se": float(res.loc["log_c12", "se"]),
        "eps_judges_ols_pooled": float(res_pool.loc["log_c12", "coef"]), "eps_gap": -1.0, "eps_gap_source": 0.0,
    }
    iv_t0 = abs_month(pd.Period(f"{cfg.iv_base_year + 1}-12", "M"))  # rolling window clear of the base year
    iv_df = use.loc[(use["t"] >= iv_t0) & use["z_c12"].notna()]
    if len(iv_df) >= 300 and iv_df["state_code"].nunique() >= 5:
        iv = twoway_fe_iv(iv_df, "log_d12", "log_c12", "z_c12", ["log_p_lag12"], "district_id", "hc_month", "state_code")
        LOG.info("disposal elasticity shift-share IV: %s", {k: round(v, 4) for k, v in iv.items()})
        out |= {f"eps_judges_iv{'' if k == 'coef' else '_' + k}": v for k, v in iv.items()}
    else:
        LOG.warning("IV sample too small (%d rows); judge elasticity not identified", len(iv_df))
    f_stat, b_iv = out.get("eps_judges_iv_first_stage_f", 0.0), out.get("eps_judges_iv", np.nan)
    identified = f_stat >= cfg.min_first_stage_f and 0.05 <= b_iv <= 1.5
    out["eps_judges_identified"] = float(identified)
    out["eps_judges"] = b_iv if identified else np.nan
    if not identified:
        LOG.warning("judge elasticity not identified (first-stage F %.1f, IV %.3f): bench levers withheld", f_stat, b_iv)
    gap = use.dropna(subset=["log_hearing_gap"]) if "log_hearing_gap" in use else use.iloc[:0]
    if len(gap) >= 300 and gap["district_id"].nunique() >= 20:
        rg = twoway_fe_ols(gap, "log_d12", [*cols[1:], "log_hearing_gap"], "district_id", "hc_month", "state_code")
        LOG.info("hearing-gap elasticity FE:\n%s", rg.round(4))
        b, se = float(rg.loc["log_hearing_gap", "coef"]), float(rg.loc["log_hearing_gap", "se"])
        if b + 1.96 * se < 0 and b <= -0.1:  # significant and economically meaningful; else structural prior -1
            out["eps_gap"], out["eps_gap_source"] = b, 1.0
        else:
            LOG.warning("eps_gap=%.3f (se %.3f) not significant/meaningful; using structural prior -1", b, se)
    return out | judge_coverage(frame)


def required_bench_additions(c0: FloatArr, d0: FloatArr, excess: FloatArr, eps: float, cfg: Config) -> FloatArr:
    """Smallest dc with sum_m D0[((c0 + dc r(m))/c0)^eps - 1] >= excess;  r(m) = appointment lag + ramp."""
    m = np.arange(1, cfg.policy_horizon + 1, dtype=np.float64)
    r = np.where(m > cfg.appointment_lag_m, 1.0 - np.exp(-(m - cfg.appointment_lag_m) / cfg.ramp_tau_m), 0.0)

    def extra(dc: FloatArr) -> FloatArr:
        return (d0[:, None] * ((1.0 + dc[:, None] * r[None, :] / c0[:, None]) ** eps - 1.0)).sum(1)

    lo, hi = np.zeros_like(c0), np.maximum(c0 * 20.0, 50.0)
    feasible = extra(hi) >= excess
    for _ in range(60):
        mid = (lo + hi) / 2.0
        ok = extra(mid) >= excess
        hi, lo = np.where(ok, mid, hi), np.where(ok, lo, mid)
    return np.where(feasible, np.ceil(hi - 1e-9), np.inf)


def policy_levers(latest: pd.DataFrame, fc: pd.DataFrame, drivers: pd.DataFrame, elast: Mapping[str, float], cfg: Config,
                  scrape: pd.DataFrame) -> pd.DataFrame:
    """Levers at the forecast origin. Hearing gap / stage mix: contemporaneous NJDG values where present, else the
    DDL scrape-time description (2019-20), which is the closest available measurement for a Dec-2018 origin."""
    h = cfg.policy_horizon
    x = latest.set_index("district_id").join(fc.set_index("district_id")).join(drivers)
    sc = scrape.reindex(x.index)
    p0, d0 = x["pending"].to_numpy(), x["d_rate_12"].to_numpy()
    c0 = np.maximum(np.nan_to_num(x["working_judges"].to_numpy()), 1.0)
    p_q90 = p0 * np.exp(x[f"growth_{h}_q90"].to_numpy())
    excess = np.maximum(p_q90 - p0, 0.0) + cfg.backlog_reduction * p0  # disposals needed so that CR_{t:t+h} >= 1
    out = pd.DataFrame(index=x.index)
    out["pending_now"] = p0
    out[f"pending_q90_t+{h}"] = p_q90
    out["cr_status_quo_q90"] = h * d0 / (h * d0 + np.maximum(p_q90 - p0, 0.0))
    out["excess_disposals_needed"] = excess
    # Lever 1: bench strength
    out["bench_additions_required"] = required_bench_additions(c0, d0, excess, elast["eps_judges"], cfg)
    out["vacant_posts"] = np.maximum(x["sanctioned_judges"].to_numpy() - c0, 0.0)
    out["needs_new_sanctioned_posts"] = out["bench_additions_required"] > out["vacant_posts"]
    # Lever 2: hearing cadence + stage bottleneck
    mu = 1.0 + excess / np.maximum(h * d0, 1e-9)
    g0 = x["hearing_gap_days"].fillna(sc["hearing_gap_at_scrape_days"]).to_numpy(dtype=np.float64)
    out["disposal_multiplier_required"] = mu
    out["hearing_gap_now_days"] = g0
    g_star = g0 * mu ** (1.0 / elast["eps_gap"])
    out["hearing_gap_target_days"] = np.maximum(g_star, 7.0)
    out["cadence_lever_feasible"] = g_star >= 7.0  # below weekly listing is not administratively reachable
    shares = latest.set_index("district_id")[list(STAGE_SHARE_COLS)].fillna(sc[list(STAGE_SHARE_COLS)])
    state_med = shares.groupby(shares.index // 1000).transform("median")
    ratio = (shares / state_med).replace([np.inf, -np.inf], np.nan)
    has = ratio.notna().any(axis=1).to_numpy()
    top = ratio.fillna(-np.inf).idxmax(axis=1).str.replace("share_", "", regex=False).to_numpy()
    out["bottleneck_stage"] = np.where(~has, "unobserved", np.where(ratio.max(axis=1).to_numpy() >= 1.15, top, "none"))
    out["bottleneck_ratio_to_hc_median"] = ratio.max(axis=1)
    # Lever 3: structural vs transient -> permanent posts vs time-bound surge capacity
    excess_inflow = x["transient_excess_12m"].to_numpy()
    out["transient_excess_inflow_12m"] = excess_inflow
    out["surge_judge_months"] = excess_inflow / np.maximum(d0 / np.maximum(c0, 1.0), 1e-9)
    out["structural_share_abs"] = x["structural_share_abs"]
    out["transient_coverage"] = excess_inflow / np.maximum(excess, 1.0)
    # time-bound surge only when the shock is quantitatively large AND TreeSHAP says it drives the q90 tail
    surge = (out["transient_coverage"] >= 0.5) & (x["transient_push"] > x["structural_push"])
    out["recommended_lever"] = np.where(surge, "surge_capacity", "permanent_bench+cadence")
    return out


# ----------------------------------------------------------------------------------------- persistence
def save_model_bundle(path: Path, suites: Mapping[tuple[str, int], QuantileSuite], cfg: Config,
                      weights: WorkloadWeights, elast: Mapping[str, float], data_label: str) -> None:
    """One LightGBM text file per (target, horizon, quantile) + manifest.json with everything needed to predict."""
    path.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "data": data_label, "lightgbm_version": lgb.__version__,
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "workload_omega": weights.omega.tolist(), "tail_hazard": weights.tail_hazard,
        "elasticities": dict(elast), "suites": [],
    }
    for (target, h), suite in suites.items():
        files = {}
        for tau, bst in suite.boosters.items():
            name = f"{target}_h{h}_q{int(round(tau * 100))}.txt"
            bst.save_model(str(path / name))
            files[str(tau)] = name
        manifest["suites"].append({"target": target, "horizon": h, "features": suite.features,  # type: ignore[union-attr]
                                   "anchor": suite.anchor, "shifts": {str(k): v for k, v in suite.shifts.items()},
                                   "drift_resid_q": {str(k): v for k, v in suite.drift_resid_q.items()},
                                   "linear": None if suite.linear is None else {
                                       "cols": suite.linear.cols, "median": suite.linear.med.tolist(),
                                       "scale": suite.linear.scale.tolist(),
                                       "coef": {str(k): v.tolist() for k, v in suite.linear.coef.items()},
                                       "shifts": {str(k): v for k, v in suite.linear.shifts.items()}},
                                   "files": files})
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info("saved model bundle (%d suites) to %s", len(suites), path)


def load_model_bundle(path: Path) -> dict[tuple[str, int], QuantileSuite]:
    manifest = json.loads((path / "manifest.json").read_text())
    def linear(s: Mapping[str, object]) -> LinearQuantileBaseline | None:
        lin = s.get("linear")
        if not lin:
            return None
        return LinearQuantileBaseline(lin["cols"], s["anchor"], np.asarray(lin["median"]), np.asarray(lin["scale"]),
                                      {float(t): np.asarray(v) for t, v in lin["coef"].items()},
                                      {float(t): v for t, v in lin["shifts"].items()})

    return {
        (s["target"], s["horizon"]): QuantileSuite(
            s["target"], s["horizon"], s["features"], s["anchor"],
            {float(t): lgb.Booster(model_file=str(path / f)) for t, f in s["files"].items()},
            {float(t): v for t, v in s["shifts"].items()},
            {float(t): v for t, v in s.get("drift_resid_q", {}).items()}, linear(s))
        for s in manifest["suites"]
    }


# ----------------------------------------------------------------------------------------- orchestration
def run(cfg: Config, ddl_parquet: Path, judges_csv: Path, njdg_csv: Path | None, edges_csv: Path | None, out: Path,
        compact: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fit_end_t = conservative_train_end(cfg)
    ddl, weights, scrape = build_ddl_panel(ddl_parquet, judges_csv, cfg, fit_end_t, compact=compact)
    scrape.to_csv(out / "hearing_at_scrape.csv")
    if njdg_csv is not None:
        panel = harmonise(ddl, load_njdg(njdg_csv), cfg, weights)
    else:
        LOG.warning("DDL-only run: no NJDG anchor, so pre-2010 pending cases are absent from the stock")
        panel = ddl.assign(coverage_kappa=1.0, state_code=ddl["district_id"] // 1000)
    w = spatial_weights(edges_csv, panel["district_id"].unique()) if edges_csv else None
    frame = engineer_features(panel, weights, cfg, w)
    frame.to_parquet(out / "panel_features.parquet", index=False)
    t_last = int(frame.loc[frame["pending"].notna(), "t"].max())

    suites: dict[tuple[str, int], QuantileSuite] = {}
    metrics: list[dict[str, object]] = []
    for target in ("growth", "cr"):
        for h in cfg.horizons:
            suite, m = fit_suite(frame, target, h, cfg, t_last)
            metrics += m
            if suite is not None:
                suites[(target, h)] = suite
    pd.DataFrame(metrics).to_csv(out / "validation_metrics.csv", index=False)
    LOG.info("validation metrics:\n%s", pd.DataFrame(metrics).round(4).to_string(index=False))

    latest = frame.loc[frame["t"] == t_last]
    young = latest["months_on_ecourts"] < 36
    if young.any():
        LOG.warning("%d districts on eCourts < 36 months at origin excluded from forecasts: %s",
                    int(young.sum()), latest.loc[young, "district_id"].tolist())
    latest = latest.loc[~young].reset_index(drop=True)
    # state at the origin (observed-stock definition), so forecasts can be graded against later data
    latest[["district_id", "pending", "instituted", "disposed", "d_rate_12", "a_trend", "working_judges"]].assign(
        instituted_12=latest["idvr_12"] * latest["d_rate_12"] * 12, origin=str(period_of(t_last))
    ).to_csv(out / "origin_state.csv", index=False)
    fc = pd.DataFrame({"district_id": latest["district_id"], "origin": str(period_of(t_last))})
    for (target, h), suite in suites.items():
        q = suite.predict(latest)
        for k, tau in enumerate(sorted(cfg.quantiles)):
            fc[f"{target}_{h}_q{int(round(tau * 100))}"] = q[:, k]
        for tau in sorted(cfg.quantiles):  # benchmarks recorded alongside so all three can be graded later
            fc[f"{target}_{h}_drift_q{int(round(tau * 100))}"] = latest[suite.anchor].to_numpy() + suite.drift_resid_q[tau]
        if suite.linear is not None:
            ql = suite.linear.predict(latest)
            for k, tau in enumerate(sorted(cfg.quantiles)):
                fc[f"{target}_{h}_linear_q{int(round(tau * 100))}"] = ql[:, k]
    fc.to_csv(out / "forecasts.csv", index=False)

    elast = estimate_elasticities(frame, cfg)
    pd.Series(elast).to_csv(out / "elasticities.csv")
    save_model_bundle(out / "model", suites, cfg, weights, elast,
                      "DDL 2010-2018 only" if njdg_csv is None else "DDL 2010-2018 + NJDG")

    ph = cfg.policy_horizon
    if ("growth", ph) not in suites or ("cr", ph) not in suites:
        LOG.error("policy horizon %d not modelled; stopping after forecasts", ph)
        return
    # flag = median path already non-clearing AND a material worst-case surge
    flagged = (fc[f"growth_{ph}_q90"] > cfg.flag_growth_q90) & (fc[f"cr_{ph}_q50"] < 1.0)
    LOG.info("flagged districts: %d / %d", int(flagged.sum()), len(fc))
    suite = suites[("growth", ph)]
    x_flag = latest.loc[flagged.to_numpy(), suite.features].set_index(latest.loc[flagged.to_numpy(), "district_id"])
    phi = shap_attribution(suite.boosters[max(cfg.quantiles)], x_flag)
    drivers = decompose_drivers(phi)
    phi.to_csv(out / "shap_q90_flagged.csv")
    drivers.to_csv(out / "drivers_flagged.csv")

    lever_path = out / "policy_levers.csv"
    if not elast["eps_judges_identified"]:
        lever_path.unlink(missing_ok=True)
        LOG.warning("policy levers not written: judge elasticity not identified (see elasticities.csv)")
        return
    a_grid = frame.pivot(index="district_id", columns="t", values="instituted")
    trend_grid = frame.pivot(index="district_id", columns="t", values="a_trend")
    cols = [t for t in a_grid.columns if t_last - 12 < t <= t_last]
    drivers["transient_excess_12m"] = (a_grid[cols] - trend_grid[cols]).clip(lower=0).sum(axis=1).reindex(drivers.index)
    levers = policy_levers(latest.loc[flagged.to_numpy()], fc.loc[flagged], drivers, elast, cfg, scrape)
    levers.to_csv(lever_path)
    LOG.info("policy levers (top 10 by q90 excess):\n%s",
             levers.sort_values("excess_disposals_needed", ascending=False).head(10).round(2).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ddl-csv-glob", help="raw DDL case CSVs, e.g. 'ddl/cases/cases_*.csv' (converted once)")
    ap.add_argument("--ddl-parquet", type=Path, default=Path("data/ddl_parquet"))
    ap.add_argument("--ddl-compact", type=Path, help="output folder of compress_ddl.py (replaces --ddl-csv-glob/--ddl-parquet)")
    ap.add_argument("--ddl-judges", type=Path)
    ap.add_argument("--njdg", type=Path)
    ap.add_argument("--edges", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--val-start", default="2022-01")
    ap.add_argument("--ddl-only", action="store_true", help="train on DDL 2010-2018 alone (no NJDG)")
    ap.add_argument("--synthetic", action="store_true", help="generate DDL/NJDG-schema synthetic inputs and run")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config(val_start=pd.Period(args.val_start, "M"))
    if args.ddl_only:
        # 2010-2018 window: 12-month horizon only (24/36 leave no room for train/cal/val with embargoes)
        # two rolling-origin folds (validation years 2016 and 2017); the 2016 fold has room for 6 calibration months
        cfg = Config(val_start=pd.Period("2017-01", "M"), horizons=(12,), policy_horizon=12, break_windows=(),
                     min_origin=pd.Period("2013-01", "M"), flag_growth_q90=float(np.log(1.05)),
                     val_folds=(("2016-01", 6), ("2017-01", 12)),
                     exclude_features=(*TRUNCATION_SENSITIVE, *JUDGE_RECORD_SENSITIVE))
    if args.synthetic:
        from synthetic_ddl import simulate

        paths = simulate(args.out / "synthetic_inputs", seed=cfg.seed)
        args.ddl_csv_glob, args.ddl_judges, args.njdg, args.edges = (
            str(paths["cases_glob"]), paths["judges"], paths["njdg"], paths["edges"])
        args.ddl_parquet = args.out / "synthetic_inputs" / "ddl_parquet"
    if args.ddl_csv_glob:
        convert_ddl_csv_to_parquet(args.ddl_csv_glob, args.ddl_parquet)
    if args.ddl_compact is not None and args.ddl_judges is None:
        args.ddl_judges = args.ddl_compact / "judges.csv.gz"
    if args.ddl_judges is None or (args.njdg is None and not args.ddl_only):
        ap.error("--ddl-judges and --njdg are required (or use --ddl-only / --synthetic)")
    if args.ddl_compact is not None:
        run(cfg, args.ddl_compact, args.ddl_judges, args.njdg, args.edges, args.out, compact=True)
    else:
        run(cfg, args.ddl_parquet, args.ddl_judges, args.njdg, args.edges, args.out)


if __name__ == "__main__":
    main()
