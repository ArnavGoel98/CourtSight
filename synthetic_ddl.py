"""Synthetic inputs in DDL-judicial-data and NJDG-aggregate schema for end-to-end pipeline tests.

Ground truth: case-level simulation 2004-2025 (pre-2010 legacy cohorts included). DDL view = filings
2010-2018 with decisions censored at the scrape date; NJDG view = monthly aggregates 2019-2025
computed independently of the pipeline's cohort-cube code.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

LOG: Final = logging.getLogger("pendency.synthetic")

SIM_START: Final = pd.Period("2004-01", "M")
SIM_END: Final = pd.Period("2025-12", "M")
DDL_START: Final = pd.Period("2010-01", "M")
DDL_END: Final = pd.Period("2018-12", "M")
SCRAPE: Final = pd.Timestamp("2019-06-30")
COVID_START: Final = pd.Period("2020-04", "M")

# (type_name string, criminal flag, base duration scale in months, mix weight)
CASE_TYPES: Final[tuple[tuple[str, bool, float, float], ...]] = (
    ("Sessions Case", True, 30.0, 0.08),
    ("Criminal Complaint C.C.", True, 20.0, 0.25),
    ("N.I. Act 138 Complaint", True, 16.0, 0.17),
    ("Bail Application", True, 1.5, 0.15),
    ("Original Suit O.S.", False, 30.0, 0.14),
    ("M.A.C.T. Claim Petition", False, 22.0, 0.07),
    ("Execution Petition", False, 18.0, 0.07),
    ("Hindu Marriage Act Petition", False, 14.0, 0.07),
)
STAGE_LABELS: Final[dict[str, str]] = {
    "service": "Summons / Notice Service", "charge": "Framing of Charge", "interlocutory": "I.A. Hearing",
    "evidence": "Prosecution Evidence", "arguments": "Final Arguments", "judgment": "For Judgment",
}
AGE_EDGES_M: Final = np.array([0, 12, 36, 60, 120, 240, 360, np.inf])


def _idx(p: pd.Period) -> int:
    return p.year * 12 + p.month - 1 - (SIM_START.year * 12 + SIM_START.month - 1)


def _to_date(month_idx: np.ndarray, rng: np.random.Generator) -> pd.DatetimeIndex:
    base = pd.Timestamp(SIM_START.start_time)
    months = pd.to_datetime(
        {"year": base.year + (base.month - 1 + month_idx) // 12, "month": (base.month - 1 + month_idx) % 12 + 1, "day": 1}
    )
    return pd.DatetimeIndex(months) + pd.to_timedelta(rng.integers(0, 28, len(month_idx)), unit="D")


def simulate(out_dir: Path, n_states: int = 5, dists_per_state: int = 8, seed: int = 7) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_t = _idx(SIM_END) + 1
    t = np.arange(n_t)
    n_d = n_states * dists_per_state
    state = np.repeat(np.arange(1, n_states + 1), dists_per_state)
    dist = np.tile(np.arange(1, dists_per_state + 1), n_states)

    # demand: trend + seasonality + idiosyncratic filing surges (transient shocks)
    lam0 = np.exp(rng.normal(np.log(90), 0.35, n_d))
    growth = rng.normal(0.012, 0.02, n_d) / 12
    lam = lam0[:, None] * np.exp(growth[:, None] * t[None, :]) * (1 + 0.08 * np.sin(2 * np.pi * t / 12))[None, :]
    for i in range(n_d):
        for _ in range(rng.integers(0, 3)):
            s0, ln = rng.integers(48, n_t - 12), rng.integers(4, 10)
            lam[i, s0 : s0 + ln] *= rng.uniform(1.5, 2.2)

    # capacity: seats (sanctioned) with tenure/vacancy spells; state 2 has a 2015-2019 recruitment freeze
    kappa = 10.0  # disposals per working judge-month at unit effectiveness
    lam_2015 = lam0 * np.exp(growth * _idx(pd.Period("2015-01", "M")))
    seats = np.maximum(3, np.round(lam_2015 * rng.uniform(0.9, 1.35, n_d) / (kappa * 0.85))).astype(int)
    eff = rng.uniform(0.75, 1.15, n_d)  # procedural effectiveness (adjournment culture, process service)
    vac_mean = rng.uniform(1.0, 8.0, n_d)
    occ = np.zeros((n_d, n_t))
    rows = []
    jid = 0
    freeze_lo, freeze_hi = _idx(pd.Period("2015-01", "M")), _idx(pd.Period("2019-12", "M"))
    for i in range(n_d):
        for s in range(seats[i]):
            cur = -int(rng.integers(0, 36))
            while cur < n_t:
                ten = 24 + int(rng.poisson(12))
                a, b = max(cur, 0), min(cur + ten, n_t) - 1
                if b >= a:
                    occ[i, a : b + 1] += 1
                    rows.append((jid, state[i], dist[i], s + 1, "Civil Judge" if s % 2 else "JMFC", a, b))
                    jid += 1
                freeze = 2.0 if (state[i] == 2 and freeze_lo <= cur + ten <= freeze_hi) else 1.0
                cur += ten + 1 + int(rng.geometric(1 / (vac_mean[i] * freeze)))
    judges = pd.DataFrame(rows, columns=["ddl_judge_id", "state_code", "dist_code", "court_no", "judge_position", "s", "e"])

    # cases: capacity-constrained multi-server queue. Monthly service capacity ~ Poisson(c_it * kappa * eff_i * covid_t);
    # served cases drawn by Gumbel top-k on priority = type urgency x age decay (negative duration dependence).
    counts = rng.poisson(lam)
    d_idx = np.repeat(np.repeat(np.arange(n_d), n_t), counts.reshape(-1))
    f_idx = np.repeat(np.tile(t, n_d), counts.reshape(-1))
    n = len(f_idx)
    probs = np.array([c[3] for c in CASE_TYPES])
    typ = rng.choice(len(CASE_TYPES), n, p=probs / probs.sum())
    logw_type = np.log(np.array([12.0 / c[2] for c in CASE_TYPES]))
    # minimum processing time (summons, pleadings, evidence): a case is not disposable before it ripens
    ripe = f_idx + np.ceil(np.array([c[2] for c in CASE_TYPES])[typ] * 0.6 * rng.gamma(2.0, 0.5, n)).astype(np.int64)
    covid = np.ones(n_t)
    covid[_idx(COVID_START) : _idx(pd.Period("2020-12", "M")) + 1] = 0.35
    covid[_idx(pd.Period("2021-01", "M")) : _idx(pd.Period("2021-06", "M")) + 1] = 0.7
    offsets = np.concatenate([[0], np.cumsum(counts.reshape(-1))])
    dec_idx = np.full(n, 10_000, dtype=np.int64)
    pend_hist = np.zeros((n_d, n_t))
    for i in range(n_d):
        pend = np.empty(0, dtype=np.int64)
        for m in range(n_t):
            pend = np.concatenate([pend, np.arange(offsets[i * n_t + m], offsets[i * n_t + m + 1])])
            elig = np.flatnonzero(ripe[pend] <= m)
            k = min(int(rng.poisson(occ[i, m] * kappa * eff[i] * covid[m])), len(elig))
            if k:
                cand = pend[elig]
                key = logw_type[typ[cand]] - 0.3 * np.log1p((m - f_idx[cand]) / 12.0) + rng.gumbel(size=len(cand))
                sel = elig[np.argpartition(-key, k - 1)[:k]]
                dec_idx[pend[sel]] = m
                pend = np.delete(pend, sel)
            pend_hist[i, m] = len(pend)
    gap_true = 30.0 * pend_hist / (np.maximum(occ, 1.0) * 600.0) / eff[:, None]
    LOG.info("simulated %d cases across %d districts; final pending median %.0f", n, n_d, np.median(pend_hist[:, -1]))

    filed = _to_date(f_idx, rng)
    decided = _to_date(dec_idx, rng)
    decided = pd.DatetimeIndex(np.where(decided < filed, filed, decided))
    crim = np.array([c[1] for c in CASE_TYPES])[typ]

    # ---- DDL view
    ddl = (f_idx >= _idx(DDL_START)) & (f_idx <= _idx(DDL_END))
    dd = pd.DataFrame({
        "ddl_case_id": np.arange(n)[ddl], "state_code": state[d_idx[ddl]], "dist_code": dist[d_idx[ddl]],
        "court_no": 1 + rng.integers(0, 3, ddl.sum()), "type_name": np.array([c[0] for c in CASE_TYPES])[typ[ddl]],
        "date_of_filing": filed[ddl], "date_of_decision": decided[ddl],
    })
    dd.loc[dd["date_of_decision"] > SCRAPE, "date_of_decision"] = pd.NaT
    pending = dd["date_of_decision"].isna().to_numpy()
    di = d_idx[ddl]
    pre_d = np.clip(0.30 + 0.8 * (1.15 - eff) + rng.normal(0, 0.03, n_d), 0.1, 0.8)
    pre = pre_d[di]
    u = rng.random(ddl.sum())
    stage = np.where(u < pre * 0.6, "service", np.where(u < pre * 0.8, "charge", np.where(u < pre, "interlocutory",
                     np.where(u < pre + (1 - pre) * 0.5, "evidence", np.where(u < pre + (1 - pre) * 0.85, "arguments", "judgment")))))
    dd["purpose_name"] = np.where(pending, pd.Series(stage).map(STAGE_LABELS).to_numpy(), "Disposed")
    gap_days = gap_true[di, _idx(pd.Period("2019-06", "M"))] * rng.lognormal(0, 0.3, ddl.sum())
    last = np.where(pending, (SCRAPE - pd.to_timedelta(rng.uniform(0, 2.5, ddl.sum()) * gap_days, unit="D")).to_numpy(),
                    dd["date_of_decision"].to_numpy())
    dd["date_last_list"] = pd.to_datetime(last)
    dd["date_next_list"] = pd.to_datetime(np.where(pending, (dd["date_last_list"] + pd.to_timedelta(gap_days, unit="D")).to_numpy(), np.datetime64("NaT")))
    dd["date_first_list"] = dd["date_of_filing"] + pd.to_timedelta(rng.integers(7, 60, ddl.sum()), unit="D")
    case_dir = out_dir / "ddl_cases"
    case_dir.mkdir(exist_ok=True)
    for yr, grp in dd.groupby(dd["date_of_filing"].dt.year):
        g = grp.copy()
        for c in ("date_of_filing", "date_of_decision", "date_first_list", "date_last_list", "date_next_list"):
            g[c] = g[c].dt.strftime("%Y-%m-%d")
        g.to_csv(case_dir / f"cases_{yr}.csv", index=False)

    j = judges.copy()
    j["start_date"] = _to_date(j["s"].to_numpy(), rng).strftime("%Y-%m-%d")
    end = _to_date(j["e"].to_numpy(), rng)
    j["end_date"] = np.where(end > SCRAPE, "", end.strftime("%Y-%m-%d"))
    j = j.loc[_to_date(j["s"].to_numpy(), rng) <= SCRAPE].drop(columns=["s", "e"])
    j.to_csv(out_dir / "judges_clean.csv", index=False)

    # ---- NJDG view: independent month-by-month computation on the full universe
    recs = []
    nb = len(AGE_EDGES_M) - 1
    for m in range(_idx(pd.Period("2019-01", "M")), n_t):
        live = (f_idx <= m) & (dec_idx > m)
        b = np.digitize(m - f_idx[live], AGE_EDGES_M[1:-1])
        ages = np.bincount(d_idx[live] * nb + b, minlength=n_d * nb).reshape(n_d, nb)
        rec = {
            "state_code": state, "dist_code": dist, "month": str(SIM_START + m),
            "pending": ages.sum(1), "pending_crim": np.bincount(d_idx[live & crim], minlength=n_d)[:],
            "instituted": np.bincount(d_idx[f_idx == m], minlength=n_d),
            "disposed": np.bincount(d_idx[dec_idx == m], minlength=n_d),
            "working_judges": occ[:, m], "sanctioned_judges": seats,
            "hearing_gap_days": gap_true[:, m] * rng.lognormal(0, 0.05, n_d),
        }
        pre_m = np.clip(pre_d + rng.normal(0, 0.02, n_d), 0.1, 0.8)
        for s, share in (("service", 0.6 * pre_m), ("charge", 0.2 * pre_m), ("interlocutory", 0.2 * pre_m),
                         ("evidence", 0.5 * (1 - pre_m)), ("arguments", 0.35 * (1 - pre_m)), ("judgment", 0.15 * (1 - pre_m))):
            rec[f"share_{s}"] = share
        for k, name in enumerate(("age_0_1", "age_1_3", "age_3_5", "age_5_10", "age_10_20", "age_20_30", "age_30p")):
            rec[name] = ages[:, k]
        recs.append(pd.DataFrame(rec))
    pd.concat(recs, ignore_index=True).to_csv(out_dir / "njdg_monthly.csv", index=False)

    edges = []
    for s in range(1, n_states + 1):
        for k in range(1, dists_per_state + 1):
            edges.append((s, k, s, k % dists_per_state + 1))
        if s < n_states:
            edges.append((s, 1, s + 1, dists_per_state // 2))
    pd.DataFrame(edges, columns=["src_state", "src_dist", "dst_state", "dst_dist"]).to_csv(out_dir / "district_edges.csv", index=False)
    return {"cases_glob": case_dir / "cases_*.csv", "judges": out_dir / "judges_clean.csv",
            "njdg": out_dir / "njdg_monthly.csv", "edges": out_dir / "district_edges.csv"}
