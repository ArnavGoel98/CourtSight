#!/usr/bin/env python3
"""What does a change of judge cost a courtroom? Event study on courtroom-month disposals, DDL 2010-2018.

Inputs: ddl_compact/detail/court_month.parquet (from `compress_ddl.py --detail`) and ddl_compact/judges.csv.gz.
Event: a judge's posting starts in a courtroom that already had an earlier posting (a handover).
Design (stacked difference-in-differences):
  - for each event, outcome = the courtroom's monthly disposals divided by its own average over the 12 months
    before the event (event months -12..-1);
  - controls = other courtrooms in the same district with no judge start within 12 months either side, scaled the
    same way over the same calendar months;
  - effect(k) = mean over events of [treated(k) - mean control(k)], for event months k = -6..12;
  - 95% intervals by resampling districts (events in one district share shocks).
Reported: the effect by event month, the cumulative disposals lost over the 12 months after a handover, and the
same split by how long the courtroom sat without a judge between postings. The match between case court numbers
and judge court numbers is checked first (share of judge-months whose courtroom shows disposals).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
PRE, POST, BASE = 6, 12, 12


def months(ym: pd.Series | np.ndarray) -> np.ndarray:
    a = np.asarray(ym, dtype=np.int64)
    return (a // 100) * 12 + a % 100 - 1


def load_judges(path: Path) -> pd.DataFrame:
    j = pd.read_csv(path)
    for c in ("start_date", "end_date"):
        raw = j[c].astype("string")
        parsed = pd.to_datetime(raw, format="%d-%m-%Y", errors="coerce")  # DDL release format
        if parsed.notna().sum() < 0.5 * raw.notna().sum():
            parsed = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
        j[c] = parsed
    j = j.loc[j["start_date"].notna()].copy()
    j["district_id"] = j["state_code"] * 1000 + j["dist_code"]
    j["s"] = j["start_date"].dt.year * 12 + j["start_date"].dt.month - 1
    j["e"] = j["end_date"].dt.year * 12 + j["end_date"].dt.month - 1
    return j


def courtroom_panel(cm: pd.DataFrame) -> pd.DataFrame:
    cm = cm.loc[cm["court_no"] >= 0].assign(t=months(cm["ym"]))
    cm = cm.loc[(cm["t"] >= 2010 * 12) & (cm["t"] <= 2018 * 12 + 11)]
    return cm.groupby(["district_id", "court_no", "t"])["disposed"].sum()


def events(j: pd.DataFrame) -> pd.DataFrame:
    j = j.sort_values(["district_id", "court_no", "s"])
    prev_end = j.groupby(["district_id", "court_no"])["e"].shift(1)
    ev = j.loc[prev_end.notna(), ["district_id", "court_no", "s"]].copy()
    ev["vacancy_months"] = (j.loc[prev_end.notna(), "s"] - prev_end[prev_end.notna()]).clip(lower=0)
    return ev.rename(columns={"s": "t0"}).drop_duplicates(["district_id", "court_no", "t0"])


def event_study(panel: pd.Series, ev: pd.DataFrame, starts: pd.DataFrame, min_base: float) -> pd.DataFrame:
    """One row per usable event: treated and control paths for k = -PRE..POST, scaled by pre-event mean."""
    wide = panel.astype(float).unstack("t", fill_value=0.0)
    cols = wide.columns.to_numpy()
    ks = np.arange(-PRE, POST + 1)
    rows = []
    starts_by = {k: g["s"].to_numpy() for k, g in starts.groupby(["district_id", "court_no"])}
    by_dist = {d: g for d, g in wide.groupby(level="district_id")}
    for r in ev.itertuples(index=False):
        if (r.district_id, r.court_no) not in wide.index or r.district_id not in by_dist:
            continue
        need = np.arange(r.t0 - BASE, r.t0 + POST + 1)
        if need[0] < cols[0] or need[-1] > cols[-1]:
            continue
        idx = need - cols[0]
        tr = wide.loc[(r.district_id, r.court_no)].to_numpy()[idx]
        base = tr[:BASE].mean()
        if base < min_base:
            continue
        g = by_dist[r.district_id]
        ctrl = []
        for (d, c), vals in zip(g.index, g.to_numpy()):
            if c == r.court_no:
                continue
            st = starts_by.get((d, c), np.array([]))
            if np.any(np.abs(st - r.t0) <= POST):
                continue
            v = vals[idx]
            b = v[:BASE].mean()
            if b >= min_base:
                ctrl.append(v / b)
        if not ctrl:
            continue
        ctrl_m = np.mean(ctrl, axis=0)
        tr_s = tr / base
        sel = BASE - PRE + np.arange(PRE + POST + 1)
        rows.append({"district_id": r.district_id, "court_no": r.court_no, "t0": r.t0,
                     "vacancy_months": r.vacancy_months, "base_disposals": base, "n_controls": len(ctrl),
                     **{f"k{k}": tr_s[s] - ctrl_m[s] for k, s in zip(ks, sel)}})
    return pd.DataFrame(rows)


def summarise(es: pd.DataFrame, draws: int, seed: int = 0) -> dict[str, object]:
    kcols = [f"k{k}" for k in range(-PRE, POST + 1)]
    post = [f"k{k}" for k in range(0, POST)]
    eff = es[kcols].mean()
    lost = -(es[post].sum(axis=1) * es["base_disposals"])  # disposals lost per event over months 0..11
    dists = es["district_id"].unique()
    rng = np.random.default_rng(seed)
    grp = {d: g for d, g in es.groupby("district_id")}
    boot_eff, boot_lost = [], []
    for _ in range(draws):
        s = pd.concat([grp[d] for d in rng.choice(dists, len(dists))])
        boot_eff.append(s[kcols].mean().to_numpy())
        boot_lost.append(float((-(s[post].sum(axis=1) * s["base_disposals"])).mean()))
    be = np.array(boot_eff)
    return {"events": int(len(es)), "districts": int(len(dists)),
            "effect_by_month": {k: [float(eff[k]), float(np.quantile(be[:, i], 0.025)), float(np.quantile(be[:, i], 0.975))]
                                for i, k in enumerate(kcols)},
            "lost_disposals_12m_mean": float(lost.mean()),
            "lost_disposals_12m_ci": [float(np.quantile(boot_lost, 0.025)), float(np.quantile(boot_lost, 0.975))],
            "lost_share_of_12m_baseline": float(lost.mean() / (12 * es["base_disposals"].mean())),
            "pre_trend_mean_k-6_to_k-1": float(es[[f"k{k}" for k in range(-PRE, 0)]].mean().mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compact", type=Path, default=HERE / "ddl_compact")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "judge_transfer")
    ap.add_argument("--min-base", type=float, default=10.0, help="minimum pre-event disposals per month")
    ap.add_argument("--draws", type=int, default=500)
    args = ap.parse_args()
    cm_path = args.compact / "detail" / "court_month.parquet"
    if not cm_path.exists():
        raise SystemExit(f"{cm_path} not found: run compress_ddl.py --detail on the DDL download first")
    args.out.mkdir(parents=True, exist_ok=True)
    panel = courtroom_panel(pd.read_parquet(cm_path))
    j = load_judges(args.compact / "judges.csv.gz")
    # linkage check: judge-months whose courtroom records any disposals that month
    jm = j.loc[j["e"].notna() & (j["e"] >= j["s"])].head(200000)
    hits = total = 0
    for r in jm.itertuples(index=False):
        for t in range(int(r.s), int(min(r.e, r.s + 11)) + 1):
            total += 1
            hits += int(panel.get((r.district_id, r.court_no, t), 0) > 0)
    link = hits / max(total, 1)
    ev = events(j)
    es = event_study(panel, ev, j[["district_id", "court_no", "s"]], args.min_base)
    res = {"courtroom_link_rate": link, "candidate_handovers": int(len(ev)), **summarise(es, args.draws)}
    for lab, sub in (("vacancy_0_1m", es.loc[es["vacancy_months"] <= 1]), ("vacancy_2m_plus", es.loc[es["vacancy_months"] >= 2])):
        if len(sub) >= 30:
            res[f"by_{lab}"] = {k: v for k, v in summarise(sub, max(args.draws // 5, 50)).items() if k != "effect_by_month"}
    # placebo: the same design with each handover moved 18 and 30 months earlier, where no judge changed.
    # A real handover cost shows up at the true date only; a similar "effect" at fake dates means the design is
    # picking up courtrooms that were declining anyway (or mean reversion from the baseline window).
    k_post = [f"k{k}" for k in range(0, POST)]
    res["placebo_mean_post_effect"] = {"true_date": float(es[k_post].mean().mean())}
    for shift in (18, 30):
        pe = event_study(panel, ev.assign(t0=ev["t0"] - shift), j[["district_id", "court_no", "s"]], args.min_base)
        res["placebo_mean_post_effect"][f"{shift}_months_earlier"] = float(pe[k_post].mean().mean())
    (args.out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "effect_by_month"}, indent=2))
    print({k: round(v[0], 3) for k, v in res["effect_by_month"].items()})


if __name__ == "__main__":
    main()
