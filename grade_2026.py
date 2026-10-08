#!/usr/bin/env python3
"""Score the locked 31.12.2026 forecast (results/forecast_2026/) once official figures are published.

Input: a CSV with columns state,pending_2026 using the state names in official_series.py (combined units summed:
Andhra Pradesh + Telangana, Jammu & Kashmir + Ladakh, Dadra & Nagar Haveli + Daman & Diu), transcribed from the
Ministry of Law & Justice answer in Parliament that reports pending cases in District & Subordinate Courts on
31.12.2026. Scoring follows PROTOCOL_2026.md exactly.
    python3 grade_2026.py --actual actual_2026.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actual", type=Path, required=True)
    ap.add_argument("--forecast-dir", type=Path, default=HERE / "results" / "forecast_2026")
    args = ap.parse_args()
    fc = pd.read_csv(args.forecast_dir / "forecast_states_2026.csv", index_col="state")
    act = pd.read_csv(args.actual, index_col="state")["pending_2026"]
    missing = sorted(set(fc.index) - set(act.index))
    if missing:
        raise SystemExit(f"actual file is missing states: {missing}")
    d = fc.join(act)
    w = d["pending_2025"] / d["pending_2025"].sum()
    g_act = np.log(d["pending_2026"] / d["pending_2025"])
    g_model = np.log(d["q50"] / d["pending_2025"])
    g_zero = pd.Series(0.0, index=d.index)
    g_nat = pd.Series(float(np.log(d["pending_2026"].sum() / d["pending_2025"].sum())), index=d.index)  # oracle
    res = {
        "primary_case_weighted_mae_model": float((np.abs(g_model - g_act) * w).sum()),
        "case_weighted_mae_no_change": float((np.abs(g_zero - g_act) * w).sum()),
        "case_weighted_mae_oracle_national_rate": float((np.abs(g_nat - g_act) * w).sum()),
        "state_interval_coverage_q10_q90": float(((d["pending_2026"] >= d["q10"]) & (d["pending_2026"] <= d["q90"])).mean()),
        "national_actual": int(d["pending_2026"].sum()),
        "national_forecast_q50": int(d["q50"].sum()),
        "national_error_pct": float(d["q50"].sum() / d["pending_2026"].sum() - 1),
    }
    meta = json.loads((args.forecast_dir / "forecast_meta.json").read_text())
    res["national_inside_range"] = bool(meta["national"]["q10"] <= res["national_actual"] <= meta["national"]["q90"])
    res["beats_no_change"] = res["primary_case_weighted_mae_model"] < res["case_weighted_mae_no_change"]
    (args.forecast_dir / "grade_2026.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
