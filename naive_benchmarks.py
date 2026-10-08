#!/usr/bin/env python3
"""District-level validation against the strongest simple forecasts, not only the DDL trend anchor.

The trend anchor extrapolates growth of the post-2010 stock, which slows mechanically as that stock matures, so it is
a weak benchmark. Here each validation fold is also scored against:
  climatology    quantiles of the target over all training rows (ignores district history)
  last_year_xs   quantiles of the target across districts in the last fully realised year before the fold
Run after pendency_forecast.py --ddl-only --out <dir>, which writes <dir>/panel_features.parquet.
    python3 naive_benchmarks.py --panel outputs/panel_features.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pendency_forecast as pf  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "ddl_2010_2018" / "naive_benchmarks.csv")
    args = ap.parse_args()
    fr = pd.read_parquet(args.panel)
    cfg = pf.Config(val_start=pd.Period("2017-01", "M"), horizons=(12,), break_windows=(),
                    min_origin=pd.Period("2013-01", "M"), val_folds=(("2016-01", 6), ("2017-01", 12)),
                    exclude_features=(*pf.TRUNCATION_SENSITIVE, *pf.JUDGE_RECORD_SENSITIVE))
    rows = []
    for target in ("growth", "cr"):
        y_col, anchor = f"y_{target}_12", f"anchor_{target}_12"
        data = fr.loc[fr[anchor].notna() & (fr["months_on_ecourts"] >= cfg.min_months_on_ecourts) & fr[y_col].notna()]
        for vs, cal in cfg.val_folds:
            v0 = pf.abs_month(pd.Period(vs, "M"))
            fit = pf._fit_one(data, target, 12, cfg, v0, cal)
            if fit is None:
                continue
            suite, _, tr, _, va = fit
            va = va.loc[va["t"] < v0 + 12]
            y = va[y_col].to_numpy()
            q = suite.predict(va)
            last = data.loc[(data["t"] >= v0 - 24) & (data["t"] < v0 - 12), y_col]
            for k, tau in enumerate(sorted(cfg.quantiles)):
                losses = {
                    "model": pf.pinball(y, q[:, k], tau),
                    "drift": pf.pinball(y, va[anchor].to_numpy() + np.quantile(tr[y_col] - tr[anchor], tau), tau),
                    "climatology": pf.pinball(y, np.full(len(y), np.quantile(tr[y_col], tau)), tau),
                    "last_year_xs": pf.pinball(y, np.full(len(y), np.quantile(last, tau)), tau),
                }
                best = min(v for kk, v in losses.items() if kk != "model")
                rows.append({"fold": vs[:4], "target": target, "tau": tau, **losses,
                             "skill_vs_drift": 1 - losses["model"] / losses["drift"],
                             "skill_vs_best_naive": 1 - losses["model"] / best})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
