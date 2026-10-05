#!/usr/bin/env python3
"""Static result charts (PNG) for the README / profile, from a results folder written by pendency_forecast.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pendency_forecast import FEATURE_FAMILY  # noqa: E402

SURFACE: Final = "#fcfcfb"
TEXT: Final = "#0b0b0b"
TEXT_2: Final = "#52514e"
GRID: Final = "#e4e3df"
SERIES: Final = "#2a78d6"
SERIES_2: Final = "#eb6834"  # categorical slot 2 (validated adjacent pair with slot 1)
REFERENCE: Final = "#8a8984"
FAMILY_LABEL: Final = {
    "flow_state": "Recent backlog trend & inflow/disposal balance",
    "capacity": "Disposal momentum (judge counts excluded)",
    "spatial": "Peer districts in the same High Court",
    "procedural": "Case stage & case age",
    "transient_shock": "Short-term filing spikes",
    "demand_trend": "Long-run filing trend",
    "other": "Trend extrapolation (baseline anchor)",
}


def _style(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=10, length=0)
    ax.figure.text(0.02, 0.95, title, fontsize=14, fontweight="bold", color=TEXT, ha="left", va="top")
    ax.figure.text(0.02, 0.885, subtitle, fontsize=10, color=TEXT_2, ha="left", va="top")


def skill_chart(res: Path, out: Path) -> None:
    """Improvement over the trend baseline, pooled over the validation years, with 95% High-Court bootstrap ranges;
    the linear quantile regression is drawn beside the LightGBM model on the same baseline."""
    m = pd.read_csv(res / "validation_metrics.csv")
    m = m.loc[m["tau"] != "interval"].copy()
    folds = [f for f in m["fold"].astype(str).unique() if f != "pooled"]
    m = m.loc[m["fold"].astype(str) == ("pooled" if "pooled" in set(m["fold"].astype(str)) else folds[-1])]
    m["tau"] = m["tau"].astype(float)
    name = {"growth": "Backlog growth", "cr": "Clearance ratio"}
    q = {0.1: "best case (q10)", 0.5: "median (q50)", 0.9: "worst case (q90)"}
    m["label"] = [f"{name[t]}, {q[tau]}" for t, tau in zip(m["target"], m["tau"])]
    m = m.iloc[::-1].reset_index(drop=True)
    # linear skill vs drift from the pinball columns (no bootstrap range for it in the metrics file)
    lin = 100 * (1 - m["pinball_linear"] / m["pinball_naive"]).to_numpy()
    mod = 100 * m["skill_vs_drift"].to_numpy()
    lo, hi = 100 * m["skill_vs_drift_lo"].to_numpy(), 100 * m["skill_vs_drift_hi"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.30, right=0.95, top=0.78, bottom=0.17)
    y = np.arange(len(m))
    ax.barh(y + 0.19, mod, height=0.36, color=SERIES, edgecolor=SURFACE, linewidth=2, label="LightGBM model")
    ax.errorbar(mod, y + 0.19, xerr=[mod - lo, hi - mod], fmt="none", ecolor=TEXT_2, elinewidth=1, capsize=2)
    ax.barh(y - 0.19, lin, height=0.36, color=SERIES_2, edgecolor=SURFACE, linewidth=2, label="Linear quantile regression")
    for yi, v, h in zip(y, mod, hi):
        ax.text(max(v, h) + 0.8, yi + 0.19, f"{v:.0f}%", va="center", fontsize=9, color=TEXT)
    for yi, v in zip(y, lin):
        ax.text(v + 0.8, yi - 0.19, f"{v:.0f}%", va="center", fontsize=9, color=TEXT_2)
    ax.set_yticks(y, m["label"])
    ax.set_xlim(min(0, lo.min() - 2), max(hi.max(), lin.max()) * 1.15)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=TEXT_2)
    ax.set_xlabel("% lower forecast error than the trend-continuation baseline (pinball loss)\n"
                  "whiskers: 95% range for the LightGBM model, resampling whole High Courts", color=TEXT_2, fontsize=9)
    years = " and ".join(folds)
    _style(ax, f"Forecast accuracy on unseen years ({years})",
           f"12-month forecasts, {int(m['n_districts'].max()):,} districts; a simple linear model does about as well")
    fig.savefig(out / "forecast_accuracy.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def driver_chart(res: Path, out: Path) -> None:
    phi = pd.read_csv(res / "shap_q90_flagged.csv", index_col=0)
    fam = pd.Series({c: FEATURE_FAMILY.get(c, "other") for c in phi.columns})
    share = phi.abs().T.groupby(fam).sum().T.sum()
    share = (share / share.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.42, right=0.95, top=0.78, bottom=0.08)
    y = np.arange(len(share))
    ax.barh(y, 100 * share.to_numpy(), height=0.62, color=SERIES, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, 100 * share.to_numpy()):
        ax.text(v + 0.8, yi, f"{v:.0f}%", va="center", fontsize=10, color=TEXT)
    ax.set_yticks(y, [FAMILY_LABEL.get(k, k) for k in share.index])
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    _style(ax, "What the worst-case model relies on",
           f"Share of TreeSHAP attribution, q90 growth model, {len(phi)} flagged districts (associations, not causes)")
    fig.savefig(out / "risk_drivers.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def clearance_chart(res: Path, out: Path) -> None:
    fc = pd.read_csv(res / "forecasts.csv")
    col = [c for c in fc.columns if c.startswith("cr_") and c.endswith("_q50")][0]
    cr = fc[col].dropna().clip(0.4, 1.3)
    below = (fc[col] < 1).mean()
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.78, bottom=0.14)
    counts, _, _ = ax.hist(cr, bins=np.arange(0.4, 1.32, 0.04), color=SERIES, edgecolor=SURFACE, linewidth=2)
    top = counts.max() * 1.32
    ax.set_ylim(0, top)
    ax.set_xlim(0.38, 1.32)
    ax.axvline(1.0, color=REFERENCE, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(0.985, top * 0.98, f"{below:.0%} below 1", ha="right", va="top", fontsize=10, color=TEXT)
    ax.text(1.015, top * 0.98, "disposing more than filed", ha="left", va="top", fontsize=10, color=TEXT_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Forecast clearance ratio for 2019, median (disposals of cases filed since 2010 ÷ filings)",
                  color=TEXT_2, fontsize=9)
    ax.set_ylabel("Districts", color=TEXT_2, fontsize=9)
    _style(ax, "Most districts forecast to file more than they dispose",
           f"{len(cr)} districts, origin Dec 2018. Understated: disposals of pre-2010 cases are not in the data")
    fig.savefig(out / "district_clearance.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path(__file__).parent / "results" / "ddl_2010_2018")
    args = ap.parse_args()
    out = args.results / "charts"
    out.mkdir(exist_ok=True)
    skill_chart(args.results, out)
    driver_chart(args.results, out)
    clearance_chart(args.results, out)
    print(f"charts written to {out}")


if __name__ == "__main__":
    main()
