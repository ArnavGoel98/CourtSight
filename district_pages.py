#!/usr/bin/env python3
"""One static page per district court, per state and an index, so each district's answer can be found, linked
and read without JavaScript (docs/d/). Called by build_site.py; everything comes from the committed results.

  docs/d/index.html                       every state and district
  docs/d/<state>/index.html               the state's districts on a map and in a table
  docs/d/<state>/<district>.html          one district: every case type, a chart, a locator map, and the caveats
                                          that apply to that district (records completeness, its state's coverage
                                          of official filings, stale pending records, rank uncertainty)
  docs/api/v1/<state>/<district>.json     the same figures as data (a static, read-only API), with docs/api/v1/index.json
  docs/site.css                           the shared stylesheet these pages link to

Pages are descriptive. Each repeats the limits that matter for it, because many readers will arrive from a search
and never see the audit.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

import build_site as B

HERE: Final = Path(__file__).resolve().parent
RES: Final = HERE / "results"
OUT: Final = HERE / "docs" / "d"
API: Final = HERE / "docs" / "api" / "v1"
ORDER: Final = ["civ_suit", "civ_family", "crim_ni138", "civ_mact", "crim_bail", "crim_sessions", "crim_magisterial",
                "civ_execution", "civ_misc", "other"]
RAMP: Final = ["--m1", "--m2", "--m3", "--m4", "--m5"]
EXTRA_CSS: Final = """
:root{--m1:#86b6ef;--m2:#5598e7;--m3:#2a78d6;--m4:#1c5cab;--m5:#0d366b;--nodata:#e9e7e1}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){--m1:#184f95;--m2:#3987e5;--m3:#6da7ec;--m4:#9ec5f4;--m5:#cde2fb;--nodata:#2a2a28}}
:root[data-theme="dark"]{--m1:#184f95;--m2:#3987e5;--m3:#6da7ec;--m4:#9ec5f4;--m5:#cde2fb;--nodata:#2a2a28}
.crumbs{font-size:.85rem;color:var(--ink-3);margin:20px 0 0}.crumbs a{color:var(--ink-2)}
.dhead{display:grid;grid-template-columns:1fr 260px;gap:24px;align-items:start}
@media (max-width:760px){.dhead{grid-template-columns:1fr}.dhead .loc{max-width:220px}}
.loc svg{width:100%;height:auto;display:block}.loc path{stroke:var(--surface);stroke-width:.8;stroke-linejoin:round}
.loc .o{fill:var(--nodata)}.loc .me{fill:var(--s1)}.loc figcaption{font-size:.78rem;color:var(--ink-3);margin-top:4px}
.smap svg{width:100%;height:auto;max-height:70vh;display:block}.smap path{stroke:var(--surface);stroke-width:.8;stroke-linejoin:round}
.smap a path:hover{stroke:var(--ink);stroke-width:1.6}
.warn{border-left:3px solid var(--neg);padding-left:12px}
.cols{columns:3 220px;column-gap:24px;font-size:.92rem}.cols a{color:var(--ink);text-decoration:none}.cols a:hover{text-decoration:underline}
.cols h2{break-after:avoid;margin-top:14px}
.tbl{overflow-x:auto}.tbl td.l{text-align:left}.tbl .sub{display:block;font-size:.75rem;color:var(--ink-3)}
.lines .dist{stroke:var(--s1);stroke-width:2.5;fill:none}.lines .nat{stroke:var(--ink-3);stroke-width:2;fill:none;stroke-dasharray:5 5}
.lines text{fill:var(--ink-2);font-size:12px}@media (max-width:560px){.lines text{font-size:19px}}.lines .lab{fill:var(--ink);font-weight:600}.lines .grid line{stroke:var(--grid)}
"""


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "x"


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def fmt_m(v: float | None) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "over half still pending"
    return f"{v:.0f} month{'s' if round(v) != 1 else ''}" if v < 24 else f"{v / 12:.1f} years"


def inr(n: float) -> str:
    """Indian digit grouping: 12,34,567."""
    s = f"{int(round(n))}"
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    return re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head) + "," + tail


def pct(v: float | None) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v * 100:.0f}%"


# ---------- geometry: the map paths are "M x y l dx dy dx dy ... z", possibly several rings per shape ----------
def rings(d: str) -> list[list[tuple[int, int]]]:
    out = []
    for part in re.findall(r"M[^M]*", d):
        nums = [int(n) for n in re.findall(r"-?\d+", part)]
        x, y = nums[0], nums[1]
        pts = [(x, y)]
        for dx, dy in zip(nums[2::2], nums[3::2]):
            x, y = x + dx, y + dy
            pts.append((x, y))
        out.append(pts)
    return out


def rdp(pts: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker line simplification (iterative). A closed ring is split at its point farthest from the
    start, since a segment from a point to itself measures no distance."""
    if len(pts) < 3:
        return pts
    if pts[0] == pts[-1]:
        k = max(range(len(pts)), key=lambda i: (pts[i][0] - pts[0][0]) ** 2 + (pts[i][1] - pts[0][1]) ** 2)
        if 0 < k < len(pts) - 1:
            return rdp(pts[:k + 1], eps) + rdp(pts[k:], eps)[1:]
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        (x1, y1), (x2, y2) = pts[i], pts[j]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5 or 1e-9
        best, k = -1.0, -1
        for m in range(i + 1, j):
            d = abs(dy * (pts[m][0] - x1) - dx * (pts[m][1] - y1)) / norm
            if d > best:
                best, k = d, m
        if best > eps:
            keep[k] = True
            stack += [(i, k), (k, j)]
    return [p for p, f in zip(pts, keep) if f]


def fit(shapes: dict[str, list[list[tuple[int, int]]]], width: int = 300,
        eps: float = 0.0) -> tuple[dict[str, str], int, int]:
    """Rescale a set of shapes to `width` units wide, as compact relative SVG paths."""
    pts = [p for rs in shapes.values() for r in rs for p in r]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    s = width / max(x1 - x0, 1)
    if (y1 - y0) * s > width * 1.3:            # tall states: cap the height instead
        s = width * 1.3 / (y1 - y0)
    out = {}
    for k, rs in shapes.items():
        parts = []
        for r in rs:
            f = [((x - x0) * s, (y - y0) * s) for x, y in r]
            q = [(round(x), round(y)) for x, y in (rdp(f, eps) if eps else f)]
            q = [q[0]] + [b for a, b in zip(q, q[1:]) if b != a]
            if len(q) < 3:
                continue
            rel = " ".join(f"{b[0] - a[0]} {b[1] - a[1]}" for a, b in zip(q, q[1:]))
            parts.append(f"M{q[0][0]} {q[0][1]}l{rel}z".replace(" -", "-"))
        out[k] = "".join(parts)
    return out, round((x1 - x0) * s), round((y1 - y0) * s)


# ---------- chart ----------
def curve_svg(dist: list[float | None], nat: list[float | None], name: str, label: str) -> str:
    W, H, l, r, t, b = 640, 250, 44, 120, 12, 30
    X = lambda m: l + m / 120 * (W - l - r)
    Y = lambda v: t + (1 - v) * (H - t - b)

    def path(c: list[float | None]) -> tuple[str, tuple[float, float] | None]:
        pts = [(i * 3, v) for i, v in enumerate(c) if v is not None]
        if not pts:
            return "", None
        d = "M" + " L".join(f"{X(m):.1f} {Y(v):.1f}" for m, v in pts)
        return d, (X(pts[-1][0]), Y(pts[-1][1]))

    dd, de = path(dist)
    nd, ne = path(nat)
    grid = "".join(f'<line x1="{l}" x2="{W - r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                   f'<text x="{l - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v * 100:.0f}%</text>' for v in (0, .25, .5, .75, 1))
    xt = "".join(f'<text x="{X(m):.1f}" y="{H - 8}" text-anchor="middle">{m // 12}{" yrs" if m == 120 else ""}</text>'
                 for m in range(0, 121, 24))
    labs = ""
    if de and ne and abs(de[1] - ne[1]) < 16:   # keep the two end labels apart
        lo, hi = sorted([de, ne], key=lambda p: p[1])
        (de, ne) = (lo, (ne[0], lo[1] + 16)) if de is lo else ((de[0], lo[1] + 16), lo)
    if de:
        labs += f'<text class="lab" x="{de[0] + 6:.1f}" y="{de[1] + 4:.1f}">{esc(name)}</text>'
    if ne:
        labs += f'<text x="{ne[0] + 6:.1f}" y="{ne[1] + 4:.1f}">All of India</text>'
    return (f'<svg viewBox="0 0 {W} {H}" role="img" class="lines"><title>Share of {esc(label.lower())} still pending, '
            f'by years since filing: {esc(name)} and all of India</title><g class="grid">{grid}</g>{xt}'
            f'<path class="nat" d="{nd}"/><path class="dist" d="{dd}"/>{labs}</svg>')


# ---------- page shell ----------
def page(title: str, desc: str, canon: str, depth: int, body: str) -> str:
    up = "../" * depth
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(desc)}">
<link rel="canonical" href="{B.BASE}{canon}">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 32 32%22%3E%3Crect width=%2232%22 height=%2232%22 rx=%227%22 fill=%22%231f5f8b%22/%3E%3Cpath d=%22M7 23h18M9 23V13m5 10V9m5 14v-7m5 7V11%22 stroke=%22white%22 stroke-width=%222.6%22 stroke-linecap=%22round%22/%3E%3C/svg%3E">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(desc)}">
<meta property="og:url" content="{B.BASE}{canon}">
<meta property="og:image" content="{B.BASE}og.png">
<meta name="twitter:card" content="summary_large_image">
<link rel="stylesheet" href="{up}site.css">
</head>
<body>
<nav><div class="wrap"><b>CourtSight</b><a href="{up}calculator.html">Calculator</a><a href="{up}districts.html">Rankings</a><a href="{up}map.html">Map</a><a href="{up}d/index.html">Districts</a><a href="{up}index.html">Findings</a><a href="{up}paper.html">Paper</a></div></nav>
<main class="wrap">
{body}
</main>
<footer><div class="wrap">CourtSight by Arnav Goel · <a href="https://github.com/ArnavGoel98/CourtSight">Code and method</a> · Data: Development Data Lab eCourts records 2010–2018 (CC BY-NC-SA 4.0); boundaries: Census 2011 via DataMeet (CC BY 4.0). Descriptive statistics, not legal advice.</div></footer>
</body>
</html>
"""


def load() -> dict[str, object]:
    t = pd.read_csv(RES / "time_to_decision" / "by_district_case_type.csv").dropna(subset=["state_name", "district_name"])
    t = t.merge(pd.read_csv(RES / "time_to_decision" / "curves_by_district_case_type.csv.gz"), on=["district_id", "case_type"], how="left")
    t = t.merge(pd.read_csv(RES / "audit" / "rank_intervals.csv"), on=["district_id", "case_type"], how="left")
    nat = pd.read_csv(RES / "time_to_decision" / "national_case_type.csv").merge(
        pd.read_csv(RES / "time_to_decision" / "curves_national_case_type.csv"), on="case_type").set_index("case_type")
    dig = pd.read_csv(RES / "time_to_decision" / "digitisation.csv").set_index("district_id")["first_complete_year"]
    wy = pd.read_csv(RES / "waiting_years" / "districts.csv").set_index("district_id")
    drv = pd.read_csv(RES / "district_drivers" / "districts.csv").set_index("district_id")
    special = set(json.loads((RES / "audit" / "summary.json").read_text())["units_that_are_court_complexes"])
    cov = pd.read_csv(RES / "audit" / "coverage_by_state.csv").set_index("state")
    act = pd.read_csv(RES / "audit" / "pending_activity.csv").set_index("state")
    from coverage_audit import state_units
    names = pd.read_csv(HERE / "ddl_compact" / "district_key.csv").groupby("state_code")["state_name"].first()
    unit_of = {names[k]: v for k, v in state_units().items()}
    held = json.loads((RES / "audit" / "summary.json").read_text())["withheld_detail"]
    return {"t": t, "nat": nat, "dig": dig, "wy": wy, "drv": drv, "special": special, "cov": cov, "act": act, "held": held,
            "unit_of": unit_of, "map": B.map_data()}


def build(sitemap: list[str]) -> int:
    D = load()
    t, nat, M = D["t"], D["nat"], D["map"]
    curve = lambda s: B._curve(s)
    if OUT.exists():
        for f in sorted(OUT.rglob("*.html"), reverse=True):
            f.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    if API.exists():
        for f in sorted(API.rglob("*.json")):
            f.unlink()
    API.mkdir(parents=True, exist_ok=True)
    api_index: list[dict[str, object]] = []
    num = lambda v, nd=3: None if v is None or pd.isna(v) else (int(v) if float(v).is_integer() and nd == 0 else round(float(v), nd))
    (HERE / "docs" / "site.css").write_text((B.SITE / "style.css").read_text() + EXTRA_CSS)

    # shapes for each (state, district), and national fifths for civil suits (as on map.html and the static map)
    codes: dict[tuple[str, str], list[str]] = {}
    for code, us in M["units"].items():
        for s, d, _ in us:
            codes.setdefault((s, d), []).append(code)
    geo = {c: rings(p) for c, p in M["paths"].items()}
    civ = t.loc[t["case_type"] == "civ_suit"].set_index(["state_name", "district_name"])
    cuts = np.quantile(civ["pending_after_3y"].dropna(), [.2, .4, .6, .8])
    bin_of = lambda v: int(np.searchsorted(cuts, v, side="right"))
    nat_civ = nat.loc["civ_suit"]

    states = {s: g for s, g in t.groupby("state_name")}
    state_slug = {s: slug(s) for s in states}
    dist_slug: dict[tuple[str, str], str] = {}
    for s, g in states.items():
        seen: set[str] = set()
        for d in sorted(g["district_name"].unique()):
            k = slug(d)
            assert k not in seen, f"two districts in {s} share the slug {k}; embed.html assumes slugs are unique"
            seen.add(k)
            dist_slug[(s, d)] = k
    n = 0

    for s, g in states.items():
        sdir = OUT / state_slug[s]
        sdir.mkdir(exist_ok=True)
        unit = D["unit_of"].get(s)
        cov_row = D["cov"].loc[unit] if unit in D["cov"].index else None
        act_v = float(D["act"].loc[s, "not_listed_in_12m"]) if s in D["act"].index else None
        state_notes = []
        if cov_row is not None and cov_row["group"] != "too small to classify":
            state_notes.append(f"{esc(unit)}'s eCourts records hold about <b>{pct(cov_row['coverage_2015_18'])}</b> of the "
                               f"cases the official figures say were filed in 2015–18"
                               + (" (the official figures for this unit include another state or territory)" if "+" in unit else "")
                               + ". Figures here describe the courts eCourts recorded.")
        elif unit is not None:
            state_notes.append("Official filing figures for this state are too small or erratic to check how complete the records are.")
        if act_v is not None and act_v >= 0.2:
            state_notes.append(f'<span class="warn">In {esc(s)}, <b>{pct(act_v)}</b> of cases recorded as pending had no hearing '
                               f'listed in the year before the data was collected, so some may in fact have been decided. '
                               f'Delays here may be overstated.</span>')
        # state map: every 2011 shape in the census states this state's districts fall in
        cstates = {M["names"][c][1] for d in g["district_name"].unique() for c in codes.get((s, d), [])}
        sshapes = {c: geo[c] for c, nm in M["names"].items() if nm[1] in cstates}
        dists = sorted(g["district_name"].unique())

        for d in dists:
            r = g.loc[g["district_name"] == d].set_index("case_type")
            did = int(r["district_id"].iloc[0])
            is_special = d in D["special"]
            prim = "civ_suit" if "civ_suit" in r.index and pd.notna(r.loc["civ_suit", "pending_after_3y"]) else \
                r["cases"].idxmax()
            p = r.loc[prim]
            label = B.LABELS[prim]
            total = int(r["cases"].sum())
            nat_p = nat.loc[prim]
            # headline
            med = None if pd.isna(p["median_months"]) else float(p["median_months"])
            head = (f"Half of the {label.split(' (')[0].lower()} cases filed here in 2010–2018 were decided within "
                    f"<b>{fmt_m(med)}</b>" if med is not None else
                    f"More than half of the {label.split(' (')[0].lower()} cases filed here in 2010–2018 were "
                    f"<b>still pending</b> when the data ends")
            head += f" (all of India: {fmt_m(nat_p['median_months'])})."
            rank_txt = ""
            if pd.notna(p.get("rank")) and pd.notna(p["pending_after_3y"]):
                rk, lo, hi, of = int(p["rank"]), int(p["rank_lo"]), int(p["rank_hi"]), int(p["of"])
                where = ("clearly among the fastest fifth" if hi <= of * 0.2 else
                         "clearly among the slowest fifth" if lo >= of * 0.8 else "")
                rank_txt = (f" <b>{pct(p['pending_after_3y'])}</b> were still pending three years after filing "
                            f"(India: {pct(nat_p['pending_after_3y'])}), which ranks {esc(d)} <b>{rk} of {of}</b> "
                            f"(1 = fewest pending; 90% range {lo}–{hi})" + (f": {where}." if where else "."))
            # tiles
            w = D["wy"].loc[did] if did in D["wy"].index else None
            tiles = [(inr(total), "cases filed 2010–2018 in this data")]
            tiles.append((fmt_m(med) if med is not None else "over half", f"median time, {B.SHORT[prim].lower()}"))
            tiles.append((pct(p["pending_after_3y"]), "still pending after 3 years"))
            if w is not None:
                tiles.append((inr(w["pending"]), "still pending when the data was collected"))
            tiles_html = "".join(f'<div class="tile"><div class="v">{esc(v)}</div><div class="l">{esc(lbl)}</div></div>'
                                 for v, lbl in tiles)
            # table
            rows = []
            for ct in ORDER:
                if ct not in r.index:
                    continue
                x = r.loc[ct]
                rng = (f'{int(x["rank"])}<span class="sub">{int(x["rank_lo"])}–{int(x["rank_hi"])} of {int(x["of"])}</span>'
                       if pd.notna(x.get("rank")) else "–")
                rows.append(f'<tr><td class="l">{esc(B.LABELS[ct])}</td><td>{inr(x["cases"])}</td>'
                            f'<td>{esc(fmt_m(None if pd.isna(x["median_months"]) else x["median_months"]))}'
                            f'<span class="sub">India {esc(fmt_m(nat.loc[ct, "median_months"]))}</span></td>'
                            f'<td>{pct(None if pd.isna(x["pending_after_1y"]) else 1 - x["pending_after_1y"])}</td>'
                            f'<td>{pct(x["pending_after_3y"])}<span class="sub">India {pct(nat.loc[ct, "pending_after_3y"])}</span></td>'
                            f'<td>{pct(x["pending_after_5y"])}</td><td>{rng}</td></tr>')
            table = ('<div class="tbl"><table><thead><tr><th>Type of case</th><th>Cases</th><th>Median time</th>'
                     '<th>Decided within 1 yr</th><th>Pending at 3 yrs</th><th>Pending at 5 yrs</th><th>Rank in India</th>'
                     f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
            chart = curve_svg(curve(p["curve_q"]), curve(nat_p["curve_q"]), d, label)
            # locator map
            mine = set(codes.get((s, d), []))
            loc = ""
            if mine and sshapes:
                fp, w_, h_ = fit(sshapes, 200, eps=0.8)
                loc = ('<figure class="loc" style="margin:0"><svg viewBox="0 0 ' + f'{w_} {h_}" role="img"><title>Location of '
                       f'{esc(d)} in {esc(s)}</title>' + "".join(f'<path class="{"me" if c in mine else "o"}" d="{pth}"/>'
                                                                   for c, pth in fp.items()) +
                       f'</svg><figcaption>{esc(d)} on the 2011 Census map'
                       + (" (shown with its 2011 parent district)" if any(rel == "carved_from" for c in mine
                          for st_, dd_, rel in M["units"][c] if dd_ == d) else "") + '</figcaption></figure>')
            # caveats specific to this district
            notes = []
            y = D["dig"].get(did)
            if y is not None and pd.notna(y) and y > 2010:
                notes.append(f"Records here are complete only from <b>{int(y)}</b>. Cases filed earlier are counted from then, "
                             "a correction for the uneven arrival of eCourts.")
            if is_special:
                notes.append("This is a <b>special court or court complex</b>, not a whole district, so it is not comparable "
                             "with district courts.")
            if pd.notna(p.get("rank")) and int(p["rank_hi"]) - int(p["rank_lo"]) > 50:
                notes.append(f"This district has few cases of this type, so its rank is uncertain "
                             f"(anywhere from {int(p['rank_lo'])} to {int(p['rank_hi'])}).")
            notes += state_notes
            notes.append('"Decided" means any final disposal: judgment, dismissal, withdrawal, settlement or transfer. '
                         "A slow district is not necessarily a badly run one: case mix, vacancies and local practice differ.")
            drv = D["drv"].loc[did] if did in D["drv"].index else None
            ctx = ""
            if drv is not None and pd.notna(drv["judge_tenure_months"]):
                ctx = (f'<p class="prose">Judges here stayed a median of <b>{drv["judge_tenure_months"]:.0f} months</b> in one '
                       f'courtroom (all districts: {D["drv"]["judge_tenure_months"].median():.0f}), and about '
                       f'<b>{drv["courts"]:.0f}</b> courtrooms were active in an average month of 2014–2017. Across districts, '
                       f'longer postings go with faster civil cases, but this is a correlation, not a cause '
                       f'(<a href="../../index.html#districts">details</a>).</p>')
            others = [x for x in dists if x != d]
            others_html = ", ".join(f'<a href="{dist_slug[(s, x)]}.html">{esc(x)}</a>' for x in others)
            q = f"#s={esc(s)}&amp;d={esc(d)}&amp;t={prim}".replace(" ", "%20")
            body = f"""<p class="crumbs"><a href="../index.html">All districts</a> › <a href="index.html">{esc(s)}</a> › {esc(d)}</p>
<header class="hero" style="padding:18px 0 4px"><div class="dhead"><div>
  <div class="kicker">{"Special court" if is_special else "District court"} · {esc(s)} · cases filed 2010–2018</div>
  <h1>How long do court cases take in {esc(d)}?</h1>
  <p class="lede">{head}{rank_txt}</p>
</div>{loc}</div></header>
<div class="tiles">{tiles_html}</div>
<div class="actions"><a class="btn" href="../../calculator.html{q}">Open the interactive calculator</a><a class="btn ghost" href="../../map.html#t={prim}&amp;m=p3&amp;s={esc(M['names'][sorted(mine)[0]][1]).replace(' ', '%20') if mine else ''}">See it on the map</a><a class="btn ghost" href="../../districts.html#t={prim}&amp;s={esc(s).replace(' ', '%20')}">Compare districts in {esc(s)}</a></div>
<section style="margin-top:36px;padding-top:28px">
<h2>{esc(label)}: share still pending, by years since filing</h2>
<div class="card">{chart}<div class="caption">Kaplan–Meier estimate. Cases still pending when the data was collected count as undecided up to that date. Dashed: all of India, same type of case.</div></div>
<h2 style="margin-top:28px">Every type of case</h2>
<div class="card">{table}<div class="caption">Rank: 1 = smallest share still pending after 3 years among all units in India with at least 200 cases of that type; the small range is where the rank lies 90% of the time once sampling noise is allowed for. Types with fewer than 200 cases are left out.</div></div>
{ctx}
<h2 style="margin-top:28px">Read this with care</h2>
<ul class="prose">{"".join(f"<li>{x}</li>" for x in notes)}</ul>
<p class="caption">Method and every check behind these numbers: <a href="../../paper.html">working paper</a> · <a href="../../index.html#audit">audit</a> · <a href="https://github.com/ArnavGoel98/CourtSight/blob/main/results/time_to_decision/by_district_case_type.csv">raw table (CSV)</a></p>
{f'<h2 style="margin-top:28px">Other districts in {esc(s)}</h2><p class="prose">{others_html}</p>' if others else ''}
</section>"""
            api = {"state": s, "district": d, "special_court": is_special,
                   "records_complete_from": None if y is None or pd.isna(y) else int(y),
                   "state_coverage_2015_18": None if cov_row is None or cov_row["group"] == "too small to classify"
                   else round(float(cov_row["coverage_2015_18"]), 3),
                   "state_pending_not_listed_12m": None if act_v is None else round(act_v, 3),
                   "page": f"{B.BASE}d/{state_slug[s]}/{dist_slug[(s, d)]}.html",
                   "case_types": {ct: {"label": B.LABELS[ct], "cases": int(r.loc[ct, "cases"]),
                                       **{k: num(r.loc[ct, k]) for k in ("median_months", "p90_months", "pending_after_1y",
                                                                          "pending_after_3y", "pending_after_5y",
                                                                          "pending_after_3y_lo", "pending_after_3y_hi")},
                                       **{k: num(r.loc[ct, k], 0) for k in ("rank", "rank_lo", "rank_hi", "of")},
                                       "india": {k: num(nat.loc[ct, k]) for k in ("median_months", "pending_after_1y",
                                                                                   "pending_after_3y", "pending_after_5y")},
                                       "curve_quarterly": curve(r.loc[ct, "curve_q"]),
                                       "india_curve_quarterly": curve(nat.loc[ct, "curve_q"])}
                                  for ct in ORDER if ct in r.index},
                   "source": "Development Data Lab eCourts records, cases filed 2010-2018 (CC BY-NC-SA 4.0); CourtSight",
                   "notes": "Kaplan-Meier estimates; curves give the share still pending every 3 months from filing."}
            (API / state_slug[s]).mkdir(exist_ok=True)
            (API / state_slug[s] / f"{dist_slug[(s, d)]}.json").write_text(json.dumps(api, separators=(",", ":"), ensure_ascii=False))
            api_index.append({"state": s, "district": d, "api": f"{state_slug[s]}/{dist_slug[(s, d)]}.json",
                              "page": api["page"]})
            desc = (f"How long court cases took in {d}, {s}: " + re.sub("<[^>]+>", "", head + rank_txt))[:300]
            rel = f"d/{state_slug[s]}/{dist_slug[(s, d)]}.html"
            (OUT / state_slug[s] / f"{dist_slug[(s, d)]}.html").write_text(
                page(f"Court case times in {d}, {s} | CourtSight", desc, rel, 2, body))
            sitemap.append(rel)
            n += 1

        # withheld units: a page that says why there is no answer
        held_here = sorted(k for k, v in D["held"].items() if v["state"] == s and v["filed_2010_2017"] >= 1000)
        for d in held_here:
            v = D["held"][d]
            k = slug(d)
            while k in {dist_slug.get((s, x)) for x in dists}:
                k += "-2"
            dist_slug[(s, d)] = k
            mine = {c for c, ws in M.get("withheld", {}).items() if d in ws}
            loc = ""
            if mine and sshapes:
                fp, w_, h_ = fit(sshapes, 200, eps=0.8)
                loc = (f'<figure class="loc" style="margin:0"><svg viewBox="0 0 {w_} {h_}" role="img"><title>Location of {esc(d)} '
                       f'in {esc(s)}</title>' + "".join(f'<path class="{"me" if c in mine else "o"}" d="{pth}"/>' for c, pth in fp.items())
                       + f'</svg><figcaption>{esc(d)} on the 2011 Census map</figcaption></figure>')
            body = f"""<p class="crumbs"><a href="../index.html">All districts</a> › <a href="index.html">{esc(s)}</a> › {esc(d)}</p>
<header class="hero" style="padding:18px 0 4px"><div class="dhead"><div>
  <div class="kicker">District court · {esc(s)} · results withheld</div>
  <h1>How long do court cases take in {esc(d)}?</h1>
  <p class="lede">The eCourts records cannot answer this. They hold <b>{inr(v['filed_2010_2017'])}</b> cases filed here in 2010–2017 but only <b>{inr(v['decided_2010_2017'])}</b> decisions recorded in those years, against <b>{inr(v['decided_2018'])}</b> in 2018 alone. Decisions before 2018 were evidently not entered, so almost every case would look stuck.</p>
</div>{loc}</div></header>
<div class="card prose" style="max-width:none">
<p>Published naively, {esc(d)} would rank among the slowest courts in India. An earlier version of this project did exactly that. A check now requires a district's records to show decisions as well as filings before its figures are used. {esc(d)} never passes, so its figures are withheld rather than shown wrong.</p>
<p>For a sense of the region, see the <a href="index.html">other districts in {esc(s)}</a>; note that their figures may not apply here.</p></div>
<div class="actions"><a class="btn ghost" href="index.html">Districts in {esc(s)}</a><a class="btn ghost" href="../../index.html#audit">How the data was checked</a></div>"""
            rel = f"d/{state_slug[s]}/{k}.html"
            (sdir / f"{k}.html").write_text(page(f"Court case times in {d}, {s}: results withheld | CourtSight",
                                                 f"Why CourtSight withholds court delay figures for {d}, {s}: the eCourts records hold "
                                                 f"almost no decisions there before 2018.", rel, 2, body))
            sitemap.append(rel)

        # state page
        civ_s = civ.loc[civ.index.get_level_values(0) == s].reset_index().dropna(subset=["pending_after_3y"])
        civ_s = civ_s.sort_values(["pending_after_3y", "district_name"])
        smap = ""
        if sshapes:
            fp, w_, h_ = fit(sshapes, 600, eps=0.7)
            parts = []
            for c, pth in fp.items():
                us = [(s2, d2) for s2, d2, _ in M["units"].get(c, []) if s2 == s and (s2, d2) in civ.index]
                vals = [(civ.loc[k, "pending_after_3y"], civ.loc[k, "cases"]) for k in us if pd.notna(civ.loc[k, "pending_after_3y"])]
                if not vals:
                    parts.append(f'<path style="fill:var(--nodata)" d="{pth}"/>')
                    continue
                v = sum(a * b for a, b in vals) / sum(b for _, b in vals)
                nm = " + ".join(d2 for _, d2 in us)
                parts.append(f'<a href="{dist_slug[us[0]]}.html"><path style="fill:var({RAMP[bin_of(v)]})" d="{pth}">'
                             f'<title>{esc(nm)}: {pct(v)} of civil suits still pending after 3 years</title></path></a>')
            smap = (f'<figure class="card smap"><svg viewBox="0 0 {w_} {h_}" role="group" aria-label="Map of {esc(s)}">'
                    + "".join(parts) + '</svg><figcaption class="caption">Civil suits still pending after 3 years, shaded by '
                    'fifths of all districts in India (darker = more pending; in dark mode, brighter). Click a district for its page.</figcaption></figure>')
        trows = "".join(
            f'<tr><td class="l"><a href="{dist_slug[(s, x.district_name)]}.html">{esc(x.district_name)}</a>'
            + (' <span class="badge">court</span>' if x.district_name in D["special"] else "")
            + f'</td><td>{inr(x.cases)}</td><td>{esc(fmt_m(None if pd.isna(x.median_months) else x.median_months))}</td>'
              f'<td>{pct(x.pending_after_3y)}</td><td>{int(x.rank) if pd.notna(x.rank) else "–"}'
              f'<span class="sub">{int(x.rank_lo) if pd.notna(x.rank) else ""}–{int(x.rank_hi) if pd.notna(x.rank) else ""}</span></td></tr>'
            for x in civ_s.itertuples())
        rest = [x for x in dists if x not in set(civ_s["district_name"])]
        lo_d, hi_d = (civ_s.iloc[0], civ_s.iloc[-1]) if len(civ_s) else (None, None)
        lede = (f"Across {len(dists)} court units, the share of civil suits still pending three years after filing ranges "
                f"from <b>{pct(lo_d['pending_after_3y'])}</b> in {esc(lo_d['district_name'])} to "
                f"<b>{pct(hi_d['pending_after_3y'])}</b> in {esc(hi_d['district_name'])} (all of India: "
                f"{pct(nat_civ['pending_after_3y'])})." if len(civ_s) > 1 else
                f"{len(dists)} court unit{'s' if len(dists) != 1 else ''} with results.")
        body = f"""<p class="crumbs"><a href="../index.html">All districts</a> › {esc(s)}</p>
<header class="hero" style="padding:18px 0 4px"><div class="kicker">State · district courts · cases filed 2010–2018</div>
<h1>How long do court cases take in {esc(s)}?</h1><p class="lede">{lede}</p></header>
{smap}
<div class="card tbl"><table><thead><tr><th>District</th><th>Civil suits</th><th>Median time</th><th>Pending at 3 yrs</th><th>Rank in India (90% range)</th></tr></thead><tbody>{trows}</tbody></table>
{f'<p class="caption">Also: ' + ", ".join(f'<a href="{dist_slug[(s, x)]}.html">{esc(x)}</a>' for x in rest) + " (too few civil suits to rank).</p>" if rest else ""}</div>
<h2 style="margin-top:28px">Read this with care</h2>
<ul class="prose">{"".join(f"<li>{x}</li>" for x in state_notes)}{f'<li>Withheld because their records hold almost no decisions before 2018: ' + ", ".join(f'<a href="{dist_slug[(s, x)]}.html">{esc(x)}</a>' for x in held_here) + ".</li>" if held_here else ""}<li>Ranks among all districts in India; districts whose ranges overlap are not clearly different. Descriptive only.</li></ul>"""
        rel = f"d/{state_slug[s]}/index.html"
        (sdir / "index.html").write_text(page(f"Court case times in {s}'s district courts | CourtSight",
                                              re.sub("<[^>]+>", "", lede)[:300], rel, 2, body))
        sitemap.append(rel)

    # index of everything
    blocks = "".join(f'<div><h2 class="h3"><a href="{state_slug[s]}/index.html">{esc(s)}</a></h2>'
                     + "".join(f'<a href="{state_slug[s]}/{dist_slug[(s, d)]}.html">{esc(d)}</a><br>'
                               for d in sorted({*g["district_name"].unique(), *(x for x, v in D["held"].items()
                                                if v["state"] == s and (s, x) in dist_slug)}, key=str.lower)) + "</div>"
                     for s, g in sorted(states.items()))
    body = f"""<header class="hero" style="padding:36px 0 8px"><div class="kicker">{n} pages · one per district court</div>
<h1>Every district court in India</h1><p class="lede">Pick a district for how long its cases took, by type of case, with the limits that apply to it. Or use the <a href="../calculator.html">calculator</a>, the <a href="../districts.html">rankings</a> or the <a href="../map.html">map</a>.</p></header>
<div class="cols">{blocks}</div>"""
    (OUT / "index.html").write_text(page("Every district court in India | CourtSight",
                                         "How long court cases took in every district court in India, one page per district, "
                                         "from 80.9 million case records filed 2010–2018.", "d/index.html", 1, body))
    sitemap.append("d/index.html")
    (API / "index.json").write_text(json.dumps({"version": 1, "base": f"{B.BASE}api/v1/", "districts": api_index},
                                               separators=(",", ":"), ensure_ascii=False))
    return n
