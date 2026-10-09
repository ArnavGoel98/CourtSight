// Minimal SVG charts: line/step chart with bands and a crosshair tooltip, and a diverging horizontal bar chart.
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function tooltip(host) {
  const t = document.createElement("div");
  t.className = "tip";
  host.appendChild(t);
  return t;
}
function placeTip(tip, host, px, py) {
  const w = host.clientWidth, tw = tip.offsetWidth;
  tip.style.left = Math.min(Math.max(px + 14, 0), w - tw) + "px";
  tip.style.top = Math.max(py - 20, 0) + "px";
}

// opts: {x:[...], series:[{name, values, color, dash, width, band:{lo,hi,color}, step}], yfmt, xfmt, tipx,
//        ymin, ymax, height, marks:[{x, label}], xticks, yticks}
function lineChart(host, opts) {
  host.innerHTML = "";
  host.classList.add("chart");
  const W = Math.max(host.clientWidth, 300), H = opts.height || Math.round(Math.min(380, Math.max(240, W * 0.5)));
  const m = {t: 16, r: opts.right ?? 24, b: 30, l: opts.left ?? 52};
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || ""}, host);
  const xs = opts.x, n = xs.length;
  const all = opts.series.flatMap(s => [...s.values, ...(s.band ? [...s.band.lo, ...s.band.hi] : [])]).filter(v => v != null);
  const ymin = opts.ymin ?? Math.min(...all), ymax = opts.ymax ?? Math.max(...all);
  const x0 = opts.xmin ?? xs[0], x1 = opts.xmax ?? xs[n - 1];
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - ymin) / (ymax - ymin) * (H - m.t - m.b);
  const g = el("g", {class: "grid"}, svg), ax = el("g", {class: "axis"}, svg);
  for (const t of opts.yticks || niceTicks(ymin, ymax, 5)) {
    el("line", {x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t)}, g);
    el("text", {x: m.l - 8, y: Y(t) + 4, "text-anchor": "end"}, ax).textContent = (opts.yfmt || String)(t);
  }
  el("line", {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b}, ax);
  for (const t of opts.xticks || xs) {
    el("text", {x: X(t), y: H - m.b + 18, "text-anchor": "middle"}, ax).textContent = (opts.xfmt || String)(t);
  }
  for (const mk of opts.marks || []) {
    el("line", {x1: X(mk.x), x2: X(mk.x), y1: m.t, y2: H - m.b, stroke: "var(--ink-3)", "stroke-dasharray": "3 4"}, svg);
    el("text", {x: X(mk.x) + 6, y: m.t + 10, class: "lab", "font-size": 11}, svg).textContent = mk.label;
  }
  const path = (vals, step) => {
    let d = "", pen = false, prev = null;
    vals.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      if (!pen) d += `M${X(xs[i])},${Y(v)}`;
      else if (step) d += `H${X(xs[i])}V${Y(v)}`;
      else d += `L${X(xs[i])},${Y(v)}`;
      pen = true; prev = v;
    });
    return d;
  };
  for (const s of opts.series) {
    if (!s.band) continue;
    const idx = s.band.lo.map((v, i) => v == null ? null : i).filter(i => i != null);
    if (!idx.length) continue;
    const d = idx.map((i, k) => `${k ? "L" : "M"}${X(xs[i])},${Y(s.band.hi[i])}`).join("") +
      idx.slice().reverse().map(i => `L${X(xs[i])},${Y(s.band.lo[i])}`).join("") + "Z";
    el("path", {d, fill: s.band.color || "var(--band)", stroke: "none"}, svg);
  }
  for (const s of opts.series) {
    el("path", {d: path(s.values, s.step), fill: "none", stroke: s.color, "stroke-width": s.width || 2,
      "stroke-dasharray": s.dash || "", "stroke-linejoin": "round", "stroke-linecap": "round", opacity: s.opacity ?? 1}, svg);
    if (s.label) {
      let i = s.values.length - 1;
      while (i > 0 && s.values[i] == null) i--;
      el("text", {x: X(xs[i]) + 6, y: Y(s.values[i]) + 4 + (s.labelDy || 0), class: "lab", "font-size": 12}, svg).textContent = s.label;
    }
  }
  // crosshair + tooltip
  const tip = tooltip(host);
  const cross = el("line", {y1: m.t, y2: H - m.b, stroke: "var(--ink-3)", opacity: 0}, svg);
  const dots = opts.series.filter(s => !s.noTip).map(s => el("circle", {r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2, opacity: 0}, svg));
  const hit = el("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"}, svg);
  const move = ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
    let i = 0, best = Infinity;
    xs.forEach((v, k) => { const d = Math.abs(X(v) - px); if (d < best) { best = d; i = k; } });
    cross.setAttribute("x1", X(xs[i])); cross.setAttribute("x2", X(xs[i])); cross.setAttribute("opacity", .6);
    const rows = [];
    opts.series.filter(s => !s.noTip).forEach((s, k) => {
      const v = s.values[i];
      dots[k].setAttribute("opacity", v == null ? 0 : 1);
      if (v != null) { dots[k].setAttribute("cx", X(xs[i])); dots[k].setAttribute("cy", Y(v)); }
      if (v != null) rows.push(`<div><span style="color:${s.color}">●</span> <span class="k">${s.name}</span> <b>${(opts.tipfmt || opts.yfmt || String)(v)}</b>${s.band && s.band.lo[i] != null ? ` <span class="k">(${(opts.tipfmt || opts.yfmt)(s.band.lo[i])}–${(opts.tipfmt || opts.yfmt)(s.band.hi[i])})</span>` : ""}</div>`);
    });
    tip.innerHTML = `<div class="k">${(opts.tipx || opts.xfmt || String)(xs[i])}</div>` + rows.join("");
    tip.style.opacity = rows.length ? 1 : 0;
    placeTip(tip, host, X(xs[i]) * r.width / W, (ev.clientY - r.top));
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerleave", () => { tip.style.opacity = 0; cross.setAttribute("opacity", 0); dots.forEach(d => d.setAttribute("opacity", 0)); });
  return {X, Y, svg};
}

// rows: [{name, value, note}] -> horizontal bars around zero; colours: positive = --neg (worse), negative = --pos
function divergingBars(host, rows, opts = {}) {
  host.innerHTML = "";
  host.classList.add("chart");
  const W = Math.max(host.clientWidth, 300), rh = 22, m = {t: 8, r: 56, b: 26, l: Math.min(150, W * 0.34)};
  const H = m.t + m.b + rows.length * rh;
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || ""}, host);
  const vals = rows.map(r => r.value), lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  const X = v => m.l + (v - lo) / (hi - lo) * (W - m.l - m.r);
  const g = el("g", {class: "grid"}, svg), ax = el("g", {class: "axis"}, svg);
  for (const t of niceTicks(lo, hi, 5)) {
    el("line", {x1: X(t), x2: X(t), y1: m.t, y2: H - m.b}, g);
    el("text", {x: X(t), y: H - 8, "text-anchor": "middle"}, ax).textContent = (opts.fmt || String)(t);
  }
  const tip = tooltip(host);
  rows.forEach((r, i) => {
    const y = m.t + i * rh, x0 = X(0), x1 = X(r.value);
    el("text", {x: m.l - 8, y: y + rh / 2 + 4, "text-anchor": "end"}, svg).textContent = r.name;
    const w = Math.max(Math.abs(x1 - x0), 1.5);
    const bar = el("rect", {x: Math.min(x0, x1), y: y + 4, width: w, height: rh - 8, rx: 3,
      fill: r.value > 0 ? "var(--neg)" : "var(--pos)"}, svg);
    el("text", {x: (r.value > 0 ? x1 + 6 : x0 + 6), y: y + rh / 2 + 4, "font-size": 11}, svg).textContent = (opts.fmt || String)(r.value);
    const hitr = el("rect", {x: 0, y, width: W, height: rh, fill: "transparent"}, svg);
    hitr.addEventListener("pointermove", ev => {
      const b = svg.getBoundingClientRect();
      tip.innerHTML = `<b>${r.name}</b><div>${r.note || ""}</div>`;
      tip.style.opacity = 1;
      bar.setAttribute("opacity", .8);
      placeTip(tip, host, ev.clientX - b.left, ev.clientY - b.top);
    });
    hitr.addEventListener("pointerleave", () => { tip.style.opacity = 0; bar.setAttribute("opacity", 1); });
  });
  el("line", {x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, stroke: "var(--ink-3)"}, svg);
}
const fmtL = v => (v / 1e5).toLocaleString("en-IN", {maximumFractionDigits: 0}) + " L";
const fmtCr = v => (v / 1e7).toFixed(2) + " cr";
const fmtPct = v => (v > 0 ? "+" : "") + Math.round(v * 100) + "%";
const fmtPct1 = v => (v > 0 ? "+" : "") + (v * 100).toFixed(1) + "%";
const onResize = fn => { let w = 0; new ResizeObserver(es => { const nw = Math.round(es[0].contentRect.width); if (nw !== w) { w = nw; fn(); } }).observe(document.body); };
