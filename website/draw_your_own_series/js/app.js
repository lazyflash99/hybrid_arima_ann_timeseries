// Page logic: drawing -> series -> linear stage -> network training loop -> charts, metric cards and verdict.

import { acf, createHybrid, verdict } from "./draw_models.js";
import { addNoise, createCanvas, toSeries } from "./canvas.js";
import { RESEARCH_URL } from "./links.js";

// the research site's colours: ARIMA (here the linear stage) and Zhang's hybrid (here the hybrid and its network part)
const COLORS = { linear: "#4c8795", hybrid: "#17637a" };
const FONT = '"Segoe UI", system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif';
const PLOT_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };

const $ = (id) => document.getElementById(id);
const state = { noiseSeed: 1, initSeed: 1, hybrid: null, series: null, timer: null, running: false, debounce: null, tick: 0 };
// read by the browser test
window.__playground = { done: false, epoch: 0, error: null };

// ---------- helpers ----------

function fmt(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return Number(v.toPrecision(4)).toString();
}
const pct = (v) => (v === null || !Number.isFinite(v) ? "–" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}%`);

function theme() {
  const css = getComputedStyle(document.documentElement);
  const v = (n) => css.getPropertyValue(n).trim();
  return { ink: v("--ink"), ink2: v("--ink-2"), line: v("--line"), held: v("--held"), bad: v("--bad") };
}

function axis(t, title, extra = {}) {
  return { title: { text: title, font: { size: 15 } }, gridcolor: t.line, zerolinecolor: t.line, linecolor: t.line,
           tickfont: { size: 14 }, automargin: true, ...extra };
}

function baseLayout(t, height, extra) {
  return { height, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
           font: { family: FONT, size: 14, color: t.ink }, hoverlabel: { font: { size: 14, family: FONT } },
           margin: { l: 80, r: 20, t: 50, b: 60 }, ...extra };
}

function value(id) { return Number($(id).value); }
function settings() {
  return { n: value("n-points"), noise: value("noise") / 100, p: value("ar-p"), diff: $("diff").checked,
           hidden: value("hidden"), lags: value("lags"), epochs: value("epochs") };
}

// ---------- pipeline ----------

function stopTraining() {
  clearTimeout(state.timer);
  state.timer = null;
  state.running = false;
  updateButtons();
}

function rebuild() {
  stopTraining();
  const s = settings();
  const { values, warnings } = toSeries(canvas.strokes, s.n);
  $("warnings").textContent = warnings.join(" ");
  window.__playground = { done: false, epoch: 0, error: null };
  if (!values) {
    state.hybrid = null;
    ["main-chart", "loss-chart", "acf-chart"].forEach((id) => Plotly.purge($(id)));
    $("cards").innerHTML = "";
    $("verdict").textContent = "";
    $("verdict").className = "badge big";
    $("verdict-text").textContent = "";
    $("ar-text").textContent = "";
    $("status").textContent = "Draw a series, or pick a preset, to fit the models.";
    window.__playground.done = true;
    return;
  }
  state.series = addNoise(values, s.noise, state.noiseSeed);
  try {
    state.hybrid = createHybrid(state.series, { p: s.p, diff: s.diff, hidden: s.hidden, lags: s.lags, seed: state.initSeed });
  } catch (err) {
    state.hybrid = null;
    $("status").textContent = `Cannot fit: ${err.message}.`;
    window.__playground = { done: true, epoch: 0, error: err.message };
    return;
  }
  describeAR(state.hybrid.ar, s);
  $("seed-text").textContent = `network seed ${state.initSeed}`;
  state.running = true;
  state.tick = 0;
  updateButtons();
  step();
}

function step() {
  const s = settings();
  const h = state.hybrid;
  const perTick = Math.max(1, Math.round(s.epochs / 60)); // about one second of animation at any epoch count
  h.train(Math.min(perTick, s.epochs - h.epoch));
  const done = h.epoch >= s.epochs;
  state.tick++;
  if (done) state.running = false;
  drawLoss();
  if (done || state.tick % 3 === 1) render();
  const last = h.history[h.history.length - 1];
  $("status").textContent = done
    ? `Finished ${h.epoch} epochs. Final network loss on the training residuals: ${fmt(last.train)} (standardised units).`
    : `Training the network: epoch ${h.epoch} of ${s.epochs}, training loss ${fmt(last.train)}.`;
  window.__playground.epoch = h.epoch;
  if (done) {
    state.running = false;
    window.__playground.done = true;
    updateButtons();
    return;
  }
  state.timer = setTimeout(step, 16);
}

function describeAR(ar, s) {
  const on = ar.diff ? "the first difference of y" : "y";
  const phi = ar.phi.map((v, i) => `φ${i + 1} = ${fmt(v)}`).join(", ");
  $("ar-text").textContent = `AR(${ar.p}) on ${on}, fitted by least squares on the first ${ar.nTrain} of ${s.n} points: ` +
    `intercept ${fmt(ar.intercept)}, ${phi}.`;
}

function updateButtons() {
  $("stop").disabled = !state.running;
  $("retrain").disabled = !state.hybrid && !canvas.strokes.length;
  $("reinit").disabled = !canvas.strokes.length;
}

// ---------- charts and cards ----------

function render() {
  const h = state.hybrid;
  if (!h) return;
  const r = h.result();
  const t = theme();
  const x = r.Lhat.map((_, i) => i);
  const show = { series: $("show-series").checked, linear: $("show-linear").checked, hybrid: $("show-hybrid").checked };
  const traces = [
    { x, y: state.series, name: "your series", line: { color: t.ink, width: 2 }, visible: show.series },
    { x, y: r.Lhat, name: "linear forecast L̂", line: { color: COLORS.linear, width: 2, dash: "dash" }, visible: show.linear },
    { x, y: r.yhat, name: "hybrid forecast L̂ + N̂", line: { color: COLORS.hybrid, width: 2.2 }, visible: show.hybrid },
    { x, y: r.e, name: "residual e", type: "bar", marker: { color: COLORS.linear }, yaxis: "y2" },
    { x, y: r.Nhat, name: "network prediction N̂", mode: "lines+markers", line: { color: COLORS.hybrid, width: 2 },
      marker: { size: 4 }, yaxis: "y2" },
    { x, y: r.remaining, name: "remaining error y − ŷ", type: "bar", marker: { color: COLORS.hybrid }, yaxis: "y3", showlegend: false },
  ];
  const panelTitle = (text, y) => ({ text, xref: "paper", yref: "paper", x: 0, y, xanchor: "left", yanchor: "bottom",
                                     showarrow: false, font: { size: 15, color: t.ink } });
  const lay = baseLayout(t, 760, {
    xaxis: axis(t, "time step", { anchor: "y3", range: [-0.5, r.n - 0.5] }),
    yaxis: axis(t, "value", { domain: [0.58, 0.96] }),
    yaxis2: axis(t, "residual", { domain: [0.3, 0.5] }),
    yaxis3: axis(t, "remaining error", { domain: [0, 0.22] }),
    bargap: 0.25,
    legend: { orientation: "h", x: 0, y: -0.09, yanchor: "top", font: { size: 14 }, itemclick: false, itemdoubleclick: false },
    margin: { l: 80, r: 20, t: 30, b: 90 },
    shapes: [{ type: "rect", xref: "x", yref: "paper", x0: r.nTrain - 0.5, x1: r.n - 0.5, y0: 0, y1: 1,
               fillcolor: t.held, line: { width: 0 }, layer: "below" }],
    annotations: [
      panelTitle("(1) Series and forecasts", 0.965),
      panelTitle("(2) Residual e = y − L̂ and the network's prediction N̂", 0.505),
      panelTitle("(3) Error left after both stages, y − ŷ", 0.225),
      { text: `held out (last ${r.n - r.nTrain} points)`, xref: "x", yref: "paper", x: r.n - 0.5, y: 1.0, xanchor: "right",
        yanchor: "bottom", showarrow: false, font: { size: 14, color: t.ink2 } },
      { text: `training (first ${r.nTrain} points)`, xref: "x", yref: "paper", x: r.nTrain - 0.5, y: 1.0, xanchor: "right",
        yanchor: "bottom", showarrow: false, font: { size: 14, color: t.ink2 } },
    ],
  });
  Plotly.react($("main-chart"), traces, lay, PLOT_CONFIG);
  drawCards(r);
  drawACF(r);
}

function drawLoss() {
  const h = state.hybrid;
  const t = theme();
  const ep = h.history.map((p) => p.epoch);
  const traces = [
    { x: ep, y: h.history.map((p) => p.train), name: "training residuals", line: { color: COLORS.hybrid, width: 2 } },
    { x: ep, y: h.history.map((p) => p.held), name: "held-out residuals (not used for fitting)", line: { color: t.bad, width: 2, dash: "dot" } },
  ];
  Plotly.react($("loss-chart"), traces, baseLayout(t, 340, {
    title: { text: "Network loss while training", font: { size: 16 }, x: 0, xanchor: "left" },
    xaxis: axis(t, "epoch", { range: [0, Math.max(settings().epochs, 1)] }),
    yaxis: axis(t, "MSE (standardised units)", { rangemode: "tozero" }),
    legend: { orientation: "h", x: 0, y: -0.3, yanchor: "top", font: { size: 14 } },
    margin: { l: 80, r: 20, t: 50, b: 100 },
  }), PLOT_CONFIG);
}

function drawACF(r) {
  const t = theme();
  const tr = (arr) => arr.slice(0, r.nTrain);
  const linear = acf(tr(r.e), Math.min(15, Math.floor(r.nTrain / 4)));
  const left = acf(tr(r.remaining), linear.values.length);
  const lags = linear.values.map((_, i) => i + 1);
  const band = 1.96 / Math.sqrt(left.n);
  Plotly.react($("acf-chart"), [
    { x: lags, y: linear.values, type: "bar", name: "residual of the linear model", marker: { color: COLORS.linear } },
    { x: lags, y: left.values, type: "bar", name: "error left by the hybrid", marker: { color: COLORS.hybrid } },
    { x: [0.5, lags.length + 0.5], y: [band, band], mode: "lines", name: `±1.96/√n`, line: { color: t.ink2, dash: "dash", width: 1.2 } },
    { x: [0.5, lags.length + 0.5], y: [-band, -band], mode: "lines", showlegend: false, line: { color: t.ink2, dash: "dash", width: 1.2 }, hoverinfo: "skip" },
  ], baseLayout(t, 340, {
    title: { text: "Autocorrelation on the training part", font: { size: 16 }, x: 0, xanchor: "left" },
    xaxis: axis(t, "lag", { dtick: 1 }),
    yaxis: axis(t, "autocorrelation"),
    barmode: "group",
    legend: { orientation: "h", x: 0, y: -0.3, yanchor: "top", font: { size: 14 } },
    margin: { l: 80, r: 20, t: 50, b: 100 },
  }), PLOT_CONFIG);
}

function drawCards(r) {
  const m = r.metrics;
  const card = (k, v, s, cls = "") => `<div class="card"><div class="k">${k}</div><div class="v ${cls}">${v}</div><div class="s">${s}</div></div>`;
  const cls = (c) => (c === null ? "" : c < 0 ? "good" : c > 0 ? "bad" : "");
  const row = (name, part) => [
    card(`Linear model MSE, ${name}`, fmt(part.linear), `over ${part.n} points`),
    card(`Hybrid MSE, ${name}`, fmt(part.hybrid), `over the same ${part.n} points`),
    card(`Change from adding the network, ${name}`, pct(part.change), "negative means the hybrid is better", cls(part.change)),
  ].join("");
  $("cards").innerHTML = row("training part", m.train) + row("held-out part", m.held);

  const v = verdict(m);
  const badge = $("verdict");
  const done = !state.running;
  badge.className = `badge big ${v === "helped" ? "good" : v === "overfit" ? "bad" : ""}`;
  badge.textContent = { helped: "Network helped", overfit: "Network overfit", none: "No clear gain" }[v];
  let text;
  if (m.train.linear !== null && m.train.linear < 1e-9) {
    text = "The linear model already forecasts the training part almost exactly, so there is nothing left for the network to learn.";
  } else if (v === "helped") {
    text = `On the held-out part the hybrid's MSE is ${Math.abs(m.held.change).toFixed(1)}% lower than the linear model's, so the pattern the network found in the training residuals carried over to new data.`;
  } else if (v === "overfit") {
    text = `The network lowered the training MSE by ${Math.abs(m.train.change).toFixed(1)}% but raised the held-out MSE by ${m.held.change.toFixed(1)}%: it fitted patterns in the training residuals that did not repeat.`;
  } else {
    text = `The held-out MSE changed by ${pct(m.held.change)} (a drop of 5% or more counts as help), so the network added little here.`;
  }
  $("verdict-text").textContent = (done ? "" : "(still training) ") + text;
}

// ---------- controls ----------

function schedule() {
  clearTimeout(state.debounce);
  state.debounce = setTimeout(rebuild, 150);
}

function bindSlider(id, fmtOut = (v) => v) {
  const el = $(id), out = $(`${id}-out`);
  el.addEventListener("input", () => { out.textContent = fmtOut(el.value); schedule(); });
}

const canvas = createCanvas($("canvas"), () => {
  document.querySelectorAll(".preset").forEach((b) => b.classList.remove("active"));
  rebuild();
});

function setTheme(name) {
  document.documentElement.dataset.theme = name;
  try { localStorage.setItem("theme", name); } catch (e) { /* storage blocked */ }
  $("theme-toggle").textContent = name === "dark" ? "Light mode" : "Dark mode";
  canvas.redraw();
  if (state.hybrid) { drawLoss(); render(); }
}

function init() {
  $("research-link").href = RESEARCH_URL;
  $("research-footer-link").href = RESEARCH_URL;
  document.querySelectorAll(".swatch").forEach((s) => {
    s.style.background = s.dataset.color === "series" ? "var(--ink)" : COLORS[s.dataset.color];
  });

  document.querySelectorAll(".preset").forEach((b) => b.addEventListener("click", () => {
    canvas.preset(b.dataset.preset);
    document.querySelectorAll(".preset").forEach((o) => o.classList.toggle("active", o === b));
  }));
  $("undo").addEventListener("click", () => canvas.undo());
  $("clear").addEventListener("click", () => canvas.clear());
  $("reroll").addEventListener("click", () => { state.noiseSeed++; rebuild(); });
  bindSlider("n-points");
  bindSlider("noise", (v) => `${v}%`);
  bindSlider("ar-p");
  bindSlider("hidden");
  bindSlider("lags");
  bindSlider("epochs");
  $("diff").addEventListener("change", schedule);
  ["show-series", "show-linear", "show-hybrid"].forEach((id) => $(id).addEventListener("change", render));
  $("stop").addEventListener("click", () => { stopTraining(); render(); window.__playground.done = true;
    $("status").textContent = `Stopped at epoch ${state.hybrid.epoch} of ${settings().epochs}; the results use the network as it is now.`; });
  $("retrain").addEventListener("click", rebuild);
  $("reinit").addEventListener("click", () => { state.initSeed++; rebuild(); });

  const saved = document.documentElement.dataset.theme
    || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  setTheme(saved);
  $("theme-toggle").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

  // start with a preset so the page shows a worked example
  document.querySelector('.preset[data-preset="Sine + trend"]').click();
}

init();
