// Section 2: forecasts and metrics of every model on every dataset, for both protocols.
import { dataset, datasetIndex, rows } from "../data.js";
import { MODELS, color, label, layout, onThemeChange, plot, themeColors } from "../charts.js";
import { dmReading, fillSelect, fmt, fmtP, pEq, range, segmented, table } from "../ui.js";

const LAYERS = [["arima", "ARIMA forecast"], ["correction", "Neural-network correction"], ["final", "Final forecasts"]];
const METRICS = [["MSE", "MSE"], ["RMSE", "RMSE"], ["MAD", "MAD"], ["MAPE", "MAPE (%)"], ["sMAPE", "sMAPE (%)"],
                 ["MASE", "MASE"], ["TheilU2", "Theil U2"], ["DA", "Direction accuracy (%)"]];

const state = { name: "Sunspot", protocol: "fixed", models: new Set(["ARIMA", "Zhang", "Extension"]),
                layers: new Set(["arima", "correction", "final"]), focus: "Extension" };
const el = {};

const available = () => MODELS.filter((m) => !(state.protocol === "rolling" && m === "Extension margin"));

function buildChecks() {
  el.models.innerHTML = "";
  for (const m of MODELS) {
    const off = !available().includes(m);
    const id = `ex-m-${m.replace(/\W/g, "")}`;
    el.models.insertAdjacentHTML("beforeend",
      `<label for="${id}"><input type="checkbox" id="${id}" value="${m}" ${state.models.has(m) && !off ? "checked" : ""} ${off ? "disabled" : ""}>` +
      `<span class="swatch" style="background:${color(m)}"></span>${label(m)}${off ? " (fixed split only)" : ""}</label>`);
  }
  el.models.querySelectorAll("input").forEach((c) => c.addEventListener("change", () => {
    c.checked ? state.models.add(c.value) : state.models.delete(c.value);
    render();
  }));
  fillSelect(el.focus, available().map((m) => ({ value: m, label: label(m) })), available().includes(state.focus) ? state.focus : "Zhang");
  state.focus = el.focus.value;
}

async function render() {
  const d = await dataset(state.name);
  const t = themeColors();
  const proto = d.models[state.protocol];
  const x = range(d.actual_test.length, 1);
  const chosen = available().filter((m) => state.models.has(m) && proto[m]);
  const arima = proto.ARIMA.forecast;
  const showCorr = state.layers.has("correction") && chosen.some((m) => m !== "ARIMA");

  const traces = [{ x, y: d.actual_test, name: "actual", line: { color: t.ink, width: 2.4 } }];
  if (state.layers.has("arima")) traces.push({ x, y: arima, name: "ARIMA forecast (layer)", line: { color: color("ARIMA"), width: 1.6, dash: "dash" } });
  if (state.layers.has("final")) {
    for (const m of chosen.filter((m) => m !== "ARIMA")) traces.push({ x, y: proto[m].forecast, name: `${label(m)} forecast`, line: { color: color(m), width: 1.8 } });
  }
  if (showCorr) {
    for (const m of chosen.filter((m) => m !== "ARIMA")) {
      const corr = proto[m].forecast.map((v, i) => (v === null || arima[i] === null ? null : v - arima[i]));
      traces.push({ x, y: corr, name: `${label(m)} correction`, xaxis: "x", yaxis: "y2", showlegend: false,
                    line: { color: color(m), width: 1.5 }, hovertemplate: `${label(m)} correction %{y:.4g}<extra></extra>` });
    }
  }
  const protoName = state.protocol === "fixed" ? "fixed split" : "rolling origin";
  const lay = layout({ title: `${d.name}: actual against forecast on the test part (${protoName}, mean over seeds)`, x: "test point", y: "value", height: 600 });
  if (showCorr) {
    lay.yaxis.domain = [0.36, 1];
    lay.yaxis2 = { ...lay.yaxis, domain: [0, 0.26], title: { text: "correction", font: { size: 15 } }, anchor: "x" };
    lay.xaxis.anchor = "y2";
    lay.shapes = [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y2", y0: 0, y1: 0, line: { color: t.ink2, width: 1 } }];
  }
  await plot(el.chart, traces, lay);

  const met = await rows("metrics.json", { protocol: state.protocol, dataset: d.name });
  const dm = await rows("dm.json", { protocol: state.protocol, dataset: d.name, versus: "Zhang" });
  const mcs = await rows("validation_mcs.json", { protocol: state.protocol, dataset: d.name });
  const get = (m) => met.find((r) => r.model === m) || {};
  const dmOf = (m) => dm.find((r) => r.model === m) || {};
  const inSet = (m) => { const r = mcs.find((q) => q.model === m); return r ? (r.in_90pct_set ? "yes" : "no") : "–"; };

  const f = get(state.focus);
  const fdm = dmOf(state.focus);
  const cards = METRICS.map(([k, name]) => `<div class="card"><div class="k">${name}</div><div class="v">${fmt(f[`${k}_mean`])}</div>` +
    `<div class="s">± ${fmt(f[`${k}_std`])} across seeds</div></div>`);
  cards.push(`<div class="card"><div class="k">Diebold–Mariano vs Zhang</div><div class="v">${state.focus === "Zhang" ? "–" : pEq(fdm.p_value)}</div>` +
    `<div class="s">${state.focus === "Zhang" ? "reference model" : dmReading(fdm.dm_stat, fdm.p_value, "Zhang")}</div></div>`);
  cards.push(`<div class="card"><div class="k">In the 90% model confidence set</div><div class="v">${inSet(state.focus)}</div>` +
    `<div class="s">"yes": cannot be told apart from the best</div></div>`);
  el.cards.innerHTML = cards.join("");

  const body = chosen.map((m) => {
    const r = get(m); const q = dmOf(m);
    const reading = m === "Zhang" ? "reference" : dmReading(q.dm_stat, q.p_value, "Zhang");
    const cls = q.p_value < 0.05 ? (q.dm_stat > 0 ? "sig" : "sig-bad") : "";
    return [`<span class="swatch" style="background:${color(m)}"></span> ${label(m)}`, fmt(r.MSE_mean), fmt(r.MSE_std), fmt(r.MASE_mean),
            fmt(r.TheilU2_mean), fmt(r.DA_mean, 3), m === "Zhang" ? "–" : fmtP(q.p_value), `<span class="${cls}">${reading}</span>`, inSet(m)];
  });
  el.table.innerHTML = table(["Model", "MSE", "MSE s.d.", "MASE", "Theil U2", "Direction %", "DM p vs Zhang", "Reading", "In MCS"], body,
                             { numeric: [1, 2, 3, 4, 5, 6] });

  const ranked = chosen.filter((m) => Number.isFinite(get(m).MSE_mean)).sort((a, b) => get(a).MSE_mean - get(b).MSE_mean);
  if (ranked.length) {
    const best = ranked[0];
    const q = dmOf(best);
    const sig = best === "Zhang" ? "" : ` Compared with Zhang's hybrid it is ${dmReading(q.dm_stat, q.p_value, "Zhang")} (${pEq(q.p_value)}).`;
    el.notice.textContent = `What to notice: among the models shown, ${label(best)} has the lowest test MSE (${fmt(get(best).MSE_mean)}).${sig}` +
      " Small differences in MSE are often not significant on short test sets.";
  } else {
    el.notice.textContent = "Select at least one model.";
  }
}

export async function init() {
  ["dataset", "models", "layers", "focus", "cards", "chart", "notice", "table"].forEach((k) => { el[k] = document.getElementById(`ex-${k}`); });
  fillSelect(el.dataset, (await datasetIndex()).map((d) => d.name), state.name);
  el.dataset.addEventListener("change", () => { state.name = el.dataset.value; render(); });
  segmented(document.getElementById("ex-protocol"), (v) => { state.protocol = v; buildChecks(); render(); });
  el.focus.addEventListener("change", () => { state.focus = el.focus.value; render(); });
  for (const [k, name] of LAYERS) {
    el.layers.insertAdjacentHTML("beforeend", `<label for="ex-l-${k}"><input type="checkbox" id="ex-l-${k}" value="${k}" checked>${name}</label>`);
  }
  el.layers.querySelectorAll("input").forEach((c) => c.addEventListener("change", () => {
    c.checked ? state.layers.add(c.value) : state.layers.delete(c.value);
    render();
  }));
  buildChecks();
  onThemeChange(render);
  await render();
}
