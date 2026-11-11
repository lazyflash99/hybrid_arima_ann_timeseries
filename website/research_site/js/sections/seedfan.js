// Section 8: spread of the forecasts across 10 random starting points.
import { getJSON } from "../data.js";
import { color, hexToRgba, label, layout, onThemeChange, plot, themeColors } from "../charts.js";
import { fillSelect, fmt, range } from "../ui.js";

const MODELS = ["Zhang", "Extension"];
const state = { name: "Sunspot" };
const el = {};

async function render() {
  const fan = await getJSON("seed_fan.json");
  const d = fan.datasets[state.name];
  const t = themeColors();
  const x = range(d.actual_test.length, 1);
  const traces = [];
  const widths = {};
  const mses = {};
  for (const m of MODELS) {
    const runs = Object.values(d.forecasts[m]);
    const lo = x.map((_, i) => Math.min(...runs.map((r) => r[i])));
    const hi = x.map((_, i) => Math.max(...runs.map((r) => r[i])));
    const avg = x.map((_, i) => runs.reduce((a, r) => a + r[i], 0) / runs.length);
    widths[m] = hi.reduce((a, v, i) => a + v - lo[i], 0) / x.length;
    mses[m] = runs.map((r) => r.reduce((a, v, i) => a + (d.actual_test[i] - v) ** 2, 0) / x.length);
    traces.push({ x, y: hi, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" });
    traces.push({ x, y: lo, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: hexToRgba(color(m), 0.28),
                  name: `${label(m)}: range across ${runs.length} seeds`, hoverinfo: "skip" });
    traces.push({ x, y: avg, name: `${label(m)}: mean forecast`, line: { color: color(m), width: 2 } });
  }
  traces.push({ x, y: d.actual_test, name: "actual", line: { color: t.ink, width: 2.2 } });
  await plot(el.chart, traces, layout({ title: `${state.name}: test forecasts across seeds 0–9 (fixed split)`, x: "test point", y: "value", height: 540 }));
  el.notice.textContent = "What to notice: a wide band means the result depends on the random start. " +
    MODELS.map((m) => `${label(m)}: average band width ${fmt(widths[m])}, test MSE from ${fmt(Math.min(...mses[m]))} to ${fmt(Math.max(...mses[m]))} across seeds`).join("; ") + ".";
}

export async function init() {
  el.chart = document.getElementById("sf-chart");
  el.notice = document.getElementById("sf-notice");
  const sel = document.getElementById("sf-dataset");
  fillSelect(sel, Object.keys((await getJSON("seed_fan.json")).datasets), state.name);
  sel.addEventListener("change", () => { state.name = sel.value; render(); });
  onThemeChange(render);
  await render();
}
