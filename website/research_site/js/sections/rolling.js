// Section 7: the rolling-origin protocol, one test point at a time.
import { getJSON } from "../data.js";
import { color, hexToRgba, label, layout, onThemeChange, plot, themeColors } from "../charts.js";
import { fmt, range, segmented } from "../ui.js";

const MODELS = ["ARIMA", "Zhang", "Extension"];
const state = { name: "Lynx", i: 0, timer: null };
const el = {};

async function render() {
  const all = await getJSON("rolling_animation.json");
  const d = all.datasets[state.name];
  const t = themeColors();
  const nTest = d.test_index.length;
  state.i = Math.min(state.i, nTest - 1);
  el.step.max = String(nTest - 1);
  el.step.value = String(state.i);
  const now = d.test_index[state.i];                 // 0-based position of the point being forecast
  const fitEnd = d.fitted_through[state.i];          // the model was fitted on positions 0 .. fitEnd-1
  el.out.textContent = `${state.i + 1} of ${nTest}`;

  const x = range(d.series.length, 1);
  const traces = [
    { x: x.slice(0, now), y: d.series.slice(0, now), name: "observed so far", line: { color: t.ink, width: 1.8 } },
    { x: x.slice(now), y: d.series.slice(now), name: "not yet observed", line: { color: t.ink2, width: 1, dash: "dot" } },
  ];
  for (const m of MODELS) {
    const xs = d.test_index.slice(0, state.i + 1).map((p) => p + 1);
    traces.push({ x: xs, y: d.forecasts[m].slice(0, state.i + 1), name: `${label(m)} one-step forecasts`, mode: "lines+markers",
                  line: { color: color(m), width: 1.6 }, marker: { size: xs.map((_, k) => (k === state.i ? 12 : 6)) } });
  }
  const lay = layout({ title: `${state.name}: forecasting test point ${state.i + 1} from data before it`, x: "time index", y: "value", height: 520 });
  const lo = Math.max(1, d.n_train - 40);
  lay.xaxis.range = [lo, d.series.length + 1];
  lay.shapes = [
    { type: "rect", xref: "x", yref: "paper", x0: 0.5, x1: fitEnd + 0.5, y0: 0, y1: 1, fillcolor: hexToRgba("#17637a", 0.10), line: { width: 0 } },
    ...d.refit_origins.map((o) => ({ type: "line", xref: "x", yref: "paper", x0: o + 0.5, x1: o + 0.5, y0: 0, y1: 1, line: { color: t.ink2, width: 1, dash: "dash" } })),
  ];
  lay.annotations = [{ x: fitEnd + 0.5, y: 1, xref: "x", yref: "paper", text: "end of training window", showarrow: false, xanchor: "right", yanchor: "bottom",
                       font: { size: 14 } }];
  await plot(el.chart, traces, lay);
  const actual = d.series[now];
  el.notice.textContent = `What to notice: the forecast of point ${now + 1} uses observations 1–${now}; the model itself was last refitted on points 1–${fitEnd} ` +
    `(dashed lines mark the ${d.refit_origins.length} refit origins). Actual value ${fmt(actual)}; forecasts: ` +
    MODELS.map((m) => `${label(m)} ${fmt(d.forecasts[m][state.i])}`).join(", ") + ". Forecasts are averaged over seeds.";
}

function stop() {
  clearInterval(state.timer); state.timer = null;
  el.play.textContent = "Play"; el.play.setAttribute("aria-pressed", "false");
}

export async function init() {
  ["chart", "notice", "play", "step"].forEach((k) => { el[k] = document.getElementById(`ro-${k}`); });
  el.out = document.getElementById("ro-step-out");
  el.step.setAttribute("aria-label", "Test point");
  el.step.addEventListener("input", () => { stop(); state.i = Number(el.step.value); render(); });
  segmented(document.getElementById("ro-dataset"), (v) => { stop(); state.name = v; state.i = 0; render(); });
  el.play.addEventListener("click", async () => {
    if (state.timer) return stop();
    const n = (await getJSON("rolling_animation.json")).datasets[state.name].test_index.length;
    if (state.i >= n - 1) state.i = 0;
    el.play.textContent = "Pause"; el.play.setAttribute("aria-pressed", "true");
    state.timer = setInterval(() => { if (state.i >= n - 1) return stop(); state.i += 1; render(); }, 900);
  });
  onThemeChange(render);
  await render();
}
