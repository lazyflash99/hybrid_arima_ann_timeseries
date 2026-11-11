// Section 1: Zhang's hybrid step by step on one dataset (fixed split, mean over seeds).
import { dataset, datasetIndex, rows } from "../data.js";
import { color, layout, onThemeChange, plot, themeColors } from "../charts.js";
import { dmReading, fillSelect, fmt, fmtP, pEq, range } from "../ui.js";

const STEPS = [
  { title: "The series", eq: "y_t = L_t + N_t",
    text: "The whole series, split in time. Models learn from the training part only and are judged on the test part. Zhang assumes each value is the sum of a linear part L and a nonlinear part N." },
  { title: "ARIMA forecasts the linear part", eq: "\\hat L_t = \\text{ARIMA forecast from } y_{t-1}, y_{t-2}, \\dots",
    text: "ARIMA predicts each test value from the values before it. It captures the linear, regular part of the series." },
  { title: "What ARIMA leaves: its errors", eq: "e_t = y_t - \\hat L_t",
    text: "The difference between the actual value and the ARIMA forecast. If the series is linear plus nonlinear, these errors should hold the nonlinear part." },
  { title: "A network forecasts the errors", eq: "\\hat N_t = f(e_{t-1}, \\dots, e_{t-n})",
    text: "A small neural network looks at the last few errors and predicts the next one. This prediction is the neural-network correction." },
  { title: "Add them back", eq: "\\hat y_t = \\hat L_t + \\hat N_t",
    text: "The final forecast is the ARIMA forecast plus the correction. It helps only if the correction actually follows the ARIMA errors." },
];

let state = { name: "Sunspot", step: 0, timer: null };
let el = {};

async function render() {
  const d = await dataset(state.name);
  const t = themeColors();
  const n = d.actual_test.length;
  const x = range(n, 1);
  const zhang = d.models.fixed.Zhang;
  const s = STEPS[state.step];
  el.stepno.textContent = `Step ${state.step + 1} of ${STEPS.length}`;
  el.title.textContent = s.title;
  katex.render(s.eq, el.eq, { throwOnError: false, displayMode: true });
  el.text.textContent = s.text;
  [...el.dots.children].forEach((b, i) => b.setAttribute("aria-current", i === state.step ? "step" : "false"));
  el.back.disabled = state.step === 0;
  el.next.disabled = state.step === STEPS.length - 1;

  let traces = [];
  let lay;
  if (state.step === 0) {
    const all = range(d.series.length, 1);
    traces = [
      { x: all.slice(0, d.n_train), y: d.series.slice(0, d.n_train), name: "training part", line: { color: t.ink2, width: 1.5 } },
      { x: all.slice(d.n_train), y: d.series.slice(d.n_train), name: "test part", line: { color: t.ink, width: 2 } },
    ];
    lay = layout({ title: `${d.name}: the whole series`, x: "time index", y: "value", height: 460 });
    el.notice.textContent = `What to notice: ${d.n_train} training points and ${n} test points. Every later step looks at the test part only.`;
  } else if (state.step === 1) {
    traces = [
      { x, y: d.actual_test, name: "actual", line: { color: t.ink, width: 2 } },
      { x, y: d.arima_forecast, name: "ARIMA forecast", line: { color: color("ARIMA"), width: 2, dash: "dash" } },
    ];
    lay = layout({ title: `${d.name}: ARIMA forecast on the test part`, x: "test point", y: "value", height: 460 });
    el.notice.textContent = "What to notice: ARIMA follows the overall shape but misses some turns.";
  } else if (state.step === 2) {
    traces = [{ type: "bar", x, y: d.arima_error, name: "ARIMA error", marker: { color: color("ARIMA") } }];
    lay = layout({ title: `${d.name}: ARIMA errors e = actual − ARIMA forecast`, x: "test point", y: "error", height: 460 });
    el.notice.textContent = "What to notice: if these bars show a pattern, a network may be able to predict them; if they look random, it cannot.";
  } else if (state.step === 3) {
    const r2 = (await rows("decomposition_r2.json", { dataset: d.name, model: "Zhang" }))[0];
    traces = [
      { x, y: d.arima_error, name: "ARIMA error", line: { color: color("ARIMA"), width: 2 } },
      { x, y: zhang.nn_correction, name: "neural-network correction", line: { color: color("Zhang"), width: 2, dash: "dot" } },
    ];
    lay = layout({ title: `${d.name}: the network's forecast of the ARIMA error`, x: "test point", y: "error", height: 460 });
    el.notice.textContent = r2
      ? `What to notice: the correction explains R² = ${fmt(r2.r2_correction_vs_arima_error, 2)} of the ARIMA error (1 = all of it, 0 = none, below 0 = worse than adding nothing; seed 0).`
      : "What to notice: a useful correction follows the ARIMA error.";
  } else {
    const m = await rows("metrics.json", { protocol: "fixed", dataset: d.name });
    const mse = (model) => (m.find((r) => r.model === model) || {}).MSE_mean;
    const dm = (await rows("dm.json", { protocol: "fixed", dataset: d.name, model: "Zhang", versus: "ARIMA" }))[0] || {};
    traces = [
      { x, y: d.actual_test, name: "actual", line: { color: t.ink, width: 2 } },
      { x, y: d.arima_forecast, name: "ARIMA forecast", line: { color: color("ARIMA"), width: 1.5, dash: "dash" } },
      { x, y: zhang.forecast, name: "final forecast (Zhang's hybrid)", line: { color: color("Zhang"), width: 2 } },
    ];
    lay = layout({ title: `${d.name}: final forecast = ARIMA forecast + correction`, x: "test point", y: "value", height: 460 });
    el.notice.textContent = `What to notice: test MSE ${fmt(mse("ARIMA"))} for ARIMA and ${fmt(mse("Zhang"))} for Zhang's hybrid ` +
      `(mean over 10 seeds). Diebold–Mariano ${pEq(dm.p_value)}: Zhang's hybrid is ${dmReading(dm.dm_stat, dm.p_value, "ARIMA")}.`;
  }
  await plot(el.chart, traces, lay);
}

function go(step) { state.step = Math.max(0, Math.min(STEPS.length - 1, step)); render(); }

function stopPlay() {
  clearInterval(state.timer); state.timer = null;
  el.play.textContent = "Play"; el.play.setAttribute("aria-pressed", "false");
}

export async function init() {
  const ids = ["dataset", "back", "next", "play", "stepno", "title", "eq", "text", "dots", "chart", "notice"];
  ids.forEach((k) => { el[k] = document.getElementById(`idea-${k}`); });
  fillSelect(el.dataset, (await datasetIndex()).map((d) => d.name), state.name);
  STEPS.forEach((s, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.setAttribute("aria-label", `Step ${i + 1}: ${s.title}`);
    b.addEventListener("click", () => { stopPlay(); go(i); });
    el.dots.appendChild(b);
  });
  el.dataset.addEventListener("change", () => { state.name = el.dataset.value; render(); });
  el.back.addEventListener("click", () => { stopPlay(); go(state.step - 1); });
  el.next.addEventListener("click", () => { stopPlay(); go(state.step + 1); });
  el.play.addEventListener("click", () => {
    if (state.timer) return stopPlay();
    if (state.step === STEPS.length - 1) go(0);
    el.play.textContent = "Pause"; el.play.setAttribute("aria-pressed", "true");
    state.timer = setInterval(() => (state.step >= STEPS.length - 1 ? stopPlay() : go(state.step + 1)), 2600);
  });
  onThemeChange(render);
  await render();
}
