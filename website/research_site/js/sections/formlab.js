// Section 6: additive versus multiplicative structure, and why variant E2 exists.
import { getJSON } from "../data.js";
import { color, label, layout, onThemeChange, plot } from "../charts.js";
import { fmt, segmented } from "../ui.js";

const state = { form: "additive", idx: 0 };
const el = {};

async function render() {
  const lab = await getJSON("form_lab.json");
  const s = lab.strengths[state.idx];
  el.out.textContent = `s = ${s}`;
  const row = lab.rows.find((r) => r.form === state.form && r.strength === s);
  const models = lab.models;
  const traces = [{
    type: "bar", x: models.map(label), y: models.map((m) => row.mse_mean[m]), marker: { color: models.map(color) },
    error_y: { type: "data", array: models.map((m) => row.mse_std[m]), visible: true, thickness: 1.5 },
    hovertemplate: "%{x}: MSE %{y:.4g}<extra></extra>",
  }];
  await plot(el.chart, traces, layout({ title: `Test MSE per model, ${state.form} form, strength ${s} (bars: mean over ${lab.seeds.length} series, whiskers: ± 1 s.d.)`,
    x: "model", y: "test MSE", height: 420, extra: { showlegend: false } }));

  const best = [...models].sort((a, b) => row.mse_mean[a] - row.mse_mean[b])[0];
  const gap = row.mse_mean.ARIMA - row.mse_mean[best];
  const within = best !== "ARIMA" && gap < Math.max(row.mse_std[best], row.mse_std.ARIMA);
  let note = `What to notice: the lowest mean MSE here is ${label(best)} (${fmt(row.mse_mean[best])}; ARIMA ${fmt(row.mse_mean.ARIMA)}). `;
  if (best !== "ARIMA") note += within ? "The gain over ARIMA is smaller than the spread across series, so it is not a reliable difference. " : "";
  if (row.arima_constant_share > 0) {
    note += `In ${Math.round(row.arima_constant_share * lab.seeds.length)} of ${lab.seeds.length} series ARIMA reduced to a constant forecast; ` +
            "then E2 and Zhang see the same information and give the same forecast. ";
  }
  el.notice.textContent = note;
}

export async function init() {
  el.chart = document.getElementById("fl-chart");
  el.notice = document.getElementById("fl-notice");
  el.out = document.getElementById("fl-strength-out");
  const lab = await getJSON("form_lab.json");
  const slider = document.getElementById("fl-strength");
  slider.max = String(lab.strengths.length - 1);
  slider.value = String(lab.strengths.length - 1);
  state.idx = lab.strengths.length - 1;
  slider.setAttribute("aria-label", "Nonlinear strength");
  slider.addEventListener("input", () => { state.idx = Number(slider.value); render(); });
  segmented(document.getElementById("fl-form"), (v) => { state.form = v; render(); });
  document.getElementById("fl-lead").innerHTML =
    "Zhang's hybrid adds a correction to the ARIMA forecast, which suits a series built as <b>L + N</b>. " +
    "If the nonlinear effect instead <b>scales</b> with the level, <b>L × (1 + N)</b>, an added correction has the wrong shape; variant <b>E2</b> multiplies the ARIMA forecast by a predicted ratio instead. " +
    "Here a persistent positive level L is combined with a nonlinear term driven by the previous value, in either form, and four models are compared on the last 60 points.";
  onThemeChange(render);
  await render();
}
