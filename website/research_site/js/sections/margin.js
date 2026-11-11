// Section 9: the scale margin and the Sunspot record peak.
import { getJSON } from "../data.js";
import { color, hexToRgba, layout, onThemeChange, plot, themeColors } from "../charts.js";
import { fmt } from "../ui.js";

const state = { idx: 0 };
const el = {};

async function render() {
  const sm = await getJSON("scale_margin.json");
  const t = themeColors();
  const m = sm.margins[state.idx];
  const key = String(m);
  const f = sm.forecasts[key];
  const base = sm.forecasts[String(sm.margins[0])];
  el.out.textContent = `m = ${m}`;
  const rec = sm.record_points;
  const traces = [
    { x: sm.years, y: sm.actual_test, name: "actual", line: { color: t.ink, width: 2.2 } },
    { x: rec.map((i) => sm.years[i]), y: rec.map((i) => sm.actual_test[i]), mode: "markers", name: "above the training maximum",
      marker: { color: t.bad, size: 11, symbol: "diamond" } },
    { x: sm.years, y: f.forecast, name: `Extension forecast, margin ${m}`, line: { color: color("Extension"), width: 2.2 } },
  ];
  if (state.idx !== 0) traces.push({ x: sm.years, y: base.forecast, name: "Extension forecast, margin 0", line: { color: color("Extension"), width: 1.2, dash: "dot" } });
  const lay = layout({ title: `Sunspot test years: Extension forecast with scale margin ${m} (mean over ${sm.seeds.length} seeds)`, x: "year", y: "sunspot number", height: 540 });
  lay.shapes = [{ type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y", y0: sm.train_min, y1: sm.train_max, fillcolor: hexToRgba("#0072B2", 0.08), line: { width: 0 } },
                { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: sm.train_max, y1: sm.train_max, line: { color: t.ink2, dash: "dash", width: 1.2 } }];
  lay.annotations = [{ xref: "paper", x: 0.01, yref: "y", y: sm.train_max, text: `training maximum ${fmt(sm.train_max)}`, showarrow: false, yanchor: "bottom", xanchor: "left", font: { size: 14 } }];
  await plot(el.chart, traces, lay);

  el.cards.innerHTML = [
    ["Test MSE, all points", fmt(f.mse), `margin 0: ${fmt(base.mse)}`],
    ["Test MSE, record points", fmt(f.mse_record_points), `margin 0: ${fmt(base.mse_record_points)}`],
    ["Highest forecast", fmt(f.max_forecast), `actual peak ${fmt(Math.max(...sm.actual_test))}`],
    ["Training range", `${fmt(sm.train_min)}–${fmt(sm.train_max)}`, "values the network has seen"],
    ["Points above that range", String(rec.length), `of ${sm.actual_test.length} test points`],
  ].map(([k, v, s]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
  el.notice.textContent = "What to notice: even with margin 0 the forecast can rise above the training maximum, because the output layer is linear. " +
    "The difficulty is that the record values become inputs the network never saw in training, where its hidden units saturate. " +
    "This is one dataset and one test set, and the margin was designed after seeing this failure, so treat the differences as illustration, not proof.";
}

export async function init() {
  ["chart", "notice", "cards"].forEach((k) => { el[k] = document.getElementById(`mg-${k}`); });
  el.out = document.getElementById("mg-margin-out");
  const sm = await getJSON("scale_margin.json");
  const slider = document.getElementById("mg-margin");
  slider.max = String(sm.margins.length - 1);
  slider.value = "0";
  slider.setAttribute("aria-label", "Scale margin");
  slider.addEventListener("input", () => { state.idx = Number(slider.value); render(); });
  document.getElementById("mg-lead").innerHTML =
    "The networks scale their inputs to the range seen in training. The 1957–59 sunspot peak is higher than anything in the training years. " +
    "The <b>scale margin</b> m maps the training range to [m, 1 − m] instead of [0, 1], leaving room for values beyond it. " +
    "The output layer is linear, so the output itself is not capped at the training range; the problem is inputs that were never seen in training.";
  onThemeChange(render);
  await render();
}
