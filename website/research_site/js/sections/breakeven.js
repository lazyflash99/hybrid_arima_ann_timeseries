// Section 5: which model wins as the nonlinear part gets stronger or the noise gets louder.
import { getJSON } from "../data.js";
import { color, label, layout, onThemeChange, plot } from "../charts.js";
import { fmt } from "../ui.js";

const el = {};

async function render() {
  const be = await getJSON("break_even.json");
  const models = be.models;
  const cell = (s, nz) => be.cells.find((c) => c.strength === s && c.noise === nz);
  const z = be.strengths.map((s) => be.noises.map((nz) => models.indexOf(cell(s, nz).winner)));
  const text = be.strengths.map((s) => be.noises.map((nz) => {
    const c = cell(s, nz);
    return `${label(c.winner)}${c.consistent ? "" : " (?)"}`;
  }));
  const hover = be.strengths.map((s) => be.noises.map((nz) => {
    const c = cell(s, nz);
    const lines = models.map((m) => `${label(m)}: ${fmt(c.mse_mean[m])}`).join("<br>");
    return `strength ${s}, noise ${nz}<br>${lines}<br>winner beat runner-up in ${c.winner_beats_runner_up_in_seeds} of ${c.n_seeds} seeds`;
  }));
  // one discrete colour per model
  const n = models.length;
  const colorscale = models.flatMap((m, i) => [[i / n, color(m)], [(i + 1) / n, color(m)]]);
  const traces = [{
    type: "heatmap", x: be.noises.map(String), y: be.strengths.map(String), z, zmin: -0.5, zmax: n - 0.5, colorscale, showscale: false,
    text, texttemplate: "%{text}", textfont: { size: 14, color: "#ffffff" }, customdata: hover, hovertemplate: "%{customdata}<extra></extra>",
    xgap: 3, ygap: 3,
  }];
  for (const m of models) traces.push({ x: [null], y: [null], type: "scatter", mode: "markers", name: `${label(m)} wins`, marker: { color: color(m), size: 14, symbol: "square" } });
  const lay = layout({ title: "Model with the lowest test MSE (mean over seeds) on synthetic series", x: "noise standard deviation", y: "strength of the nonlinear part", height: 560 });
  lay.xaxis.type = "category"; lay.yaxis.type = "category";
  await plot(el.chart, traces, lay);

  const counts = Object.fromEntries(models.map((m) => [m, be.cells.filter((c) => c.winner === m).length]));
  const consistent = be.cells.filter((c) => c.consistent).length;
  el.notice.textContent = `What to notice: ${models.map((m) => `${label(m)} wins ${counts[m]}`).join(", ")} of ${be.cells.length} cells. ` +
    "With no or weak nonlinearity ARIMA is hard to beat; the Extension pays off once the nonlinear part is strong and the noise is not overwhelming. " +
    `Only ${consistent} cells have a winner that beat the runner-up in every seed; cells marked (?) are close calls.`;
}

export async function init() {
  el.chart = document.getElementById("be-chart");
  el.notice = document.getElementById("be-notice");
  const be = await getJSON("break_even.json");
  document.getElementById("be-lead").innerHTML =
    "We generated series of a known form, <i>y<sub>t</sub> = 0.6 y<sub>t−1</sub> − 0.3 y<sub>t−2</sub> + strength · sin(y<sub>t−1</sub>) + noise</i>, " +
    `for ${be.strengths.length} strengths and ${be.noises.length} noise levels, ${be.seeds.length} series each, and compared ARIMA, Zhang's hybrid and the Extension ` +
    "on the last 60 points. Hover over a cell to see every model's error.";
  onThemeChange(render);
  await render();
}
