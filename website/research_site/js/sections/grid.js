// Section 10: the linear x nonlinear x combiner grid.
import { datasetIndex, getJSON } from "../data.js";
import { layout, onThemeChange, plot } from "../charts.js";
import { fillSelect, fmt, fmtP, table } from "../ui.js";

const COMBINERS = ["additive", "multiplicative", "joint-input", "validation-weighted"];
const COMBINER_TEXT = {
  "additive": "linear forecast + nonlinear model of its error (Zhang's form)",
  "multiplicative": "linear forecast × nonlinear model of the ratio",
  "joint-input": "one nonlinear model sees past values, the linear forecast and past errors (the Extension's form)",
  "validation-weighted": "linear forecast + a weighted correction, weight chosen on validation data",
};
const state = { name: "Lynx", combiner: "additive" };
const el = {};

async function render() {
  const designs = (await getJSON("grid_designs.json")).filter((r) => r.dataset === state.name);
  const singles = (await getJSON("grid_single_models.json")).filter((r) => r.dataset === state.name && r.MSE !== null);
  const best = singles.reduce((a, b) => (b.MSE < a.MSE ? b : a));
  const lead = (await getJSON("grid_leaderboard.json")).filter((r) => r.dataset === state.name);
  const when = (await getJSON("grid_when.json")).find((r) => r.dataset === state.name);

  const rowsC = designs.filter((r) => r.combiner === state.combiner);
  const lins = [...new Set(designs.map((r) => r.linear))].sort();
  const nls = [...new Set(designs.map((r) => r.nonlinear))].sort();
  const z = lins.map((l) => nls.map((n) => { const r = rowsC.find((q) => q.linear === l && q.nonlinear === n); return r ? r.MSE / best.MSE : null; }));
  const text = z.map((row) => row.map((v) => (v === null ? "not defined" : fmt(v, 3))));
  const traces = [{
    type: "heatmap", x: nls, y: lins, z, zmid: 1, zmin: 0, zmax: 2, colorscale: "RdBu",
    text, texttemplate: "%{text}", textfont: { size: 14 }, hovertemplate: "%{y} + %{x}: %{text} × best single model<extra></extra>",
    colorbar: { title: { text: "MSE ÷ best single", font: { size: 14 } }, tickfont: { size: 14 } }, xgap: 2, ygap: 2,
  }];
  await plot(el.chart, traces, layout({ title: `${state.name}, ${state.combiner}: test MSE relative to the best single model (${best.model})`,
    x: "nonlinear model", y: "linear model", height: 520, extra: { margin: { l: 130, r: 20, t: 56, b: 70 } } }));

  const flat = z.flat().filter((v) => v !== null);
  const below = flat.filter((v) => v < 1).length;
  el.notice.textContent = `What to notice: ${COMBINER_TEXT[state.combiner]}. ${below} of ${flat.length} designs here have a lower test MSE than the best single model ` +
    "(blue cells, below 1). Values are means over seeds; the colour scale is capped at 2.";
  el.baseline.textContent = when.simple_baseline_wins
    ? `A simple single model (${when.best_single}) is at least as good as every hybrid on ${state.name}.`
    : `Best hybrid on ${state.name}: ${when.best_hybrid}. Best single model: ${when.best_single}.`;

  const ranked = [...rowsC].sort((a, b) => a.MSE - b.MSE).slice(0, 12);
  const body = ranked.map((r, i) => {
    const l = lead.find((q) => q.linear === r.linear && q.nonlinear === r.nonlinear && q.combiner === r.combiner);
    const p = l ? fmtP(l.dm_p) : "not tested";
    const sig = l ? (l.dm_p < 0.05 ? (l.dm_stat > 0 ? '<span class="sig">significantly better</span>' : '<span class="sig-bad">significantly worse</span>')
                               : "not significant") : "–";
    return [i + 1, `${r.linear} + ${r.nonlinear}`, fmt(r.MSE), fmt(r.MSE / best.MSE, 3), r.n_runs, p, sig];
  });
  el.table.innerHTML = table(["Rank", "Design", "Test MSE", "÷ best single", "Runs", "DM p vs best single", "Reading"], body, { numeric: [0, 2, 3, 4, 5] }) +
    `<p class="notice">The significance test was run on the ten best designs per dataset over all combiners; other rows show "not tested". ` +
    "The best design is chosen after seeing test errors, so its advantage is optimistic.</p>";
}

export async function init() {
  ["chart", "notice", "baseline", "table"].forEach((k) => { el[k] = document.getElementById(`gr-${k}`); });
  const ds = document.getElementById("gr-dataset");
  const cb = document.getElementById("gr-combiner");
  fillSelect(ds, (await datasetIndex()).map((d) => d.name), state.name);
  fillSelect(cb, COMBINERS, state.combiner);
  ds.addEventListener("change", () => { state.name = ds.value; render(); });
  cb.addEventListener("change", () => { state.combiner = cb.value; render(); });
  onThemeChange(render);
  await render();
}
