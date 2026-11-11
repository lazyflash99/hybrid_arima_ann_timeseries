// Section 3: residual plots drawn from the saved test errors, plus the stored test results.
import { dataset, datasetIndex, getJSON, rows } from "../data.js";
import { MODELS, color, label, layout, onThemeChange, plot, themeColors } from "../charts.js";
import { acf, finite, qq } from "../stats.js";
import { fillSelect, fmt, fmtP, pEq, range, table } from "../ui.js";

const state = { name: "Sunspot", model: "ARIMA" };
const el = {};
// which stored test row belongs to which model's test errors
const TEST_ROW = { ARIMA: "ARIMA test", Zhang: "Hybrid test" };

function badge(name, p, okWhenAbove, meaningLow, meaningHigh) {
  if (p === null || p === undefined || !Number.isFinite(p)) return `<span class="badge">${name}: series too short for this test</span>`;
  const low = p < 0.05;
  const good = okWhenAbove ? !low : low;
  return `<span class="badge ${good ? "good" : "bad"}">${name}: ${pEq(p)}, ${low ? meaningLow : meaningHigh}</span>`;
}

function badgesFor(r, title) {
  if (!r) return "";
  return `<span class="badge"><b>${title}</b></span>` +
    badge("Ljung–Box (lag 12)", r.ljung_box_p_12, true, "autocorrelation left", "no autocorrelation") +
    badge("Jarque–Bera", r.jarque_bera_p, true, "not normal", "normality not rejected") +
    badge("ARCH-LM", r.arch_lm_p, true, "changing variance", "no change in variance") +
    badge("BDS", r.bds_p, false, "dependence found", "no dependence found") +
    badge("Neural-network test", r.nn_linearity_p, false, "nonlinearity found", "no nonlinearity found");
}

async function render() {
  const d = await dataset(state.name);
  const t = themeColors();
  const err = d.models.fixed[state.model].final_error;
  const e = finite(err);
  const n = e.length;
  const x = range(err.length, 1);
  const c = color(state.model);
  const who = `${label(state.model)} errors`;

  await plot(el.time, [{ type: "bar", x, y: err, name: who, marker: { color: c } }],
             layout({ title: `${who} over the test part`, x: "test point", y: "error", height: 340, extra: { showlegend: false } }));

  const lags = Math.min(20, Math.floor(n / 2));
  const r = acf(e, lags);
  const band = 1.96 / Math.sqrt(n);
  const outside = r.filter((v) => Math.abs(v) > band).length;
  await plot(el.acf, [
    { type: "bar", x: range(lags, 1), y: r, name: "autocorrelation", marker: { color: c } },
    { x: [0.5, lags + 0.5], y: [band, band], mode: "lines", name: "±1.96/√n", line: { color: t.bad, dash: "dash", width: 1.5 } },
    { x: [0.5, lags + 0.5], y: [-band, -band], mode: "lines", showlegend: false, line: { color: t.bad, dash: "dash", width: 1.5 }, hoverinfo: "skip" },
  ], layout({ title: "Autocorrelation of the errors", x: "lag", y: "autocorrelation", height: 340 }));

  await plot(el.hist, [{ type: "histogram", x: e, nbinsx: Math.max(8, Math.round(Math.sqrt(n) * 1.5)), name: who, marker: { color: c } }],
             layout({ title: "Histogram of the errors", x: "error", y: "count", height: 340, extra: { showlegend: false } }));

  const q = qq(e);
  const ends = [q.theo[0], q.theo[q.theo.length - 1]];
  await plot(el.qq, [
    { x: q.theo, y: q.sample, mode: "markers", name: "errors", marker: { color: c, size: 7 } },
    { x: ends, y: ends.map((v) => q.icpt + q.slope * v), mode: "lines", name: "normal reference", line: { color: t.bad, width: 1.5 } },
  ], layout({ title: "Normal QQ plot", x: "normal quantile", y: "error quantile", height: 340 }));

  const diag = await rows("residual_diagnostics.json", { dataset: d.name });
  const train = diag.find((r) => r.residuals === "ARIMA train");
  const test = diag.find((r) => r.residuals === TEST_ROW[state.model]);
  el.badges.innerHTML = badgesFor(train, "Tests on ARIMA's training errors:") +
    (test ? `<br>${badgesFor(test, `Tests on ${who} (test part, n = ${test.n}):`)}` : "");
  const findings = await getJSON("residual_findings.json");
  el.finding.textContent = findings[d.name] || "";
  el.notice.textContent = `What to notice: ${outside} of ${lags} autocorrelation bars lie outside the dashed band (about ${fmt(0.05 * lags, 2)} ` +
    "would be expected by chance). Points bending away from the QQ line mean heavier tails than a normal distribution. " +
    (test ? "" : "Stored tests cover ARIMA's and Zhang's errors; for this model only the plots are shown.");
}

async function renderTable() {
  const all = await getJSON("nonlinearity_vs_improvement.json");
  const yn = (b) => (b ? "yes" : "no");
  el.table.innerHTML = table(["Dataset", "Nonlinearity found in ARIMA errors?", "Zhang's hybrid beat ARIMA?", "Both give the same answer?", "ARIMA MSE", "Zhang MSE"],
    all.map((r) => [r.dataset, yn(r.nonlinearity_detected), yn(r.hybrid_improved), yn(r.both), fmt(r.mse_arima), fmt(r.mse_zhang)]), { numeric: [4, 5] }) +
    `<p class="notice">What to notice: the test and the outcome agree on ${all.filter((r) => r.both).length} of ${all.length} datasets, ` +
    "so detecting nonlinearity in ARIMA's errors did not reliably predict whether the hybrid would help.</p>";
}

export async function init() {
  ["dataset", "model", "badges", "finding", "time", "acf", "hist", "qq", "notice", "table"].forEach((k) => { el[k] = document.getElementById(`res-${k}`); });
  fillSelect(el.dataset, (await datasetIndex()).map((d) => d.name), state.name);
  fillSelect(el.model, MODELS.map((m) => ({ value: m, label: label(m) })), state.model);
  el.dataset.addEventListener("change", () => { state.name = el.dataset.value; render(); });
  el.model.addEventListener("change", () => { state.model = el.model.value; render(); });
  onThemeChange(render);
  await Promise.all([render(), renderTable()]);
}
