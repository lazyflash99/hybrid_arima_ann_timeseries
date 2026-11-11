// Section 4: training series and its first difference, with the stored stationarity tests.
import { dataset, datasetIndex, getJSON } from "../data.js";
import { layout, onThemeChange, plot, themeColors } from "../charts.js";
import { diff } from "../stats.js";
import { fillSelect, fmt, fmtP, pEq, range, table } from "../ui.js";

const state = { name: "AirPassengers" };
const el = {};

const verdict = (r) => ({
  adf: r.adf_p < 0.05 ? "stationary" : "unit root (not stationary)",
  kpss: r.kpss_p < 0.05 ? "not stationary" : "stationary",
  pp: r.pp_p < 0.05 ? "stationary" : "unit root (not stationary)",
});

async function render() {
  const d = await dataset(state.name);
  const t = themeColors();
  const y = d.series.slice(0, d.n_train);
  const dy = diff(y);
  const traces = [
    { x: range(y.length, 1), y, name: "training series", line: { color: t.ink, width: 1.6 } },
    { x: range(dy.length, 2), y: dy, name: "first difference (change from one step to the next)", xaxis: "x2", yaxis: "y2", line: { color: t.accent, width: 1.3 } },
  ];
  const lay = layout({ title: `${d.name}: training series (top) and its first difference (bottom)`, x: "", y: "value", height: 560 });
  lay.grid = { rows: 2, columns: 1, pattern: "independent", roworder: "top to bottom" };
  lay.xaxis2 = { ...lay.xaxis, title: { text: "time index", font: { size: 15 } } };
  lay.yaxis2 = { ...lay.yaxis, title: { text: "change", font: { size: 15 } } };
  await plot(el.chart, traces, lay);

  const st = (await getJSON("stationarity.json")).find((r) => r.dataset === d.name);
  const v = verdict(st);
  const agree = st.adf_kpss !== "disagree";
  el.badges.innerHTML = [
    `<span class="badge">ADF: ${pEq(st.adf_p)}, ${v.adf}</span>`,
    `<span class="badge">KPSS: ${pEq(st.kpss_p)}, ${v.kpss}</span>`,
    `<span class="badge">Phillips–Perron: ${pEq(st.pp_p)}, ${v.pp}</span>`,
    `<span class="badge ${agree ? "good" : "bad"}">${agree ? "ADF and KPSS agree" : "ADF and KPSS disagree: the pipeline follows ADF"}</span>`,
    `<span class="badge">AutoARIMA differences: d = ${st.d}</span>`,
    `<span class="badge">Seasonal strength: ${st.seasonal_strength === null ? "no seasonal period" : fmt(st.seasonal_strength, 2)}</span>`,
    `<span class="badge ${st.arima_passes_ljung_box ? "good" : "bad"}">Selected ARIMA ${st.arima_passes_ljung_box ? "passes" : "fails"} Ljung–Box</span>`,
  ].join("");
  el.notice.textContent = `What to notice: if the top series drifts while the bottom one moves around a constant level, differencing removed the drift. ` +
    `Here AutoARIMA uses d = ${st.d} difference${st.d === 1 ? "" : "s"}. ADF and KPSS test opposite null hypotheses, so they can disagree.`;
}

async function renderTable() {
  const all = await getJSON("stationarity.json");
  el.table.innerHTML = table(["Dataset", "ADF p", "KPSS p", "PP p", "ADF and KPSS", "d chosen", "Seasonal strength", "ARIMA passes Ljung–Box"],
    all.map((r) => ({ highlight: r.adf_kpss === "disagree",
      cells: [r.dataset, fmtP(r.adf_p), fmtP(r.kpss_p), fmtP(r.pp_p), r.adf_kpss, r.d, r.seasonal_strength === null ? "–" : fmt(r.seasonal_strength, 2),
              r.arima_passes_ljung_box ? "yes" : "no"] })), { numeric: [1, 2, 3, 5, 6] }) +
    `<p class="notice">ADF and Phillips–Perron: p below 0.05 means stationary. KPSS: p below 0.05 means not stationary (KPSS p-values are bounded to 0.01–0.10). ` +
    "When the two disagree, the pipeline takes d from ADF.</p>";
}

export async function init() {
  ["dataset", "badges", "chart", "notice", "table"].forEach((k) => { el[k] = document.getElementById(`st-${k}`); });
  fillSelect(el.dataset, (await datasetIndex()).map((d) => d.name), state.name);
  el.dataset.addEventListener("change", () => { state.name = el.dataset.value; render(); });
  onThemeChange(render);
  await Promise.all([render(), renderTable()]);
}
