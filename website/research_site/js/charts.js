// Chart helpers: one colour per model on every chart (colour-blind-safe palette), theme-aware layouts,
// and a registry so every chart is redrawn when the theme changes.

export const MODELS = ["ARIMA", "ANN", "Zhang", "Extension", "Extension margin", "E1", "E2", "E3", "E4", "E5", "E6", "E7"];

// Muted academic palette, chosen to stay distinguishable with colour-vision deficiency
export const MODEL_COLORS = {
  "ARIMA": "#4c8795",
  "ANN": "#a6783d",
  "Zhang": "#17637a",
  "Extension": "#b45f3c",
  "Extension margin": "#876b91",
  "E1": "#3e806d",
  "E2": "#b28a43",
  "E3": "#76556e",
  "E4": "#528b83",
  "E5": "#946b63",
  "E6": "#4d6582",
  "E7": "#6d9aa2",
};

export const MODEL_LABELS = {
  "Zhang": "Zhang's hybrid",
  "Extension": "Extension",
  "Extension margin": "Extension + margin",
};
export const label = (m) => MODEL_LABELS[m] || m;
export const color = (m) => MODEL_COLORS[m] || "#777777";

export function themeColors() {
  const css = getComputedStyle(document.documentElement);
  const v = (name) => css.getPropertyValue(name).trim();
  return { ink: v("--ink"), ink2: v("--ink-2"), line: v("--line"), surface: v("--surface"), surface2: v("--surface-2"),
           bad: v("--bad"), good: v("--good"), accent: v("--accent") };
}

const FONT = '"Segoe UI", system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif';

export function layout({ title, x, y, height = 420, extra = {} }) {
  const t = themeColors();
  const axis = (text) => ({ title: { text, font: { size: 15 } }, gridcolor: t.line, zerolinecolor: t.line, linecolor: t.line,
                            tickfont: { size: 14 }, automargin: true });
  return {
    title: { text: title, font: { size: 17 }, x: 0, xanchor: "left", pad: { l: 4 } },
    height,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: FONT, size: 14, color: t.ink },
    xaxis: axis(x),
    yaxis: axis(y),
    // legend about 58 px below the plot area, whatever the chart height
    legend: { orientation: "h", y: -58 / Math.max(height - 166, 120), yanchor: "top", x: 0, font: { size: 14 } },
    margin: { l: 70, r: 20, t: 56, b: 110 },
    hoverlabel: { font: { size: 14, family: FONT } },
    ...extra,
  };
}

export const CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };

export function plot(el, traces, lay) {
  return Plotly.react(el, traces, lay, CONFIG);
}

// sections register a redraw function; the theme toggle calls them all
const redraws = [];
export function onThemeChange(fn) { redraws.push(fn); }
export function redrawAll() { redraws.forEach((fn) => fn()); }

export function hexToRgba(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}
