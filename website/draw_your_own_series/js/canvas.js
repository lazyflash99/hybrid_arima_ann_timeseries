// Drawing canvas: mouse strokes, presets, undo/clear, and conversion of the drawing to N evenly spaced values.
// Coordinates are stored normalised to [0, 1]; the value axis runs from 0 (bottom) to 100 (top).

import { gaussian, mulberry32 } from "./draw_models.js";

const BINS = 400;          // fine bins used to turn strokes into a function of time
const GAP_WARN = 0.04;     // warn when an empty stretch is longer than 4% of the drawn span
const BACK_WARN = 0.02;    // warn when strokes run backwards by more than 2% of the width in total

export const PRESETS = {
  "Sine + trend": (x) => 30 + 14 * Math.sin(2 * Math.PI * 4.5 * x) + 40 * x,
  "Sawtooth": (x) => 22 + 56 * ((5 * x) % 1),
  "Random walk": (() => {
    const rng = mulberry32(7);
    const w = [0];
    for (let i = 1; i < 300; i++) w.push(w[i - 1] + gaussian(rng));
    const lo = Math.min(...w), hi = Math.max(...w);
    return (x) => 15 + (70 * (w[Math.min(299, Math.round(x * 299))] - lo)) / (hi - lo);
  })(),
  "Quiet then spike": (() => {
    const rng = mulberry32(21);
    const wiggle = Array.from({ length: 300 }, () => gaussian(rng));
    return (x) => 35 + 1.5 * wiggle[Math.min(299, Math.round(x * 299))] + 45 * Math.exp(-(((x - 0.86) / 0.025) ** 2));
  })(),
};

export function createCanvas(el, onChange) {
  let strokes = [];
  let current = null;
  const ctx = el.getContext("2d");

  function resize() {
    const r = el.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    el.width = Math.round(r.width * dpr);
    el.height = Math.round(r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function draw() {
    const w = el.clientWidth, h = el.clientHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = css("--grid");
    ctx.lineWidth = 1;
    for (let i = 1; i < 10; i++) {
      ctx.beginPath(); ctx.moveTo((i * w) / 10, 0); ctx.lineTo((i * w) / 10, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, (i * h) / 10); ctx.lineTo(w, (i * h) / 10); ctx.stroke();
    }
    ctx.fillStyle = css("--ink-2");
    ctx.font = "14px 'Segoe UI', system-ui, sans-serif";
    ctx.fillText("value ↑", 8, 18);
    ctx.fillText("time →", w - 58, h - 10);
    if (!strokes.length && !current) {
      ctx.font = "17px 'Segoe UI', system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Hold the mouse button and draw a curve from left to right, or pick a preset", w / 2, h / 2);
      ctx.textAlign = "left";
    }
    ctx.strokeStyle = css("--ink");
    ctx.lineWidth = 2.5;
    ctx.lineJoin = ctx.lineCap = "round";
    for (const s of current ? [...strokes, current] : strokes) {
      ctx.beginPath();
      s.forEach(([x, y], i) => (i ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h)));
      ctx.stroke();
    }
  }

  function pos(ev) {
    const r = el.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height))];
  }

  el.addEventListener("mousedown", (ev) => {
    if (ev.button !== 0) return;
    current = [pos(ev)];
    draw();
  });
  el.addEventListener("mousemove", (ev) => {
    if (!current) return;
    current.push(pos(ev));
    draw();
  });
  const finish = () => {
    if (!current) return;
    if (current.length > 1) strokes.push(current);
    current = null;
    draw();
    onChange();
  };
  window.addEventListener("mouseup", finish);
  new ResizeObserver(resize).observe(el);

  return {
    redraw: draw,
    get strokes() { return strokes; },
    setStrokes(s) { strokes = s; draw(); onChange(); },
    undo() { strokes = strokes.slice(0, -1); draw(); onChange(); },
    clear() { strokes = []; draw(); onChange(); },
    preset(name) {
      const f = PRESETS[name];
      const pts = Array.from({ length: 300 }, (_, i) => {
        const x = 0.01 + (0.98 * i) / 299;
        return [x, 1 - f(x) / 100];
      });
      strokes = [pts];
      draw();
      onChange();
    },
  };
}

// strokes -> { values (length n, 0-100 scale), warnings[] } or { values: null, warnings } if there is too little drawn
export function toSeries(strokes, n) {
  const pts = strokes.flat();
  const warnings = [];
  if (pts.length < 5) return { values: null, warnings: [] };
  let back = 0;
  for (const s of strokes) for (let i = 1; i < s.length; i++) back += Math.max(0, s[i - 1][0] - s[i][0]);
  if (back > BACK_WARN || strokes.length > 1 && overlaps(strokes)) {
    warnings.push("Part of the drawing runs backwards or overlaps itself in time; where it does, the heights are averaged.");
  }
  const xs = pts.map((p) => p[0]);
  const lo = Math.min(...xs), hi = Math.max(...xs);
  if (hi - lo < 0.2) return { values: null, warnings: ["Draw across at least a fifth of the canvas width."] };
  // average the height in fine bins, then fill empty bins by straight lines
  const sum = new Array(BINS).fill(0), cnt = new Array(BINS).fill(0);
  for (const [x, y] of pts) {
    const b = Math.min(BINS - 1, Math.floor(((x - lo) / (hi - lo)) * BINS));
    sum[b] += 100 * (1 - y);
    cnt[b]++;
  }
  const known = [];
  for (let b = 0; b < BINS; b++) if (cnt[b]) known.push([b, sum[b] / cnt[b]]);
  let longest = 0;
  for (let i = 1; i < known.length; i++) longest = Math.max(longest, known[i][0] - known[i - 1][0] - 1);
  if (longest / BINS > GAP_WARN) warnings.push("There is a gap in the drawing; it was filled with a straight line.");
  const at = (b) => {
    let j = known.findIndex((k) => k[0] >= b);
    if (j === -1) return known[known.length - 1][1];
    if (j === 0 || known[j][0] === b) return known[j][1];
    const [b0, v0] = known[j - 1], [b1, v1] = known[j];
    return v0 + ((v1 - v0) * (b - b0)) / (b1 - b0);
  };
  const values = Array.from({ length: n }, (_, i) => at((i * (BINS - 1)) / (n - 1)));
  return { values, warnings };
}

function overlaps(strokes) {
  const spans = strokes.map((s) => [Math.min(...s.map((p) => p[0])), Math.max(...s.map((p) => p[0]))]).sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < spans.length; i++) if (spans[i][0] < spans[i - 1][1] - 0.01) return true;
  return false;
}

// adds noise with standard deviation level * sd(series), from a fixed seed
export function addNoise(values, level, seed) {
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const sd = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length);
  const rng = mulberry32(1000 + seed);
  return values.map((v) => v + level * sd * gaussian(rng));
}
