// Formatting and control helpers shared by the sections.

export function fmt(v, digits = 4) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return Number(v.toPrecision(digits)).toString();
}

export function fmtP(p) {
  if (p === null || p === undefined || !Number.isFinite(p)) return "–";
  return p < 0.001 ? "< 0.001" : p.toFixed(3);
}

// "p = 0.042" or "p < 0.001"
export function pEq(p) {
  if (p === null || p === undefined || !Number.isFinite(p)) return "p = –";
  return p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`;
}

// plain-language reading of a Diebold-Mariano result (positive statistic = model has the lower error)
export function dmReading(stat, p, ref) {
  if (p === null || p === undefined || !Number.isFinite(p)) return "not tested";
  if (p >= 0.05) return `not significantly different from ${ref}`;
  return stat > 0 ? `significantly better than ${ref}` : `significantly worse than ${ref}`;
}

export function fillSelect(el, options, value) {
  el.innerHTML = "";
  for (const o of options) {
    const opt = document.createElement("option");
    opt.value = typeof o === "string" ? o : o.value;
    opt.textContent = typeof o === "string" ? o : o.label;
    el.appendChild(opt);
  }
  if (value !== undefined) el.value = value;
}

// a group of toggle buttons where exactly one is pressed
export function segmented(el, onChange) {
  const buttons = [...el.querySelectorAll("button")];
  const set = (value) => buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.value === value)));
  buttons.forEach((b) => b.addEventListener("click", () => { set(b.dataset.value); onChange(b.dataset.value); }));
  return { set, get: () => buttons.find((b) => b.getAttribute("aria-pressed") === "true").dataset.value };
}

export function table(headers, body, opts = {}) {
  const numeric = opts.numeric || [];
  const th = headers.map((h, i) => `<th${numeric.includes(i) ? ' class="num"' : ""} scope="col">${h}</th>`).join("");
  const tr = body.map((r) => {
    const cls = r.highlight ? ' class="highlight"' : "";
    const cells = (r.cells || r).map((c, i) => `<td${numeric.includes(i) ? ' class="num"' : ""}>${c}</td>`).join("");
    return `<tr${cls}>${cells}</tr>`;
  }).join("");
  return `<table class="data"><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table>`;
}

export const range = (n, start = 0) => Array.from({ length: n }, (_, i) => i + start);
