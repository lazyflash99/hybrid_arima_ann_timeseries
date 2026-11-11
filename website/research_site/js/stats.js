// Small statistics used to draw the residual charts from the saved errors.

export const mean = (x) => x.reduce((a, b) => a + b, 0) / x.length;

export function acf(x, maxLag) {
  const m = mean(x);
  const d = x.map((v) => v - m);
  const denom = d.reduce((a, v) => a + v * v, 0);
  const out = [];
  for (let k = 1; k <= maxLag; k++) {
    let s = 0;
    for (let t = k; t < d.length; t++) s += d[t] * d[t - k];
    out.push(s / denom);
  }
  return out;
}

// inverse of the standard normal distribution function (Acklam's rational approximation, error below 1.2e-9)
export function normInv(p) {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const lo = 0.02425, hi = 1 - lo;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > hi) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// normal QQ points: theoretical quantiles against sorted values, plus the line through the quartiles
export function qq(x) {
  const s = [...x].sort((u, v) => u - v);
  const n = s.length;
  const theo = s.map((_, i) => normInv((i + 0.5) / n));
  const q = (arr, p) => arr[Math.min(arr.length - 1, Math.max(0, Math.round(p * (arr.length - 1))))];
  const slope = (q(s, 0.75) - q(s, 0.25)) / (q(theo, 0.75) - q(theo, 0.25));
  const icpt = q(s, 0.25) - slope * q(theo, 0.25);
  return { theo, sample: s, slope, icpt };
}

export const diff = (x) => x.slice(1).map((v, i) => v - x[i]);
export const finite = (x) => x.filter((v) => v !== null && Number.isFinite(v));
