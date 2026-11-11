// Models for the "Draw your own series" demo, in plain JavaScript with no dependencies (runs in the browser and in Node).
//   linear stage: AR(p) fitted by ordinary least squares, optionally on the first difference
//   neural stage: small MLP (logistic hidden units, linear output) on lagged residuals, trained with Adam
//   hybrid forecast: y_hat_t = L_hat_t + N_hat_t
// Everything is fitted on the first nTrain points only; forecasts at time t use values up to t-1 only.

export const TRAIN_FRACTION = 0.7;

// ---------- random numbers (fixed seeds) ----------

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng) {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

export function nTrainFor(n) {
  return Math.floor(TRAIN_FRACTION * n);
}

// ---------- small linear algebra ----------

// solves A x = b (A square) by Gaussian elimination with partial pivoting; a tiny ridge keeps flat inputs solvable
export function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) M[i][i] += 1e-9;
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    const d = M[c][c] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / d;
      if (f !== 0) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / (row[i] || 1e-12));
}

// ---------- linear stage: AR(p) by OLS ----------

// z is the modelled series: y itself, or its first difference (z_t = y_t - y_{t-1}, z_0 undefined)
function modelled(y, diff) {
  return diff ? y.map((v, t) => (t === 0 ? null : v - y[t - 1])) : y.slice();
}

// fits z_t = c + phi_1 z_{t-1} + ... + phi_p z_{t-p} on targets t < nTrain only
export function fitAR(y, { p, diff = false, nTrain }) {
  const z = modelled(y, diff);
  const start = p + (diff ? 1 : 0); // first t whose lags all exist
  const k = p + 1;
  const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty = new Array(k).fill(0);
  let rows = 0;
  for (let t = start; t < nTrain; t++) {
    const x = [1];
    for (let i = 1; i <= p; i++) x.push(z[t - i]);
    for (let a = 0; a < k; a++) {
      Xty[a] += x[a] * z[t];
      for (let b = 0; b < k; b++) XtX[a][b] += x[a] * x[b];
    }
    rows++;
  }
  if (rows < k) throw new Error(`not enough training points for AR(${p})`);
  const coef = solve(XtX, Xty);
  return { p, diff, start, nTrain, intercept: coef[0], phi: coef.slice(1), rows };
}

// one-step-ahead forecast L_hat_t from actual values before t; null where the lags do not exist
export function predictAR(model, y) {
  const z = modelled(y, model.diff);
  return y.map((_, t) => {
    if (t < model.start) return null;
    let v = model.intercept;
    for (let i = 1; i <= model.p; i++) v += model.phi[i - 1] * z[t - i];
    return model.diff ? y[t - 1] + v : v;
  });
}

export function residuals(y, fitted) {
  return y.map((v, t) => (fitted[t] === null ? null : v - fitted[t]));
}

// ---------- neural stage: MLP on lagged residuals ----------

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class MLP {
  constructor(nIn, nHidden, seed) {
    const rng = mulberry32(seed);
    const u = (lim) => (2 * rng() - 1) * lim;
    const l1 = Math.sqrt(6 / (nIn + nHidden));
    const l2 = Math.sqrt(6 / (nHidden + 1));
    this.nIn = nIn;
    this.nHidden = nHidden;
    this.W1 = Array.from({ length: nHidden }, () => Array.from({ length: nIn }, () => u(l1)));
    this.b1 = new Array(nHidden).fill(0);
    this.w2 = Array.from({ length: nHidden }, () => u(l2));
    this.b2 = 0;
    this.rng = mulberry32(seed + 7919); // batch order
    this.step = 0;
    this.m = this.zeros();
    this.v = this.zeros();
  }

  zeros() {
    return { W1: this.W1.map((r) => r.map(() => 0)), b1: this.b1.map(() => 0), w2: this.w2.map(() => 0), b2: 0 };
  }

  forward(x) {
    const h = this.W1.map((row, j) => sigmoid(row.reduce((s, w, i) => s + w * x[i], this.b1[j])));
    return { h, out: h.reduce((s, hj, j) => s + hj * this.w2[j], this.b2) };
  }

  predict(x) {
    return this.forward(x).out;
  }

  loss(X, Y) {
    if (!X.length) return null;
    let s = 0;
    for (let i = 0; i < X.length; i++) s += (this.predict(X[i]) - Y[i]) ** 2;
    return s / X.length;
  }

  // one pass over the training rows in shuffled mini-batches; Adam updates
  trainEpoch(X, Y, { lr = 0.01, batch = 16 } = {}) {
    const idx = X.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (let s = 0; s < idx.length; s += batch) {
      const part = idx.slice(s, s + batch);
      const g = this.zeros();
      for (const i of part) {
        const { h, out } = this.forward(X[i]);
        const d = (2 * (out - Y[i])) / part.length;
        g.b2 += d;
        for (let j = 0; j < this.nHidden; j++) {
          g.w2[j] += d * h[j];
          const dh = d * this.w2[j] * h[j] * (1 - h[j]);
          g.b1[j] += dh;
          for (let k = 0; k < this.nIn; k++) g.W1[j][k] += dh * X[i][k];
        }
      }
      this.adam(g, lr);
    }
    return this.loss(X, Y);
  }

  adam(g, lr) {
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    this.step++;
    const c1 = 1 - b1 ** this.step, c2 = 1 - b2 ** this.step;
    const upd = (val, grad, m, v) => {
      const mm = b1 * m + (1 - b1) * grad;
      const vv = b2 * v + (1 - b2) * grad * grad;
      return [val - (lr * (mm / c1)) / (Math.sqrt(vv / c2) + eps), mm, vv];
    };
    for (let j = 0; j < this.nHidden; j++) {
      for (let k = 0; k < this.nIn; k++) {
        [this.W1[j][k], this.m.W1[j][k], this.v.W1[j][k]] = upd(this.W1[j][k], g.W1[j][k], this.m.W1[j][k], this.v.W1[j][k]);
      }
      [this.b1[j], this.m.b1[j], this.v.b1[j]] = upd(this.b1[j], g.b1[j], this.m.b1[j], this.v.b1[j]);
      [this.w2[j], this.m.w2[j], this.v.w2[j]] = upd(this.w2[j], g.w2[j], this.m.w2[j], this.v.w2[j]);
    }
    [this.b2, this.m.b2, this.v.b2] = upd(this.b2, g.b2, this.m.b2, this.v.b2);
  }
}

// rows (t, e_{t-1..t-q}) -> e_t, scaled with the mean and standard deviation of the training residuals only
export function lagData(e, q, nTrain) {
  const train = e.slice(0, nTrain).filter((v) => v !== null);
  const mean = train.reduce((s, v) => s + v, 0) / train.length;
  const sd = Math.sqrt(train.reduce((s, v) => s + (v - mean) ** 2, 0) / train.length) || 1;
  const sc = (v) => (v - mean) / sd;
  const rows = [];
  for (let t = q; t < e.length; t++) {
    const lags = [];
    for (let i = 1; i <= q; i++) lags.push(e[t - i]);
    if (lags.some((v) => v === null) || e[t] === null) continue;
    rows.push({ t, x: lags.map(sc), y: sc(e[t]) });
  }
  const tr = rows.filter((r) => r.t < nTrain);
  const ho = rows.filter((r) => r.t >= nTrain);
  return {
    mean, sd, rows,
    Xtrain: tr.map((r) => r.x), Ytrain: tr.map((r) => r.y),
    Xheld: ho.map((r) => r.x), Yheld: ho.map((r) => r.y),
  };
}

// ---------- the whole hybrid ----------

// sets up both stages; call train() repeatedly, then result()
export function createHybrid(y, { p = 2, diff = false, hidden = 4, lags = 2, seed = 1, lr = 0.01 } = {}) {
  const n = y.length;
  const nTrain = nTrainFor(n);
  const ar = fitAR(y, { p, diff, nTrain });
  const Lhat = predictAR(ar, y);
  const e = residuals(y, Lhat);
  const data = lagData(e, lags, nTrain);
  if (data.Xtrain.length < 5) throw new Error("too few training rows for the network; lower the AR order or the lags");
  const net = new MLP(lags, hidden, seed);
  const history = [{ epoch: 0, train: net.loss(data.Xtrain, data.Ytrain), held: net.loss(data.Xheld, data.Yheld) }];
  let epoch = 0;

  function train(epochs = 1) {
    for (let i = 0; i < epochs; i++) {
      epoch++;
      const tl = net.trainEpoch(data.Xtrain, data.Ytrain, { lr });
      history.push({ epoch, train: tl, held: net.loss(data.Xheld, data.Yheld) });
    }
    return epoch;
  }

  function result() {
    const Nhat = new Array(n).fill(null);
    for (const r of data.rows) Nhat[r.t] = net.predict(r.x) * data.sd + data.mean;
    const yhat = y.map((_, t) => (Nhat[t] === null ? null : Lhat[t] + Nhat[t]));
    return { n, nTrain, ar, Lhat, e, Nhat, yhat, remaining: y.map((v, t) => (yhat[t] === null ? null : v - yhat[t])),
             metrics: compareMSE(y, Lhat, yhat, nTrain), epoch, history };
  }

  return { train, result, ar, net, data, get epoch() { return epoch; }, history };
}

// MSE of the linear model and of the hybrid over the same points (those where the hybrid exists), per part
export function compareMSE(y, Lhat, yhat, nTrain) {
  const part = (lo, hi) => {
    let sl = 0, sh = 0, k = 0;
    for (let t = lo; t < hi; t++) {
      if (yhat[t] === null) continue;
      sl += (y[t] - Lhat[t]) ** 2;
      sh += (y[t] - yhat[t]) ** 2;
      k++;
    }
    if (!k) return { linear: null, hybrid: null, change: null, n: 0 };
    const linear = sl / k, hybrid = sh / k;
    return { linear, hybrid, change: linear > 1e-12 ? (100 * (hybrid - linear)) / linear : null, n: k };
  };
  return { train: part(0, nTrain), held: part(nTrain, y.length) };
}

// "helped", "overfit" or "none", from the percent changes in MSE (negative = hybrid better)
export function verdict(m) {
  const tr = m.train.change, ho = m.held.change;
  if (ho === null || tr === null) return "none";
  if (ho <= -5) return "helped";
  if (tr < -1 && ho > 1) return "overfit";
  return "none";
}

export function acf(x, maxLag) {
  const v = x.filter((a) => a !== null);
  const n = v.length;
  const mean = v.reduce((s, a) => s + a, 0) / n;
  const den = v.reduce((s, a) => s + (a - mean) ** 2, 0);
  const out = [];
  for (let k = 1; k <= maxLag; k++) {
    let s = 0;
    for (let t = k; t < n; t++) s += (v[t] - mean) * (v[t - k] - mean);
    out.push(den > 0 ? s / den : 0);
  }
  return { values: out, n };
}
