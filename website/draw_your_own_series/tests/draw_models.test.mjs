// Unit tests for js/draw_models.js.   Run:  node --test website/draw_your_own_series/tests/
import assert from "node:assert/strict";
import test from "node:test";

import { MLP, createHybrid, fitAR, gaussian, lagData, mulberry32, predictAR } from "../js/draw_models.js";

function ar2(n, seed) {
  const rng = mulberry32(seed);
  const y = [0, 0];
  for (let t = 2; t < n; t++) y.push(0.6 * y[t - 1] - 0.3 * y[t - 2] + gaussian(rng));
  return y;
}

test("AR(2) by least squares recovers the coefficients of a known AR(2) series", () => {
  const y = ar2(5000, 11);
  const m = fitAR(y, { p: 2, nTrain: y.length });
  console.log(`    fitted phi = [${m.phi.map((v) => v.toFixed(3)).join(", ")}], true [0.6, -0.3], intercept ${m.intercept.toFixed(3)}`);
  assert.ok(Math.abs(m.phi[0] - 0.6) < 0.05);
  assert.ok(Math.abs(m.phi[1] + 0.3) < 0.05);
  assert.ok(Math.abs(m.intercept) < 0.1);
});

test("AR with differencing recovers the coefficient of a known ARI(1,1) series", () => {
  const rng = mulberry32(5);
  const y = [0, 0];
  for (let t = 2; t < 5000; t++) y.push(y[t - 1] + 0.5 * (y[t - 1] - y[t - 2]) + gaussian(rng));
  const m = fitAR(y, { p: 1, diff: true, nTrain: y.length });
  console.log(`    fitted phi on differences = ${m.phi[0].toFixed(3)}, true 0.5`);
  assert.ok(Math.abs(m.phi[0] - 0.5) < 0.05);
});

test("the MLP lowers its training loss on a known nonlinear residual", () => {
  const rng = mulberry32(3);
  const e = [0.1];
  for (let t = 1; t < 300; t++) e.push(0.9 * Math.sin(3 * e[t - 1]) + 0.1 * gaussian(rng));
  const d = lagData(e, 1, 210);
  const net = new MLP(1, 6, 1);
  const before = net.loss(d.Xtrain, d.Ytrain);
  let after = before;
  for (let i = 0; i < 300; i++) after = net.trainEpoch(d.Xtrain, d.Ytrain, { lr: 0.01 });
  console.log(`    training loss (scaled units): ${before.toFixed(4)} -> ${after.toFixed(4)}`);
  assert.ok(after < 0.5 * before);
});

test("no look-ahead: changing a future point does not change earlier forecasts", () => {
  const rng = mulberry32(9);
  const y = Array.from({ length: 100 }, (_, t) => 50 + 20 * Math.sin(t / 5) + 3 * gaussian(rng));
  const k = 85; // a held-out point (training is the first 70)
  const run = (series) => {
    const h = createHybrid(series, { p: 3, hidden: 5, lags: 3, seed: 2 });
    h.train(100);
    return h.result();
  };
  const a = run(y);
  const changed = y.slice();
  changed[k] += 40;
  const b = run(changed);
  for (let t = 0; t <= k; t++) {
    assert.equal(a.Lhat[t], b.Lhat[t], `L_hat changed at t=${t}`);
    assert.equal(a.yhat[t], b.yhat[t], `hybrid forecast changed at t=${t}`);
  }
  assert.notEqual(a.yhat[k + 1], b.yhat[k + 1], "the point after the change should see it");
  console.log(`    changed y[${k}]: forecasts for t <= ${k} identical, forecast at t = ${k + 1} changed`);
});

test("no look-ahead inside the AR forecast: with fixed coefficients, L_hat_t ignores y_t and later", () => {
  const y = ar2(200, 4);
  const m = fitAR(y, { p: 2, nTrain: 140 });
  const changed = y.slice();
  changed[60] = 1e6;
  const a = predictAR(m, y), b = predictAR(m, changed);
  for (let t = 0; t <= 60; t++) assert.equal(a[t], b[t]);
  assert.notEqual(a[61], b[61]);
});

test("fitting uses only the first 70%: changing held-out points leaves both stages unchanged", () => {
  const rng = mulberry32(12);
  const y = Array.from({ length: 120 }, (_, t) => 30 + t * 0.2 + 5 * Math.sin(t / 3) + gaussian(rng));
  const z = y.map((v, t) => (t >= 84 ? v + 100 * rng() : v));
  const ha = createHybrid(y, { p: 2, hidden: 3, lags: 2, seed: 4 });
  const hb = createHybrid(z, { p: 2, hidden: 3, lags: 2, seed: 4 });
  ha.train(50); hb.train(50);
  assert.deepEqual(ha.ar.phi, hb.ar.phi);
  assert.equal(ha.ar.intercept, hb.ar.intercept);
  assert.deepEqual(ha.net.W1, hb.net.W1);
  assert.equal(ha.data.mean, hb.data.mean);
  assert.equal(ha.data.sd, hb.data.sd);
});

test("fixed seeds give identical results", () => {
  const y = ar2(150, 8).map((v) => v + 10);
  const r1 = createHybrid(y, { seed: 3 }); r1.train(40);
  const r2 = createHybrid(y, { seed: 3 }); r2.train(40);
  assert.deepEqual(r1.result().yhat, r2.result().yhat);
});
