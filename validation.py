"""
Evaluation protocols. The test part is never used to choose anything.

  fixed    Zhang's protocol: one fit on the training part, sizes from the registry, one-step forecasts on the test part.
  rolling  chronological train / validation / test: hyperparameters are chosen only on validation folds inside the
           training part, then an expanding-window walk-forward over the test part with a refit every k origins.
           Also reports iterated multi-step forecasts for h in MODES[mode]["multi_h"].

Usage:  python validation.py [--quick]
Writes the files of results/2_model_comparison/: a test MSE overview, the chosen hyperparameters, significance tests,
the model confidence set, mean and standard deviation per window, every run, and the raw forecasts.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import arima_model as am
from ann_model import NeuralNetwork
from config import MODES, VAL_FRAC, WORKERS, results_path
from data_loader import DATASETS
from extension_model import ExtensionHybrid
from extensions import (DecompositionHybrid, GatedHybrid, JointARNN, MultiplicativeHybrid, SkipCombination,
                        ValidatedCombination, WeightedCombination)
from hybrid_model import HybridModel

MODELS_FIXED = ["ARIMA", "ANN", "Zhang", "Extension", "Extension margin", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]
MODELS_ROLLING = ["ARIMA", "ANN", "Zhang", "Extension", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]
DETERMINISTIC = {"ARIMA"}


def build(model, d, p, seed, restarts=20):
    """p: dict with hidden, lags and (Extension only) margin, weight_decay, errors."""
    h, k = p.get("hidden"), p.get("lags")
    if model == "ARIMA":
        return am.AutoARIMA(verbose=False)
    if model == "ANN":
        return NeuralNetwork(k, h, seed=seed)
    if model == "Zhang":
        return HybridModel(am.AutoARIMA(verbose=False), k, h, seed=seed)
    if model in ("Extension", "Extension margin"):
        return ExtensionHybrid(am.AutoARIMA(verbose=False), n_series_lags=k, n_residual_lags=p["errors"], n_hidden=h,
                               weight_decay=p["weight_decay"], scale_margin=p["margin"], n_restarts=restarts, seed=seed)
    if model == "E1":
        return SkipCombination(k, p["errors"], h, seed=seed)
    cls = dict(E2=MultiplicativeHybrid, E3=DecompositionHybrid, E4=GatedHybrid, E5=JointARNN,
               E6=WeightedCombination, E7=ValidatedCombination)[model]
    return cls(k, h, seed=seed)


def registry_params(model, d):
    e = d["ext_kwargs"]
    if model == "ARIMA":
        return {}
    if model == "Extension":
        return dict(lags=e["n_series_lags"], errors=e["n_residual_lags"], hidden=e["n_hidden"],
                    weight_decay=e["weight_decay"], margin=0.0)
    if model == "Extension margin":
        return dict(registry_params("Extension", d), margin=0.2)
    if model == "E1":
        return dict(lags=e["n_series_lags"], errors=e["n_residual_lags"], hidden=e["n_hidden"])
    return dict(lags=d["ann"][0], hidden=d["ann"][1])


def param_grid(model, d, grid):
    if model == "ARIMA":
        return [{}]
    lags = sorted(set(grid["lags"] + ([d["period"]] if 1 < d["period"] <= 12 else [])))
    errors = d["ext_kwargs"]["n_residual_lags"]
    if model == "Extension":
        return [dict(hidden=h, lags=k, margin=m, weight_decay=w, errors=errors)
                for h, k, m, w in itertools.product(grid["hidden"], lags, grid["margin"], grid["weight_decay"])]
    extra = dict(errors=errors) if model == "E1" else {}
    return [dict(hidden=h, lags=k, **extra) for h, k in itertools.product(grid["hidden"], lags)]


# ------------------------------------------------------------------------------- splits
def validation_folds(n_train, n_folds, val_frac=VAL_FRAC):
    """Chronological (fit_end, val_end) pairs inside the training part: fit on [0, fit_end), score on [fit_end, val_end)."""
    n_val = max(n_folds, int(round(n_train * val_frac)))
    edges = np.linspace(n_train - n_val, n_train, n_folds + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def refit_origins(n_train, n_total, n_refits):
    """Test indices at which the model is refitted: n_train first, then evenly spaced."""
    n_test = n_total - n_train
    k = max(1, n_test // max(1, n_refits))
    return [n_train + i * k for i in range(n_refits) if n_train + i * k < n_total]


# ------------------------------------------------------------------------------- forecasting
def walk_forward(make_model, y, n_train, n_refits):
    """One-step forecasts for y[n_train:]; each block uses a model fitted only on data before the block."""
    origins = refit_origins(n_train, len(y), n_refits)
    out, fitted = np.full(len(y) - n_train, np.nan), []
    for i, start in enumerate(origins):
        end = origins[i + 1] if i + 1 < len(origins) else len(y)
        model = make_model().fit(y[:start])
        out[start - n_train:end - n_train] = model.predict_one_step(y[:end])[start:end]
        fitted.append((start, end, model))
    return out, fitted


def iterated_forecast(model, history, h):
    """h-step forecast by feeding forecasts back in. The placeholder at the end never influences its own forecast."""
    z = list(history)
    for _ in range(h):
        z.append(z[-1])
        z[-1] = float(model.predict_one_step(np.asarray(z))[-1])
    return z[-1]


def metrics(a, f, train):
    e = a - f
    prev = np.r_[train[-1], a[:-1]]
    naive = np.mean(np.abs(np.diff(train)))
    rw_mse = np.mean((a - prev) ** 2)
    return dict(MSE=np.mean(e ** 2), MAD=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)),
                MAPE=100 * np.mean(np.abs(e) / np.maximum(np.abs(a), 1e-8)),
                sMAPE=100 * np.mean(2 * np.abs(e) / np.maximum(np.abs(a) + np.abs(f), 1e-8)),
                MASE=np.mean(np.abs(e)) / naive, TheilU2=np.sqrt(np.mean(e ** 2) / rw_mse),
                DA=100 * np.mean(np.sign(f - prev) == np.sign(a - prev)))


# ------------------------------------------------------------------------------- jobs (run in worker processes)
def tune_job(job):
    name, model, mode = job
    cfg, d = MODES[mode], DATASETS[name]
    am.DEFAULT_SUBSET = d["arima_subset"]
    y = d["load"]()[:d["n_train"]]                    # only the training part reaches this function
    folds, best = validation_folds(len(y), cfg["val_folds"]), (np.inf, None)
    for p in param_grid(model, d, cfg["grid"]):
        try:
            errs = []
            for fit_end, val_end in folds:
                m = build(model, d, p, 0, cfg["restarts_tune"]).fit(y[:fit_end])
                errs.append(np.mean((y[fit_end:val_end] - m.predict_one_step(y[:val_end])[fit_end:val_end]) ** 2))
            score = float(np.mean(errs))
        except Exception:
            score = np.nan
        if np.isfinite(score) and score < best[0]:
            best = (score, p)
    return dict(dataset=name, model=model, val_mse=best[0], params=best[1])


def final_job(job):
    protocol, name, model, seed, params, mode = job
    cfg, d = MODES[mode], DATASETS[name]
    am.DEFAULT_SUBSET = d["arima_subset"]
    y, n = d["load"](), d["n_train"]
    head = dict(protocol=protocol, dataset=name, model=model, seed=seed)
    start = time.time()
    try:
        restarts = 20 if protocol == "fixed" else cfg["restarts_final"]
        n_refits = 1 if protocol == "fixed" else cfg["n_refits"]
        f, fitted = walk_forward(lambda: build(model, d, params, seed, restarts), y, n, n_refits)
        if not np.isfinite(f).all():
            raise ValueError("non-finite forecasts")
    except Exception as ex:
        return head, None, [dict(head, status=f"not run: {type(ex).__name__}: {ex}")]
    a, train = y[n:], y[:n]
    rows = []
    for win, h in {**d["horizons"], "all test points": len(a)}.items():
        rows.append(dict(head, status="ok", window=win, h=1, **metrics(a[:h], f[:h], train)))
    if protocol == "rolling":
        origins = np.unique(np.linspace(n, len(y) - max(cfg["multi_h"]), min(30, len(a) - max(cfg["multi_h"]))).astype(int))
        for h in cfg["multi_h"]:
            fc, act = [], []
            for t in origins:
                model_t = next(m for s, e, m in fitted if s <= t < e)
                fc.append(iterated_forecast(model_t, y[:t], h)); act.append(y[t + h - 1])
            if np.isfinite(fc).all():
                rows.append(dict(head, status="ok", window=f"iterated h={h}", h=h, **metrics(np.array(act), np.array(fc), train)))
    for r in rows:
        r["seconds"] = time.time() - start
    return head, f, rows


# ------------------------------------------------------------------------------- statistics
def diebold_mariano(e1, e2):
    """Squared-error DM test, h = 1, Harvey-Leybourne-Newbold correction. Positive = model 2 has the lower loss."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if np.allclose(d, 0):
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(d.var(ddof=1) / n) * np.sqrt((n - 1) / n)
    return float(stat), float(2 * student_t.sf(abs(stat), n - 1))


def model_confidence_set(losses, alpha=0.1, boot=1000, block=5, seed=0):
    """Hansen's MCS, range statistic, circular block bootstrap. losses: dict name -> per-step loss."""
    names, rng = list(losses), np.random.RandomState(seed)
    L = np.column_stack([losses[k] for k in names])
    n, keep, p_mcs = len(L), list(range(len(names))), {}
    idx = np.array([np.r_[[(s + j) % n for s in rng.randint(0, n, n // block + 1) for j in range(block)]][:n] for _ in range(boot)])
    max_p = 0.0
    while len(keep) > 1:
        X = L[:, keep]
        bar = X.mean(0)
        boot_bar = X[idx].mean(1)
        d_ij = bar[:, None] - bar[None, :]
        var = ((boot_bar[:, :, None] - boot_bar[:, None, :] - d_ij) ** 2).mean(0) + np.eye(len(keep))
        t = np.abs(d_ij) / np.sqrt(var)
        boot_t = np.abs(boot_bar[:, :, None] - boot_bar[:, None, :] - d_ij) / np.sqrt(var)
        max_p = max(max_p, float((boot_t.max((1, 2)) >= t.max()).mean()))
        worst = int(np.argmax((d_ij / np.sqrt(var)).max(1)))
        p_mcs[names[keep[worst]]] = max_p
        keep.pop(worst)
    p_mcs[names[keep[0]]] = 1.0
    return {k: (p_mcs[k], p_mcs[k] >= alpha) for k in names}


def compare(forecasts, boot):
    """DM against Zhang and ARIMA and the MCS, on errors averaged over the seeds of each model (h = 1)."""
    dm, mcs = [], []
    for (protocol, name), g in forecasts.groupby_keys().items():
        d = DATASETS[name]
        a = d["load"]()[d["n_train"]:]
        err = {m: a - np.mean(fs, axis=0) for m, fs in g.items()}
        loss = {m: e ** 2 for m, e in err.items()}
        for m, e in err.items():
            for ref in ("Zhang", "ARIMA"):
                if ref in err and m != ref:
                    s, p = diebold_mariano(err[ref], e)
                    dm.append(dict(protocol=protocol, dataset=name, model=m, versus=ref, dm_stat=s, p_value=p,
                                   model_mse=np.mean(e ** 2), ref_mse=np.mean(err[ref] ** 2)))
        for m, (p, kept) in model_confidence_set(loss, boot=boot).items():
            mcs.append(dict(protocol=protocol, dataset=name, model=m, mcs_p=p, in_90pct_set=kept))
    return pd.DataFrame(dm), pd.DataFrame(mcs)


def write_overview():
    """Mean test MSE over seeds, all test points, one row per protocol and dataset and one column per model."""
    df = pd.read_csv(results_path("validation_results.csv"))
    ok = df[(df.status == "ok") & (df.window == "all test points")]
    table = ok.groupby(["protocol", "dataset", "model"], sort=False)["MSE"].mean().unstack("model")
    table.reset_index().to_csv(results_path("validation_overview.csv"), index=False)
    return table


class Forecasts:
    def __init__(self):
        self.store = {}

    def add(self, protocol, name, model, seed, f):
        self.store.setdefault((protocol, name), {}).setdefault(model, {})[seed] = f

    def groupby_keys(self):
        return {k: {m: [v[s] for s in sorted(v)] for m, v in g.items()} for k, g in self.store.items()}

    def save(self):
        for protocol in {k[0] for k in self.store}:
            arrays = {f"{name}|{m}|{s}": f for (p, name), g in self.store.items() if p == protocol
                      for m, v in g.items() for s, f in v.items()}
            np.savez_compressed(results_path(f"forecasts_{protocol}.npz"), **arrays)


def main(mode="full"):
    cfg = MODES[mode]
    names = cfg["datasets"] or list(DATASETS)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        tuned = list(pool.map(tune_job, [(n, m, mode) for n in names for m in MODELS_ROLLING]))
        print(f"tuning finished in {time.time() - t0:.0f} s", flush=True)
        # a model whose every validation fit failed keeps its registry sizes (no test data involved)
        chosen = {(r["dataset"], r["model"]): (r["params"] if r["params"] is not None
                                               else registry_params(r["model"], DATASETS[r["dataset"]])) for r in tuned}
        jobs = []
        for n in names:
            for m in MODELS_FIXED:
                jobs += [("fixed", n, m, s, registry_params(m, DATASETS[n]), mode) for s in ([0] if m in DETERMINISTIC else cfg["seeds"])]
            for m in MODELS_ROLLING:
                jobs += [("rolling", n, m, s, chosen[(n, m)], mode) for s in ([0] if m in DETERMINISTIC else cfg["seeds"])]
        results = list(pool.map(final_job, jobs, chunksize=2))
    print(f"evaluation finished in {time.time() - t0:.0f} s", flush=True)
    store, rows = Forecasts(), []
    for head, f, rs in results:
        rows += rs
        if f is not None:
            store.add(head["protocol"], head["dataset"], head["model"], head["seed"], f)
    store.save()
    df = pd.DataFrame(rows)
    df.to_csv(results_path("validation_results.csv"), index=False)
    ok = df[df.status == "ok"]
    measures = ["MSE", "MAD", "RMSE", "MAPE", "sMAPE", "MASE", "TheilU2", "DA"]
    summary = ok.groupby(["protocol", "dataset", "model", "window", "h"], sort=False)[measures].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(results_path("validation_summary.csv"), index=False)
    pd.DataFrame([dict(r, params=str(r["params"])) for r in tuned]).to_csv(results_path("validation_tuning.csv"), index=False)
    dm, mcs = compare(store, cfg["mcs_boot"])
    dm.to_csv(results_path("validation_dm.csv"), index=False)
    mcs.to_csv(results_path("validation_mcs.csv"), index=False)
    bad = df[df.status != "ok"].groupby(["protocol", "dataset", "model", "status"]).size()
    print("runs not completed:\n" + (bad.to_string() if len(bad) else "none"))
    table = write_overview()
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(5).to_string())
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main("quick" if "--quick" in sys.argv else "full")
