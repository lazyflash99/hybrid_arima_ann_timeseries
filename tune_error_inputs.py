"""
Selects the number of past ARIMA errors given to the networks, for the hybrid (Zhang, Eq. 7: n past errors) and
for both extensions (r past errors next to the past values and L_hat_t), with the training data only:

  1. fit part = first 80 % of the training data, validation part = last 20 %
  2. for every value in GRID and every seed: train on the fit part, forecast the validation part one step ahead
  3. choose the value with the lowest mean validation MSE
  4. test errors for every value are printed as a sensitivity check only (never used to choose)

Hidden nodes, past-value inputs, training and margins are unchanged.
Current values: hybrid n = 4 / 7 / 7, extensions r = 2 / 3 / 3 (sunspot / lynx / exchange rate).
Usage:  python tune_error_inputs.py   (writes results_error_inputs.csv; --report re-prints from it)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from config import results_path
import pandas as pd

GRID = [1, 2, 3, 4, 7]
SEEDS = range(10)
VAL_FRAC = 0.2
MODELS = ["Hybrid", "Extension 1", "Extension 2"]


def current_value(model, cfg):
    return cfg["ann"][0] if model == "Hybrid" else cfg["ext_kwargs"]["n_residual_lags"]


def make(model, k, seed, cfg):
    from arima_model import AutoARIMA
    from hybrid_model import HybridModel
    from extension_model import ExtensionHybrid
    if model == "Hybrid":
        return HybridModel(AutoARIMA(verbose=False), n_inputs=k, n_hidden=cfg["ann"][1], seed=seed)
    ext = dict(cfg["ext_kwargs"], n_residual_lags=k, seed=seed)
    return ExtensionHybrid(AutoARIMA(verbose=False), scale_margin=0.2 if model == "Extension 2" else 0.0, **ext)


def run_one(job):
    from run_extension_experiments import EXPERIMENTS
    name, model, k, seed, stage = job
    cfg = EXPERIMENTS[name]
    y, n = cfg["load"](), cfg["n_train"]
    row = {"dataset": name, "model": model, "k": k, "seed": seed, "stage": stage}
    if stage == "validation":
        n_fit = int(round(n * (1 - VAL_FRAC)))
        f = make(model, k, seed, cfg).fit(y[:n_fit]).predict_one_step(y[:n])[n_fit:]
        a = y[n_fit:n]
        row["MSE"], row["MAD"] = np.mean((a - f) ** 2), np.mean(np.abs(a - f))
        return row
    f = make(model, k, seed, cfg).fit(y[:n]).predict_one_step(y)[n:]
    a = y[n:]
    for h_name, h in cfg["horizons"].items():
        row[f"{h_name} MSE"] = np.mean((a[:h] - f[:h]) ** 2)
        row[f"{h_name} MAD"] = np.mean(np.abs(a[:h] - f[:h]))
    return row


def report(df):
    from run_extension_experiments import EXPERIMENTS
    val = df[df.stage == "validation"].groupby(["dataset", "model", "k"])["MSE"].mean()
    test = df[df.stage == "test"]
    print("===== Step 1: validation MSE (last 20 % of training), mean of 10 seeds; [current], *chosen* =====")
    chosen = {}
    for name in EXPERIMENTS:
        scale = 1e5 if name == "Exchange rate" else 1.0
        print(f"\n{name}" + (" (x 1e-5)" if scale != 1 else ""))
        for model in MODELS:
            v = val.loc[(name, model)]
            chosen[(name, model)] = int(v.idxmin())
            cur = current_value(model, EXPERIMENTS[name])
            cells = []
            for k in GRID:
                txt = f"{v[k] * scale:.4g}"
                txt = f"[{txt}]" if k == cur else txt
                txt = f"*{txt}*" if k == chosen[(name, model)] else txt
                cells.append(f"k={k}: {txt:>12}")
            print(f"  {model:12} " + "  ".join(cells))

    print("\n===== Step 2: test errors, mean of 10 seeds (sensitivity only; [current], *chosen*) =====")
    for name, cfg in EXPERIMENTS.items():
        cols = [f"{h} {m}" for h in cfg["horizons"] for m in ("MSE", "MAD")]
        scale = {c: (1e5 if name == "Exchange rate" and c.endswith("MSE") else 1.0) for c in cols}
        print(f"\n{name}" + (" (MSE x 1e5)" if name == "Exchange rate" else ""))
        print(f"  {'':16}" + "".join(f"{c:>21}" for c in cols))
        for model in MODELS:
            means = test[(test.dataset == name) & (test.model == model)].groupby("k")[cols].mean()
            cur = current_value(model, cfg)
            for k in GRID:
                mark = ("[c]" if k == cur else "   ") + ("*" if k == chosen[(name, model)] else " ")
                print(f"  {model:12} k={k}{mark}" + "".join(f"{means.loc[k, c] * scale[c]:>21.5g}" for c in cols))


def main():
    from run_extension_experiments import EXPERIMENTS
    jobs = [(name, model, k, s, stage) for stage in ("validation", "test") for name in EXPERIMENTS
            for model in MODELS for k in GRID for s in SEEDS]
    with ProcessPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(run_one, jobs, chunksize=2))
    df = pd.DataFrame(rows)
    df.to_csv(results_path("results_error_inputs.csv"), index=False)
    report(df)


if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        report(pd.read_csv(results_path("results_error_inputs.csv")))
    else:
        main()
