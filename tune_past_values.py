"""
Selects the number of past values given to the networks, for the stand-alone ANN (paper: 4 / 7 / 7) and for
both extensions (past values next to L_hat_t and the past errors), with the training data only:

  1. fit part = first 80 % of the training data, validation part = last 20 %
  2. for every value in GRID and every seed: train on the fit part, forecast the validation part one step ahead
  3. choose the value with the lowest mean validation MSE
  4. test errors for every value are printed as a sensitivity check only (never used to choose)

Hidden nodes, past-error inputs, training and margins are unchanged.
Usage:  python tune_past_values.py   (writes results_past_values.csv; --report re-prints from it)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from config import results_path
import pandas as pd

GRID = [2, 4, 7, 10]
SEEDS = range(10)
VAL_FRAC = 0.2
MODELS = ["ANN alone", "Extension 1", "Extension 1 Improved"]


def make(model, k, seed, cfg):
    from arima_model import AutoARIMA
    from ann_model import NeuralNetwork
    from extension_model import ExtensionHybrid
    if model == "ANN alone":
        return NeuralNetwork(k, cfg["ann"][1], seed=seed)
    ext = dict(cfg["ext_kwargs"], n_series_lags=k, seed=seed)
    return ExtensionHybrid(AutoARIMA(verbose=False), scale_margin=0.2 if "Improved" in model else 0.0, **ext)


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
    for name, cfg in EXPERIMENTS.items():
        scale = 1e5 if name == "Exchange rate" else 1.0
        print(f"\n{name}" + (" (x 1e-5)" if scale != 1 else ""))
        for model in MODELS:
            v = val.loc[(name, model)]
            chosen[(name, model)] = int(v.idxmin())
            cells = []
            for k in GRID:
                txt = f"{v[k] * scale:.4g}"
                txt = f"[{txt}]" if k == cfg["ann"][0] else txt
                txt = f"*{txt}*" if k == chosen[(name, model)] else txt
                cells.append(f"k={k}: {txt:>12}")
            print(f"  {model:22} " + "  ".join(cells))

    print("\n===== Step 2: test errors, mean of 10 seeds (sensitivity only; [c] current, * chosen) =====")
    for name, cfg in EXPERIMENTS.items():
        cols = [f"{h} {m}" for h in cfg["horizons"] for m in ("MSE", "MAD")]
        scale = {c: (1e5 if name == "Exchange rate" and c.endswith("MSE") else 1.0) for c in cols}
        print(f"\n{name}" + (" (MSE x 1e5)" if name == "Exchange rate" else ""))
        print(f"  {'':28}" + "".join(f"{c:>21}" for c in cols))
        for model in MODELS:
            means = test[(test.dataset == name) & (test.model == model)].groupby("k")[cols].mean()
            for k in GRID:
                mark = ("[c]" if k == cfg["ann"][0] else "   ") + ("*" if k == chosen[(name, model)] else " ")
                print(f"  {model:22} k={k:<2}{mark}" + "".join(f"{means.loc[k, c] * scale[c]:>21.5g}" for c in cols))

    # overall: extensions with validation-chosen past values vs the current hybrid (results_extensions.csv)
    hyb = pd.read_csv(results_path("results_extensions.csv"))
    hyb = hyb[hyb.model == "Zhang hybrid"]
    for model in MODELS[1:]:
        for label, pick in (("current", lambda d: EXPERIMENTS[d]["ann"][0]),
                            ("validation-chosen", lambda d: chosen[(d, model)])):
            wins = total = 0
            for name in EXPERIMENTS:
                cols = [c for c in test.columns if c.endswith(("MSE", "MAD")) and test[test.dataset == name][c].notna().any()]
                m = test[(test.dataset == name) & (test.model == model) & (test.k == pick(name))][cols].mean()
                h = hyb[hyb.dataset == name][cols].mean()
                wins += int((m < h).sum())
                total += len(cols)
            print(f"{model:22} {label:18} past values: lower error than the hybrid in {wins}/{total}")


def main():
    from run_extension_experiments import EXPERIMENTS
    jobs = [(name, model, k, s, stage) for stage in ("validation", "test") for name in EXPERIMENTS
            for model in MODELS for k in GRID for s in SEEDS]
    with ProcessPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(run_one, jobs, chunksize=2))
    df = pd.DataFrame(rows)
    df.to_csv(results_path("results_past_values.csv"), index=False)
    report(df)


if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        report(pd.read_csv(results_path("results_past_values.csv")))
    else:
        main()
