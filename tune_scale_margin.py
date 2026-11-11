"""
Selects the scale margin m of Extension 2 using the training data only.

  1. Split each training set into a fit part (first 80 %) and a validation part (last 20 %).
  2. For every m in GRID and every seed, train Extension 2 on the fit part and forecast the validation part
     one step ahead, using actual past values.
  3. Choose m with the lowest mean validation MSE over the seeds.
  4. Retrain on the full training set with that m and evaluate once on the test set.

The test errors for every m are also printed, as a sensitivity check only: they are never used to choose m.
Usage:  python tune_scale_margin.py   (writes results_margin_tuning.csv)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # one thread per worker process

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from config import results_path
import pandas as pd

GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
SEEDS = range(10)
VAL_FRAC = 0.2


def run_one(job):
    """Runs one fit for (dataset, margin, seed, stage) and returns its errors. Imports are local to each worker process."""
    from arima_model import AutoARIMA
    from extension_model import ExtensionHybrid
    from run_extension_experiments import EXPERIMENTS
    name, m, seed, stage = job
    cfg = EXPERIMENTS[name]
    y = cfg["load"]()
    n = cfg["n_train"]
    ext = dict(cfg["ext_kwargs"], seed=seed)
    model = ExtensionHybrid(AutoARIMA(verbose=False), scale_margin=m, **ext)
    if stage == "validation":
        n_fit = int(round(n * (1 - VAL_FRAC)))
        f = model.fit(y[:n_fit]).predict_one_step(y[:n])[n_fit:]
        a = y[n_fit:n]
        return dict(dataset=name, m=m, seed=seed, stage=stage,
                    MSE=np.mean((a - f) ** 2), MAD=np.mean(np.abs(a - f)))
    f = model.fit(y[:n]).predict_one_step(y)[n:]
    a = y[n:]
    row = dict(dataset=name, m=m, seed=seed, stage=stage)
    for h_name, h in cfg["horizons"].items():
        row[f"{h_name} MSE"] = np.mean((a[:h] - f[:h]) ** 2)
        row[f"{h_name} MAD"] = np.mean(np.abs(a[:h] - f[:h]))
    return row


def main():
    from run_extension_experiments import EXPERIMENTS
    jobs = [(name, m, s, stage) for stage in ("validation", "test")
            for name in EXPERIMENTS for m in GRID for s in SEEDS]
    with ProcessPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(run_one, jobs, chunksize=2))
    df = pd.DataFrame(rows)
    df.to_csv(results_path("results_margin_tuning.csv"), index=False)
    report(df, EXPERIMENTS)


def report(df, experiments):
    val = df[df.stage == "validation"].groupby(["dataset", "m"])[["MSE", "MAD"]].agg(["mean", "std"])
    test = df[df.stage == "test"]
    print("\n===== Step 1: validation MSE (last 20 % of the training data), mean over 10 seeds =====")
    chosen = {}
    for name in experiments:
        v = val.loc[name][("MSE", "mean")]
        chosen[name] = float(v.idxmin())
        scale = 1e5 if name == "Exchange rate" else 1.0
        cells = "  ".join(f"m={m:<4} {v[m] * scale:9.4g}" for m in GRID)
        print(f"{name:13} {cells}   -> chosen m = {chosen[name]}")
    print("(exchange-rate MSE x 1e5)")

    print("\n===== Step 2: test errors, mean over 10 seeds (chosen m marked with *) =====")
    for name, cfg in experiments.items():
        cols = [f"{h} {k}" for h in cfg["horizons"] for k in ("MSE", "MAD")]
        means = test[test.dataset == name].groupby("m")[cols].mean()
        scale = {c: (1e5 if name == "Exchange rate" and c.endswith("MSE") else 1.0) for c in cols}
        print(f"\n{name}" + (" (MSE x 1e5)" if name == "Exchange rate" else ""))
        print(f"{'m':>8}" + "".join(f"{c:>24}" for c in cols))
        for m in GRID:
            mark = "*" if m == chosen[name] else " "
            print(f"{m:>7}{mark}" + "".join(f"{means.loc[m, c] * scale[c]:>24.5g}" for c in cols))


if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        from run_extension_experiments import EXPERIMENTS
        report(pd.read_csv(results_path("results_margin_tuning.csv")), EXPERIMENTS)
    else:
        main()
