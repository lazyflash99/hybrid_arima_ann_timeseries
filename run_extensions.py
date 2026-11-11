"""
Compares the extensions in extensions.py with the current models (ARIMA, ANN, Zhang hybrid, Extension)
on the three data sets. Every model with a neural network is run with SEEDS different starting weights;
tables show the mean (and std) of MSE / MAD over the seeds, one-step-ahead, on the test horizons of the paper.
Results are also written to results_extensions.csv.   Usage:  python run_extensions.py
"""
import numpy as np
from config import results_path
import pandas as pd

from arima_model import AutoARIMA
from ann_model import NeuralNetwork
from hybrid_model import HybridModel
from extension_model import ExtensionHybrid
from extensions import (SkipCombination, MultiplicativeHybrid, DecompositionHybrid,
                        GatedHybrid, JointARNN, WeightedCombination, ValidatedCombination)
from run_extension_experiments import EXPERIMENTS

SEEDS = range(10)
CURRENT = ["ARIMA", "ANN", "Zhang hybrid", "Extension"]


def models(cfg, seed):
    p, q = cfg["ann"]
    ext = dict(cfg["ext_kwargs"], seed=seed)
    return {
        "ANN": lambda: NeuralNetwork(p, q, seed=seed),
        "Zhang hybrid": lambda: HybridModel(AutoARIMA(verbose=False), p, q, seed=seed),
        "Extension": lambda: ExtensionHybrid(AutoARIMA(verbose=False), **ext),
        "Extension + skip": lambda: ExtensionHybrid(AutoARIMA(verbose=False), output="skip", **ext),
        "Extension + shortcut": lambda: ExtensionHybrid(AutoARIMA(verbose=False), shortcut=True, **ext),
        "Extension + margin": lambda: ExtensionHybrid(AutoARIMA(verbose=False), scale_margin=0.2, **ext),
        "E1 Skip combination": lambda: SkipCombination(ext["n_series_lags"], ext["n_residual_lags"],
                                                       ext["n_hidden"], seed=seed),
        "E2 Multiplicative": lambda: MultiplicativeHybrid(p, q, seed=seed),
        "E3 Wavelet decomposition": lambda: DecompositionHybrid(p, q, seed=seed),
        "E4 Gated": lambda: GatedHybrid(p, q, seed=seed),
        "E5 Joint AR+ANN": lambda: JointARNN(p, q, seed=seed),
        "E6 Weighted combination": lambda: WeightedCombination(p, q, seed=seed),
        "E7 Validated combination": lambda: ValidatedCombination(p, q, seed=seed),
    }


def errors(actual, forecast, horizons):
    out = {}
    for h_name, h in horizons.items():
        d = actual[:h] - forecast[:h]
        out[(h_name, "MSE")] = np.mean(d ** 2)
        out[(h_name, "MAD")] = np.mean(np.abs(d))
    return out


def run(name, cfg):
    y = cfg["load"]()
    n = cfg["n_train"]
    y_train, y_test = y[:n], y[n:]
    f_arima = AutoARIMA(verbose=False).fit(y_train).predict_one_step(y)[n:]
    rows = [{"dataset": name, "model": "ARIMA", "seed": 0, **errors(y_test, f_arima, cfg["horizons"])}]
    for seed in SEEDS:
        for model_name, make in models(cfg, seed).items():
            f = make().fit(y_train).predict_one_step(y)[n:]
            rows.append({"dataset": name, "model": model_name, "seed": seed, **errors(y_test, f, cfg["horizons"])})
        print(f"   {name}: seed {seed} done", flush=True)
    return rows


def report(df):
    """df: one row per (dataset, model, seed), error columns named '<horizon> MSE' / '<horizon> MAD'."""
    summary = {}
    for name, part in df.groupby("dataset", sort=False):
        cells = [c for c in part.columns if c.endswith((" MSE", " MAD")) and part[c].notna().any()]
        scale = 1e5 if name == "Exchange rate" else 1.0
        grouped = part.groupby("model", sort=False)[cells]
        mean, std = grouped.mean(), grouped.std().fillna(0.0)
        print(f"\n===== {name} (mean ± std over {len(SEEDS)} seeds{', MSE x 1e5' if scale != 1 else ''}) =====")
        print(f"{'':26}" + "".join(f"{c:>24}" for c in cells))
        for model in mean.index:
            vals = []
            for c in cells:
                k = scale if c.endswith("MSE") else 1.0
                vals.append(f"{mean.loc[model, c] * k:.4g} ±{std.loc[model, c] * k:.2g}")
            print(f"{model:26}" + "".join(f"{v:>24}" for v in vals))
        best_current = mean.loc[CURRENT].min()
        for model in mean.index:
            if model in CURRENT:
                continue
            tally = summary.setdefault(model, [0, 0, 0])
            tally[0] += int((mean.loc[model] < mean.loc["Zhang hybrid"]).sum())
            tally[1] += int((mean.loc[model] < best_current).sum())
            tally[2] += len(cells)
    print("\n===== Summary: cells (data set x horizon x MSE/MAD) where the extension's mean error is lower =====")
    print(f"{'':26}{'vs Zhang hybrid':>18}{'vs best current model':>24}")
    for model, (vz, vb, tot) in summary.items():
        print(f"{model:26}{f'{vz}/{tot}':>18}{f'{vb}/{tot}':>24}")


if __name__ == "__main__":
    import sys
    if "--report" not in sys.argv:                      # --report: rebuild the tables from the saved CSV
        all_rows = []
        for name, cfg in EXPERIMENTS.items():
            all_rows += run(name, cfg)
        df = pd.DataFrame(all_rows)
        df.columns = [" ".join(c) if isinstance(c, tuple) else c for c in df.columns]
        df.to_csv(results_path("results_extensions.csv"), index=False)
    report(pd.read_csv(results_path("results_extensions.csv")))
