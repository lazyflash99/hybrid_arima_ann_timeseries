"""
Exports the results as JSON for the project website, into website/data/: one file per dataset (series, forecasts,
ARIMA forecast, neural-network correction, errors) and one file per result table. The website is kept apart from
the code; nothing in the code reads these files.   Usage:  python website/export_json.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arima_model as am  # noqa: E402
from config import WEBSITE_DATA_DIR, results_path  # noqa: E402
from data_loader import DATASETS  # noqa: E402


def out_path(name):
    os.makedirs(WEBSITE_DATA_DIR, exist_ok=True)
    return os.path.join(WEBSITE_DATA_DIR, name)


def clean(a):
    return [None if not np.isfinite(v) else round(float(v), 6) for v in np.asarray(a, float)]


def table(csv, out):
    path = results_path(csv)
    if os.path.exists(path):
        pd.read_csv(path).replace({np.nan: None}).to_json(out_path(out), orient="records", double_precision=6)
        return True
    return False


def main():
    fixed = dict(np.load(results_path("forecasts_fixed.npz"))) if os.path.exists(results_path("forecasts_fixed.npz")) else {}
    rolling = dict(np.load(results_path("forecasts_rolling.npz"))) if os.path.exists(results_path("forecasts_rolling.npz")) else {}
    written = 0
    for name, d in DATASETS.items():
        keys = [k for k in fixed if k.startswith(name + "|")]
        if not keys:
            continue
        y, n = d["load"](), d["n_train"]
        am.DEFAULT_SUBSET = d["arima_subset"]
        L = am.AutoARIMA(verbose=False).fit(y[:n]).predict_one_step(y)[n:]
        models = {}
        for proto, store in (("fixed", fixed), ("rolling", rolling)):
            for m in sorted({k.split("|")[1] for k in store if k.startswith(name + "|")}):
                fs = np.array([store[k] for k in store if k.startswith(f"{name}|{m}|")])
                f = fs.mean(0)
                models.setdefault(proto, {})[m] = dict(forecast=clean(f), forecast_seed_std=clean(fs.std(0)),
                                                       nn_correction=clean(f - L), final_error=clean(y[n:] - f))
        out = dict(name=name, source=d["source"], frequency=d["frequency"], n_train=n, transform=d["transform"], series=clean(y),
                   actual_test=clean(y[n:]), arima_forecast=clean(L), arima_error=clean(y[n:] - L), models=models)
        with open(out_path(name.replace(" ", "_").lower() + ".json"), "w", encoding="utf-8") as f:
            json.dump(out, f)
        written += 1
    tables = {"stationarity.csv": "stationarity.json", "validation_overview.csv": "validation_overview.json", "validation_summary.csv": "validation_summary.json", "validation_dm.csv": "validation_dm.json",
              "validation_mcs.csv": "validation_mcs.json", "validation_tuning.csv": "validation_tuning.json", "residual_diagnostics.csv": "residual_diagnostics.json",
              "nonlinearity_vs_improvement.csv": "nonlinearity_vs_improvement.json", "decomposition_r2.csv": "decomposition_r2.json",
              "grid_results.csv": "grid_results.json", "grid_single_models.csv": "grid_single_models.json",
              "grid_leaderboard.csv": "grid_leaderboard.json", "grid_when.csv": "grid_when.json"}
    n_tables = sum(table(c, j) for c, j in tables.items())
    print(f"wrote {written} dataset files and {n_tables} tables to {WEBSITE_DATA_DIR}")


if __name__ == "__main__":
    main()
