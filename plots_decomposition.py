"""
Forecast decomposition plots on the test part (fixed protocol, seed 0) for Zhang's hybrid, the Extension and the
best-performing variant (lowest validation MSE among E1-E7; the test part is not used to choose it).
Files: <dataset>_zhang.png, <dataset>_extension.png, <dataset>_best_variant.png.
The neural-network correction is the final forecast minus the ARIMA forecast. Panels:
  (1) actual values, ARIMA forecast and final forecast
  (2) ARIMA error and the neural-network correction over time
  (3) correction against ARIMA error, with R^2 (how much of the ARIMA error the correction explains)
  (4) error left after the correction
  (5) change in squared error at each step due to the correction (below 0 = the correction helped)
Usage:  python plots_decomposition.py [--quick]
Writes figures/2_forecast_decomposition/*.png and results/3_residuals_and_decomposition/2_correction_fit.csv.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import arima_model as am
from config import MODES, WORKERS, figure_path, results_path
from data_loader import DATASETS
from validation import build, registry_params

ACTUAL, LINEAR, FINAL, CORRECTION, ERROR = "#0b0b0b", "#1f4e9c", "#d95f02", "#0f8a5f", "#52514e"


def best_variant(name):
    t = pd.read_csv(results_path("validation_tuning.csv"))
    t = t[(t.dataset == name) & t.model.isin([f"E{i}" for i in range(1, 8)]) & t.val_mse.notna()]
    return t.sort_values("val_mse").iloc[0]["model"] if len(t) else None


def r_squared(e, n):
    ss = np.sum((e - e.mean()) ** 2)
    return float(1 - np.sum((e - n) ** 2) / ss) if ss > 0 else np.nan


def decompose(name, model):
    d = DATASETS[name]
    am.DEFAULT_SUBSET = d["arima_subset"]
    y, n = d["load"](), d["n_train"]
    arima = am.AutoARIMA(verbose=False).fit(y[:n])
    L = arima.predict_one_step(y)[n:]
    f = build(model, d, registry_params(model, d), 0).fit(y[:n]).predict_one_step(y)[n:]
    return y[n:], L, f


ROLES = {"zhang": "Zhang's hybrid", "extension": "Extension"}


def figure(job):
    name, model, role = job
    try:
        a, L, f = decompose(name, model)
    except Exception as ex:
        print(f"{name} {model}: not plotted ({type(ex).__name__}: {ex})")
        return None
    title = ROLES.get(role, f"best-performing variant ({model}, lowest validation error among E1-E7)")
    e, N = a - L, f - L
    x = np.arange(1, len(a) + 1)
    fig, ax = plt.subplots(5, 1, figsize=(8, 10))
    ax[0].plot(x, a, color=ACTUAL, lw=1.6, label="actual")
    ax[0].plot(x, L, color=LINEAR, lw=1.1, label="ARIMA forecast")
    ax[0].plot(x, f, color=FINAL, lw=1.1, ls="--", label="final forecast")
    ax[0].legend(frameon=False, fontsize=8, ncol=3); ax[0].set_title(f"{name}: {title}", fontsize=10, loc="left")
    ax[1].plot(x, e, color=LINEAR, lw=1, label="ARIMA error (actual - ARIMA forecast)"); ax[1].plot(x, N, color=CORRECTION, lw=1, ls="--", label="neural-network correction")
    ax[1].legend(frameon=False, fontsize=8, ncol=2)
    r2 = r_squared(e, N)
    ax[2].scatter(e, N, s=10, color=CORRECTION); lim = [min(e.min(), N.min()), max(e.max(), N.max())]
    ax[2].plot(lim, lim, color=ERROR, lw=.7); ax[2].set_title(f"neural-network correction (y) against ARIMA error (x), R squared = {r2:.3f}", fontsize=9, loc="left")
    ax[3].plot(x, a - f, color=ERROR, lw=1); ax[3].axhline(0, color=ERROR, lw=.5); ax[3].set_title("error left after the correction (actual - final forecast)", fontsize=9, loc="left")
    gain = (a - f) ** 2 - e ** 2
    ax[4].bar(x, gain, color=[CORRECTION if g < 0 else FINAL for g in gain])
    ax[4].set_title("change in squared error due to the correction (below 0 = better)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(figure_path("decomposition", f"{name}_{role}".replace(" ", "_").lower() + ".png"), dpi=110)
    plt.close(fig)
    return dict(dataset=name, model=model, r2_correction_vs_arima_error=r2, mse_arima=float(np.mean(e ** 2)), mse_model=float(np.mean((a - f) ** 2)),
                share_steps_improved=float(np.mean(gain < 0)))


def main(mode="full"):
    names = MODES[mode]["datasets"] or list(DATASETS)
    jobs = [(n, "Zhang", "zhang") for n in names] + [(n, "Extension", "extension") for n in names]
    jobs += [(n, best_variant(n), "best_variant") for n in names if best_variant(n)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = [r for r in pool.map(figure, jobs) if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(results_path("decomposition_r2.csv"), index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main("quick" if "--quick" in sys.argv else "full")
